import logging
import numpy as np
import torch
import torch.nn as nn
import math
from collections import defaultdict

from utils.utils import MergeLayer, get_sentence_embeddings
from modules.cache import Cache
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.cache_updater import get_cache_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode
from gnn.models import CompGCN


class MTG_model(torch.nn.Module):
  def __init__(self, neighbor_finder, dim, num_nodes, num_edges, device, max_pool, graph_dict, sentence_embeddings_dict, hist_wind=7, n_layers=2,
               n_heads=2, dropout=0.1, use_cache=True,
               message_dimension=100,
               cache_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="mean",
               cache_updater_type="gru",
               dyrep=False):
    super(MTG_model, self).__init__()

    self.dim = dim
    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.max_pool = max_pool
    self.hist_wind = hist_wind
    self.logger = logging.getLogger(__name__)

    node_features = np.random.rand(num_nodes, self.dim)
    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    self.n_node_features = self.node_raw_features.shape[1]
    self.edge_raw_features = torch.nn.Parameter(torch.zeros(num_edges, self.dim)).to(device)
    self.n_nodes = num_nodes
    self.n_edge_features = dim
    self.num_rels = num_edges
    self.embedding_dimension = dim
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type

    self.use_cache = use_cache
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.entity_cache = None

    self.mean_time_shift_src = mean_time_shift_src
    self.std_time_shift_src = std_time_shift_src
    self.mean_time_shift_dst = mean_time_shift_dst
    self.std_time_shift_dst = std_time_shift_dst

    if self.use_cache:
      self.entity_cache_dimension = dim
      raw_message_dimension = 2 * self.entity_cache_dimension + self.n_edge_features + \
                              self.time_encoder.dimension
      message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
      self.entity_cache = Cache(n_nodes=self.n_nodes,
                           cache_dimension=self.entity_cache_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device)
      self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)
      self.entity_cache_updater = get_cache_updater(module_type=cache_updater_type,
                                               cache=self.entity_cache,
                                               message_dimension=message_dimension,
                                               cache_dimension=self.entity_cache_dimension,
                                               device=device)

    self.embedding_module_type = embedding_module_type

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=self.node_raw_features,
                                                 edge_features=self.edge_raw_features,
                                                 cache=self.entity_cache,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_cache=use_cache,
                                                 n_neighbors=self.n_neighbors)

    if self.use_cache:
      self.rel_raw_message_dimension = 2 * self.entity_cache_dimension + dim + \
                              self.time_encoder.dimension
      self.rel_cache = Cache(n_nodes=num_edges,
                           cache_dimension=self.entity_cache_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.rel_message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device)
      self.rel_message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=self.rel_raw_message_dimension,
                                                   message_dimension=self.rel_raw_message_dimension)
      self.rel_cache_updater = get_cache_updater(module_type=cache_updater_type,
                                               cache=self.rel_cache,
                                               message_dimension=self.rel_raw_message_dimension,
                                               cache_dimension=self.entity_cache_dimension,
                                               device=device)

    self.sentence_size = 384
    self.text_embedding_size = dim
    self.textEmbeddingLayer = torch.nn.Linear(self.sentence_size, self.text_embedding_size)
    self.memory = Memory(n_nodes=self.n_nodes, n_rels=self.num_rels, memory_dimension=dim, input_dimension=dim, device=device)
    # MLP to compute probability on an edge given two node embeddings
    self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                     self.n_node_features, 1)

    self.sentence_embeddings_dict = sentence_embeddings_dict
    self.gnn_model = CompGCN(h_dim=dim*2, num_ents=self.n_nodes, num_rels=num_edges, sentence_size=self.sentence_size,  dropout=dropout, seq_len=hist_wind, maxpool=1, use_edge_node=0, use_gru=1, attn='')
    self.gnn_model.graph_dict = graph_dict
    self.gnn_model.sentence_embeddings_dict = sentence_embeddings_dict
    # self.init_weights()

  def init_weights(self):
      for p in self.parameters():
          if p.data.ndimension() >= 2:
              nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
          else:
              stdv = 1. / math.sqrt(p.size(0))
              p.data.uniform_(-stdv, stdv)

  def compute_temporal_embeddings(self, source_nodes, destination_nodes, edge_times,
                                  edge_idxs, story_ids, n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """
    source_nodes = np.array(source_nodes)
    destination_nodes = np.array(destination_nodes)
    edge_times = np.array(edge_times)
    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times])
    rels = np.concatenate([edge_idxs])

    cache = None
    time_diffs = None
    if self.use_cache:
      cache = self.entity_cache.get_cache(list(range(self.n_nodes)))
      last_update = self.entity_cache.last_update
      rel_cache = self.rel_cache.get_cache(list(range(self.num_rels)))

      ### Compute differences between the time the cache of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs],
                             dim=0)

    # Compute the embeddings using the embedding module

    if len(edge_times) > 0:
        node_embedding = self.embedding_module.compute_embedding(cache=cache,
                                                                 source_nodes=nodes,
                                                                 timestamps=timestamps,
                                                                 n_layers=self.n_layers,
                                                                 n_neighbors=n_neighbors,
                                                                 time_diffs=time_diffs)

        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples:]

        unique_nodes = np.unique(nodes)
        unique_node_embedding = self.embedding_module.compute_embedding(cache=cache,
                                                                 source_nodes=unique_nodes,
                                                                 timestamps=np.array([timestamps[0] for i in range(unique_nodes.shape[0])]),
                                                                 n_layers=self.n_layers,
                                                                 n_neighbors=n_neighbors,
                                                                 time_diffs=time_diffs)

        unique_rels = np.unique(edge_idxs)
        unique_rel_cache = self.rel_cache.get_cache(unique_rels)
        self.memory.update_memory(unique_node_embedding, unique_rel_cache, unique_nodes, unique_rels)

        unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
        unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
        unique_rels, rel_id_to_messages = self.get_rel_raw_messages(destination_nodes,
                                                                        destination_node_embedding,
                                                                        source_nodes,
                                                                        source_node_embedding,
                                                                        edge_times, edge_idxs, story_ids)

        self.entity_cache.store_raw_messages(unique_sources, source_id_to_messages)
        self.entity_cache.store_raw_messages(unique_destinations, destination_id_to_messages)
        self.rel_cache.store_raw_messages(unique_rels, rel_id_to_messages)

        self.update_entity_cache(positives, self.entity_cache.messages)
        self.update_rel_cache(rels, self.rel_cache.messages)

        self.entity_cache.clear_messages(positives)
        self.rel_cache.clear_messages(rels)

    all_nodes = np.unique(list(range(self.n_nodes)))
    all_nodes_cache = self.entity_cache.get_cache(all_nodes)
    all_nodes_memory = self.memory.get_nodes_memory(all_nodes)
    all_nodes_embeddings = torch.cat([all_nodes_cache, all_nodes_memory], dim=1)

    all_rels = np.unique(list(range(self.num_rels)))
    all_rels_cache = self.entity_cache.get_cache(all_rels)
    all_rels_memory = self.memory.get_rels_memory(all_rels)
    all_rels_embeddings = torch.cat([all_rels_cache, all_rels_memory], dim=1)


    return all_nodes_embeddings, all_rels_embeddings

  def predict(self, source_nodes, destination_nodes,  edge_idxs, edge_times, time_idx, story_ids_batch, n_neighbors=20):
    all_nodes_cache, all_rels_cache = self.compute_temporal_embeddings(
      source_nodes, destination_nodes, edge_times, edge_idxs, story_ids_batch, n_neighbors)
    t_list = torch.tensor([time_idx + self.hist_wind])
    pred = self.gnn_model(t_list, all_nodes_cache, all_rels_cache)

    return pred

  def update_entity_cache(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    # Update the cache with the aggregated messages
    self.entity_cache_updater.update_cache(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)

  def update_rel_cache(self, rels, rel_messages):
    # Aggregate messages for the same relation types
    unique_rels, unique_messages, unique_timestamps = \
      self.rel_message_aggregator.aggregate(
        rels,
        rel_messages)
    if len(unique_rels) > 0:
      unique_messages = self.rel_message_function.compute_message(unique_messages)

    # Update the cache with the aggregated messages
    self.rel_cache_updater.update_cache(unique_rels, unique_messages,
                                      timestamps=unique_timestamps)

  def get_updated_cache(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    updated_cache, updated_last_update = self.entity_cache_updater.get_updated_cache(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_cache, updated_last_update

  def get_updated_rel_cache(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_rels, unique_rel_messages, unique_timestamps = \
      self.rel_message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.rel_message_function.compute_message(unique_rel_messages)

    updated_cache, updated_last_update = self.rel_cache_updater.get_updated_cache(unique_rels,
                                                                                 unique_rel_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_rel_cache, updated_rel_last_update

  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    # rel_cache = self.edge_raw_features[edge_idxs]
    rel_cache = self.rel_cache.get_cache(edge_idxs)

    source_cache = source_node_embedding
    destination_cache = destination_node_embedding

    source_time_delta = edge_times - self.entity_cache.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    source_message = torch.cat([source_cache, destination_cache, rel_cache,
                                source_time_delta_encoding],
                               dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages


  def get_rel_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs, story_ids):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    source_cache = source_node_embedding
    destination_cache = destination_node_embedding

    source_time_delta = edge_times - self.entity_cache.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    sentence_embeddings = get_sentence_embeddings(self.sentence_embeddings_dict, story_ids, self.sentence_size, device=self.device)
    text_embeddings = self.textEmbeddingLayer(sentence_embeddings)
    rel_message = torch.cat([source_cache, destination_cache, text_embeddings,
                                source_time_delta_encoding], dim=1)
    messages = defaultdict(list)
    unique_rels = np.unique(source_nodes)

    for i in range(len(edge_idxs)):
      messages[edge_idxs[i]].append((rel_message[i], edge_times[i]))

    return unique_rels, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
