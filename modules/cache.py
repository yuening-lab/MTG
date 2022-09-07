import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy


class Cache(nn.Module):

  def __init__(self, n_nodes, cache_dimension, input_dimension, message_dimension=None,
               device="cpu", combination_method='sum'):
    super(Cache, self).__init__()
    self.n_nodes = n_nodes
    self.cache_dimension = cache_dimension
    self.input_dimension = input_dimension
    self.message_dimension = message_dimension
    self.device = device

    self.combination_method = combination_method

    self.__init_cache__()

  def __init_cache__(self):
    """
    Initializes the cache to all zeros. It should be called at the start of each epoch.
    """
    # Treat cache as parameter so that it is saved and loaded together with the model
    self.cache = nn.Parameter(torch.zeros((self.n_nodes, self.cache_dimension)).to(self.device),
                               requires_grad=False)
    self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                    requires_grad=False)

    self.messages = defaultdict(list)

  def store_raw_messages(self, nodes, node_id_to_messages):
    for node in nodes:
      self.messages[node].extend(node_id_to_messages[node])

  def get_cache(self, node_idxs):
    return self.cache[node_idxs, :]

  def set_cache(self, node_idxs, values):
    self.cache[node_idxs, :] = values

  def get_last_update(self, node_idxs):
    return self.last_update[node_idxs]

  def backup_cache(self):
    messages_clone = {}
    for k, v in self.messages.items():
      messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

    return self.cache.data.clone(), self.last_update.data.clone(), messages_clone

  def restore_cache(self, cache_backup):
    self.cache.data, self.last_update.data = cache_backup[0].clone(), cache_backup[1].clone()

    self.messages = defaultdict(list)
    for k, v in cache_backup[2].items():
      self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

  def detach_cache(self):
    self.cache.detach_()

    # Detach all stored messages
    for k, v in self.messages.items():
      new_node_messages = []
      for message in v:
        new_node_messages.append((message[0].detach(), message[1]))

      self.messages[k] = new_node_messages

  def clear_messages(self, nodes):
    for node in nodes:
      self.messages[node] = []
