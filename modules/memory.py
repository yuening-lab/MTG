import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy


class Memory(nn.Module):

  def __init__(self, n_nodes, n_rels, memory_dimension, input_dimension,
               device="cpu", combination_method='sum'):
    super(Memory, self).__init__()
    self.n_nodes = n_nodes
    self.n_rels = n_rels
    self.memory_dimension = memory_dimension
    self.input_dimension = input_dimension
    # self.message_dimension = message_dimension
    self.device = device

    self.combination_method = combination_method

    self.W_mem_node = torch.nn.Linear(input_dimension, memory_dimension)
    self.W_mem_rel = torch.nn.Linear(input_dimension, memory_dimension)

    self.__init_memory__()

  def __init_memory__(self):
    """
    Initializes the memory to all zeros. It should be called at the start of each epoch.
    """
    # Treat memory as parameter so that it is saved and loaded together with the model
    self.time = 0
    self.entity_memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                               requires_grad=False)
    self.rel_memory = nn.Parameter(torch.zeros((self.n_rels, self.memory_dimension)).to(self.device),
                               requires_grad=False)

    # self.source_messages = defaultdict(list)
    # self.target_messages = defaultdict(list)
    # self.rel_messages = defaultdict(list)

  def detach_memory(self):
    self.entity_memory.detach_()
    self.rel_memory.detach_()
    # Detach all stored messages

  def update_memory(self, nodes_embeddings, rels_embeddings, nodes_ids , rels_ids):
      if self.time > 1:
          self.entity_memory[:]  = torch.div(self.entity_memory, (self.time+1)/self.time)
          self.rel_memory[:]  = torch.div(self.rel_memory ,(self.time+1)/self.time)
      nodes_new_memory = self.W_mem_node(nodes_embeddings)
      # print (self.entity_memory[nodes_ids].shape)
      # print (torch.div(nodes_new_memory,(self.time+1)).shape)
      self.entity_memory[nodes_ids] += torch.div(nodes_new_memory,(self.time+1))
      rels_new_memory = self.W_mem_rel(rels_embeddings)
      self.rel_memory[rels_ids] += torch.div(rels_new_memory,(self.time+1))
      self.time += 1

  def get_nodes_memory(self, nodes_ids):
      return self.entity_memory[nodes_ids]


  def get_rels_memory(self, rels_ids):
      return self.rel_memory[rels_ids]





  # def store_source_messages(self, nodes, node_id_to_messages):
  #   for node in nodes:
  #     self.source_messages[node].extend(node_id_to_messages[node])
  # def store_target_messages(self, nodes, node_id_to_messages):
  #   for node in nodes:
  #     self.target_messages[node].extend(node_id_to_messages[node])
  # def store_target_messages(self, nodes, node_id_to_messages):
  #   for rel in rels:
  #     self.target_messages[rel].extend(rel_id_to_messages[node])
  #
  # def get_source_memory(self, node_idxs):
  #   return self.source_memory[node_idxs, :]
  # def get_target_memory(self, node_idxs):
  #   return self.target_memory[node_idxs, :]
  # def get_rel_memory(self, node_idxs):
  #   return self.rel_memory[node_idxs, :]
  #
  # def set_source_memory(self, node_idxs, values):
  #   self.source_memory[node_idxs, :] = values
  # def set_target_memory(self, node_idxs, values):
  #   self.target_memory[node_idxs, :] = values
  # def set_rel_memory(self, edge_idxs, values):
  #   self.rel_memory[edge_idxs, :] = values
  #
  # def get_last_update(self, node_idxs):
  #   return self.last_update[node_idxs]
  #
  # def backup_memory(self):
  #   messages_clone = {}
  #   for k, v in self.messages.items():
  #     messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]
  #
  #   return self.memory.data.clone(), self.last_update.data.clone(), messages_clone
  #
  # def restore_source_memory(self, memory_backup):
  #   self.source_memory.data, self.source_last_update.data = memory_backup[0].clone(), memory_backup[1].clone()
  #
  #   self.source_messages = defaultdict(list)
  #   for k, v in memory_backup[2].items():
  #     self.source_messages[k] = [(x[0].clone(), x[1].clone()) for x in v]
  # def restore_target_memory(self, memory_backup):
  #   self.target_memory.data, self.target_last_update.data = memory_backup[0].clone(), memory_backup[1].clone()
  #
  #   self.target_messages = defaultdict(list)
  #   for k, v in memory_backup[2].items():
  #     self.target_messages[k] = [(x[0].clone(), x[1].clone()) for x in v]
  # def restore_rel_memory(self, rel_memory_backup):
  #   self.rel_memory.data, self.rel_last_update.data = memory_backup[0].clone(), memory_backup[1].clone()
  #
  #   self.rel_messages = defaultdict(list)
  #   for k, v in memory_backup[2].items():
  #     self.rel_messages[k] = [(x[0].clone(), x[1].clone()) for x in v]
  #
  # def detach_memory(self):
  #   self.source_memory.detach_()
  #   self.target_memory.detach_()
  #   self.memory.detach_()
  #   # Detach all stored messages
  #   for k, v in self.source_messages.items():
  #     new_node_messages = []
  #     for message in v:
  #       new_node_messages.append((message[0].detach(), message[1]))
  #
  #     self.source_messages[k] = new_node_messages
  #
  #   for k, v in self.target_messages.items():
  #     new_node_messages = []
  #     for message in v:
  #       new_node_messages.append((message[0].detach(), message[1]))
  #
  #     self.target_messages[k] = new_node_messages
  #
  #   for k, v in self.rel_messages.items():
  #     new_rel_messages = []
  #     for message in v:
  #       new_rel_messages.append((message[0].detach(), message[1]))
  #
  #     self.rel_messages[k] = new_rel_messages
  #
  # def clear_messages(self, nodes):
  #   for node in self.source_messages:
  #     self.source_messages[node] = []
  #   for node in self.target_messages:
  #     self.target_messages[node] = []
  #   for rel in self.rel_messages:
  #     self.rel_messages[rel] = []
