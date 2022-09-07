from torch import nn
import torch


class CacheUpdater(nn.Module):
  def update_cache(self, unique_node_ids, unique_messages, timestamps):
    pass


class SequenceCacheUpdater(CacheUpdater):
  def __init__(self, cache, message_dimension, cache_dimension, device):
    super(SequenceCacheUpdater, self).__init__()
    self.cache = cache
    self.layer_norm = torch.nn.LayerNorm(cache_dimension)
    self.message_dimension = message_dimension
    self.device = device

  def update_cache(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return

    assert (self.cache.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update cache to time in the past"

    cache = self.cache.get_cache(unique_node_ids)
    self.cache.last_update[unique_node_ids] = timestamps

    updated_cache = self.cache_updater(unique_messages, cache)

    self.cache.set_cache(unique_node_ids, updated_cache)

  def get_updated_cache(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return self.cache.cache.data.clone(), self.cache.last_update.data.clone()

    assert (self.cache.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update cache to time in the past"

    updated_cache = self.cache.cache.data.clone()
    updated_cache[unique_node_ids] = self.cache_updater(unique_messages, updated_cache[unique_node_ids])

    updated_last_update = self.cache.last_update.data.clone()
    updated_last_update[unique_node_ids] = timestamps

    return updated_cache, updated_last_update


class GRUCacheUpdater(SequenceCacheUpdater):
  def __init__(self, cache, message_dimension, cache_dimension, device):
    super(GRUCacheUpdater, self).__init__(cache, message_dimension, cache_dimension, device)

    self.cache_updater = nn.GRUCell(input_size=message_dimension,
                                     hidden_size=cache_dimension)


class RNNCacheUpdater(SequenceCacheUpdater):
  def __init__(self, cache, message_dimension, cache_dimension, device):
    super(RNNCacheUpdater, self).__init__(cache, message_dimension, cache_dimension, device)

    self.cache_updater = nn.RNNCell(input_size=message_dimension,
                                     hidden_size=cache_dimension)


def get_cache_updater(module_type, cache, message_dimension, cache_dimension, device):
  if module_type == "gru":
    return GRUCacheUpdater(cache, message_dimension, cache_dimension, device)
  elif module_type == "rnn":
    return RNNCacheUpdater(cache, message_dimension, cache_dimension, device)
