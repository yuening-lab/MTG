import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from gnn.aggregators import *
from gnn.utils import *
from gnn.modules import *
import time
import math
import random
import itertools
import collections


class CompGCN(nn.Module):
    def __init__(self, h_dim, num_ents, num_rels, sentence_size,  dropout=0, seq_len=10, maxpool=1, use_edge_node=0, use_gru=1, attn=''):
        super().__init__()
        self.h_dim = h_dim
        self.sentence_size = sentence_size
        self.text_embedding_size = h_dim
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # initialize rel and ent embedding
        self.rel_embeds = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.ent_embeds = nn.Parameter(torch.Tensor(num_ents, h_dim))

        self.word_embeds = None
        self.global_emb = None
        self.ent_map = None
        self.rel_map = None
        self.word_graph_dict = None
        self.graph_dict = None
        self.sentence_embeddings_dict = None
        self.aggregator= aggregator_event(h_dim*2, dropout, num_ents, num_rels, self.sentence_size, self.text_embedding_size, seq_len, maxpool, attn)
        if use_gru:
            # self.encoder = nn.GRU(3*h_dim, h_dim, batch_first=True)
            self.encoder = nn.GRU(4*h_dim, 2*h_dim, batch_first=True)
        else:
            # self.encoder = nn.RNN(3*h_dim, h_dim, batch_first=True)
            self.encoder = nn.RNN(4*h_dim, 2*h_dim, batch_first=True)
        # self.linear_r = nn.Linear(h_dim, self.num_rels)
        self.linear_r = nn.Linear(h_dim*2, 1)
        self.threshold = 0.5
        self.out_func = torch.sigmoid
        self.criterion = nn.BCELoss()
        self.init_weights()


    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, t_list, ent_memory, rel_memory):
        pred, idx, _ = self.__get_pred_embeds(t_list, ent_memory, rel_memory)
        # loss = self.criterion(pred, true_prob_r[idx])
        # return loss
        return pred

    def __get_pred_embeds(self, t_list, ent_memory, rel_memory):
        sorted_t, idx = t_list.sort(0, descending=True)
        # print (t_list)
        embed_seq_tensor, len_non_zero = self.aggregator(sorted_t, ent_memory, rel_memory, self.ent_embeds,
                            self.rel_embeds, self.graph_dict, self.sentence_embeddings_dict)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(embed_seq_tensor,
                                                               len_non_zero,
                                                               batch_first=True)
        _, feature = self.encoder(packed_input)
        feature = feature.squeeze(0)

        if torch.cuda.is_available():
            feature = torch.cat((feature, torch.zeros(len(t_list) - len(feature), feature.size(-1)).cuda()), dim=0)
        else:
            feature = torch.cat((feature, torch.zeros(len(t_list) - len(feature), feature.size(-1))), dim=0)

        pred = self.linear_r(feature)
        pred = self.out_func(pred)
        # print (pred)
        # quit()
        return pred, idx, feature

    def predict(self, t_list, true_prob_r):
        pred, idx, feature = self.__get_pred_embeds(t_list)
        pred = pred.float()
        # print (pred.type())
        if true_prob_r is not None:
            loss = self.criterion(pred, true_prob_r[idx].float())
        else:
            loss = None
        return loss, pred, feature

    def evaluate(self, t, true_prob_r):
        loss, pred, _ = self.predict(t, true_prob_r)
        # print(true_prob_r)
        # print(pred)
        # quit()
        prob_rel = pred.view(-1)
        sorted_prob_rel, prob_rel_idx = prob_rel.sort(0, descending=True)
        # print(sorted_prob_rel)
        # quit()
        if torch.cuda.is_available():
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()).cuda())
        else:
            sorted_prob_rel = torch.where(sorted_prob_rel > self.threshold, sorted_prob_rel, torch.zeros(sorted_prob_rel.size()))
        nonzero_prob_idx = torch.nonzero(sorted_prob_rel,as_tuple=False).view(-1)
        nonzero_prob_rel_idx = prob_rel_idx[:len(nonzero_prob_idx)]

        # target
        true_prob_r = true_prob_r.view(-1)
        nonzero_rel_idx = torch.nonzero(true_prob_r,as_tuple=False) # (x,1)->(x)
        sorted_true_rel, true_rel_idx = true_prob_r.sort(0, descending=True)
        nonzero_true_rel_idx = true_rel_idx[:len(nonzero_rel_idx)]
        return nonzero_true_rel_idx, nonzero_prob_rel_idx, loss, true_prob_r, prob_rel
