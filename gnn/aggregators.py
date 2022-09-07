import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from gnn.utils import *
from gnn.propagations import *
from gnn.modules import *
from utils.utils import get_sentence_embeddings
import time
from itertools import groupby

class aggregator_event(nn.Module):
    def __init__(self, h_dim, dropout, num_nodes, num_rels, sentence_size, text_embedding_size,  seq_len=10, maxpool=1, attn=''):
        super().__init__()
        self.h_dim = h_dim  # feature
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.maxpool = maxpool

        self.sentence_size = sentence_size
        self.text_embedding_size = text_embedding_size

        self.textEmbeddingLayer = torch.nn.Linear(sentence_size, text_embedding_size)

        out_feat = int(h_dim // 2)
        self.re_aggr1 = CompGCN_dg(h_dim, out_feat, h_dim, out_feat, sentence_size, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        self.re_aggr2 = CompGCN_dg(out_feat, h_dim, out_feat, h_dim, sentence_size, True, F.relu, self_loop=True, dropout=dropout) # to be defined
        if attn == 'add':
            self.attn = Attention(h_dim, 'add')
        elif attn == 'dot':
            self.attn = Attention(h_dim, 'dot')
        else:
            self.attn = Attention(h_dim, 'general')

    def forward(self, t_list, ent_memory, rel_memory, ent_embeds, rel_embeds, graph_dict, sentence_embeddings_dict):
        times = list(graph_dict.keys())
        times.sort(reverse=False)  # 0 to future
        time_unit = times[1] - times[0]
        time_list = []
        len_non_zero = []
        nonzero_idx = torch.nonzero(t_list, as_tuple=False).view(-1)
        t_list = t_list[nonzero_idx]  # usually no duplicates
        for tim in t_list:
            length = times.index(tim)
            if (self.seq_len) <= length:
                time_list.append(torch.LongTensor(
                    times[length - self.seq_len:length]))
                len_non_zero.append(self.seq_len)
            else:
                time_list.append(torch.LongTensor(times[:length]))
                len_non_zero.append(length)

        unique_t = torch.unique(torch.cat(time_list))
        t_idx = list(range(len(unique_t)))
        time_to_idx = dict(zip(unique_t.cpu().numpy(), t_idx))
        # entity graph
        g_list = [graph_dict[tim.item()] for tim in unique_t]
        batched_g = dgl.batch(g_list)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batched_g = batched_g.to(device) # torch.device('cuda:0')
        batched_g.ndata['h']= torch.cat([ent_embeds[batched_g.ndata['id']].view(-1, ent_embeds.shape[1]),
                                        ent_memory[batched_g.ndata['id']].view(-1, ent_embeds.shape[1])], dim=1)
        if torch.cuda.is_available():
            type_data = batched_g.edata['type'].cuda()
        else:
            type_data = batched_g.edata['type']

        story_ids = batched_g.edata['sid'].tolist()
        s_embeddings = get_sentence_embeddings(sentence_embeddings_dict, story_ids, self.sentence_size, device)
        batched_g.edata['s_h'] = s_embeddings
        batched_g.edata['e_h'] = torch.cat([rel_embeds.index_select(0, type_data), rel_memory.index_select(0, type_data)], dim=1)
        self.re_aggr1(batched_g, False)
        self.re_aggr2(batched_g, False)

        # cpu operation for nodes
        g_node_embs = batched_g.ndata.pop('h').data.cpu()
        g_node_ids = batched_g.ndata['id'].view(-1)
        max_query_ent = 0
        num_nodes = len(g_node_ids)
        c_g_node_ids = g_node_ids.data.cpu().numpy()
        c_unique_ent_id = list(set(c_g_node_ids))
        ent_gidx_dict = {} # entid: [[gidx],[word_idx]]

        # cpu operation on edges
        g_edge_embs = batched_g.edata.pop('e_h').data.cpu() ####
        g_edge_types = batched_g.edata['type'].view(-1)
        num_edges = len(g_edge_types)
        max_query_rel = 0
        c_g_edge_types = g_edge_types.data.cpu().numpy()
        c_unique_type_id = list(set(c_g_edge_types))
        type_gidx_dict = {}

        # initialize a batch
        Q_mx_ent = g_node_embs.view(num_nodes , 1, self.h_dim)
        Q_mx_rel = g_edge_embs.view(num_edges , 1, self.h_dim)
        Q_mx = torch.cat((Q_mx_ent, Q_mx_rel), dim=0)
        # H_mx = torch.zeros((num_nodes + num_edges, max_query, self.h_dim))

        if torch.cuda.is_available():
            Q_mx = Q_mx.cuda()

        output = Q_mx
        batched_g.ndata['h'] = output[:num_nodes].view(-1, self.h_dim)
        batched_g.edata['e_h'] = output[num_nodes:].view(-1, self.h_dim)
        if self.maxpool == 1:
            global_node_info = dgl.max_nodes(batched_g, 'h')
            global_edge_info = dgl.max_edges(batched_g, 'e_h')
        else:
            global_node_info = dgl.mean_nodes(batched_g, 'h')
            global_edge_info = dgl.mean_edges(batched_g, 'e_h')

        global_node_info = torch.cat((global_node_info, global_edge_info), -1)
        embed_seq_tensor = torch.zeros(len(len_non_zero), self.seq_len, 2*self.h_dim)
        if torch.cuda.is_available():
            embed_seq_tensor = embed_seq_tensor.cuda()
        for i, times in enumerate(time_list):
            for j, t in enumerate(times):
                embed_seq_tensor[i, j, :] = global_node_info[time_to_idx[t.item()]]
        embed_seq_tensor = self.dropout(embed_seq_tensor)
        return embed_seq_tensor, len_non_zero
