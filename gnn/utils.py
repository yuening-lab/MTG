import numpy as np
import os
from math import log
import scipy.sparse as sp

import dgl
import torch
import pickle
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, recall_score, precision_score, fbeta_score, hamming_loss, zero_one_loss, precision_score, roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import jaccard_score

 
def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

 
def get_data_with_t(data, time, pred_wind):
    target_times = [i for i in range(time, time+pred_wind)]
    # triples = []
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] in target_times]
    return np.array(triples)


def get_data_idx_with_t_r(data, t,r):
    for i, quad in enumerate(data):
        if quad[3] == t and quad[1] == r:
            return i
    return None

 
def load_quadruples(inPath, lead_time, fileName, fileName2=None, fileName3=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3]) 
            if time < lead_time-1:
                continue
            quadrupleList.append([head, rel, tail, time-(lead_time-1)])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)

    if fileName3 is not None:
        with open(os.path.join(inPath, fileName3), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()
    
    return np.asarray(quadrupleList), np.asarray(times)
 
'''
Customized collate function for Pytorch data loader
'''
 
def collate_4(batch):
    batch_data = [item[0] for item in batch]
    r_prob = [item[1] for item in batch]
    return [batch_data, r_prob]

def collate_6(batch):
    inp0 = [item[0] for item in batch]
    inp1 = [item[1] for item in batch]
    inp2 = [item[2] for item in batch]
    inp3 = [item[3] for item in batch]
    inp4 = [item[4] for item in batch]
    inp5 = [item[5] for item in batch]
    return [inp0, inp1, inp2, inp3, inp4, inp5]


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor

def move_dgl_to_cuda(g):
    if torch.cuda.is_available():
        g.ndata.update({k: cuda(g.ndata[k]) for k in g.ndata})
        g.edata.update({k: cuda(g.edata[k]) for k in g.edata})

 
'''
Get sorted r to make batch for RNN (sorted by length)
'''
def get_sorted_r_t_graphs(t, r, r_hist, r_hist_t, graph_dict, word_graph_dict, reverse=False):
    r_hist_len = torch.LongTensor(list(map(len, r_hist)))
    if torch.cuda.is_available():
        r_hist_len = r_hist_len.cuda()
    r_len, idx = r_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(r_len,as_tuple=False))
    r_len_non_zero = r_len[:num_non_zero]
    idx_non_zero = idx[:num_non_zero]  
    idx_zero = idx[num_non_zero-1:]  
    if torch.max(r_hist_len) == 0:
        return None, None, r_len_non_zero, [], idx, num_non_zero
    r_sorted = r[idx]
    r_hist_t_sorted = [r_hist_t[i] for i in idx]
    g_list = []
    wg_list = []
    r_ids_graph = []
    r_ids = 0 # first edge is r 
    for t_i in range(len(r_hist_t_sorted[:num_non_zero])):
        for tim in r_hist_t_sorted[t_i]:
            try:
                wg_list.append(word_graph_dict[r_sorted[t_i].item()][tim])
            except:
                pass

            try:
                sub_g = graph_dict[r_sorted[t_i].item()][tim]
                if sub_g is not None:
                    g_list.append(sub_g)
                    r_ids_graph.append(r_ids) 
                    r_ids += sub_g.number_of_edges()
            except:
                continue
    if len(wg_list) > 0:
        batched_wg = dgl.batch(wg_list)
    else:
        batched_wg = None
    if len(g_list) > 0:
        batched_g = dgl.batch(g_list)
    else:
        batched_g = None
    
    return batched_g, batched_wg, r_len_non_zero, r_ids_graph, idx, num_non_zero
 
 

'''
Loss function
'''
# Pick-all-labels normalised (PAL-N)
def soft_cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax(dim=-1) # pred (batch, #node/#rel)
    pred = pred.type('torch.DoubleTensor')
    if torch.cuda.is_available():
        pred = pred.cuda()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

 
'''
Generate/get (r,t,s_count, o_count) datasets 
'''
def get_scaled_tr_dataset(num_nodes, path='../data/', dataset='india', set_name='train', seq_len=7, num_r=None):
    import pandas as pd
    from scipy import sparse
    file_path = '{}{}/tr_data_{}_sl{}_rand_{}.npy'.format(path, dataset, set_name, seq_len, num_r)
    if not os.path.exists(file_path):
        print(file_path,'not exists STOP for now')
        exit()
    else:
        print('load tr_data ...',dataset,set_name)
        with open(file_path, 'rb') as f:
            [t_data, r_data, r_hist, r_hist_t, true_prob_s, true_prob_o] = pickle.load(f)
    t_data = torch.from_numpy(t_data)
    r_data = torch.from_numpy(r_data)
    true_prob_s = torch.from_numpy(true_prob_s.toarray())
    true_prob_o = torch.from_numpy(true_prob_o.toarray())
    return t_data, r_data, r_hist, r_hist_t, true_prob_s, true_prob_o
 
'''
Empirical distribution
'''
def get_true_distributions(path, data, all_data, lead_time, pred_wind, num_nodes, num_rels, targetRels, dataset='india', set_name='train'):
    """ (# of s-related triples) / (total # of triples) """
     
    file_path = '{}{}/true_probs_{}_lt{}_pw{}.npy'.format(path, dataset, set_name, lead_time, pred_wind)
    if not os.path.exists(file_path):
        print('build true distributions...',dataset,set_name)
        time_l = list(set(data[:,-1]))
        time_l = sorted(time_l,reverse=False)
        true_prob_r = None
        for cur_t in time_l:
            triples = get_data_with_t(all_data, cur_t, pred_wind)
            true_r = np.zeros(1)
            for s,r,o in triples:
                if r in targetRels:
                    true_r[0] += 1
            if (np.sum(true_r) > 0):
                true_r = true_r / np.sum(true_r)
            if true_prob_r is None:
                true_prob_r = true_r.reshape(1, 1)
            else:
                true_prob_r = np.concatenate((true_prob_r, true_r.reshape(1, 1)), axis=0)
        with open(file_path, 'wb') as fp:
            pickle.dump([true_prob_r], fp)
    else:
        print('load true distributions...',dataset,set_name)
        with open(file_path, 'rb') as f:
            [true_prob_r] = pickle.load(f)
    true_prob_r = torch.from_numpy(true_prob_r)
    return true_prob_r

 

'''
Evaluation metrics
'''
# Label based
 
def print_eval_metrics(true_rank_l, prob_rank_l, true_prob_l, pred_prob_l, prt=True):
    true_prob_l = [x[0] for x in true_prob_l]
    pred_prob_l = [x[0] for x in pred_prob_l]
    pred_l = [x>0.5 for x in pred_prob_l]
    recall = recall_score(true_prob_l, pred_l)
    precision = precision_score(true_prob_l, pred_l)
    f1 = f1_score(true_prob_l, pred_l) 
    bac = balanced_accuracy_score(true_prob_l, pred_l)
    acc = accuracy_score(true_prob_l, pred_l)
    try:
        auc = roc_auc_score(true_prob_l, pred_prob_l)
    except ValueError:
        auc = 0.5
    hloss = hamming_loss(true_prob_l, pred_l)
    if prt:
        print("Rec : {:.4f}".format(recall))
        print("Precision : {:.4f}".format(precision))
        print("F1 : {:.4f}".format(f1))
        print("BAC : {:.4f}".format(bac))
        print("Acc : {:.4f}".format(acc))
        print("auc : {:.4f}".format(auc))
        print("hamming loss: {:.4f}".format(hloss))
    return hloss, recall, precision ,f1, bac, acc, auc

def print_hit_eval_metrics(total_ranks):
    total_ranks += 1
    mrr = np.mean(1.0 / total_ranks) 
    mr = np.mean(total_ranks)
    hits = []
    for hit in [1, 3, 10]: # , 20, 30
        avg_count = np.mean((total_ranks <= hit))
        hits.append(avg_count)
        print("Hits @ {}: {:.4f}".format(hit, avg_count))
    # print("MRR: {:.4f} | MR: {:.4f}".format(mrr,mr))
    return hits, mrr, mr

