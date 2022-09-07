import math
import logging
import time
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import json
from pathlib import Path
from tqdm import tqdm

from model.mtg import MTG_model
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder, soft_cross_entropy
from utils.data_processing import get_data, compute_time_statistics
from utils.event_data_processing import *
from utils.eval import print_eval_metrics

torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":
  ### Argument and global variables
  parser = argparse.ArgumentParser('MTG')
  parser.add_argument('-d', '--data', type=str, help='Dataset name', default='THA')
  parser.add_argument('-lt', '--leadtime', type=int, help='lead time', default=1)
  parser.add_argument('-pw', '--predwind', type=int, help='pred wind', default=1)
  parser.add_argument('-hw', '--hist_wind', type=int, help='hist window', default=7)
  parser.add_argument('--bs', type=int, default=1, help='Batch_size')
  parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
  parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
  parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
  parser.add_argument('--n_epoch', type=int, default=20, help='Number of epochs')
  parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
  parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
  parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
  parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
  parser.add_argument('--drop_out', type=float, default=0.2, help='Dropout probability')
  parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
  parser.add_argument('--dim', type=int, default=32, help='Dimensions of the node embedding')
  parser.add_argument('--backprop_every', type=int, default=128, help='Every how many batches to '
                                                                    'backprop')
  parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
    "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
  parser.add_argument('--message_function', type=str, default="identity", choices=[
    "mlp", "identity"], help='Type of message function')
  parser.add_argument('--cache_updater', type=str, default="gru", choices=[
    "gru", "rnn"], help='Type of cache updater')
  parser.add_argument('--aggregator', type=str, default="mean", help='Type of message '
                                                                          'aggregator')
  parser.add_argument('--max_pool', type=str, default=True)
  parser.add_argument('--message_dim', type=int, default=32, help='Dimensions of the messages')
  parser.add_argument('--cache_dim', type=int, default=32, help='Dimensions of the cache for '
                                                                  'each user')
  parser.add_argument('--uniform', action='store_true',
                      help='take uniform sampling from temporal neighbors')
  try:
    args = parser.parse_args()
  except:
    parser.print_help()
    sys.exit(0)

  BATCH_SIZE = args.bs
  LEAD_TIME = args.leadtime
  NUM_NEIGHBORS = args.n_degree
  NUM_NEG = 1
  NUM_EPOCH = args.n_epoch
  NUM_HEADS = args.n_head
  DROP_OUT = args.drop_out
  GPU = args.gpu
  DATA = args.data
  NUM_LAYER = args.n_layer
  LEARNING_RATE = args.lr
  NODE_DIM = args.dim
  TIME_DIM = args.dim
  # USE_MEMORY = True
  MESSAGE_DIM = args.message_dim
  MEMORY_DIM = args.cache_dim

  Path("./saved_models/").mkdir(parents=True, exist_ok=True)
  Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
  MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
  get_checkpoint_path = lambda \
      epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

  ### set up logger
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger()
  logger.setLevel(logging.DEBUG)
  Path("log/").mkdir(parents=True, exist_ok=True)
  fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
  fh.setLevel(logging.DEBUG)
  ch = logging.StreamHandler()
  ch.setLevel(logging.WARN)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  fh.setFormatter(formatter)
  ch.setFormatter(formatter)
  logger.addHandler(fh)
  logger.addHandler(ch)
  logger.info(args)

  ### Extract data for training, validation and testing
  feature_size = args.dim
  with open(f'data/{args.data}/stat.txt', 'r') as fr:
      for line in fr:
        line_split = line.split()
      num_nodes, num_rels = int(line_split[0]), int(line_split[1])

  target_rels = json.load(open(f'data/{args.data}/targetRelIds.json'))
  x_data, y_data = generate_data(f'data/{args.data}/quadruple_idx.txt', num_rels, target_rels, feature_size)
  train_data_x_list, train_data_y_list = \
  divide_data_online(x_data, y_data, lead_time = args.leadtime, pred_wind = args.predwind)
  # cuts to split each set of data to training, validation and test set
  cuts = []
  cuts.append((0, 316, 354, 392))
  cuts.append((392, 708, 746, 784))
  cuts.append((784, 1402, 1440, 1478))
  cuts.append((1478, 1784, 1822, 1860))
  cuts.append((1860, 2168, 2206, 2244))
  n_sets = len(cuts)

  all_sources, all_destinations, all_timestamps = generate_all(f'data/{args.data}/quadruple_idx.txt')

  with open(f'data/{args.data}/dg_dict.txt', 'rb') as f:
      graph_dict = pickle.load(f)

  sentence_embedding_file = f'data/{args.data}/sentence_embeddings_{args.data}.npy'
  sentence_embeddings_dict = np.load(sentence_embedding_file, allow_pickle=True)

  full_ngh_finder = get_neighbor_finder(f'data/{args.data}/quadruple_idx.txt', args.uniform, num_nodes)

  # Set device
  device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
  device = torch.device(device_string)

  # Compute time statistics
  mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(all_sources, all_destinations, all_timestamps)

  recall_list  = []
  precision_list = []
  f1_list  = []
  bac_list  = []
  hloss_list = []
  acc_list = []
  auc_list = []

  for i in range(args.n_runs):
    results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)
    y_trues_test_list = []
    y_hats_test_list = []
    for s in range(n_sets):
        model = MTG_model(neighbor_finder = full_ngh_finder, dim=feature_size,
              num_nodes=num_nodes,
              num_edges=num_rels, device=device, max_pool = args.max_pool,
              graph_dict = graph_dict, sentence_embeddings_dict = sentence_embeddings_dict, hist_wind = args.hist_wind,
              n_layers=NUM_LAYER,
              n_heads=NUM_HEADS, dropout=DROP_OUT, use_cache=True,
              message_dimension=MESSAGE_DIM, cache_dimension=MEMORY_DIM,
              embedding_module_type=args.embedding_module,
              message_function=args.message_function,
              aggregator_type=args.aggregator,
              cache_updater_type=args.cache_updater,
              n_neighbors=NUM_NEIGHBORS,
              mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst)

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        model = model.to(device)
        all_data_x, all_data_y = train_data_x_list[s], train_data_y_list[s]
        cut_1, cut_2, cut_3, cut_4 = cuts[s]
        logger.info('num of training instances: {}'.format(cut_2))

        best_hloss = float('inf')
        no_improvement = 0
        early_stopper = EarlyStopMonitor(max_round=args.patience)
        for epoch in range(NUM_EPOCH):
          ### Training
          # Reinitialize cache and memory of the model at the start of each epoch
          model.entity_cache.__init_cache__()
          model.rel_cache.__init_cache__()
          model.memory.__init_memory__()
          # Train using only training graph
          model.set_neighbor_finder(full_ngh_finder)
          m_loss = []
          logger.info('start {} epoch'.format(epoch))
          batch_idx = 0
          cut_1 = 0
          for k in tqdm(range(cut_1, cut_2-args.hist_wind, args.backprop_every)):
            loss = 0.
            num_samples = 0
            optimizer.zero_grad()
            # Custom loop to allow to perform backpropagation only every a certain number of batches
            while num_samples < args.backprop_every:
              if batch_idx >= cut_2-args.hist_wind:
                  break
              sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, story_ids_batch = \
              get_batch_data(all_data_x[batch_idx])
              t_h = batch_idx + args.hist_wind
              if t_h not in graph_dict:
                  batch_idx += 1
                  continue
              num_samples += 1
              y_true = torch.tensor([[all_data_y[batch_idx+args.hist_wind]]], requires_grad=False).float().to(device)
              y_hat = model.predict(sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, batch_idx, story_ids_batch)
              loss += criterion(y_hat, y_true)
              batch_idx += 1
            if loss > 0.:
                loss.backward(retain_graph=False)
                optimizer.step()
                m_loss.append(loss.item())

            # Detach cache and memory after 'args.backprop_every' number of batches so we don't backpropagate to
            # the start of time
            model.entity_cache.detach_cache()
            model.rel_cache.detach_cache()
            model.memory.detach_memory()
          # validation
          valid_loss = 0.
          y_hats_valid = []
          y_trues_valid = []
          for k in range(cut_2-args.hist_wind, cut_3-args.hist_wind):
              sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, story_ids_batch = \
              get_batch_data(all_data_x[k])
              t_h = k + args.hist_wind
              if t_h not in graph_dict:
                  continue
              y_true = torch.tensor([[all_data_y[k+args.hist_wind]]], requires_grad=False).float().to(device)
              y_hat = model.predict(sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, k, story_ids_batch)
              valid_loss += criterion(y_hat, y_true)
              y_hats_valid.append(y_hat.item())
              y_trues_valid.append(y_true.item())

          hloss, recall, precision ,f1, bac, acc, auc = print_eval_metrics(y_trues_valid, y_hats_valid)
          if hloss < best_hloss:
            no_improvement = 0
            best_hloss = hloss
            valid_loss = 0.
            y_hats_test = []
            y_trues_test = []
            for k in range(cut_3-args.hist_wind, cut_4-args.hist_wind):
              sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, story_ids_batch = \
              get_batch_data(all_data_x[k])
              t_h = k + args.hist_wind
              if t_h not in graph_dict:
                  continue
              y_true = torch.tensor([[all_data_y[k+args.hist_wind]]], requires_grad=False).float().to(device)
              y_hat = model.predict(sources_batch, destinations_batch, edge_idxs_batch, timestamps_batch, k, story_ids_batch)
              valid_loss += criterion(y_hat, y_true)
              y_hats_test.append(y_hat.item())
              y_trues_test.append(y_true.item())

            _,_,_,_,_,_,_ = print_eval_metrics(y_trues_test, y_hats_test)

          else:
            no_improvement += 1
            print ('no_improvement:', no_improvement)

          if no_improvement > args.patience:
            logger.info('No improvement over {} epochs, stop training'.format(args.patience))
            break

        y_trues_test_list.extend(y_trues_test)
        y_hats_test_list.extend(y_hats_test)

    test_hloss, test_recall, test_precision, test_f1, test_bac, test_acc, test_auc = \
    print_eval_metrics(y_trues_test_list, y_hats_test_list)
    recall_list.append(test_recall)
    precision_list.append(test_precision)
    f1_list.append(test_f1)
    bac_list.append(test_bac)
    acc_list.append(test_acc)
    auc_list.append(test_auc)
    hloss_list.append(test_hloss)

  print('finish training, results ....')
  # save average results
  recall_list = np.array(recall_list)
  precision_list = np.array(precision_list)
  f1_list = np.array(f1_list)
  bac_list = np.array(bac_list)
  hloss_list = np.array(hloss_list)
  acc_list = np.array(acc_list)
  auc_list = np.array(auc_list)

  recall_avg, recall_std = recall_list.mean(0), recall_list.std(0)
  precision_avg, precision_std = precision_list.mean(0), precision_list.std(0)
  f1_avg, f1_std = f1_list.mean(0), f1_list.std(0)
  bac_avg, bac_std = bac_list.mean(0), bac_list.std(0)
  acc_avg, acc_std = acc_list.mean(0), acc_list.std(0)
  hloss_avg, hloss_std = hloss_list.mean(0), hloss_list.std(0)
  auc_avg, auc_std = auc_list.mean(0), auc_list.std(0)

  print('--------------------')
  print("Rec  weighted: {:.4f}".format(recall_avg))
  print("Precision  weighted: {:.4f}".format(precision_avg))
  print("F1   weighted: {:.4f}".format(f1_avg))
  beta=2
  print("BAC  weighted: {:.4f}".format(bac_avg))
  print("Accuracy  weighted: {:.4f}".format(acc_avg))
  print("hamming loss: {:.4f}".format(hloss_avg))
  print("auc : {:.4f}".format(auc_avg))

  # save results
  result = '{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}\n'.format(recall_avg, precision_avg, f1_avg, bac_avg, acc_avg, auc_avg)
  with open('results.csv','a') as fd:
      fd.write(result)
