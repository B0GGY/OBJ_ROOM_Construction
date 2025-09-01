from sklearn.datasets import make_blobs, make_moons, make_circles
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score
from time import time
import numpy as np
from Spectral_Clustering import NJW
from Eigen_gap import ISCEIGN
from CH_based import ISC_CH
from Quality_based import ISCQuatlity
from Valgren_ISC import IncrementalSP
from ISC import NewIncrementalSPDivision
import argparse


def get_args_parser():
  parser = argparse.ArgumentParser("Project Residual Stream", add_help=False)
  parser.add_argument("--n_cluster", default=4, type=int, help="number of clusters")

  return parser


args = get_args_parser()
args = args.parse_args()

n_cluster = args.n_cluster
cluster_std=1.5
n_samples=n_cluster*20

aris_sc = []
aris_our = []
aris_valgren = []
aris_eigen = []
aris_quality = []
aris_ch = []


for i in tqdm(range(200)):#200
  x1, y1 = make_blobs(n_samples=n_samples, n_features=2, centers=n_cluster, random_state=i, cluster_std=cluster_std)#n_cluster*5

  # original SC
  y_pred = NJW(x1, n_cluster)
  ari = adjusted_rand_score(y1, y_pred)
  # print('baseline', ari)
  # print(ari)
  aris_sc.append(ari)

  # Our ISC

  our_sp = NewIncrementalSPDivision(sim_t=0.95, use_sklearn=False, rou=14)

  for i in x1:
    our_sp.input_signal(i)
  predicted_labels = [0 for _ in range(len(y1))]

  for c_i,i in enumerate(our_sp.graph.cluster_representatives):
    data_points = our_sp.graph.get_data_point(c_i)
    for j in data_points:
      idx = np.argwhere(x1==our_sp.graph.g.vs[j]['features'])[0][0]
      predicted_labels[idx] = c_i

  ari = adjusted_rand_score(y1, predicted_labels)
  aris_our.append(ari)

  # Valgren ISC
  valgren_sp = IncrementalSP(sim_threshold=0.85, use_sklearn=False, rou=14)

  for i in x1:
    valgren_sp.input_signal(i)
  predicted_labels_valgren = [0 for _ in range(len(y1))]

  for c_i,i in enumerate(valgren_sp.graph.cluster_representatives):
    data_points = valgren_sp.graph.get_data_point(c_i)
    for j in data_points:
      idx = np.argwhere(x1==valgren_sp.graph.g.vs[j]['features'])[0][0]
      predicted_labels_valgren[idx] = c_i
      # print(test_sp.graph.g.vs[j]['features'])
  ari = adjusted_rand_score(y1, predicted_labels_valgren)
  aris_valgren.append(ari)

  # Eigen gap based SC
  datas = x1.tolist()
  eigen_sp = ISCEIGN(init_data=datas[:-2], use_sklearn=False, rou=1)
  _ = eigen_sp.input_signal(datas[-2])
  predicted_labels = eigen_sp.input_signal(datas[-1])
  ari = adjusted_rand_score(y1, predicted_labels)
  aris_eigen.append(ari)

  # Quality based SC
  quality_sp = ISCQuatlity(init_data=datas[:-2], use_sklearn=False, rou=1)#
  _ = quality_sp.input_signal(datas[-2])
  predicted_labels = quality_sp.input_signal(datas[-1])
  ari = adjusted_rand_score(y1, predicted_labels)
  aris_quality.append(ari)

  # CH based SC

  ch_sp = ISC_CH(init_data=datas[:-2], use_sklearn=False, rou=1)#
  _ = ch_sp.input_signal(datas[-2])
  predicted_labels = ch_sp.input_signal(datas[-1])
  ari = adjusted_rand_score(y1, predicted_labels)
  aris_ch.append(ari)
  # print(len(np.unique(predicted_labels)), ari)

print('\n')
print('Original {}'.format(np.mean(aris_sc)))
print('Our {}'.format(np.mean(aris_our)))
print('Valgren {}'.format(np.mean(aris_valgren)))
print('Eigen {}'.format(np.mean(aris_eigen)))
print('Quality {}'.format(np.mean(aris_quality)))
print('CH {}'.format(np.mean(aris_ch)))
