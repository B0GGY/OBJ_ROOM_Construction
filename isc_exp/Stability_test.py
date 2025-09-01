from ISC import NewIncrementalSPDivision
from Valgren_ISC import IncrementalSP
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score
import random

n_cluster = 4
cluster_std=1.5
n_samples=n_cluster*20
num_iter = 10


aris_our = []
aris_valgren = []

for i in tqdm(range(200)):#200
  x1_, y1_ = make_blobs(n_samples=n_samples, n_features=2, centers=n_cluster, random_state=i, cluster_std=cluster_std)#n_cluster*5
  # x1 = x1.tolist()

  for _ in range(num_iter):
    zip_data = list(zip(x1_, y1_))
    random.shuffle(zip_data)

    x1 = np.zeros_like(x1_)
    y1 = np.zeros_like(y1_)
    for i,j in enumerate(zip_data):
      x1[i] = j[0]
      y1[i] = j[1]

    our_sp = NewIncrementalSPDivision(sim_t=0.95, use_sklearn=False, rou=14)

    for i in x1:
      our_sp.input_signal(i)
    predicted_labels = [0 for _ in range(len(y1))]

    for c_i,i in enumerate(our_sp.graph.cluster_representatives):
      data_points = our_sp.graph.get_data_point(c_i)
      for j in data_points:
        # print(test_sp.graph.g.vs[j]['features'])
        # print(np.argwhere(x1==test_sp.graph.g.vs[j]['features']))
        idx = np.argwhere(x1==our_sp.graph.g.vs[j]['features'])[0][0]
        # print(idx)
        predicted_labels[idx] = c_i
    ari = adjusted_rand_score(y1, predicted_labels)
    aris_our.append(ari)

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


print('\n')

print('Our mean {}'.format(np.mean(aris_our)))
print('Valgren mean {}'.format(np.mean(aris_valgren)))
print('Our var {}'.format(np.var(aris_our)))
print('Valgren var {}'.format(np.var(aris_valgren)))

