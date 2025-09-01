import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from scipy.linalg import eigh

def cal_sim_row(x, data_list):
  # return x@np.array(self.data_set).T
  a = []
  for y in data_list:
    a.append(cal_sim(x, y))
  return np.array(a)

def cal_sim(x,y, rou=1):
  rou = rou#15
  dist = np.linalg.norm(x-y)
  return np.exp(-(dist**2/(2*rou**2)))


def cal_sim_mat(x):
  upper_part = np.zeros((x.shape[0],x.shape[0]))
  for i in range(x.shape[0]):
    for j in range(i, x.shape[0]):
      upper_part[i,j] = cal_sim(x[i], x[j])
  # print(upper_part+upper_part.T-np.identity(x.shape[0]))
  return upper_part+upper_part.T-np.identity(x.shape[0])

def NJW(x, cluster_n, random_state=0, use_sklearn=False):

  affinity_matrix = cal_sim_mat(x)
  if use_sklearn:

    degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix))
    laplacian_normalized = np.eye(affinity_matrix.shape[0]) - D_inv_sqrt @ affinity_matrix @ D_inv_sqrt
    model = SpectralClustering(n_clusters=cluster_n, affinity='precomputed', assign_labels='kmeans', random_state=random_state)#
    labels = model.fit_predict(affinity_matrix)

    eigvals, eigvecs = eigh(laplacian_normalized)
    embedding = eigvecs[:, :cluster_n]
  else:

    # affinity_matrix, degree_matrix, laplacian_normalized = self.get_matrices(x)
    degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix))
    laplacian_normalized = np.eye(affinity_matrix.shape[0]) - D_inv_sqrt @ affinity_matrix @ D_inv_sqrt
    eigvals, eigvecs = eigh(laplacian_normalized)
    embedding = eigvecs[:, :cluster_n]
    kmeans = KMeans(n_clusters=cluster_n, random_state=random_state)
    labels = kmeans.fit_predict(embedding)

  return labels#, embedding, eigvecs, eigvals, np.sqrt(degree_matrix), np.sqrt(degree_matrix)-affinity_matrix, affinity_matrix