import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from scipy.linalg import eigh


class ISCEIGN:
  def __init__(self, random_state=0, init_data = [], use_sklearn = True, metric_type='dist', rou=1):
      self.k = 1#

      self.represetative_reliability = [] # list to store r
      self.metric_type = metric_type
      self.random_state = random_state
      # self.n_neighbors = 20
      self.rou=rou
      self.reinit_flag = True
      self.representatives = {}

      self.cluster_labels = []
      self.data_set = list(init_data) # set of all input data
      self.eigen_gap = []
      self.eigen_values = []
      self.use_sklearn = use_sklearn

      # self.initialization()

  def initialization(self):

    labels, X, Z, eig_values, D, L, A = self.NJW(np.array(self.data_set))
    self.cluster_labels = labels.tolist()
    self.eigen_values = eig_values
    self.update_reliability(labels, X)
    for cluster_id in range(self.k):
        representative_points = self.get_representatives(cluster_id, X, labels, L, D)
        self.representatives[cluster_id] = representative_points
    # print('representatives {}'.format(self.representatives))


  def NJW(self, x):

    if self.use_sklearn:

      affinity_matrix, degree_matrix, laplacian_normalized = self.get_matrices(x)
      model = SpectralClustering(n_clusters=self.k, affinity='precomputed', assign_labels='kmeans', random_state=self.random_state)#
      labels = model.fit_predict(affinity_matrix)

      eigvals, eigvecs = eigh(laplacian_normalized)
      embedding = eigvecs[:, :self.k]
    else:

      affinity_matrix, degree_matrix, laplacian_normalized = self.get_matrices(x)
      eigvals, eigvecs = eigh(laplacian_normalized)
      embedding = eigvecs[:, :self.k]
      kmeans = KMeans(n_clusters=self.k, random_state=self.random_state)
      labels = kmeans.fit_predict(embedding)

    return labels, embedding, eigvecs, eigvals, np.sqrt(degree_matrix), np.sqrt(degree_matrix)-affinity_matrix, affinity_matrix

  def get_matrices(self, x):

    affinity_matrix = self.cal_sim_mat(x)

    degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix))
    laplacian_normalized = np.eye(np.array(x).shape[0]) - D_inv_sqrt @ affinity_matrix @ D_inv_sqrt

    return affinity_matrix, degree_matrix, laplacian_normalized

  def cal_sim_row(self, x, data_list):
    # return x@np.array(self.data_set).T
    a = []
    for y in data_list:
      a.append(self.cal_sim(x, y))
    return np.array(a)

  def cal_sim(self,x,y):

    if self.metric_type == 'dist':
      rou = self.rou#15
      dist = np.linalg.norm(x-y)
      return np.exp(-(dist**2/(2*rou**2)))
    elif self.metric_type == 'sim':
      return x@y
    else:
      raise Exception('metric type error')

  def cal_sim_mat(self,x):
    upper_part = np.zeros((x.shape[0],x.shape[0]))
    for i in range(x.shape[0]):
      for j in range(i, x.shape[0]):
        upper_part[i,j] = self.cal_sim(x[i], x[j])
    # print(upper_part+upper_part.T-np.identity(x.shape[0]))
    return upper_part+upper_part.T-np.identity(x.shape[0])

  def input_signal(self, x):
    self.data_set.append(x)
    labels, X, Z, eig_values, D, L, A = self.NJW(np.array(self.data_set))
    self.k = self.calculate_eigen_gap(eig_values)
    return labels

  def get_delta_lambda(self):

    labels, X, Z, eig_values, D, L, A = self.NJW(np.array(self.data_set))
    # A, degree_matrix, laplacian_normalized = self.get_matrices(x)
    delta_c = A[:,-1]
    for i in range(self.k):
      delta_c_ij = delta_c[i]
      x_i = X[:,i].reshape(-1, 1)
      r_ij = np.zeros_like(x_i)
      r_ij[i] = -1
      r_ij[-1] = 1
      lambda_i = self.eigen_values[i]
      diag_vij = np.zeros_like(D)
      diag_vij[i,i]=1
      diag_vij[-1,-1]=1
      delta_lambda = delta_c_ij*((x_i.T@(r_ij@r_ij.T-lambda_i*diag_vij)@x_i)/(x_i.T@D@x_i))
    #   print('delta lambda {}'.format(delta_lambda))
    # print(self.eigen_values[:self.k])
    # print(eig_values[:self.k])
    # print('-------')


  def get_distance2cluster(self, cluster_id, input_data):
    # calculate average distance to clusters
    representatives = self.representatives[cluster_id]
    Dis = 0
    for r in representatives:
      Dis += np.linalg.norm(r - input_data)
    return Dis/len(representatives)

  def update_reliability(self, labels, X):

    self.representative_reliability = []
    for data_id, data_point in enumerate(self.data_set):
      r_i = 0
      for cluster_id in range(self.k):
        X_ij = X[data_id, cluster_id]
        M_i = max(X[data_id, :])
        r_i += X_ij**2/M_i**2
      self.representative_reliability.append(r_i)
    # print(self.representative_reliability)

  def get_representatives(self, j, X, labels, L, D):

    eigen_value_list = []
    X_j = X[labels==j]
    local_datas = np.array(self.data_set)[labels==j]
    A_j, D_j, L_j = self.get_matrices(local_datas)

    for i in range(X_j.shape[1]):
      eigen_value_list.append((X_j[:,i].T@L_j@X_j[:,i])/(X_j[:,i].T@D_j@X_j[:,i]))
    # print(eigen_value_list)
    kci=self.get_k(eigen_value_list)
    print('get {} representative points for cluster {}'.format(kci, j))
    # print(np.argmax(self.representative_reliability))
    mask = labels == j
    R_j = self.representative_reliability*mask

    indices = np.argpartition(R_j, -kci)[-kci:]
    sorted_indices = indices[np.argsort(R_j[indices])[::-1]]
    representatives = []

    for id in sorted_indices:
      representatives.append(self.data_set[id])
    return representatives

  def get_k(self, eigen_values):
    # calculate k through eigen gap
    if len(eigen_values) < 2:
      return 1
    max_eigen = 0
    min_i = None
    self.eigen_gap = []
    for i in range(len(eigen_values)):
      lambda_diff = eigen_values[i+1]-eigen_values[i]
      print('lambda diff {}'.format(lambda_diff))
      self.eigen_gap.append(lambda_diff)
      if lambda_diff > max_eigen:
        max_eigen = lambda_diff
        min_i = i+1
    # print(max_eigen, min_i)
    return min_i

  def calculate_eigen_gap(self, eigen_values):
    # calculate k through eigen gap
    # print('******')
    # print(eigen_values)
    if len(eigen_values) < 2:
      return 1
    max_eigen = 0
    min_i = None
    self.eigen_gap = []
    for i in range(len(eigen_values)-1):
      lambda_diff = eigen_values[i+1]-eigen_values[i]
      # print('lambda diff {}'.format(lambda_diff))
      self.eigen_gap.append(lambda_diff)
      if lambda_diff > max_eigen:
        max_eigen = lambda_diff
        min_i = i+1
    # print(max_eigen, min_i)
    return min_i