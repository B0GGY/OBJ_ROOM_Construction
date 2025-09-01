import igraph as ig
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import eigh
from sklearn.cluster import SpectralClustering, KMeans

class SCGraphValgren:
  def __init__(self):
    self.g = ig.Graph()
    self.cluster_representatives=[] # save the idx of representative node
    self.data_points = [] # save the idx of data point node
    self.g.vs['features']=[]

  def add_data_point(self, x):
    self.g.add_vertex()
    self.data_points.append(self.g.vs[-1].index)
    self.g.vs[-1]['features']=x

  def add_representative(self, x):
    self.g.add_vertex()
    self.cluster_representatives.append(self.g.vs[-1].index)
    self.g.vs[-1]['features']=x

  def add_edge(self, idx1, idx2):
    self.g.add_edge(idx1, idx2)

  def add_new_cluster(self, x, y):
    self.add_representative(y)
    for i in x:
      self.add_data_point(i)
      self.g.add_edge(self.cluster_representatives[-1], self.data_points[-1])

  def get_representatives(self):
    results = []
    for i in range(len(self.cluster_representatives)):
      results.append(self.g.vs[self.cluster_representatives[i]]['features'])
    return results

  def check_existance(self, datas):
    representatives = self.get_representatives()
    cids = []
    for cid, r in enumerate(representatives):
      if self.is_in(r, datas):
        # return cid
        cids.append(cid)
    if len(cids) == 0:
      return None
    else:
      return cids

  def is_in(self, data, data_list):
      for d in data_list:
        if np.array_equal(d, data):
          return True
      return False

  def get_data_point(self, c_id):
    # print('cid {}'.format(c_id))
    # print(len(self.cluster_representatives), c_id)
    c_idx = self.cluster_representatives[c_id]
    return self.g.neighbors(c_idx, mode='all')

  def get_data_point_idx(self, x):
    for i in self.data_points:
      if np.array_equal(self.g.vs[i]['features'], x):
        return i

  def cluster_update(self, x, c_id):
    self.add_data_point(x)
    self.g.add_edge(self.cluster_representatives[c_id], self.data_points[-1])
    new_representative_feature = self.get_representative_feature(c_id)
    # print('new feature {}'.format(new_representative_feature))
    self.g.vs[self.cluster_representatives[c_id]]['features'] = new_representative_feature


  def cal_sim(self,x,y):
      rou = 15
      dist = np.linalg.norm(x-y)
      return np.exp(-(dist**2/(2*rou**2)))

  def cal_sim_mat(self,x):
    upper_part = np.zeros((x.shape[0],x.shape[0]))
    for i in range(x.shape[0]):
      for j in range(i, x.shape[0]):
        upper_part[i,j] = self.cal_sim(x[i], x[j])
    # print(upper_part+upper_part.T-np.identity(x.shape[0]))
    return upper_part+upper_part.T-np.identity(x.shape[0])

  def cluster_check(self, t):
    split_cluster_id = []
    for i in range(len(self.cluster_representatives)):
      cluster_data_points = self.get_data_point(i)
      if len(cluster_data_points) == 0:
        continue
      cluster_features = np.array(self.g.vs[cluster_data_points]['features'])
      cluster_sim = self.cal_sim_mat(cluster_features)
      cluster_sim[cluster_sim<0]=0
      inner_sim = np.max(np.min(cluster_sim, axis=1))
      if inner_sim<t:
        split_cluster_id.append(i)
    return split_cluster_id

  def cluster_split(self, label, cid):
    # print(label)
    representative_idx = self.cluster_representatives[cid]
    cluster_data_points = self.get_data_point(cid)
    cluster_features = np.array(self.g.vs[cluster_data_points]['features'])
    for i in range(len(list(set(label)))-1):
      j = list(set(label))[i]
      self.add_representative(None)
      # print('label {}'.format(j))
      C_n=np.array(cluster_features)[np.array(label)==j]

      for k in cluster_data_points:
        if self.is_in(self.g.vs[k]['features'], C_n):
          self.g.delete_edges([(representative_idx, k)])
          self.g.add_edge(self.cluster_representatives[-1], k)

      new_representative_feature = self.get_representative_feature(-1)
      self.g.vs[self.cluster_representatives[-1]]['features'] = new_representative_feature
      # print('target data point idx {}'.format(target_data_point_idx))


  def get_representative_feature(self, c_id):
    cluster_data_points = self.get_data_point(c_id)
    cluster_features = np.array(self.g.vs[cluster_data_points]['features'])

    cluster_sim = self.cal_sim_mat(cluster_features)
    cluster_sim[cluster_sim<0]=0
    try:
      idx = np.argmax(np.min(cluster_sim, axis=1))#np.argmax(np.sum(cluster_sim, axis=0))
      result = cluster_features[idx]
    except:
      return None

    # result = np.mean(cluster_features, axis=0)
    # print('cc{}'.format(result.shape[0]))
    try:
      _=result.shape[0]
      # print(result)
    except:
      # print(result)
      # print(cluster_features)
      # print(c_id)
      # print(self.cluster_representatives)
      raise ValueError
    return result#np.mean(cluster_features, axis=0)

  def is_in(self, data, data_list):
    for d in data_list:
      if np.array_equal(d, data):
        return True
    return False

  def cluster_merge(self):

    cluster_num = len(self.cluster_representatives)
    merge_idxs = np.random.choice(cluster_num, 2, replace=False)
    target_idx = self.cluster_representatives[merge_idxs[0]]
    source_idx = self.cluster_representatives[merge_idxs[1]]
    cluster_data_points = self.get_data_point(merge_idxs[1])
    # old_cluser_representative = self.cluster_representatives.copy()
    for k in cluster_data_points:
      self.g.delete_edges([(source_idx, k)])
      self.g.add_edge(target_idx, k)
    self.g.delete_vertices(source_idx)
    del self.cluster_representatives[merge_idxs[1]]
    for i in range(len(self.cluster_representatives)):
      compansation = 0
      if self.cluster_representatives[i] > source_idx:#old_cluser_representative[merge_idxs[1]]:
        compansation += 1
      self.cluster_representatives[i] -= compansation

    for i in range(len(self.data_points)):
      compansation = 0

      if self.data_points[i] > source_idx:#old_cluser_representative[merge_idxs[1]]:
        compansation += 1
      # print(self.graph.data_points[i], compansation)
      self.data_points[i] -= compansation
    if merge_idxs[0] > merge_idxs[1]:
      merge_idxs[0] -= 1
      # target_idx -= 1
    new_representative_feature = self.get_representative_feature(merge_idxs[0])
    self.g.vs[self.cluster_representatives[merge_idxs[0]]]['features'] = new_representative_feature

  def check_existance_data_Point(self, x):
    data_idx = self.get_data_point_idx(x)
    if data_idx is None:
      return None
    clusters = self.g.neighbors(data_idx, mode='all')

    if len(clusters) == 0:
      return None
    else:
      return clusters[0]

  def data_point_move(self, x, source_idx, target_idx):

    data_idx = self.get_data_point_idx(x)

    target_node_idx = self.cluster_representatives[target_idx]
    self.g.delete_edges([(source_idx, data_idx)])
    self.g.add_edge(target_node_idx, data_idx)
    new_representative_feature = self.get_representative_feature(target_idx)
    self.g.vs[self.cluster_representatives[target_idx]]['features'] = new_representative_feature

  def graph_vis(self, with_embs = False):
    # plt.ion()
    fig, ax = plt.subplots()
    if not with_embs:
      ig.plot(self.g, target=ax,
              vertex_label=['{}'.format(i) for i in range(self.g.vcount())]
              # ,edge_label=['{}, {}'.format(way_point,i) for i, way_point in enumerate(self.g.es['way_point'])]
              )
    else:
      ig.plot(self.g, target=ax,
              vertex_label=['{}, {}'.format(i, self.g.vs[i]['features']) for i in range(self.g.vcount())]
              # ,edge_label=['{}, {}'.format(way_point,i) for i, way_point in enumerate(self.g.es['way_point'])]
              )
    plt.show()




class IncrementalSP:
    def __init__(self, sim_threshold=0.85, use_sklearn = False, rou=15, random_state=0):
        self.data_set = []
        self.cluster_n = 1
        self.affinity_mat = np.zeros((0,0))
        self.sim_threshold = sim_threshold
        self.cluster_content_save = {}
        self.simplified_idx = 0
        self.graph = SCGraphValgren()
        # self.sp_model = SpectralClustering(affinity='precomputed', n_clusters=4, assign_labels='kmeans'
        self.use_sklearn = use_sklearn
        self.rou = rou
        self.random_state = random_state

    def input_signal(self, x):
        self.data_set.append(x)
        # print('len here {}'.format(len(self.data_set)))
        new_row = self.cal_sim_row(x)
        # print('data set len {}'.format(len(self.data_set)))
        # print(self.affinity_mat)

        if self.affinity_mat.shape[0] == 0:
            self.affinity_mat = np.array([new_row])
            # self.affinity_mat = np.array([0])
        else:
            new_mat_size = self.affinity_mat.shape[0] + 1
            tmp_mat = np.identity(new_mat_size)
            tmp_mat[:new_mat_size-1, :new_mat_size-1]=self.affinity_mat

            tmp_mat[-1,:] = new_row
            tmp_mat[:,-1] = new_row.T
            tmp_mat[tmp_mat<0]=0
            self.affinity_mat = tmp_mat
            # self.affinity_mat = self.affinity_mat - np.identity(self.affinity_mat.shape[0])
            # print(self.affinity_mat)
            new_node = False
            N_n = []  # the min similarity value for cluster representative
            R_n = []


            while self.min_N_n(N_n) < self.sim_threshold:
                label = self.cluster(self.cluster_n, self.affinity_mat)
                # label = self.cluster()
                # label = self.label_alignment(label)

                for i in range(self.cluster_n):
                    C_n=np.array(self.data_set)[np.array(label)==i]

                    cluster_sim = self.cal_sim_mat(C_n)
                    cluster_sim[cluster_sim<0]=0

                    R_n.append(np.argmax(np.min(cluster_sim, axis=1)))#np.argmax(np.sum(cluster_sim, axis=0))
                    N_n.append(np.max(np.min(cluster_sim, axis=1)))
                    # print(R_n, N_n)

                if self.min_N_n(N_n) < self.sim_threshold:
                    new_node = True
                    self.cluster_n += 1
                    # print(N_n)
                    N_n = []
                    R_n = []
            # print('cluster_n {}'.format(self.cluster_n))

            for i in range(self.cluster_n):
              C_n=np.array(self.data_set)[np.array(label)==i]
              representative=C_n[R_n[i]]
              if len(self.graph.cluster_representatives)==0:
                for j in range(self.cluster_n):
                  C_n=np.array(self.data_set)[np.array(label)==j]
                  representative=C_n[R_n[j]]
                  self.graph.add_new_cluster(C_n, representative)
                self.data_set = self.graph.get_representatives()

                self.affinity_mat = self.cal_sim_mat(np.array(self.data_set))
                self.affinity_mat[self.affinity_mat<0]=0

              else:

                if not self.graph.check_existance(C_n) is None:
                  cids = self.graph.check_existance(C_n)

                  representative_point = []
                  for cid in cids:
                    representative_point.append(self.graph.get_representative_feature(cid))


                  for p in C_n:
                    if not self.is_in(p, representative_point):
                      if not self.graph.check_existance_data_Point(p) is None:
                        self.graph.data_point_move(p, self.graph.check_existance_data_Point(p), cid)
                      else:
                        self.graph.cluster_update(p, cid)

                else:
                  self.graph.add_new_cluster(C_n, representative)

            cluster_ids = self.graph.cluster_check(self.sim_threshold)
            for cid in cluster_ids:
              cluster_data_points = self.graph.get_data_point(cid)
              cluster_features = np.array(self.graph.g.vs[cluster_data_points]['features'])
              local_affinity_mat = self.cal_sim_mat(cluster_features)
              local_R_n = []
              local_N_n = []
              local_cluster_n = 1
              local_new_node = False
              while self.min_N_n(local_N_n) < self.sim_threshold:
                label = self.cluster(local_cluster_n, local_affinity_mat)

                for i in range(local_cluster_n):
                  # print('cluser_feature size {}'.format(len(cluster_features)))
                  # print('label size {}'.format(len(label)))
                  C_n=np.array(cluster_features)[np.array(label)==i]

                  cluster_sim = self.cal_sim_mat(C_n)
                  cluster_sim[cluster_sim<0]=0

                  local_R_n.append(np.argmax(np.min(cluster_sim, axis=1)))

                  local_N_n.append(np.max(np.min(cluster_sim, axis=1)))
                  # print(R_n, N_n)

                if self.min_N_n(local_N_n) < self.sim_threshold:
                  local_new_node = True
                  local_cluster_n += 1
                  # print(N_n)
                  local_N_n = []
                  local_R_n = []
              if local_new_node:
                self.graph.cluster_split(label, cid)

            self.data_set = self.graph.get_representatives()
            self.affinity_mat = self.cal_sim_mat(np.array(self.data_set))
            self.affinity_mat[self.affinity_mat<0]=0
        self.random_reduce()

    def random_reduce(self):

      if np.random.rand() < 0.1 and self.cluster_n >1:
        self.cluster_n -= 1
        self.graph.cluster_merge()

        self.data_set = self.graph.get_representatives()
        self.affinity_mat = self.cal_sim_mat(np.array(self.data_set))
        self.affinity_mat[self.affinity_mat<0]=0


    def is_in(self, data, data_list):
      for d in data_list:
        if np.array_equal(d, data):
          return True
      return False

    def min_N_n(self, N_n):
      if len(N_n) == 0:
        return 0
      else:
        return min(N_n)

    def cal_sim_row(self, x):
        # return x@np.array(self.data_set).T
        a = []
        for y in self.data_set:
          a.append(self.cal_sim(x, y))
        return np.array(a)

    def cal_sim(self,x,y):
      rou = 15
      dist = np.linalg.norm(x-y)
      return np.exp(-(dist**2/(2*rou**2)))

    def cal_sim_mat(self,x):
      upper_part = np.zeros((x.shape[0],x.shape[0]))
      for i in range(x.shape[0]):
        for j in range(i, x.shape[0]):
          upper_part[i,j] = self.cal_sim(x[i], x[j])
      # print(upper_part+upper_part.T-np.identity(x.shape[0]))
      return upper_part+upper_part.T-np.identity(x.shape[0])


    def cluster(self, n_cluster, affinity_mat):
        # sp_model = SpectralClustering(affinity='precomputed', n_clusters=n_cluster, assign_labels='kmeans')

        # label = sp_model.fit_predict(affinity_mat)
        label = self.NJW(affinity_mat, n_cluster)
        return label

    def NJW(self, affinity_matrix, cluster_n):

      if self.use_sklearn:

        degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix))
        laplacian_normalized = np.eye(affinity_matrix.shape[0]) - D_inv_sqrt @ affinity_matrix @ D_inv_sqrt
        model = SpectralClustering(n_clusters=cluster_n, affinity='precomputed', assign_labels='kmeans', random_state=self.random_state)#
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
        kmeans = KMeans(n_clusters=cluster_n, random_state=self.random_state)
        labels = kmeans.fit_predict(embedding)

      return labels#, embedding, eigvecs, eigvals, np.sqrt(degree_matrix), np.sqrt(degree_matrix)-affinity_matrix, affinity_matrix
