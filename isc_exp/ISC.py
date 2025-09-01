import numpy as np
import igraph as ig
from matplotlib import pyplot as plt
from scipy.linalg import eigh
from sklearn.cluster import SpectralClustering, KMeans


class SCGraph:
  def __init__(self):
    self.g = ig.Graph()
    self.cluster_representatives=[]
    self.data_points = []
    self.g.vs['features']=[]
    self.add_representative(None)

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

  def cluster_update(self, x, c_id):
    self.add_data_point(x)
    self.g.add_edge(self.cluster_representatives[c_id], self.data_points[-1])
    new_representative_feature = self.get_representative_feature(c_id)
    # print('new feature {}'.format(new_representative_feature))
    self.g.vs[self.cluster_representatives[c_id]]['features'] = new_representative_feature
    # cluster_data_points = self.get_data_point(c_id)
    # print(cluster_data_points)

  def get_data_point(self, c_id):
    c_idx = self.cluster_representatives[c_id]
    return self.g.neighbors(c_idx, mode='all')

  def get_representative_feature(self, c_id):
    cluster_data_points = self.get_data_point(c_id)
    cluster_features = np.array(self.g.vs[cluster_data_points]['features'])
    result = np.mean(cluster_features, axis=0)
    # print('cc{}'.format(result.shape[0]))
    try:
      _=result.shape[0]
      # print(result)
    except:

      raise ValueError
    return result#np.mean(cluster_features, axis=0)

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
    # plt.pause(2)
    # plt.ioff()

class NewIncrementalSPDivision:
    def __init__(self, metric_type='dist', sim_t = 0.85, use_sklearn=True, random_state=0, rou=15):
        self.data_set = [] # set of all input data
        self.current_set = [] # set of currently processing data
        # self.labels = [] # labels are corresponding to elements in self.data_set
        self.cluster_n = 1
        self.affinity_mat = np.zeros((0,0))
        self.sim_threshold = sim_t
        self.cluster_content_save = {}
        self.simplified_idx = 0
        self.existing_cluster_num = 0
        self.metric_type = metric_type

        # self.sp_model = SpectralClustering(affinity='precomputed', n_clusters=4, assign_labels='kmeans')
        self.graph = SCGraph()
        self.rou = rou # 15
        self.use_sklearn = use_sklearn
        self.random_state = random_state


    def input_signal(self, x):
        self.data_set.append(x)
        # self.labels.append(None)
        new_cluster = False
        min_N = 0
        if len(self.graph.cluster_representatives)>1:  # check whether the new data belongs to existing cluster
          # representative_list = self.get_representative()
          # print(self.affinity_mat)
          sim_list = []
          for representative_idx in range(len(self.graph.cluster_representatives)-1):
            representative_content = self.graph.g.vs[self.graph.cluster_representatives[representative_idx]]['features']
            sim = self.cal_sim(x, representative_content)
            # print(representative_content)
            # print('comaring input feature {} to existing representative features sim {}'.format(x,sim))
            sim_list.append(sim)
          comparison_result = np.array(sim_list)>self.sim_threshold
          if np.sum(comparison_result) == 0:
            if len(self.current_set)==0: # means x is the first data

              # print('add {} pos1'.format(x))
              self.graph.cluster_update(x,-1)
              self.current_set.append(x)
              new_row = self.cal_sim_row(x, self.current_set)
              self.affinity_mat = np.array([new_row])
              # print(self.affinity_mat)
            else:

              # print('add {} pos2'.format(x))
              self.graph.cluster_update(x,-1)
              self.current_set.append(x)
              new_mat_size = self.affinity_mat.shape[0]+1
              tmp_mat = np.zeros((new_mat_size, new_mat_size))
              tmp_mat[:new_mat_size-1, :new_mat_size-1]=self.affinity_mat
              tmp_mat[-1,:] = self.cal_sim_row(x, self.current_set)
              tmp_mat[:,-1] = self.cal_sim_row(x, self.current_set).T
              tmp_mat[tmp_mat<0]=0
              self.affinity_mat = tmp_mat
              # print(self.affinity_mat)
              N_n = []

              def min_N_n():
                if len(N_n) == 0:
                  return 0
                else:
                  return min(N_n)

              while min_N_n()< self.sim_threshold:
                label = self.cluster(self.cluster_n)
                # print('cluster n {}'.format(self.cluster_n))
                # print('label {}'.format(label))
                for i in range(self.cluster_n):
                  C_n = np.array(self.current_set)[np.array(label)==i]
                  representative=np.mean(C_n,axis=0)
                  min_sim = 1
                  min_idx = 0
                  for j in range(C_n.shape[0]):
                    sim = self.cal_sim(C_n[j], representative)
                    if sim < min_sim:
                      min_sim = sim
                      min_idx = j
                  N_n.append(min_sim)
                  # print('N_n {}'.format(N_n))

                if min_N_n()<self.sim_threshold:
                  self.cluster_n += 1
                  new_cluster = True
                  N_n=[]

              if new_cluster:
                # print('cluster n {}'.format(self.cluster_n))
                # print('label {}'.format(label))
                # self.graph
                old_NCD_idx = self.graph.cluster_representatives[-1]
                old_NCD_cluster_idx = len(self.graph.cluster_representatives)-1

                all_data_points = self.graph.get_data_point(old_NCD_cluster_idx)
                tmp=[]
                for idx in all_data_points:
                  tmp.append(self.graph.g.vs[idx]['features'])
                # print(tmp)

                for i in range(self.cluster_n-1):

                  data_points=[]
                  cluster_features = np.array(self.current_set)[label==i]

                  all_data_points = self.graph.get_data_point(old_NCD_cluster_idx)
                  # print(len(self.current_set), len(all_data_points))
                  for idx in all_data_points:
                    # print(self.graph.g.vs[idx]['features'], cluster_features)
                    if self.graph.g.vs[idx]['features'] in cluster_features:
                      data_points.append(idx)

                  self.graph.add_representative(None)

                  for idx in data_points:
                    self.graph.add_edge(self.graph.cluster_representatives[-1], idx)
                    self.graph.g.delete_edges([(old_NCD_idx, idx)])

                  new_representative_feature = self.graph.get_representative_feature(-1)
                  self.graph.g.vs[self.graph.cluster_representatives[-1]]['features'] = new_representative_feature

                new_representative_feature = self.graph.get_representative_feature(old_NCD_cluster_idx)
                self.graph.g.vs[old_NCD_idx]['features'] = new_representative_feature
                self.graph.add_representative(None)


                self.current_set = []
                self.affinity_mat = np.zeros((0,0))
                self.cluster_n = 1

          elif np.sum(comparison_result) == 1:
            cluster_id = np.argwhere(comparison_result>0)[0][0]
            # print('cluster id {}'.format(cluster_id))
            self.graph.cluster_update(x, cluster_id)
          elif np.sum(comparison_result) > 1:
            recluster_cluster_idx = np.argwhere(comparison_result>0).tolist()
            recluster_datapoint_idx = []
            for idx in recluster_cluster_idx:
              recluster_datapoint_idx.extend(self.graph.get_data_point(idx[0]))

            for idx in recluster_datapoint_idx:
              self.graph.add_edge(self.graph.cluster_representatives[-1], idx)
            recluster_cluster_idx.reverse()
            old_cluser_representative = self.graph.cluster_representatives.copy()
            for idx in recluster_cluster_idx:
              self.graph.g.delete_vertices(self.graph.cluster_representatives[idx[0]])
              del self.graph.cluster_representatives[idx[0]]

            for i in range(len(self.graph.cluster_representatives)):
              compansation = 0
              for del_idx in recluster_cluster_idx:
                if self.graph.cluster_representatives[i] > old_cluser_representative[del_idx[0]]:
                  compansation += 1
              self.graph.cluster_representatives[i] -= compansation

            for i in range(len(self.graph.data_points)):
              compansation = 0
              for del_idx in recluster_cluster_idx:
                if self.graph.data_points[i] > old_cluser_representative[del_idx[0]]:
                  compansation += 1
              # print(self.graph.data_points[i], compansation)
              self.graph.data_points[i] -= compansation

            self.graph.cluster_update(x, -1)

            new_representative_feature = self.graph.get_representative_feature(-1)
            self.graph.g.vs[self.graph.cluster_representatives[-1]]['features'] = new_representative_feature

            self.current_set = []
            self.affinity_mat = np.zeros((0,0))

            unprocessed_data_idx = self.graph.get_data_point(-1)
            # print('unprocessed data {}'.format(unprocessed_data_idx))

            for idx in unprocessed_data_idx:
              self.current_set.append(self.graph.g.vs[idx]['features'])

            self.affinity_mat = self.cal_sim_mat(np.array(self.current_set))
            self.affinity_mat[self.affinity_mat<0]=0


            N_n = []

            def min_N_n():
              if len(N_n) == 0:
                return 0
              else:
                return min(N_n)

            while min_N_n()< self.sim_threshold:
              label = self.cluster(self.cluster_n)
              # print('cluster n {}'.format(self.cluster_n))
              # print('label {}'.format(label))
              for i in range(self.cluster_n):
                C_n = np.array(self.current_set)[np.array(label)==i]
                representative=np.mean(C_n,axis=0)
                min_sim = 1
                min_idx = 0
                for j in range(C_n.shape[0]):
                  sim = self.cal_sim(C_n[j], representative)
                  if sim < min_sim:
                    min_sim = sim
                    min_idx = j
                N_n.append(min_sim)
                # print('N_n {}'.format(N_n))

              if min_N_n()<self.sim_threshold:
                self.cluster_n += 1
                new_cluster = True
                N_n=[]

            if new_cluster:
              # print('cluster n {}'.format(self.cluster_n))
              # print('label {}'.format(label))
              # self.graph
              old_NCD_idx = self.graph.cluster_representatives[-1]
              old_NCD_cluster_idx = len(self.graph.cluster_representatives)-1

              all_data_points = self.graph.get_data_point(old_NCD_cluster_idx)
              tmp=[]
              for idx in all_data_points:
                tmp.append(self.graph.g.vs[idx]['features'])
              # print(tmp)

              for i in range(self.cluster_n-1):

                data_points=[]
                cluster_features = np.array(self.current_set)[label==i]

                # print('c_feature {}'.format(cluster_features))
                all_data_points = self.graph.get_data_point(old_NCD_cluster_idx)

                for idx in all_data_points:
                  # print(self.graph.g.vs[idx]['features'], cluster_features)
                  if self.graph.g.vs[idx]['features'] in cluster_features:
                    data_points.append(idx)

                self.graph.add_representative(None)

                for idx in data_points:
                  self.graph.add_edge(self.graph.cluster_representatives[-1], idx)
                  self.graph.g.delete_edges([(old_NCD_idx, idx)])

                new_representative_feature = self.graph.get_representative_feature(-1)
                self.graph.g.vs[self.graph.cluster_representatives[-1]]['features'] = new_representative_feature


              new_representative_feature = self.graph.get_representative_feature(old_NCD_cluster_idx)
              self.graph.g.vs[old_NCD_idx]['features'] = new_representative_feature
              self.graph.add_representative(None)


              self.current_set = []
              self.affinity_mat = np.zeros((0,0))
              self.cluster_n = 1


        else:

          if len(self.current_set)==0: # means x is the first data

            self.graph.cluster_update(x,-1)
            self.current_set.append(x)
            new_row = self.cal_sim_row(x, self.current_set)
            self.affinity_mat = np.array([new_row])
            # print(self.affinity_mat)
          else:

            self.graph.cluster_update(x,-1)
            self.current_set.append(x)
            new_mat_size = self.affinity_mat.shape[0]+1
            tmp_mat = np.zeros((new_mat_size, new_mat_size))
            tmp_mat[:new_mat_size-1, :new_mat_size-1]=self.affinity_mat
            tmp_mat[-1,:] = self.cal_sim_row(x, self.current_set)
            tmp_mat[:,-1] = self.cal_sim_row(x, self.current_set).T
            tmp_mat[tmp_mat<0]=0
            self.affinity_mat = tmp_mat
            # print(self.affinity_mat)
            N_n = []

            def min_N_n():
              if len(N_n) == 0:
                return 0
              else:
                return min(N_n)

            while min_N_n()< self.sim_threshold:
              label = self.cluster(self.cluster_n)
              # print('cluster n {}'.format(self.cluster_n))
              # print('label {}'.format(label))
              for i in range(self.cluster_n):
                C_n = np.array(self.current_set)[np.array(label)==i]
                representative=np.mean(C_n,axis=0)
                min_sim = 1
                min_idx = 0
                for j in range(C_n.shape[0]):
                  sim = self.cal_sim(C_n[j], representative)
                  if sim < min_sim:
                    min_sim = sim
                    min_idx = j
                N_n.append(min_sim)
                # print('N_n {}'.format(N_n))

              if min_N_n()<self.sim_threshold:
                self.cluster_n += 1
                new_cluster = True
                N_n=[]

            if new_cluster:

              old_NCD_idx = self.graph.cluster_representatives[-1]
              old_NCD_cluster_idx = len(self.graph.cluster_representatives)-1

              all_data_points = self.graph.get_data_point(old_NCD_cluster_idx)
              tmp=[]
              for idx in all_data_points:
                tmp.append(self.graph.g.vs[idx]['features'])
              # print(tmp)

              for i in range(self.cluster_n-1):

                data_points=[]
                cluster_features = np.array(self.current_set)[label==i]

                all_data_points = self.graph.get_data_point(old_NCD_cluster_idx)

                # print(len(self.current_set), len(all_data_points))
                for idx in all_data_points:
                  # print(self.graph.g.vs[idx]['features'], cluster_features)
                  if self.graph.g.vs[idx]['features'] in cluster_features:
                    data_points.append(idx)

                self.graph.add_representative(None)

                for idx in data_points:
                  self.graph.add_edge(self.graph.cluster_representatives[-1], idx)
                  self.graph.g.delete_edges([(old_NCD_idx, idx)])

                new_representative_feature = self.graph.get_representative_feature(-1)
                self.graph.g.vs[self.graph.cluster_representatives[-1]]['features'] = new_representative_feature

              new_representative_feature = self.graph.get_representative_feature(old_NCD_cluster_idx)
              self.graph.g.vs[old_NCD_idx]['features'] = new_representative_feature
              self.graph.add_representative(None)


              self.current_set = []
              self.affinity_mat = np.zeros((0,0))
              self.cluster_n = 1

    def cal_sim_row(self, x, data_list):
        # return x@np.array(self.data_set).T
        a = []
        for y in data_list:
          a.append(self.cal_sim(x, y))
        return np.array(a)

    def cal_sim(self,x,y):

      if self.metric_type == 'dist':
        rou = self.rou
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

    def content_replacement(self, R):
        for i in range(self.simplified_idx, self.cluster_n-1):
          representative_content = self.cluster_content_save[i][R[i]]
          cluster_idx = np.array(self.data_set)[np.array(label)==i]
          for idx in cluster_idx:
            if self.data_set[idx] != representative_content:
              self.data_set.remove(idx)
          # print(representative_content)
        self.simplified_idx = self.cluster_n
        # self.affinity_mat = np.array(self.data_set)@np.array(self.data_set).T
        self.affinity_mat = self.cal_sim_mat(np.array(self.data_set))
        self.affinity_mat[self.affinity_mat<0]=0

    def cluster(self, cluster_n):

        label = self.NJW(self.affinity_mat, cluster_n)

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
