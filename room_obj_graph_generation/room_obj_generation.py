import numpy as np
from sklearn.cluster import SpectralClustering
from matplotlib import pyplot as plt
from place_obj_generation import OPGraphCreationTest
import igraph as ig
from adjustText import adjust_text
from sklearn.cluster import SpectralClustering, KMeans
from scipy.linalg import eigh

class SCGraph:
  def __init__(self):
    self.g = ig.Graph()
    self.cluster_representatives=[]
    self.data_points = []
    self.g.vs['features']=[]
    self.g.vs['name'] = []
    self.add_representative(None)

  def add_data_point(self, x, obj_name=None):
    self.g.add_vertex()
    self.data_points.append(self.g.vs[-1].index)
    self.g.vs[-1]['features']=x
    self.g.vs[-1]['name']=obj_name

  def add_representative(self, x):
    self.g.add_vertex()
    self.cluster_representatives.append(self.g.vs[-1].index)
    self.g.vs[-1]['features']=x
    self.g.vs[-1]['name']='RC{}'.format(self.g.vs[-1].index)

  def add_edge(self, idx1, idx2):
    self.g.add_edge(idx1, idx2)

  def cluster_update(self, x, c_id, obj_name=None):
    self.add_data_point(x, obj_name)
    self.g.add_edge(self.cluster_representatives[c_id], self.data_points[-1])
    new_representative_feature = self.get_representative_feature(c_id)
    # print('new feature {}'.format(new_representative_feature))
    self.g.vs[self.cluster_representatives[c_id]]['features'] = new_representative_feature

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
      # print(result)
      # print(cluster_features)
      # print(c_id)
      # print(self.cluster_representatives)
      raise ValueError
    return result#np.mean(cluster_features, axis=0)

  def graph_vis(self, with_embs = False):

    layout = self.g.layout("fruchterman_reingold")
    # layout = self.g.layout("random")
    coords = layout.coords

    x, y = zip(*coords)
    fig, ax = plt.subplots(figsize=(10, 10))

    for edge in self.g.es:
      source, target = edge.tuple
      if 'RC' in self.g.vs[source]['name'] and 'RC' in self.g.vs[target]['name']:

        ax.plot(
            [x[source], x[target]],
            [y[source], y[target]],
            color="gray",
            lw=3
        )
      else:
        ax.plot(
            [x[source], x[target]],
            [y[source], y[target]],
            color="darkgray",
            lw=2
        )

    for i, (xi, yi) in enumerate(zip(x, y)):
      if 'RC' in self.g.vs[i]['name']:
        ax.scatter(xi, yi, s=600, c="lightblue", edgecolors=(0, 0, 0, 0), zorder=3)
      else:
        ax.scatter(xi, yi, s=300, c="salmon", edgecolors=(0, 0, 0, 0), zorder=3)

    texts = []
    for i, (xi, yi) in enumerate(zip(x, y)):
      if not 'RC' in self.g.vs[i]['name']:
        texts.append(ax.text(
            xi, yi,
            f"{self.g.vs[i]['name']}", fontsize=13, ha="center", va="center", zorder=5,fontweight='bold'
        ))
      else:
        ax.text(
            xi, yi,
            f"{self.g.vs[i]['name']}", fontsize=14, ha="center", va="center", zorder=5,fontweight='bold'
        )
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="black"))

    ax.axis("off")

    plt.show()

  def get_feature(self, name):
    # print('searching {}'.format(name))
    # print(self.g.vs['name'])
    for i in range(self.g.vcount()):
      if self.g.vs[i]['name'] == name:
        # print('{}-{}'.format(name, i))
        return self.g.vs[i]['features']
    # print('cant find {}'.format(name))
    return None

  def get_cluster_content(self, c_id):
    name_list =[]
    cluster_data_points = self.get_data_point(c_id)
    for idx in cluster_data_points:
      name_list.append(self.g.vs[idx]['name'])
    # print('cc {}'.format(name_list))
    return name_list

  def room_connectivity(self, geo_graph):
    # print(self.cluster_representatives)
    connectivity_mat = np.zeros((len(self.cluster_representatives), len(self.cluster_representatives)))
    for i in range(len(self.cluster_representatives)):

      current_room_objs = self.get_cluster_content(i)
      for j in range(i+1, len(self.cluster_representatives)):

        other_room_objs = self.get_cluster_content(j)
        connect_falg = False
        for c_obj in current_room_objs:
          for o_obj in other_room_objs:

            vpath = geo_graph.graph.get_shortest_paths(geo_graph.get_node_idx_by_name(c_obj), to=geo_graph.get_node_idx_by_name(o_obj), weights=geo_graph.graph.es['path_finding_weight'], output='vpath')[0]
            vpath = geo_graph.path_check(vpath[1:-2])
            connect_falg = True
            for node in vpath:
              if node == 'place':
                continue
              elif node in current_room_objs or node in other_room_objs:
                continue
              else:
                connect_falg = False
                break
        if connect_falg:
          connectivity_mat[i,j]=1
          connectivity_mat[j,i]=1
    for i in range(len(self.cluster_representatives)):
      for j in range(i+1, len(self.cluster_representatives)):
        if connectivity_mat[i,j] == 1:
          self.add_edge(self.cluster_representatives[i], self.cluster_representatives[j])
    return connectivity_mat


class NewIncrementalSPDivision:
    def __init__(self,item_list_dict, metric_type='dist', sim_t = 0.85, use_sklearn=True):
        self.data_set = [] # set of all input data
        self.current_set = [] # set of currently processing data
        self.current_name_set = []
        # self.labels = [] # labels are corresponding to elements in self.data_set
        self.cluster_n = 1
        self.affinity_mat = np.zeros((0,0))
        self.sim_threshold = sim_t
        self.cluster_content_save = {}
        self.simplified_idx = 0
        self.existing_cluster_num = 0
        self.metric_type = metric_type
        item_list_dict = item_list_dict
        self.geo_graph = OPGraphCreationTest(item_list_dict, dist_param=2)
        self.geo_graph.get_graph()

        self.graph = SCGraph()
        self.use_sklearn = use_sklearn


    def input_signal(self, x, obj_name):
        self.data_set.append(x)
        # self.labels.append(None)
        new_cluster = False
        min_N = 0
        if len(self.graph.cluster_representatives)>1:  # check whether the new data belongs to existing cluster

          sim_list = []
          for representative_idx in range(len(self.graph.cluster_representatives)-1):
            representative_content = self.graph.g.vs[self.graph.cluster_representatives[representative_idx]]['features']
            # sim = self.cal_sim(x, representative_content)
            cluster_content = self.graph.get_cluster_content(representative_idx)

            sim = self.cal_sim(obj_name, cluster_content, unattached=True, given_feature=x)

            # print('comaring input feature {} to existing representative features sim {}'.format(x,sim))
            sim_list.append(sim)
          comparison_result = np.array(sim_list)>self.sim_threshold
          if np.sum(comparison_result) == 0:
            if len(self.current_set)==0: # means x is the first data

              self.graph.cluster_update(x,-1, obj_name)
              self.current_set.append(x)
              self.current_name_set.append(obj_name)
              # new_row = self.cal_sim_row(x, self.current_set)
              new_row = self.cal_sim_row(obj_name, self.current_name_set)
              self.affinity_mat = np.array([new_row])

            else:

              self.graph.cluster_update(x,-1, obj_name)
              self.current_set.append(x)
              self.current_name_set.append(obj_name)
              new_mat_size = self.affinity_mat.shape[0]+1
              tmp_mat = np.zeros((new_mat_size, new_mat_size))
              tmp_mat[:new_mat_size-1, :new_mat_size-1]=self.affinity_mat
              tmp_mat[-1,:] = self.cal_sim_row(obj_name, self.current_name_set)
              tmp_mat[:,-1] = self.cal_sim_row(obj_name, self.current_name_set).T
              tmp_mat[tmp_mat<0]=0

              self.affinity_mat = tmp_mat
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
                  C_n_name = self.masked_selector(self.current_name_set, label, i)
                  representative=np.mean(C_n,axis=0)
                  min_sim = 1
                  min_idx = 0
                  for j in range(C_n.shape[0]):
                    # sim = self.cal_sim(C_n[j], representative)
                    sim = self.cal_sim(C_n_name[j], C_n_name)
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


                  # print('cluster n {}/ {}'.format(i, self.cluster_n))
                  new_representative_feature = self.graph.get_representative_feature(-1)
                  self.graph.g.vs[self.graph.cluster_representatives[-1]]['features'] = new_representative_feature

                new_representative_feature = self.graph.get_representative_feature(old_NCD_cluster_idx)
                self.graph.g.vs[old_NCD_idx]['features'] = new_representative_feature
                self.graph.add_representative(None)


                self.current_set = []
                self.current_name_set = []
                self.affinity_mat = np.zeros((0,0))
                self.cluster_n = 1

          elif np.sum(comparison_result) == 1:
            cluster_id = np.argwhere(comparison_result>0)[0][0]
            # print('cluster id {}'.format(cluster_id))
            self.graph.cluster_update(x, cluster_id, obj_name)
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

            self.graph.cluster_update(x, -1, obj_name)

            new_representative_feature = self.graph.get_representative_feature(-1)
            self.graph.g.vs[self.graph.cluster_representatives[-1]]['features'] = new_representative_feature

            self.current_set = []
            self.current_name_set = []
            self.affinity_mat = np.zeros((0,0))

            unprocessed_data_idx = self.graph.get_data_point(-1)
            # print('unprocessed data {}'.format(unprocessed_data_idx))

            for idx in unprocessed_data_idx:
              self.current_set.append(self.graph.g.vs[idx]['features'])
              self.current_name_set.append(self.graph.g.vs[idx]['name'])

            self.affinity_mat = self.cal_sim_mat(np.array(self.current_name_set))
            self.affinity_mat[self.affinity_mat<0]=0

            N_n = []

            def min_N_n():
              if len(N_n) == 0:
                return 0
              else:
                return min(N_n)

            while min_N_n()< self.sim_threshold:
              label = self.cluster(self.cluster_n)

              for i in range(self.cluster_n):
                C_n = np.array(self.current_set)[np.array(label)==i]
                C_n_name = self.masked_selector(self.current_name_set, label, i)
                representative=np.mean(C_n,axis=0)
                min_sim = 1
                min_idx = 0
                for j in range(C_n.shape[0]):
                  # sim = self.cal_sim(C_n[j], representative)
                  sim = self.cal_sim(C_n_name[j], C_n_name)
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
                # print(len(self.current_set), len(all_data_points))
                # print('all data {}'.format(all_data_points))
                for idx in all_data_points:
                  # print(self.graph.g.vs[idx]['features'], cluster_features)
                  if self.graph.g.vs[idx]['features'] in cluster_features:
                    data_points.append(idx)

                self.graph.add_representative(None)


                for idx in data_points:
                  self.graph.add_edge(self.graph.cluster_representatives[-1], idx)
                  self.graph.g.delete_edges([(old_NCD_idx, idx)])


                # print('cluster n {}/ {}'.format(i, self.cluster_n))
                new_representative_feature = self.graph.get_representative_feature(-1)
                self.graph.g.vs[self.graph.cluster_representatives[-1]]['features'] = new_representative_feature


              new_representative_feature = self.graph.get_representative_feature(old_NCD_cluster_idx)
              self.graph.g.vs[old_NCD_idx]['features'] = new_representative_feature
              self.graph.add_representative(None)

              self.current_set = []
              self.current_name_set = []

              self.affinity_mat = np.zeros((0,0))
              self.cluster_n = 1


        else:

          if len(self.current_set)==0: # means x is the first data

            self.graph.cluster_update(x,-1, obj_name)
            self.current_set.append(x)
            self.current_name_set.append(obj_name)

            new_row = self.cal_sim_row(obj_name, self.current_name_set)
            self.affinity_mat = np.array([new_row])
            # print('affinity mat')
            # print(self.affinity_mat)
          else:

            self.graph.cluster_update(x,-1, obj_name)
            self.current_set.append(x)
            self.current_name_set.append(obj_name)
            new_mat_size = self.affinity_mat.shape[0]+1
            tmp_mat = np.zeros((new_mat_size, new_mat_size))
            tmp_mat[:new_mat_size-1, :new_mat_size-1]=self.affinity_mat

            tmp_mat[-1,:] = self.cal_sim_row(obj_name, self.current_name_set)
            tmp_mat[:,-1] = self.cal_sim_row(obj_name, self.current_name_set).T
            tmp_mat[tmp_mat<0]=0

            self.affinity_mat = tmp_mat
            # print('affinity mat')
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
                C_n_name = self.masked_selector(self.current_name_set, label, i)
                representative=np.mean(C_n,axis=0)
                min_sim = 1
                min_idx = 0
                for j in range(C_n.shape[0]):

                  sim = self.cal_sim(C_n_name[j], C_n_name)
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

                # print('cluster n {}/ {}'.format(i, self.cluster_n))
                new_representative_feature = self.graph.get_representative_feature(-1)
                self.graph.g.vs[self.graph.cluster_representatives[-1]]['features'] = new_representative_feature

              new_representative_feature = self.graph.get_representative_feature(old_NCD_cluster_idx)
              self.graph.g.vs[old_NCD_idx]['features'] = new_representative_feature
              self.graph.add_representative(None)


              self.current_set = []
              self.current_name_set = []
              self.affinity_mat = np.zeros((0,0))
              self.cluster_n = 1


    def cal_sim_row(self, x, data_list):
        # return x@np.array(self.data_set).T
        a = []
        for y in data_list:
          a.append(self.cal_sim(x, [y]))
        return np.array(a)


    def cal_sim_ori(self,x,y):
      if self.metric_type == 'dist':
        rou = 15
        dist = np.linalg.norm(x-y)
        return np.exp(-(dist**2/(2*rou**2)))
      elif self.metric_type == 'sim':
        return x@y
      else:
        raise Exception('metric type error')

    def cal_sim(self, target_item_name, check_item_names, unattached=False, given_feature=None):

      check_features = []
      if not unattached:
        target_item_feature = self.graph.get_feature(target_item_name)
      else:
        target_item_feature = given_feature

      for check_item_name in check_item_names:
        tmp = self.graph.get_feature(check_item_name)
        if tmp is None:
          raise Exception('check item {} not in graph'.format(check_item_name))
        check_features.append(tmp)
      average_feature = np.mean(check_features, axis=0)
      # print('target {}'.format(target_item_feature.shape))
      # print('check {}'.format(average_feature.shape))
      semantic_sim = self.cal_sim_ori(target_item_feature, average_feature)

      dist = 0
      for check_item_name in check_item_names:
        dist += self.geo_graph.get_item_dist(target_item_name, check_item_name)
      avg_dist = dist/len(check_item_names)
      geo_sim = np.exp(-avg_dist/(2*self.geo_graph.dist_param**2))
      # print('geo_dim {}'.format(geo_sim))
      # print('sim {}'.format(semantic_sim*geo_sim))
      return semantic_sim*geo_sim



    def cal_sim_mat(self,x):
      upper_part = np.zeros((len(x),len(x)))
      for i in range(len(x)):
        for j in range(i, len(x)):
          upper_part[i,j] = self.cal_sim(x[i], [x[j]])
      # print(upper_part+upper_part.T-np.identity(x.shape[0]))
      return upper_part+upper_part.T-np.identity(len(x))

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
        model = SpectralClustering(n_clusters=cluster_n, affinity='precomputed', assign_labels='kmeans')#
        labels = model.fit_predict(affinity_matrix)

        eigvals, eigvecs = eigh(laplacian_normalized)
        embedding = eigvecs[:, :cluster_n]
      else:

        degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix))
        laplacian_normalized = np.eye(affinity_matrix.shape[0]) - D_inv_sqrt @ affinity_matrix @ D_inv_sqrt
        eigvals, eigvecs = eigh(laplacian_normalized)
        embedding = eigvecs[:, :cluster_n]
        kmeans = KMeans(n_clusters=cluster_n)
        labels = kmeans.fit_predict(embedding)

      return labels#, embedding, eigvecs, eigvals, np.sqrt(degree_matrix), np.sqrt(degree_matrix)-affinity_matrix, affinity_matrix

    def masked_selector(self, a,b, mask_value):
      result,_=zip(*list(filter(lambda pair: pair[1]==mask_value, zip(a,b))))
      return result

if __name__ == '__main__':

    np.random.seed(0)
    room_cluster_data = np.load('obj_room_features.npy')
    room_cluster_data = np.concatenate((room_cluster_data, room_cluster_data[5:10]))
    room_cluster_data = np.concatenate((room_cluster_data, room_cluster_data[35:40]))

    item_list_dict = {'Bedroom': ['Bed', 'Blanket', 'Nightstand', 'Pillows', 'Wardrobe'],
                      'Bathroom': ['Bathtub', 'Shower shelf', 'Toilet', 'Toilet sink', 'Towel rack'],

                      'Kitchen': ['Dish rack', 'Kitchen sink', 'Microwave', 'Refrigerator', 'Stove'],
                      'Living Room': ['Armchair', 'Coffee table', 'Sofa', 'TV remote holder', 'TV stand'],
                      # 'gym': ['Beach press', 'Cable Machine', 'Dumbbell', 'Roman chair', 'Spinning bike'],
                      # 'Bedroom2': ['Bed2', 'Blanket2', 'Nightstand2', 'Pillows2', 'Wardrobe2']
                      }

    item_list_dict_ori = {'Bathroom': ['Bathtub', 'Shower shelf', 'Toilet', 'Toilet sink', 'Towel rack'],
                          'Bedroom': ['Bed', 'Blanket', 'Nightstand', 'Pillows', 'Wardrobe'],
                          'Dining Room': ['Chairs', 'Charger plates', 'Dining table', 'Place mats', 'Salt and pepper shakers'],
                          'Kitchen': ['Dish rack', 'Kitchen sink', 'Microwave', 'Refrigerator', 'Stove'],
                          'Laundry': ['Clothes basket', 'Detergent dispenser', 'Dryer', 'Drying rack', 'Washer'],
                          'Living Room': ['Armchair', 'Coffee table', 'Sofa', 'TV remote holder', 'TV stand'],
                          'Study': ['Bookcase', 'Desk lamp', 'Desk organizer', 'Desk', 'Monitor stand'],
                          'gym': ['Beach press', 'Cable Machine', 'Dumbbell', 'Roman chair', 'Spinning bike'],
                          'Bedroom2': ['Bed2', 'Blanket2', 'Nightstand2', 'Pillows2', 'Wardrobe2'],
                          'gym2': ['Beach press2', 'Cable Machine2', 'Dumbbell2', 'Roman chair2', 'Spinning bike2']
                          }

test_sp = NewIncrementalSPDivision(item_list_dict, 'sim', sim_t=0.018, use_sklearn=False)#0.018

feature_list = []

def find_item_idx(item_name_input, item_list_dict):
  global feature_list
  for i, r in enumerate(item_list_dict.keys()):
    for j, item_name in enumerate(item_list_dict[r]):
      if item_name_input == item_name:
        feature_list.append(room_cluster_data[i*5+j])
        return i*5+j

  return None

for r in item_list_dict.keys():
  for item in item_list_dict[r]:
    test_sp.input_signal(room_cluster_data[find_item_idx(item, item_list_dict_ori)], item)

feature_list = np.array(feature_list)

sim_mat = feature_list@feature_list.T

dist_mat = test_sp.geo_graph.get_dist_mat()
overall_mat = sim_mat*dist_mat

connectivity = test_sp.graph.room_connectivity(test_sp.geo_graph)
# print(connectivity)
test_sp.graph.graph_vis()
