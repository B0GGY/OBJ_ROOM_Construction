import igraph as ig
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)

class OPGraphCreationTest:
    def __init__(self, item_list_dict, dist_param=2):
        self.graph = ig.Graph()
        self.graph.vs['node_type'] = []
        self.graph.vs['item_name'] = []
        self.graph.es['path_finding_weight'] = []
        # self.get_graph(item_list_dict)
        self.place_node_n = 0
        self.item_node_n = 0
        self.item_list_dict = item_list_dict
        self.dist_param = dist_param

    def get_node_idx_by_name(self,node_name):
        index = self.graph.vs.find(name=node_name).index
        return index

    def add_place_vertice(self, add_num):
        if add_num>0:
            self.graph.add_vertices(['place_node{}'.format(i) for i in range(self.place_node_n, self.place_node_n+add_num)])
            # edge_list = []
            for i in range(self.place_node_n, self.place_node_n+add_num):
                self.graph.vs[self.get_node_idx_by_name('place_node{}'.format(i))]['node_type'] = 'place'
                if i < self.place_node_n+add_num-1:
                    #edge_list.append(('place_node{}'.format(i), 'place_node{}'.format(i + 1)))
                    self.graph.add_edge('place_node{}'.format(i), 'place_node{}'.format(i + 1))
                    self.graph.es[self.graph.get_eid('place_node{}'.format(i), 'place_node{}'.format(i + 1))]['path_finding_weight'] = 1

            # self.graph.add_edges(edge_list)
            if self.place_node_n != 0:
                # print(self.place_node_n)
                # print(add_num)
                self.graph.add_edge('place_node{}'.format(self.place_node_n - 1), 'place_node{}'.format(self.place_node_n))
                self.graph.es[self.graph.get_eid('place_node{}'.format(self.place_node_n - 1), 'place_node{}'.format(self.place_node_n))][
                    'path_finding_weight'] = 1
            self.place_node_n += add_num

    def add_item_vertice(self, item_name, item_node_n):
        self.graph.add_vertex('{}'.format(item_name))
        self.graph.vs[self.get_node_idx_by_name(item_name)]['node_type']='item'
        for i in range(item_node_n):
            self.graph.add_edge('{}'.format(item_name),'place_node{}'.format(self.place_node_n-1-i))
            self.graph.es[self.graph.get_eid('{}'.format(item_name),'place_node{}'.format(self.place_node_n-1-i))][
                'path_finding_weight'] = 1000
        self.item_node_n += 1

    def place_v_count(self):
        place_v_n = 0
        for i in range(self.graph.vcount()):
            print(self.graph.vs[i]['node_type'])
            if self.graph.vs[i]['node_type'] == 'place':
                place_v_n+=1
        return place_v_n

    def get_graph(self):

        first_item = True
        last_item_n = 0
        for room in self.item_list_dict.keys():
            if not first_item:
                room_separation_node_n = np.random.choice([4,5,6,7])
                self.add_place_vertice(room_separation_node_n)
            for item in self.item_list_dict[room]:
                item_node_n = np.random.choice([1,2,3,4],p=[0.1,0.4,0.4,0.1])
                is_covisible = np.random.choice([True,False],p=[0.2, 0.8])

                if is_covisible and not first_item:
                    covisible_node_n = np.random.choice([i for i in range(1, min(item_node_n,last_item_n)+1)])
                    new_node_n = item_node_n - covisible_node_n
                    self.add_place_vertice(new_node_n)
                    self.add_item_vertice(item, item_node_n)
                else:
                    item_seperation_node_n = np.random.choice([4,5,6,7])
                    self.add_place_vertice(item_seperation_node_n)
                    self.add_place_vertice(item_node_n)
                    self.add_item_vertice(item, item_node_n)


                first_item = False
                last_item_n = item_node_n
        # add some edge for loop
        self.graph.add_edge('place_node{}'.format(3), 'place_node{}'.format(90))
        self.graph.add_edge('place_node{}'.format(86), 'place_node{}'.format(124))
        self.graph.es[
            self.graph.get_eid('place_node{}'.format(3), 'place_node{}'.format(90))][
            'path_finding_weight'] = 1
        self.graph.es[
            self.graph.get_eid('place_node{}'.format(86), 'place_node{}'.format(124))][
            'path_finding_weight'] = 1


    def get_item_dist(self, source, target):
        source_idx = self.get_node_idx_by_name(source)
        target_idx = self.get_node_idx_by_name(target)
        vpath = self.graph.get_shortest_paths(source_idx, to=target_idx, weights=self.graph.es['path_finding_weight'], output='vpath')
        dist = 0
        for v in vpath[0]:
            if self.graph.vs[v]['node_type'] == 'place':
                dist+=1

        return dist

    def graph_visualization(self):

        layout = self.graph.layout("tree")
        coords = layout.coords


        x, y = zip(*coords)
        fig, ax = plt.subplots(figsize=(100, 100))

        for edge in self.graph.es:
          source, target = edge.tuple
          ax.plot(
              [x[source], x[target]],
              [y[source], y[target]],
              color="gray",
              lw=1
          )

        ax.scatter(x, y, s=500, c="lightblue", edgecolors="black", zorder=3)

        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.text(
                xi, yi + 0.05,
                f"{self.graph.vs[i]['name']}", fontsize=10, ha="center", va="center", zorder=4
            )

        ax.axis("off")

        plt.show()

    def get_dist_mat(self):
        item_list = []

        for room in self.item_list_dict.keys():
            for item in self.item_list_dict[room]:
                item_list.append(item)
        item_dist_mat = np.zeros((len(item_list),len(item_list)))
        for i in range(len(item_list)):
            for j in range(i, len(item_list)):
                item_dist_mat[i,j]=self.get_item_dist(item_list[i], item_list[j])
        item_dist_mat = item_dist_mat+item_dist_mat.T
        item_dist_mat_normalized = np.exp(-item_dist_mat/(2*self.dist_param**2))

        return item_dist_mat_normalized

    def path_check(self, path):

      path_list = []
      for node_id in path:
        neighbors = self.graph.neighbors(node_id, mode="all")
        place_flag = True
        for i in neighbors:
          if self.graph.vs[i]['node_type'] == 'item':
            place_flag = False
            path_list.append(self.graph.vs[i]['name'])
        if place_flag:
          path_list.append('place')

      return path_list


if __name__ == '__main__':
    item_list_dict = {'Bathroom': ['Bathtub', 'Toilet', 'Bathroom Sink', 'Shower shelf', 'Towel rack'],
                      'Bedroom': ['Nightstand', 'Bedside lamp', 'Bed', 'Wardrobe', 'Dresser'],
                      'Living Room': ['Sofa', 'Coffee table', 'TV stand', 'Armchair', 'Floor lamp'],
                      'Kitchen': ['Refrigerator', 'Stove', 'Microwave', 'Kitchen Sink', 'Utensil holder']}
    test_graph = OPGraphCreationTest(item_list_dict)
    test_graph.get_graph()
    test_graph.get_item_dist('Bathtub', 'Utensil holder')
    dist_mat = test_graph.get_dist_mat()
    # test_graph.graph_visualization()
    plt.imshow(dist_mat)
    plt.show()