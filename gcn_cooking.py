import networkx as nx
import yaml
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.community.modularity_max import greedy_modularity_communities


list_key = []
list_value_isA = []
list_value_HasProperty = []
list_value_AtLocation = []
list_isA = []
list_HasProperty = []
list_AtLocation = []
list_edge_isA = []

with open('/home/jeni/Desktop/gcn/test1.yaml') as f:
    yaml_data = yaml.load(f, Loader = yaml.FullLoader)
# print(yaml_data,'\n')
    
for key, value in yaml_data.items():
    list_key.append(key)
    list_value_isA.append(value['isA'][0])
    # list_value_HasProperty.append(value['HasProperty'][0])
    # list_value_AtLocation.append(value['AtLocation'])
# print(list_key,'\n')
# print(list_value_isA,'\n')
# print(list_value_HasProperty,'\n')
# print(list_value_AtLocation,'\n')

for i in range(len(list_key)):
    # tuple_isA = tuple([list_key[i] , list_value_isA[i]])
    # list_isA.append(tuple_isA)
    dict_isA = {"isA": list_value_isA[i]}
    tuple_node_isA = tuple([list_key[i], dict_isA])
    list_isA.append(tuple_node_isA)
    
    #edges
    tuple_edge_isA = tuple([list_key[i],list_value_isA[i]])
    list_edge_isA.append(tuple_edge_isA)
# print(list_isA)
# no_duplicate_isA = list(set(list_value_isA))
# print(no_duplicate)
# list_node = list_key + no_duplicate_isA    
    


# for num,node_name in enumerate(list_node):
#     print(num, node_name)





g = nx.Graph()

# G.add_nodes_from([(1, dict(size=11)), (2, {"color": "blue"})])

g.add_nodes_from(list_isA)
g.add_edges_from(list_edge_isA, weight=0.063, relation='isA')

for node in g.nodes(data=True):
  print(node)
print(g.edges())


communities = greedy_modularity_communities(g) # Greedy modularity communities: starting each nodes to maximize
g.number_of_nodes(), g.number_of_edges()


colors = np.zeros(g.number_of_nodes())
# colors = np.array([])



for i, com in enumerate(communities):
    # print(com)
    list_colors = np.full(len(list(com)),0)
    colors[list_colors] = i
    # colors = np.append(colors, list_colors)
    # print(i, list(com),'\n',list_colors) # Coloring the com
    # print(len(list_colors), list_colors)
print(colors)
    
    
    
# n_classes = np.unique(colors).shape[0] #np.unique -> flatting one dimension shape
# labels = np.eye(n_classes)[colors.astype(int)] # eye: diagonal with 1 else 0 (2d array)
# # print(labels)
# fig, ax = plt.subplots(figsize=(10,10)) # Determine the size of the figure.
# # pos = nx.spring_layout(g, k=5/np.sqrt(g.number_of_nodes()))
# kwargs = {"cmap":'gist_rainbow', "edge_color":'gray'} # kwargs = keyword arguments (save as dictionary type with a parameter name)
# nx.draw(g, with_labels = True, ax=ax, node_color=colors, **kwargs)

# plt.show()