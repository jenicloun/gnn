import matplotlib.pyplot as plt
# create number for each group to allow use of colormap
from itertools import count
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

# Change the location

with open('/home/jeni/Desktop/gcn/test1.yaml') as f:
    yaml_data = yaml.load(f, Loader = yaml.FullLoader)
# print(yaml_data,'\n')
    
for key, value in yaml_data.items():
    list_key.append(key)
    list_value_isA.append(value['isA'][0])
    # print(key)
    # list_value_HasProperty.append(value['HasProperty'][0])
    # list_value_AtLocation.append(value['AtLocation'])
# print(list_key,'\n')
# print(list_value_isA,'\n')
# print(list_value_HasProperty,'\n')
# print(list_value_AtLocation,'\n')

for i in range(len(list_key)):
    tuple_isA = tuple([list_key[i] , list_value_isA[i]])
    # tuple_HasProperty = tuple([list_key[i], list_value_HasProperty[i]])
    # tuple_AtLocation = tuple([list_key[i], list_AtLocation[i]])
    list_isA.append(tuple_isA)
    # list_HasProperty.append(tuple_HasProperty)
#     # list_AtLocation.append(tuple_AtLocation)
# print(list_isA)
# print(list_HasProperty)

g = nx.Graph()
g.add_nodes_from(list_key)
g.add_nodes_from(list_value_isA)
g.add_edges_from(list_isA, weight=0.063, relation = 'isA')
# g.add_edges_from(list_HasProperty, weight=0.002, label = 'HasProperty')
degree = nx.degree(g)
nx.draw(g, node_size=[50 + v[1]*50 for v in degree], with_labels =True)
g.edges()

plt.show()

# get unique groups
groups = set(nx.get_node_attributes(g,'group').values())
mapping = dict(zip(sorted(groups),count()))
nodes1 = g.nodes()
colors = [mapping[g.node[n]['group']] for n in nodes1]

# drawing nodes and edges separately so we can capture collection for colobar
pos = nx.spring_layout(g)
ec = nx.draw_networkx_edges(g, pos, alpha=0.2)
nc = nx.draw_networkx_nodes(g, pos, nodelist=nodes1, node_color=colors, 
                            with_labels=False, node_size=100, cmap=plt.cm.jet)
plt.colorbar(nc)
plt.axis('off')
plt.show()