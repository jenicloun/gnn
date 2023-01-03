import yaml
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.nn.functional as f
from torch_geometric.nn import GCNConv
from torch.nn import Linear, ReLU




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
g.add_edges_from(list_isA, weight=0.03, relation = 'isA')
degree = nx.degree(g)

print(g.number_of_nodes(), g.number_of_edges())

g.edges()
fig, ax = plt.subplots(figsize=(10,10))
nx.draw(g, node_size=[50 + v[1]*50 for v in degree], ax=ax, with_labels =True)
plt.show()

#### training & testing ####
# train_set, test_set = train_test_split(g, train_size = 10)
# val_set, test_set = train_test_split(test_set, train_size = 10)
# print(train_set.shape, val_set.shape, test_set.shape)















