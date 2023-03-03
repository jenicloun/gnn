import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Dataset
import os
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid, Amazon


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = "./data"

# exist_ok == True인 경우 error 발생 없이 넘어감, 없을 경우에만 생성
# exist_ok == False인 경우 이미 해당 디렉토리가 존재하는 경우 exception error뜸
os.makedirs(data_dir, exist_ok= True) 

### Cora ### (Class: 7)
dataset = Planetoid(root=data_dir, name= 'Cora')
# Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])


### CiteSeer ### (Class: 6)
# dataset = Planetoid(root=data_dir, name="CiteSeer")
# Data(x=[3327, 3703], edge_index=[2, 9104], y=[3327], train_mask=[3327], val_mask=[3327], test_mask=[3327])

### PubMed ### (Class: 3)
# dataset = Planetoid(root=data_dir, name= 'PubMed')
# Data(x=[19717, 500], edge_index=[2, 88648], y=[19717], train_mask=[19717], val_mask=[19717], test_mask=[19717])


# Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.x
# Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.tx
# Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.allx
# Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.y
# Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.ty
# Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.ally
# Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.graph
# Downloading https://github.com/kimiyoung/planetoid/raw/master/data/ind.cora.test.index
# Processing...
# Done!
# Cora()

# dataset = Amazon(root= data_dir, name=)

print(dataset)


data0 = dataset[0]
print(data0)

# edge_index = data0.edge_index.numpy()

# print(edge_index.shape)
# Data(x=[19717, 500], edge_index=[2, 88648], y=[19717], train_mask=[19717], val_mask=[19717], test_mask=[19717])
# (2, 88648)

# Node 30이 연결된 edge들의 종류
# edge_example = edge_index[:, np.where(edge_index[0]==30)[0]]
# print(edge_example)

# Cora node dataset features 1433개, Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])

### For node features (Node features are composed of 0s and 1s.)
# node_feature = data0.x[0][:150]
# print(node_feature) 
# print(len(node_feature)) # Total 1433


### For edge features
edge_features = data0.y
print(edge_features)
print(len(edge_features)) 

# Cora edge dataset features 2708개, Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
# e.g.) tensor([3, 4, 4,  ..., 3, 3, 3])

# import collections
# counter = collections.Counter(data0.y.numpy())
# counter = dict(counter)
# print(counter)
# count = [x[1] for x in sorted(counter.items())]
# plt.figure(figsize=(10, 6))
# plt.bar(range(7), count)
# plt.xlabel("class", size=20)
# plt.show()

# from data_read_jeni import dataset2
# print(dataset2)