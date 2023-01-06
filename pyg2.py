import numpy as np
import torch
import torch_geometric.datasets as datasets
import torch_geometric.data as data
import torch_geometric.transforms as transforms
import networkx as nx
from torch_geometric.utils.convert import to_networkx


embeddings = torch.rand((100, 16), dtype=torch.float)

rows = np.random.choice(100, 500)
cols = np.random.choice(100, 500)
edges = torch.tensor([rows, cols])

edges_attr = np.random.choice(3,500)
ys = torch.rand((100)).round().long()
graph = data.Data(x=embeddings, edge_index=edges, edge_attr=edges_attr, y=ys)
print(graph)

for prop in graph:
    print(prop)

vis = to_networkx(graph)

node_labels = graph.y.numpy()

import matplotlib.pyplot as plt
plt.figure(1,figsize=(10,9)) 
nx.draw(vis, cmap=plt.get_cmap('Set3'),node_color = node_labels,node_size=70,linewidths=6)
# plt.show()

##Batch
graph2 = graph
batch = data.Batch().from_data_list([graph, graph2])

##Cluster
#cluster = data.ClusterData(graph, 5)
#clusterloader = data.ClusterLoader(cluster)

##Sampler
sampler = data.NeighborSampler(graph.edge_index, sizes=[3,10], batch_size=4,
                                  shuffle=False)
                        
for s in sampler:
    print(s)
    break


print("Batch size:", s[0])
print("Number of unique nodes involved in the sampling:",len(s[1]))
print("Number of neighbors sampled:", len(s[2][0].edge_index[0]), len(s[2][1].edge_index[0]))








