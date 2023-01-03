import torch
import community as community_louvain
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
import numpy as np

g = nx.karate_club_graph()

x = torch.eye(g.number_of_nodes(), dtype=torch.float)
adj = nx.to_scipy_sparse_matrix(g).tocoo()
row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
edge_index = torch.stack([row, col], dim=0)
# Compute communities.
partition = community_louvain.best_partition(g)
y = torch.tensor([partition[i] for i in range(g.number_of_nodes())])
# Select a single training node for each community
# (we just use the first one).
train_mask = torch.zeros(y.size(0), dtype=torch.bool)
for i in range(int(y.max()) + 1):
    train_mask[(y == i).nonzero(as_tuple=False)[0]] = True
data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)
from torch_geometric.nn import GCNConv
LAYER= GCNConv(data.num_features, 2)
embedding = LAYER(data.x, data.edge_index)
embedding_np = embedding.detach().numpy()
# plot_features(embedding_np)