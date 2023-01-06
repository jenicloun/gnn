import torch
from torch_geometric.data import Data


edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

dataset = Data(x=x, edge_index=edge_index.t().contiguous())

print(dataset) # x = [3, 1] -> Total 3 nodes and 1 node attribute
print(dataset.edge_index) #edge_index = [2, 4] -> Total 4/2= 2
print(dataset.x)