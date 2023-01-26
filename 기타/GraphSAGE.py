import torch_geometric
from torch_geometric.datasets import Planetoid

## Additional import commands
import networkx as nx
import torch_geometric.data as data


use_cuda_if_available = False
dataset = Planetoid(root="tutorial1",name= "Cora")
print("Kind of dataset:",dataset)
# print("number of graphs:\t\t",len(dataset))
# print("number of classes:\t\t",dataset.num_classes)
# print("number of node features:\t",dataset.num_node_features)
# print("number of edge features:\t",dataset.num_edge_features)
# print(dataset.data)
# print("edge_index:\t\t",dataset.data.edge_index.shape)
print("edge_index:",dataset.data.edge_index)
# print("\n")
# print("train_mask:\t\t",dataset.data.train_mask.shape)
print("train_mask:",dataset.data.train_mask)
print("\n")
print("x:\t\t",dataset.data.x.shape)
print("data_x:",dataset.data.x)
print("\n")
print("y:\t\t",dataset.data.y.shape)
print("data.y:",dataset.data.y)




# node_labels = dataset.y.numpy()

# import matplotlib.pyplot as plt
# plt.figure(1,figsize=(10,9)) 
# nx.draw(dataset, cmap=plt.get_cmap('Set3'),node_size=70,linewidths=6)
# plt.show()

# Graph Network
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

data = dataset[0]

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv = SAGEConv(dataset.num_features,
                             dataset.num_classes,
                             aggr="max") # max, mean, add ...)

    def forward(self):
        x = self.conv(data.x, data.edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_available else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
print(device)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

best_val_acc = test_acc = 0
# for epoch in range(1,100):
#     train()
#     _, val_acc, tmp_test_acc = test()
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         test_acc = tmp_test_acc
#     log = 'Epoch: {:03d}, Val: {:.4f}, Test: {:.4f}'
    
#     if epoch % 10 == 0:
#         print(log.format(epoch, best_val_acc, test_acc))