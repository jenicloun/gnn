import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import os


class Graphdataset(Dataset):
    def __init__(self, path):
        nf_csv = pd.read_csv(path)
        # print("[Node Features CSV]\n",nf_csv)
        node_features = torch.Tensor(nf_csv.values)
        # print('\n[Node_features]\n',node_features)
        self.x = node_features[:,2:]
     

    def __len__(self): # data size return
        return len(self.x)


    def __getitem__(self, index): # 
        return super().__getitem__(index)



ds = Graphdataset('/home/jeni/Desktop/gcn/dataset/node_features/nf1.csv')
print(len(ds)) # data size = 8
print(ds.x)
print(ds.x.size()) # torch.Size([8,13])




class Randomposdata(Dataset):
    def __init__(self, file_path):
        nf_csv = pd.read_csv(file_path)
        # print("[Node Features CSV]\n",nf_csv)
        node_features = torch.Tensor(nf_csv.values)
        # print('\n[Node_features]\n',node_features)
        self.x = node_features[:,8:]

    def switch_position(self):
        tensor_matrix = self.x
        numpy_matrix = tensor_matrix.numpy()
        rearrange_arr = np.rollaxis(numpy_matrix,0,1)
        tensor_data = torch.tensor(rearrange_arr)
        self.tensor_data = tensor_data

        return tensor_data

    def save_data(self,filepath):
        # pathList = []
       
        # folders = os.listdir(homepath)
        # for folder in folders:
        #     foldername = folder
        #     files = os.listdir(homepath+folder)
        #     for file in files:
        #         filename=file
        #         path = homepath+foldername+'/'+filename

        os.chdir(filepath)
        for i in range(0, len(self.x)):
            final_path = filepath + 'nf_ex'+str(i)+'.csv'
            np.savetxt(final_path, self.tensor_data, delimiter =',', fmt="%d")        


d1 = Randomposdata('/home/jeni/Desktop/gcn/dataset/node_features/nf1.csv')
# print(d1.switch_position())
# print(d1.save_data('/home/jeni/Desktop/gcn/dataset/node_features/'))


## Creating OwnDatasets
import os.path as osp
import torch
from torch_geometric.data import InMemoryDataset, download_url

class MyDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None):
        self.data_list = data_list
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])










## GCNConv layer
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv

from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

## EdgeConv import
import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)

### GCN

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, node_feature_size, feature_size):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(node_feature_size, feature_size)
        self.conv2 = GCNConv(feature_size, feature_size)
        self.conv3 = GCNConv(feature_size, feature_size)

    def forward(self, data, attr = 'x'):
        x, edge_index = getattr(data, attr), data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training = self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training = self.training)
        x = self.conv3(x, edge_index)
        
        return x