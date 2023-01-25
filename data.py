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

    def switch_numpy(self):
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
# print(d1.switch_numpy())
print(d1.switch_numpy())
# print(d1.save_data('/home/jeni/Desktop/gcn/dataset/node_features/'))




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