### GCN
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def MLP(layers, input_dim, dropout = 0.0):
    mlp_layers = [torch.nn.Linear(input_dim, layers[0])]

    for layer_num in range(0, len(layers)-1):
        mlp_layers.append(torch.nn.ReLU())
        mlp_layers.append(torch.nn.Linear(layers[layer_num], layers[layer_num + 1]))

    return torch.nn.Sequential(*mlp_layers)