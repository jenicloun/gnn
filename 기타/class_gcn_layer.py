### GCN
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv

# Build NN
class Net_ex(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.flatten(x)
        Net_Out = self.layer(x)
        return Net_Out


# Build Train Function
def train(dataloader, model, loss_fn, optimizer):
    pbar = tqdm(dataloader, desc=f'Training')
    for batch, (X, y) in enumerate(pbar):
        # X, y = X.to(device), y.to(device)
 
        # Feedforward
        pred = model(X)
 
        # Calc. Loss
        loss = loss_fn(pred, y)
 
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 

# Build Test Function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, correct = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            # X, y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    print(f'Test Accuracy: {(100*correct):>0.1f}%     Loss: {loss:>8f} \n')
 
    return 100*correct, loss


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

# def MLP(layers, input_dim, dropout = 0.0):
#     mlp_layers = [torch.nn.Linear(input_dim, layers[0])]

#     for layer_num in range(0, len(layers)-1):
#         mlp_layers.append(torch.nn.ReLU())
#         mlp_layers.append(torch.nn.Linear(layers[layer_num], layers[layer_num + 1]))

#     return torch.nn.Sequential(*mlp_layers)

class MLP(nn.Module) :
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.flatten(x)
        Net_Out = self.layer(x)
        return Net_Out