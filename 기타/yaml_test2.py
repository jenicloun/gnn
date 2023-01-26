import csv
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import count




# Parsing
import pandas as pd
Ground = pd.read_csv("/home/jeni/Desktop/gcn/box_test1.csv")
print(Ground)

#### parsing relation

parse_r = Ground[["ID","Relation_In","Relation_On","Relation_Attach"]]

# print(parse_g)

splited1 = parse_r["Relation_In"].str.split(',',expand=True)
splited2 = parse_r["Relation_On"].str.split(',',expand=True)
splited3 = parse_r["Relation_Attach"].str.split(',',expand=True)

# print(splited1)

splited1.rename(columns= lambda x: "Relation_In_" + str(x), inplace=True) # Switch int to str
splited2.rename(columns= lambda x: "Relation_On_" + str(x), inplace=True)
splited3.rename(columns= lambda x: "Relation_Attach_" + str(x), inplace=True)

gt = pd.concat([Ground["ID"],splited1,splited2,splited3], axis=1)

# fill "NaN" data as "0"
df = gt.fillna(0)

### save
# df.to_csv('/home/jeni/Desktop/gcn/gt_box_test.csv')

print("\nGT:\n",df)


### parsing nodes

parse_n = Ground[["ID","Type","Property_V","Property_C","Property_G","Property_Color"]]
print(parse_n)


node_onehot = pd.get_dummies(parse_n)
print(node_onehot)

### save
# node_onehot.to_csv('/home/jeni/Desktop/gcn/gt_node_features.csv')

# Creating_graph

G = nx.Graph()


with open('/home/jeni/Desktop/gcn/gt_box_test.csv','r') as f:
    data = csv.reader(f)
    headers = next(data)
    for row in tqdm(data):
        for i in range(1,len(row)-1):
            G.add_node(row[i])
            G.add_node(row[i+1])
            if row[0] != row[i+1]:
                if G.has_edge(row[1],row[i+1]):
                    G[row[1]][row[i+1]]['weight'] += 1
                else:
                    G.add_edge(row[1],row[i+1], weight = 1)
            
# Remove selfloop data      
G.remove_edges_from(nx.selfloop_edges(G)) # Not for self
G.remove_node('0') # Nan data remove
       


G_nodes = G.number_of_nodes()
G_edges = G.number_of_edges()
print("Nodes =", G_nodes, "Edges =", G_edges)


# Figure size
plt.figure(figsize=(7,6))

# Node position (short for pos)
pos = nx.shell_layout(G)
# pos = nx.spring_layout(G)


# Node size
degree = nx.degree(G)


# nx.draw(G, pos, node_size = [50+v[1]*100 for v in degree], edge_color="black", with_labels = True)

nx.draw(G, pos, edge_color="black", with_labels = True)

# plt.show()


### Pandas -> Tensor
node_onehot.drop(columns='ID', inplace=True) #inplace means change original data or not
# node_tensor = node_features.values
# print("node_tensor" '\n' ,node_tensor)

### Making GNN - Link prediction

# One-hot to tensor
import torch
node_features = torch.Tensor(node_onehot.values)
# print(node_features)

# ########################################################## Testing 
import torch
from torch import nn
from torch.nn import Linear, ReLU


class GCNLayer(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(adj_matrix, node_feats) #torch.bmm: Batch Matrix Multiplication
        node_feats = node_feats / num_neighbours
        return node_feats

# node_feats = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
# adj_matrix = torch.Tensor([[[1, 1, 0, 0],
#                             [1, 1, 1, 1],
#                             [0, 1, 1, 1],
#                             [0, 1, 1, 1]]])

# print("Node features:\n", node_feats)
# print("\nAdjacency matrix:\n", adj_matrix)

# layer = GCNLayer(c_in=2, c_out=2)
# layer.projection.weight.data = torch.Tensor([[1., 0.], [0., 1.]])
# layer.projection.bias.data = torch.Tensor([0., 0.])

# with torch.no_grad():
#     out_feats = layer(node_feats, adj_matrix)

# print("Adjacency matrix", adj_matrix)
# print("Input features", node_feats)
# print("Output features", out_feats)

#GCN Layer integer, GAT Layer probability

test1 = pd.read_csv('/home/jeni/Desktop/gcn/Node_property.csv')
test1.drop(columns='ID', inplace=True) #inplace means change original data or not
# node_tensor = node_features.values
# print("node_tensor" '\n' ,node_tensor)

### Making GNN - Link prediction

# One-hot to tensor
import torch
node_features = torch.Tensor(test1.values)
print(node_features)

# relation graph = G
A = nx.adjacency_matrix(G)
Adj =A.todense()
print(Adj)




