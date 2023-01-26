import torch
import pandas as pd


## Node features
nf_csv = pd.read_csv('/home/jeni/Desktop/gcn/gt_node_features_re.csv')
nf_re = nf_csv[["Type_Bowl", "Type_Box","Type_Table","Property_V_Velcro","Property_C_Contain","Property_G_Graspable"]]
print("[Node Features CSV]\n",nf_re)
node_features = torch.Tensor(nf_re.values)
print('\n[Node_features]\n',node_features)








## Torch import
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




edge_index = torch.tensor([1,0])


x = node_features

# conv = GCNConv(4,1)
# x = conv(x,edge_index)

print(x.size()) ## torch.Size([8,6])
# print(x.size(dim=1)) # 6
print(x, '\n')



# ## Error check
# from torch_scatter import scatter_add
# num_nodes = 8
# embed_size = 4

# # src = torch.randint(0, num_nodes, (num_nodes, embed_size)) # 범위 내 정수를 균등하게 생성
# # src_index = torch.tensor([0,0,0,1,1,0,0,0]) 
# # tmp = torch.index_select(src, 0, src_index) # shape [num_edges, embed_size]
# # print("input: ")
# # print(tmp)
# target_index = torch.tensor([0,0,0,0,0,1,1,0]) # Target_index = goal = (Type_Box,Property_V:1 (velcro(o)), Property_C:0 (contain(o)), Property_G (graspable))
# aggr = scatter_add(node_features, target_index, 0) # shape [num_nodes, embed_size]

# # print("agg out:")
# print(aggr)

# # behind the sence, torch.scatter_add is used
# # repeat the edge_index
# index2 = target_index.expand((embed_size, target_index.size(0))).T
# # same result by using torch.scatter_add
# aggr2 = torch.zeros(num_nodes, embed_size, dtype=node_features.dtype).scatter_add(0, index2, node_features)














