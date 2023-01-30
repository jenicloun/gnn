import os
import torch
import pandas as pd
import torch_geometric
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Dataset

## Check dataframe
class LoadDataFrame(Dataset):
    ## Load DataFrame
    def __init__(self, DATA_PATH):
        data = pd.read_csv(DATA_PATH)
        self.data = data
        # print(data.head)

    ## Example of general information about the dataset
    def general_info(self):
        print(self.data.shape)
        print(self.data["Position_t4"].value_counts()) # value) 6 zeros 2 ones

    ## Quick check with versions
    def checkversion(self):
        print(f"Torch version: {torch.__version__}") 
        print(f"Cuda available: {torch.cuda.is_available()}") 
        print(f"Torch geometric version: {torch_geometric.__version__}") 


## Dataset
class MakeDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path

        # Search path
        search_path = os.path.join(os.getcwd(), self.root_path)
        self.search_path = search_path


    # Creating sample data
    def rand_sample(self, file_name, folder_name, save_dir, n):

        # Read node feature datasets
        node_path = os.path.join(self.search_path, folder_name, file_name)
        nf_csv = pd.read_csv(node_path, index_col=0)  # index_col drops 'Unnamed' column

        # Make random samples with torch
        for i in range(n):
            p = torch.randint(0,2,(5,))
            z = torch.zeros(3)
            concat = torch.cat((p,z),0).tolist()
            
        # Transform property from list values / Switch only Property_V for while
            nf_csv['Property_V_Velcro'] = concat # Only list can switch values of the column
        

        # Save files
            final_path = os.path.join(self.search_path, save_dir, 'nf_ex' + str(i) + '.csv')
            nf_csv.to_csv(final_path)  
            
    # Getting node features
    def node_feature(self, csv_file, root_dir):
        # Search path
        node_path = os.path.join(self.search_path, root_dir, csv_file)

        # Read csv file to tensor
        nf_csv = pd.read_csv(node_path, index_col=0)
        nf_drop = nf_csv.drop(labels='ID',axis=1) # drop the "ID" column / axis=0 (row), axis=1(column)
        self.x = torch.Tensor(nf_drop.values) # dataframe to tensor

        return self.x

    # Data size return
    def __len__(self): 
        return len(self.x)


    # Sampling one specific data from dataset
    def __getitem__(self, index): 
        part_x = torch.FloatTensor(self.x[index])
        return part_x

    # Getting edge_features - edge_index, edge_attribute
    def edge_feature(self, csv_file, root_dir):
        # Search path
        edge_path = os.path.join(self.search_path, root_dir, csv_file)

        # Read csv file to tensor
        ef_csv = pd.read_csv(edge_path, index_col=0)
        ef = ef_csv.drop(labels='ID',axis=1) # drop the "ID" column / axis=0 (row), axis=1(column)
        

        # edge_index: [2, num_edges], edge_attr: [num_edges, dim_edge_features]
        
        ####################### Recommend to change #################
        ## Edge index

        list_i = []
        list_c = []
       
        for index in range(len(ef.columns)):
            for column in range(len(ef.columns)):
                if ef.iat[index, column] == 1:    # Recommend to change '.iat' to speed up
                    list_i.append(index)
                    list_c.append(column)
        
        tensor_i = torch.tensor(list_i)
        tensor_c = torch.tensor(list_c)
        edge_index = torch.cat((tensor_i, tensor_c), dim=0).reshape(2,len(tensor_i))
         

        ############################################################
        ## Edge attribute # on, in, attach 임의로 정해서 만들어 놓기
        # edge_attr = torch.Tensor(ef.values)
        edge_attr = torch.randint(2,(8,3))
        
        return edge_index, edge_attr

 

## Print

make_data = MakeDataset(root_path='dataset')


# print(make_data.rand_sample(folder_name='node_features',file_name='nf1.csv',save_dir='node_features', n=13))
x = make_data.node_feature(csv_file='nf1.csv', root_dir='node_features')
edge_index, edge_attr = make_data.edge_feature(csv_file='ef0.csv', root_dir='edge_features')

# print(edge_attr)
# data = x, edge_index, edge_attr


# print("Node Feature:\n",x) #Number of nodes: 8, node feature: 13 (8,13)
# print("\nEdge index:\n",edge_index) #(2,8)
# print("\nEdge attr:\n", edge_attr) #shape [8,8]


## Making graph data
from torch_geometric.data import Data
dataset = Data(x=x, edge_index= edge_index, edge_attr=edge_attr) # Data(x=[8, 13], edge_index=[2, 8], edge_attr=[8, 8])

print("Node Feature:\n",dataset.x) #Number of nodes: 8, node feature: 13 (8,13)
print("\nEdge index:\n",dataset.edge_index) #(2,8)
print("\nEdge attr:\n", dataset.edge_attr) #shape [8,8]

# print(dataset.x) 
# print(dataset.keys) # 'x', 'edge_attr', 'edge_index'
# print(dataset.num_nodes) # 8
# print(dataset.has_isolated_nodes()) #False
# print(dataset.has_self_loops()) #False
# print(dataset.is_directed()) #True


####################################################### GNN #################################################################

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim


# Basically the same as the baseline except we pass edge features 

#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"


#### Action input embedding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# word_to_index = {"pick":0, "place":1, "pour":2}
# embs = nn.Embedding(1,3)  ## nn.Embedding(num_embeddings, embedding_dim) 
# lookup_tensor = torch.tensor([word_to_index["pick"]], dtype=torch.long)
# pick_embed = embs(lookup_tensor)
# print(pick_embed)



#### ActionModel

class ActionModel(nn.Module):
    def __init__(self, hidden_dim, num_action, node_feature_size, edge_feature_size):
        super(ActionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        
        self.convs = [GATConv(in_channels=self.node_feature_size, out_channels=self.hidden_dim, edge_dim=self.edge_feature_size),
                      GATConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, edge_dim=self.edge_feature_size)]
        
        self.action_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.num_action),
            nn.LeakyReLU(),
        )
        self.node_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv in self.convs[:-1]:
            x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr) # adding edge features here!
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr) # edge features here as well
        
        action_input_emb = x.mean()                                    #(x의 피쳐? 들을 합치는과정 avg)
        node_features = x
                
        action_prob = nn.Softmax(self.action_layers(action_input_emb))
        
        
        node_scores = []
        for feature in node_features:
            node_scores.append(nn.Sigmoid(self.node_layers(feature)))
        return action_prob, node_scores   

#test
hidden_dim = 64
num_action = 3 # 액션 개수 [pick, place, pour]
node_feature_size = 13#노드 feature 크기
edge_feature_size = 3 #노드 사이의 relation 개수 [on, in, attach]

model = ActionModel(hidden_dim, num_action, node_feature_size, edge_feature_size)

test_data = dataset

test_action_prob, test_node_scores = model(test_data)

L_Action = nn.CrossEntropyLoss().to(device)
L_NodeScore = nn.CrossEntropyLoss().to(device)

action_prob_target = [1, 0, 0]
node_scores_target = [1, 0, 0, 0, 0, 0, 0, 0]

L_total = L_Action(test_action_prob, action_prob_target) + L_NodeScore(test_node_scores, node_scores_target)
print(L_total)

