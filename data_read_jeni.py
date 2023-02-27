import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pickle
import natsort

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Dataset
import os
import natsort
import random
import networkx as nx
import matplotlib.pyplot as plt
import PIL
import re


## Dataset
class LoadDataset(Dataset):
    def __init__(self, problem):
        # Search path
        FILEPATH, _ = os.path.split(os.path.realpath(__file__))
        search_path = os.path.join(FILEPATH,'seq_dataset',problem)

        example = 'ex_1_2_3_4_5'
        example_search_path = os.path.join(FILEPATH,'seq_dataset',problem, example)
        
    
        self.FILEPATH = FILEPATH
        self.search_path = search_path
        self.example_search_path = example_search_path
        self.problem = problem
        self.example = example

        print("\n==========================================INIT======================================================")
        print("\n[File path]",FILEPATH)
        print("\n[Search_path]",search_path)
        print("\n[Example]", example)
        print("\n[Problem]", problem)
        print("\n==========================================INIT======================================================")

            
    # Getting node features
    def node_feature(self,i):
        # Search path
        node_path = os.path.join(self.search_path, 'node_features')
        nf_file_list = natsort.natsorted(os.listdir(node_path))
        nf_path = os.path.join(node_path, nf_file_list[i])

        # Read csv file to tensor
        nf_csv = pd.read_csv(nf_path, index_col=0)
        nf = torch.Tensor(nf_csv.values) # dataframe to tensor
        self.x = nf.to(dtype=torch.float32)

        return self.x

    # Data size return
    def __len__(self): 
        return len(self.x)


    # Sampling one specific data from dataset
    def __getitem__(self, index): 
        part_x = torch.FloatTensor(self.x[index])
        return part_x


    # Getting edge_features - edge_index, edge_attribute
    def edge_index(self, order, i):
    
        # Edge index path
        index_path = os.path.join(self.search_path,'edge_features', order, 'edge_index')
        ei_file_list = natsort.natsorted(os.listdir(index_path))
        ei_path = os.path.join(index_path, ei_file_list[i])
    
        # Read csv file to tensor
        ef = pd.read_csv(ei_path, index_col=0)

        list_i = []
        list_c = []
        ID_list = list(map(int, ef.columns))
        for index in range(len(ID_list)):
            for column in range(len(ID_list)):
                if ef.iat[index, column] == 1:    # Recommend to change '.iat' to speed up
                    list_i.append(ID_list[index])
                    list_c.append(ID_list[column])
    
        tensor_i = torch.tensor(list_i)
        tensor_c = torch.tensor(list_c)
        edge_tensor = torch.cat((tensor_i, tensor_c), dim=0).reshape(2,len(tensor_i))
        edge_index = edge_tensor.to(dtype=torch.int64)

########################################### 정리 중 ###########################################################
        # print("\n[Edge feature]\n", ef)
        # print("\n[ID list]\n", ID_list) # column을 list로 가지고 있다... 코드 바꿔도 됨
        # print("\nedge_tensor\n", edge_tensor)
        # print("\n[Edge index]\n", edge_index) # Node pair를 tensor로 가지고 있음
        return edge_index
    


    def edge_attr(self,order, i):
        # Edge attribute path
        attr_path = os.path.join(self.search_path,'edge_features', order, 'edge_attr')
        ea_file_list = natsort.natsorted(os.listdir(attr_path))
        ea_path = os.path.join(attr_path,ea_file_list[i])

    
        ea_csv = pd.read_csv(ea_path, index_col=0)
        ea = torch.Tensor(ea_csv.values) # dataframe to tensor
        edge_attr = ea.to(dtype = torch.float32)

########################################### 정리 중 ###########################################################
        # print("ea_csv",ea_csv) # 안에 내용이 전부 동일함
        # print("ea", ea)
        # print("edge attr", edge_attr)


        return edge_attr
    
if __name__ == '__main__':

    ## Print
    make_data = LoadDataset(problem = 'stacking_5')
  
    ######Train
    x_train = make_data.node_feature(i=0) # nf0.csv
    edge_index_train = make_data.edge_index(order= '1_2_3_4_5', i=0) # ei0.csv
    edge_attr_train = make_data.edge_attr(order= '1_2_3_4_5', i=0) # ea0.csv

    # print(x_train)
    # print(edge_index_train)
    # print(edge_attr_train)

    # ######Test
    x_test = make_data.node_feature(i=0) # nf0.csv

    edge_index_test = make_data.edge_index(order= '4_1_3_2_5', i=5) # ei5.csv
    edge_attr_test = make_data.edge_attr(order= '4_1_3_2_5', i=5) # ea5.csv


    # # Making graph data
    dataset = Data(x=x_train, edge_index= edge_index_train, edge_attr= edge_attr_train) 
    dataset2 = Data(x= x_test, edge_index= edge_index_test, edge_attr= edge_attr_test)

    # print("\nNode Feature:\n",dataset.x) 
    # print("\nEdge index:\n",dataset.edge_index) 
    # print("\nEdge attr:\n", dataset.edge_attr) 
    # print("Node Feature:\n",dataset2.x) 
    # print("\nEdge index:\n",dataset2.edge_index) 
    # print("\nEdge attr:\n", dataset2.edge_attr) 


def to_fully_connected(state_edge_index, state_edge_attr):
    edge_index_template = np.ones((9, 9), dtype=int)
    for idx in range(9):
        edge_index_template[idx][idx] = 0
        # print(state_edge_index.size(1))

    for idx in range(state_edge_index.size(1)):
        src, dest = (state_edge_index[0][idx].item(), state_edge_index[1][idx].item())
        edge_index_template[src][dest] = 0
        #edge_index_template[src][dest] = 1


    for src in range(9):
        for dest in range(9):
            if edge_index_template[src][dest] == 1:
                state_edge_index = torch.cat((state_edge_index, torch.tensor([[src],[dest]])), dim=1)
                #state_edge_index[0].append(src)
                #state_edge_index[1].append(dest)
                state_edge_attr = torch.cat((state_edge_attr, torch.zeros(1, 13)), dim=0)
                #state_edge_attr.append(np.zeros((13), dtype=int))

    #print(state_edge_index.shape)
    #print(state_edge_attr.shape)
    #input()
    #print(block_order, edge_index_template)
    return state_edge_index, state_edge_attr




def stacking_5_dataset():
    stacking_dataset = []

    make_data = LoadDataset(problem = 'stacking_5')
    make_data.node_feature(i=0)

    action_sequence = ['pick','place','pick','place','pick','place','pick','place']
    action_encoder = {'pick':[1, 0, 0], 'place':[0, 1, 0], 'pour':[0, 0, 1]}
    target_object_sequence = [4, 5, 3, 4, 2, 3, 1, 2]

    block_order_list = os.path.join(make_data.search_path, 'edge_features')


    for block_order in os.listdir(block_order_list):
        goal_edge_index= make_data.edge_index(order = block_order, i=8)
        goal_edge_attr = make_data.edge_attr(order = block_order, i=8)
        
        # print("\n[Block order]\n", block_order)
        # print("\n[Goal edge index]\n",goal_edge_index)
        # print("\n[Goal edge attr]\n",goal_edge_attr)
        
        goal_edge_index, goal_edge_attr = to_fully_connected(goal_edge_index, goal_edge_attr)

    #     block_order_num = list(map(int, block_order.split('_')))
    #     #print(block_order_num)

        for i in range(8):
            state_edge_index= make_data.edge_index(order=block_order, i= i)
            state_edge_attr = make_data.edge_attr(order=block_order, i= i)
     
            state_edge_index, state_edge_attr = to_fully_connected(state_edge_index, state_edge_attr)
            
            # print("\n[Block order]\n", block_order)
            # print("\n[Goal edge index]\n",goal_edge_index)
            # print("\n[State edge index]\n", state_edge_index)
            # print("\n[State edge attr]\n", state_edge_attr)

    #         action_code = torch.Tensor(action_encoder[action_sequence[i]])

    #         target_object_index = block_order_num[target_object_sequence[i]-1]
    #         target_object_score = np.zeros(x.shape[0], dtype=int)
    #         target_object_score[target_object_index] = 1
    #         target_object_score = torch.from_numpy(target_object_score).type(torch.FloatTensor)

    #         #write data to dictionary
    #         graph_dict_data = {'input':{'state':{},
    #                                     'goal':{}
    #                                     },
    #                         'target':{'action':[],
    #                                     'object':[]
    #                                     },
    #                         'info':str()
    #                                     }
            
    #         graph_dict_data['input']['state']['x'] = x
    #         graph_dict_data['input']['state']['edge_index'] = state_edge_index
    #         graph_dict_data['input']['state']['edge_attr'] = state_edge_attr

    #         graph_dict_data['input']['goal']['x'] = x
    #         graph_dict_data['input']['goal']['edge_index'] = goal_edge_index
    #         graph_dict_data['input']['goal']['edge_attr'] = goal_edge_attr            

    #         graph_dict_data['target']['action'] = action_code
    #         graph_dict_data['target']['object'] = target_object_score

    #         graph_dict_data['info'] = block_order
                       
    #         stacking_dataset.append(graph_dict_data)

    # return stacking_dataset


stacking_dataset = stacking_5_dataset()
# print("the num of data:", len(stacking_dataset)) #120X8=960
# print(stacking_dataset[0])


#print(stacking_dataset[8]['input']['state']['edge_index'])
#print(stacking_dataset[8]['input']['goal']['edge_index'])
#print(stacking_dataset[8]['info'])\
#save each graph in json file
# for i, g in enumerate(stacking_dataset):
#     file_path = "./stacking_dataset/stacking_"+str(i)
#     with open(file_path, "wb") as outfile:
#         pickle.dump(g, outfile)
'''
with open("./stacking_dataset/stacking_"+str(10), "rb") as file:
    load_data = pickle.load(file)
    print(load_data)
'''