import os
import torch
import pandas as pd
import torch_geometric
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Dataset
import time
import os
import natsort

class MakeDataset(Dataset):
    def __init__(self, problem, i):
        
        root_path = ['edge_features/edge_index','edge_features/edge_attr','node_features','action_sequence'] # 0,1,2,3

        # Search path
        FILEPATH, _ = os.path.split(os.path.realpath(__file__))
        search_path = os.path.join(FILEPATH, problem, root_path[i])
        file_list = os.listdir(search_path)
        order_file_list = natsort.natsorted(file_list) 
        problem_path = os.path.join(FILEPATH, problem)

        self.problem_path = problem_path
        self.root_path = root_path
        self.search_path = search_path
        self.order_file_list = order_file_list

        print("[Search path]:",search_path)
        print("\n[Order file list]:",order_file_list)


    def edge_feature(self, i): # i -> Number of the file lists e.g.) ['0_ef0.csv', '1_ef_pick4.csv']

        # Search path
        edge_path = os.path.join(self.search_path, self.order_file_list[i])
        print(f"\n[{self.order_file_list[i]}]\n")

        # Read csv file without new index column
        self.ef_csv = pd.read_csv(edge_path, index_col=0) 
        return self.ef_csv 
    

    def pick(self, i, obj1): # obj = ID number
        
        edge_path = os.path.join(self.search_path, self.order_file_list[i])
        ef_csv = pd.read_csv(edge_path, index_col=0)

        # Data type) column :'object', index = 'int64'
        pick_csv = ef_csv

        # Preconditions
        if obj1 in range(1,len(pick_csv.columns)-1): # Not robot-hand and table (The very first:robot-hand, The very last:table)
            if pick_csv.loc[obj1,'0'] == 0 and pick_csv.loc[0,f'{obj1}'] == 0: # obj1 is has not relationship with robot-hand

                # Remove 'on' relation (Table) 
                pick_csv.loc[obj1,f'{len(pick_csv.columns)-1}'] = 0
                pick_csv.loc[len(pick_csv.columns)-1,f'{obj1}'] = 0

                # Add 'in-grasp' relation (Robot-hand)
                pick_csv.loc[obj1,'0'] = 1
                pick_csv.loc[0,f'{obj1}'] = 1

                self.obj1 = obj1
                self.pick_csv = pick_csv
               
                print(f'\n[ef_pick{str(self.obj1)}.csv] \n') 

                return  self.pick_csv
            
            else:
                print("\n----Check the '.csv' file again----\nFile lists:", self.file_list)
            
        else:
            print("\n----Cannot pick this object----\n")

        
    def place(self, i, obj1, obj2):
        edge_path = os.path.join(self.search_path, self.order_file_list[i])
        ef_csv = pd.read_csv(edge_path, index_col=0)
        self.obj1 = obj1
        self.obj2 = obj2
        place_csv = ef_csv
        
        # Check obj1 and obj2 range
        if obj1 in range(1,len(place_csv.columns)-1) and obj2 in range(1,len(place_csv.columns)-1):
            # 'in-grasp' relation (Robot hand O -> X) , object are not equal
            if place_csv.loc[obj1,'0'] == 1 and place_csv.loc[0,f'{obj1}'] == 1:
                # Check obj1 and obj2 are equal
                if obj1 != obj2:

                    # Add 'on' relation (Table) 
                    place_csv.loc[obj1,f'{obj2}'] = 1
                    place_csv.loc[obj2,f'{obj1}'] = 1

                    # Remove 'in-grasp' relation (Robot hand)
                    place_csv.loc[obj1,'0'] = 0
                    place_csv.loc[0,f'{obj1}'] = 0

                    self.place_csv = place_csv
                    print(f'\n[ef_place_{str(self.obj1)}_on_{str(self.obj2)}.csv] \n') 

                    return self.place_csv
                else:
                    print("----Object1 and object2 are equal----")
            else:
                print("\n----Robot hand does not hold obj1. Please check the '.csv' file again----\nFile lists:", self.file_list)
        else:
            print("----Cannot place this object----")
    
        
    def save_file(self, action):
       
        if action == 'pick':
            action_pick = 'ef_pick'+ str(self.obj1) + '.csv'
            self.pick_csv.to_csv(os.path.join(self.search_path, action_pick))
            print("\n", action_pick,"is saved")

        elif action == 'place':
            action_place = 'ef_place' + str(self.obj1) + '_on_'+ str(self.obj2)+ '.csv'
            self.place_csv.to_csv(os.path.join(self.search_path, action_place))
            print("\n", action_place,"is saved")

        else:
            print("----Wrong action----")

    def init_edge_attr(self):
        list_attr = []
        list_r = []
        list_l = []

        # Dataframe
        ef = self.ef_csv

        # Collect index and column which value is 1 / Table column, index = 7
        for index in range(len(ef.index)):
            for column in range(len(ef.index)):
                if ef.iat[index, column] == 1:    
                    list_attr.append((index, column))
                    if column == 8:
                        list_r.append(column)
                        list_on_table_r = [1 for i in range(len(list_r))]
                    elif index == 8:
                        list_l.append(index)
                        list_on_table_l = [0 for i in range(len(list_l))]
                        list_on_r = list_on_table_r + list_on_table_l
                        list_on_l = list_on_table_l +  list_on_table_r 
                        

                    # Only table has a relationship
                        list_0 = [0 for i in range(len(list_attr))]

                        edge_attr_csv = pd.DataFrame({'ID': list_attr, 'rel_on_right':list_on_r, 'rel_on_left': list_on_l, \
                                                      'rel_in_right':list_0, 'rel_in_left': list_0, 'rel_attach':list_0, \
                                                      'rel_in_grasp':list_0, 'rel_grasp': list_0})

    
                    # Save path
                        final_path = os.path.join(self.problem_path, self.root_path[1], 'init_ea0.csv')
                        edge_attr_csv.to_csv(final_path, index = False) # Remove index when file is saved
                      
                    
        print("\n[init_ea0.csv]\n",edge_attr_csv)        
        print("\n----Edge attribute is saved----")

    def make_edge_index_change(self, i):
        
        # Search path
        edge_index_path = os.path.join(self.search_path, self.order_file_list[i])

        # Read csv file to tensor
        ef = pd.read_csv(edge_index_path, index_col=0)


        # edge_index: [2, num_edges], edge_attr: [num_edges, dim_edge_features]
        
        ####################### Recommend to change ################
        ## Edge index
        list_attr = []
        list_i = []
        list_c = []
       
        for index in range(len(ef.columns)):
            for column in range(len(ef.columns)):
                if ef.iat[index, column] == 1:    # Recommend to change '.iat' to speed up
                    list_i.append(index)
                    list_c.append(column)
                    list_attr.append((index, column))
        
        list_0 = [0 for i in range(len(list_attr))]
               
        print(f"\n[Edge attribute: {self.order_file_list[i]}]\n")
        edge_attr_csv = pd.DataFrame({'ID': list_attr, 'rel_on_right':list_0, 'rel_on_left': list_0, \
                                      'rel_in_right':list_0, 'rel_in_left': list_0, 'rel_attach':list_0, \
                                      'rel_in_grasp':list_0, 'rel_grasp':list_0})
        # Save file
        # final_path = os.path.join(self.problem_path, self.root_path[1], 'ea'+ str(i) + '.csv')
        # edge_attr_csv.to_csv(final_path, index = False) # Remove index when file is saved

        return edge_attr_csv

    def Call(self,file1, file2, file3): # i = range(0,8)
        nf_path = os.path.join(self.problem_path, self.root_path[2], file1)
        ef_index_path = os.path.join(self.problem_path, self.root_path[0], file2)
        ef_attr_path = os.path.join(self.problem_path, self.root_path[1],file3)

        node_feature = pd.read_csv(nf_path)
        edge_index = pd.read_csv(ef_index_path)
        edge_attr = pd.read_csv(ef_attr_path)

        self.node_feature = node_feature
        self.edge_index = edge_index
        self.edge_attr = edge_attr

        print("\n[Node feature]:\n", node_feature)
        print("\n[Edge index]:\n", edge_index)
        print("\n[Edge_attribute]:\n", edge_attr)

    def make_graph(self):
        import networkx as nx
        import matplotlib.pyplot as plt
        list_edge_index = []
        list_edge_attr = []


        nodes = self.node_feature['ID'].tolist()
        ea = self.edge_attr['ID'].to_list()
        print("[ea]:",ea)
        column = self.edge_attr.columns
    
        for i in range(len(ea)):
            ei = eval(ea[i])
            list_edge_index.append(ei)
            
            for j in range(len(column)):
                if self.edge_attr.at[i, column[j]] == 1:
                    list_edge_attr.append(column[j])
        
           
        print("\n[List edge index]:",list_edge_index)
        print("\n[List edge attribute]:",list_edge_attr)


        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(list_edge_index)
        pos = nx.shell_layout(g)
        
        # edge_labels = nx.get_edge_attributes(g, list_edge_attr)
        nx.draw(g, pos, with_labels = True)

        plt.title("Present state")
        plt.show()

        
    

### Checking paths
# Root path: ['edge_features/edge_index','edge_features/edge_attr','node_features','action_sequence'] # 0,1,2,3
# file_path: ['```1.csv`,'```0.csv`]

make_data = MakeDataset(problem = 'stacking_5/1_2_3_4_5', i=0)
make_data.Call(file1='nf0.csv', file2='ef0.csv', file3='ea0.csv')
print(make_data.make_graph())




# print(make_data.edge_feature(i=0))
# make_data.init_edge_attr()

# print(make_data.make_edge_index_change(i=8))




# print(make_data.pick(i=2, obj1=1))
# print(make_data.place(i=1, obj1=1, obj2=2))  # e.g.) obj1=3, obj2=4 -> obj1->obj2


# print(make_data.pick(i=0, obj1= 2))


# print(make_data.place(i=3,obj1=2, obj2=3))


# make_data.save_file(action='pick')
# make_data.save_file(action='place')





### Creating folders ###
import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory.'  +  directory)
 



# # Stacking blocks 
# for i in range(1,8):
#     for j in range(1,8):
#         if i != j:
#             folder_name = f"{i}_on_{j}"
#             break

# createFolder(f'/home/jeni/Desktop/dataloader/seq_dataset/stacking_5/action_sequence')
# createFolder(f'/home/jeni/Desktop/dataloader/seq_dataset/stacking_5/edge_features/edge_attr')
# createFolder(f'/home/jeni/Desktop/dataloader/seq_dataset/stacking_5/edge_features/edge_index')
# createFolder(f'/home/jeni/Desktop/dataloader/seq_dataset/stacking_5/node_features')
            








