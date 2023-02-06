import os
import torch
import pandas as pd
import torch_geometric
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Dataset
import time

class MakeDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path

        # Search path
        search_path = os.path.join(os.getcwd(), self.root_path)
        self.search_path = search_path
        print(search_path)



    def edge_feature(self, csv_file, root_dir):
        # Search path
        edge_path = os.path.join(self.search_path, root_dir, csv_file)

        # Read csv file to tensor
        self.ef_csv = pd.read_csv(edge_path, index_col=0)
        # ef = ef_csv.drop(labels='ID',axis=1) # drop the "ID" column / axis=0 (row), axis=1(column)
        
        return self.ef_csv

    def pick(self, root_dir, csv_file, obj): # obj = ID number

        edge_path = os.path.join(self.search_path, root_dir, csv_file)
        ef_csv = pd.read_csv(edge_path, index_col=0)
        # self.ef = self.ef_csv.at[obj,'8'] 
        # self.ef2 = self.ef_csv.at[7,f'{obj}']

        # Data type) column :'object', index = 'int64'
        pick_csv = ef_csv
        
        # Remove 'on' relation (Table) 
        pick_csv.loc[obj-1,'8'] = 0
        pick_csv.loc[7,f'{obj}'] = 0

        # Add 'in-grasp' relation (Robot hand)
        pick_csv.loc[obj-1,'9'] = 1
        pick_csv.loc[8,f'{obj}'] = 1

        self.obj = obj
        self.pick_csv = pick_csv
        # return self.ef , ex_time , self.ef2 , self.pick_csv
        return self.pick_csv

    
    def save_file(self, save_dir):

        # Action) pick
        final_path = os.path.join(self.search_path, save_dir, 'ef_pick' + str(self.obj) + '.csv')
        self.pick_csv.to_csv(final_path)
            
        # # Action) place
        # final_path = os.path.join(self.search_path, save_dir, 'ef_place' + str(self.obj) + '.csv')
        # self.place_csv.to_csv(final_path)
            
        # # Action) pour
        # final_path = os.path.join(self.search_path, save_dir, 'ef_pour' + str(self.obj) + '.csv')
        # self.pour_csv.to_csv(final_path)

    def init_edge_attr(self, save_dir):
        list_attr = []
        list_r = []
        list_l = []

        # ID drop
        ef = self.ef_csv.drop(labels='ID',axis=1) 

        # Collect index and column which value is 1 / Table column, index = 7
        for index in range(len(ef.index)):
            for column in range(len(ef.index)):
                if ef.iat[index, column] == 1:    
                    list_attr.append((index, column))
                    if column == 7:
                        list_r.append(column)
                        list_on_table_r = [1 for i in range(len(list_r))]
                    elif index == 7 :
                        list_l.append(index)
                        list_on_table_l = [0 for i in range(len(list_l))]
                        list_on_r = list_on_table_r + list_on_table_l
                        list_on_l = list_on_table_l +  list_on_table_r 
                        

                        # Only table has a relationship
                        list_0 = [0 for i in range(len(list_attr))]
                        edge_attr_csv = pd.DataFrame({'ID': list_attr, 'rel_on_right':list_on_r, 'rel_on_left': list_on_l, 'rel_in_right':list_0, 'rel_in_left': list_0, 'rel_attach':list_0, 'rel_in_grasp':list_0})

        # Save path
        final_path = os.path.join(self.search_path, save_dir, 'init_ea0' + '.csv')
        edge_attr_csv.to_csv(final_path)
        print("\nEdge attribute is saved")
    

    def pick_edge_attr(self):
        pass

make_data = MakeDataset(root_path='dataset')
print(make_data.edge_feature(csv_file='ef0.csv', root_dir='edge_features/edge_index'))
print(make_data.init_edge_attr(save_dir='edge_features/edge_attr'))

# print(make_data)



# Save files) Action_pick
# n = 7
# for i in range(1,n+1):
#     print(make_data.pick(csv_file='ef0.csv', root_dir='edge_features/edge_index',obj=i)) # 1~5 : box, 6~7:bowl, 8: Table, 9: Robot hand (pick 1~7)
#     print(make_data.save_file(save_dir='edge_features/edge_index'))


