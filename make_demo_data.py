import os
import torch
import pandas as pd
import torch_geometric
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Dataset
import time
import os

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

    def pick(self, csv_file, obj): # obj = ID number

        edge_path = os.path.join(self.search_path, 'edge_features', 'edge_index', csv_file)
        ef_csv = pd.read_csv(edge_path, index_col=0)

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
        return self.pick_csv


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
    

    def place(self,obj1,obj2,csv_file):
        # path = './dataset/edge_features/edge_index'
        # file_list = os.listdir(path)
        # # for i in range(len(file_list)):
        # #     csv_file = file_list[i]
        edge_path = os.path.join(self.search_path, 'edge_features','edge_index', csv_file)
        ef_csv = pd.read_csv(edge_path, index_col=0)
    
        place_csv = ef_csv
        
        # 'in-grasp' relation (robot hand o -> x) 
        if place_csv.loc[obj1-1,'9'] == 1 and obj1 != obj2:
            # Add 'on' relation (Table) 
            place_csv.loc[obj1-1,f'{obj2}'] = 1
            place_csv.loc[obj2-1,f'{obj1}'] = 1

            # Remove 'in-grasp' relation (Robot hand)
            place_csv.loc[obj1-1,'9'] = 0
            place_csv.loc[8,f'{obj1}'] = 0

            self.place_csv = place_csv
            return self.place_csv

        else:
            print("----Cannot place----")
        # self.obj = obj
        # self.place_csv = place_csv

        
            # return self.place_csv

    
    def pick_edge_attr(self):
        pass
        # list_attr = []
        # list_r = []
        # list_l = []

        # # ID drop
        # ef = self.ef_csv.drop(labels='ID',axis=1) 

        # # Collect index and column which value is 1 / Table column, index = 7
        # for index in range(len(ef.index)):
        #     for column in range(len(ef.index)):
        #         if ef.iat[index, column] == 1:    
        #             list_attr.append((index, column))
        # print(list_attr)

    def place_attr(self):
        list_attr = []
        list_r = []
        list_l = []

        # ID drop
        ef = self.place_csv.drop(labels='ID',axis=1) 

        # Collect index and column which value is 1 / Table column, index = 7
        for index in range(len(ef.index)):
            for column in range(len(ef.index)):
                if ef.iat[index, column] == 1:    
                    list_attr.append((index, column))
        print(list_attr)

        
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

# Object의 범주 구하기

make_data = MakeDataset(root_path='dataset')
# print(make_data.edge_feature(csv_file='ef0.csv', root_dir='edge_features/edge_index'))
# print(make_data.pick(csv_file='ef0.csv',obj=5))
# print(make_data.pick_edge_attr())
print(make_data.place(obj1=1, obj2=4, csv_file='ef_pick1.csv')) # 지금 순서가 9번 hand가 8번 table위에 있던 1번을 들어서 4번에 놓는 상황
print(make_data.place_attr())

# All of the list directories
# import os
# path = './dataset/edge_features/edge_index'
# file_list = os.listdir(path)
# for i in range(len(file_list)):
#     print(file_list[i])



# Make initial edge attribute
# print(make_data.init_edge_attr(save_dir='edge_features/edge_attr'))

# print(make_data)



# Save files) Action_pick
# n = 7
# for i in range(1,n+1):
#     print(make_data.pick(csv_file='ef0.csv', root_dir='edge_features/edge_index',obj=i)) # 1~5 : box, 6~7:bowl, 8: Table, 9: Robot hand (pick 1~7)
#     print(make_data.save_file(save_dir='edge_features/edge_index'))


