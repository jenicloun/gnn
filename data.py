import os
import torch
import pandas as pd
import torch_geometric
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
        

        # edge_index: [2, num_edges], edge_attr: [num_edges, num_edge_features]
        
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
        ## Edge attribute
        edge_attr = torch.Tensor(ef.values)
        
        return edge_index, edge_attr

## Print

make_data = MakeDataset(root_path='dataset')


# print(make_data.rand_sample(folder_name='node_features',file_name='nf1.csv',save_dir='node_features', n=13))
print(make_data.node_feature(csv_file='nf1.csv', root_dir='node_features'))
print(make_data.edge_feature(csv_file='ef1.csv', root_dir='edge_features'))


#### Data Loader
from torch.utils.data import DataLoader

# train_dataloader = DataLoader(training_set, batch_size= 16, shuffle= True)
# test_dataloader = DataLoader(test_set, batch_size= 16, shuffle= True)