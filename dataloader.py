import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import os
import torch
import torch_geometric
import torchvision.transforms as transforms
import random

class LoadDataFrame(Dataset):
    ## Load DataFrame
    def __init__(self):
        DATA_PATH = "/home/jeni/Desktop/gcn/dataset/node_features/nf1.csv"
        data = pd.read_csv(DATA_PATH)
        self.data = data
        # print(data.head)

    ## General information about the dataset
    def general_info(self):
        print(self.data.shape)
        print(self.data["Position_t4"].value_counts()) # value) 6 zeros 2 ones

    ## Quick check iwth versions
    def checkversion(self):
        print(f"Torch version: {torch.__version__}") # 1.10.2
        print(f"Cuda available: {torch.cuda.is_available()}") #True
        print(f"Torch geometric version: {torch_geometric.__version__}") #2.0.4

#Test1
a = LoadDataFrame()
# print(a.__init__())
# print(a.general_info())
# print(a.checkversion())

# print(os.listdir()) # Current path lists ['dataloader.py', 'dataset', '.git', 'tutorail1', 'tutorial1', 'appendix', '기타', 'data.py']
# print(os.getcwd()) # Current path state '/home/jeni/Desktop/gcn'


## Generate a Dataset
class GraphDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.csv_file = csv_file
        self.root_dir = root_dir

        root_path = os.path.join(os.getcwd(),'dataset')
        node_path = os.path.join(root_path, self.root_dir, self.csv_file)

        nf_csv = pd.read_csv(node_path)
        node_features = torch.Tensor(nf_csv.values)
        self.x = node_features[:,2:]
     

    def __len__(self): # data size return
        return len(self.x)


    def __getitem__(self, index): # Sampling one specific data from dataset
        part_x = torch.FloatTensor(self.x[index])
        return part_x

#'/home/jeni/Desktop/gcn/dataset/node_features/nf1.csv'
# root_dir = 'node_features'
# csv_file = 'nf1.csv'

# print(os.path.join(os.getcwd(), 'dataset',root_dir, csv_file))

ds = GraphDataset(csv_file='nf1.csv', root_dir='node_features')
# print(len(ds)) # data size = 8
# print(ds.x)
# print(ds.x.size()) # torch.Size([8,13])
# print(ds.__getitem__(1)) # index: 0~7




class Randomposdata(Dataset):
    def __init__(self, file_path):
        nf_csv = pd.read_csv(file_path)
        # print("[Node Features CSV]\n",nf_csv)
        node_features = torch.Tensor(nf_csv.values)
        # print('\n[Node_features]\n',node_features)
        self.x = node_features[:,8:]

    def switch_position(self):
        tensor_matrix = self.x
        numpy_matrix = tensor_matrix.numpy()
        rearrange_arr = np.rollaxis(numpy_matrix,0,1)
        tensor_data = torch.tensor(rearrange_arr)
        self.tensor_data = tensor_data

        return tensor_data


    @property
    def raw_file_names(self):
        return 'nf1.csv'

    @property
    def processed_file_names(self):
        return 'not_implemented'

    def download(self):
        pass

    def save_data(self,filepath):
        # pathList = []
       
        # folders = os.listdir(homepath)
        # for folder in folders:
        #     foldername = folder
        #     files = os.listdir(homepath+folder)
        #     for file in files:
        #         filename=file
        #         path = homepath+foldername+'/'+filename

        os.chdir(filepath)
        for i in range(0, len(self.x)):
            final_path = filepath + 'nf_ex'+str(i)+'.csv'
            np.savetxt(final_path, self.tensor_data, delimiter =',', fmt="%d")     

    # def process(self):
    #     self.data = pd.read_csv(self.raw_paths[0])   
    #     for index, col in tqdm(self.data.iterrows(), total=self.data.shape[0]):
    #         col_obj = col['Position_t2']

    #         node_features = self._get_node_features(col_obj) # Get node features
    #         edge_features = self._get_edge_features(col_obj) # Get edge features
    #         edge_index = self._get_adjacency_info(col_obj) # Get adjacency info
    #         label = self._get_labels(col["Position_t4"]) # Get labels info

    #     # Create data object
    #     data = Data(x=node_features,
    #                 edge_index=edge_index,
    #                 edge_attr=edge_features,
    #                 y=label,
    #                 positions=col["Position_t2"]
    #                 )
    #     torch.save(data,
    #                 os.path.join(self.processed_dir,
    #                             f'data_{index}.pt'))

    # def _get_node_features(self, col):
    #     # Return a matrix as a 2d array of the shape [Number of nodes, Node feature size]

    #     all_node_features = []

   

d1 = Randomposdata('/home/jeni/Desktop/gcn/dataset/node_features/nf1.csv')
# print(d1.switch_position())
# print(d1.save_data('/home/jeni/Desktop/gcn/dataset/node_features/'))

###Test the dataset
# dataset = Randomposdata(root="dataset/")

#print(dataset[0].edge_index.t())
#print(dataset[0].x)
#print(dataset[0].edge_attr)
#print(dataset[0].y)


## Switch only Property_V for while
class MakeDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.csv_file = csv_file
        self.root_dir = root_dir

        # Search path
        root_path = os.path.join(os.getcwd(),'dataset')
        search_path = os.path.join(root_path, self.root_dir, self.csv_file)

        self.root_path = root_path
        self.search_path = search_path

        # Read csv file
        self.nf_csv = pd.read_csv(search_path, index_col=0) # index_col drops 'Unnamed' column
        

    def rand_sample(self, save_dir, n):
        # Make random samples with torch
        for i in range(n):
            p = torch.randint(0,2,(5,))
            z = torch.zeros(3)
            concat = torch.cat((p,z),0).tolist()
            
        # Transform property from list values
            self.nf_csv['Property_V_Velcro'] = concat # Only list can switch values of the column
            new_nf = self.nf_csv
            # print(new_nf)

        # Save files
            self.save_dir = save_dir
            final_path = os.path.join(self.root_path, self.save_dir, 'nf_ex'+str(i)+'.csv')
            new_nf.to_csv(final_path)  
            # print(new_nf)



make_data = MakeDataset(csv_file='nf1.csv', root_dir='node_features')
# print(make_data.rand_sample(save_dir='node_features', n=11))


