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




class MakeDataset(Dataset):
    def __init__(self, problem, example):
        # Search path
        FILEPATH, _ = os.path.split(os.path.realpath(__file__))
        search_path = os.path.join(FILEPATH, problem, example)
    
       

        self.FILEPATH = FILEPATH
        self.search_path = search_path
        self.problem = problem
        self.example = example
      

        print("\n==========================================INIT======================================================")
        print("\n[File path]",FILEPATH)
        print("\n[Search_path]",search_path)
        print("\n[Example]", example)
        print("\n[Problem]", problem)
        print("\n==========================================INIT======================================================")
    

    def pick(self, file_num, obj1): # obj = ID number
        # Choose sample
        sample_inx_path = os.path.join(self.search_path,'edge_index')
        file_list = natsort.natsorted(os.listdir(sample_inx_path))

        edge_path = os.path.join(self.search_path, 'test/edge_index',file_list[file_num])
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

          
            
                print(f'\n[ef_pick{str(obj1)}.csv] \n') 

        
                file_name = 'ei'+str(file_num)+'.csv'
                save_path = os.path.join(self.search_path,'test','edge_index')
                # createFolder(save_path)
                # pick_csv.to_csv(os.path.join(save_path,file_name))
                   

                self.file_num = file_num
                self.pick_csv = pick_csv
            
                return pick_csv
            
            else:
                print("\n----Check the '.csv' file again----\nFile lists:", file_list[file_num])
            
        else:
            print("\n----Cannot pick this object----\n")


        
    def place(self, obj1, obj2): 
        place_csv = self.pick_csv
        
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

                    
                    print(f'\n[ef_place_{str(obj1)}_on_{str(obj2)}.csv] \n') 


                    file_name = 'ei'+str(self.file_num+1)+'.csv'
                    save_path = os.path.join(self.search_path,'test','edge_index')
                    createFolder(save_path)
                    place_csv.to_csv(os.path.join(save_path,file_name))
                    return place_csv
                
                else:
                    print("----Object1 and object2 are equal----")
            else:
                print("\n----Robot hand does not hold obj1. Please check the '.csv' file again----\nFile lists:", self.file_list[self.file_num+1])
        else:
            print("----Cannot place this object----")


    def pour(self, obj1, obj2):
        file = 'ei'+str(self.file_num)+'.csv'
        edge_path = os.path.join(self.search_path, 'test/edge_index',file)
        pour_csv = pd.read_csv(edge_path, index_col=0)
        # print(pour_csv,'\n')
        
        # Node feature 따라서 만들고 obj1은 bowl이어야 함
        if obj1 == 6 or obj1 == 7 :
            placed_obj = 1
            while placed_obj < len(pour_csv.columns)-1 : # placed_obj 1 ~ 7
                placed_obj = placed_obj + 1

                # Relation with obj1 (bowl) must be relative
                if pour_csv.loc[obj1,f'{placed_obj}'] == 1 and pour_csv.loc[placed_obj,f'{obj1}'] == 1:
                    # print(placed_obj)

                    # Remove relation with obj1
                    pour_csv.loc[obj1,f'{placed_obj}'] = 0
                    pour_csv.loc[placed_obj,f'{obj1}'] = 0

                    # Add relation with obj2
                    pour_csv.loc[obj2,f'{placed_obj}'] = 1
                    pour_csv.loc[placed_obj,f'{obj2}'] = 1
                    
                    # Repeat all applicable 'placed_obj' while if statement is true
                    continue
                print(pour_csv)
               
                
                # file_name = 'ei'+str(self.file_num)+'.csv'
                # save_path = os.path.join(self.search_path,'test','edge_index')
                # createFolder(save_path)
                # pour_csv.to_csv(os.path.join(save_path,file_name))
                
            else:
                    print("----Nothing Change----")
    
        else:
            print("----Object is not a bowl----")

    

        
    def save_file(self, action):
       
        if action == 'pick':
            action_pick = 'ef_'+ str(i) + '.csv'
            self.pick_csv.to_csv(os.path.join(self.search_path, action_pick))
            print("\n", action_pick,"is saved")

        elif action == 'place':
            action_place = 'ef_ '+ str(i) + '.csv'
            self.place_csv.to_csv(os.path.join(self.search_path, action_place))
            print("\n", action_place,"is saved")

        elif action == 'pour':
            print("\n", action, "is saved")

        else:
            print("----Wrong action----")


    ##########################Call informations################################

    def sample_data(self, i): # i = range(0,8)
        # Node feature path
        nf_path = os.path.join(self.FILEPATH, self.problem , 'node_features/nf0.csv')

        # Edge index path
        index_path = os.path.join(self.search_path, 'edge_index')
        ei_file_list = natsort.natsorted(os.listdir(index_path))
        ei_path = os.path.join(index_path,ei_file_list[i])

        # Edge attribute path
        attr_path = os.path.join(self.search_path, 'edge_attr')
        ea_file_list = natsort.natsorted(os.listdir(attr_path))
        ea_path = os.path.join(attr_path,ea_file_list[i])

        node_feature = pd.read_csv(nf_path, index_col=0)
        edge_index = pd.read_csv(ei_path, index_col=0)
        edge_attr = pd.read_csv(ea_path, index_col=0)

        self.node_feature = node_feature
        self.edge_index = edge_index
        self.edge_attr = edge_attr

        # print("\n[Node feature]:\n", node_feature)
        # print("\n[Edge index]:\n", edge_index)
        # print("\n[Edge_attribute]:\n", edge_attr)
        



    ############################################# Make Edge indexes##########################################
    def make_edge_index(self, i):
        list_num = [1,2,3,4,5]
        list_index = []

        unique = 0
        while unique < 120:
            unique =  unique + 1
            # list(1~5 사이 숫자들 랜덤 배열) - 중복 허용해서 나옴
            sample_list = random.sample(list_num,5)
            list_index.append(sample_list)
            seen = []
            
            # 중복 제거하고 120개 전부 출력될 때까지 돌림
            unique_list = [x for x in list_index if x not in seen and not seen.append(x)] 
            unique = len(unique_list) 
                
            
            # Data type 맞추기 - index) int - list, column) string - list
            str_sample = [str(x) for x in sample_list]       
            index_list = [0] + sample_list+ [6,7,8]
            column_list = ['0'] + str_sample + ['6','7','8']
            

            # Change Index and column
            self.edge_index.index = index_list 
            self.edge_index.index.name = "ID" # Setting index name = 'ID'
            self.edge_index.columns = column_list 
            change_edge_index = self.edge_index
            # print(change_edge_index)
            
            # Make folders
            folder_name = f"{str_sample[0]}_{str_sample[1]}_{str_sample[2]}_{str_sample[3]}_{str_sample[4]}"
            self.str_sample = str_sample
            print(folder_name)
            print(change_edge_index)
          
            # # SAVE PATH (edge_index)
            save_inx_path = os.path.join(self.FILEPATH, self.problem, 'edge_features',folder_name, 'edge_index')
            createFolder(save_inx_path)
            
            # # SAVE CSV FILE (edge_index) #root_path[0]인 경우만
            save_csv = os.path.join(save_inx_path,'ei' +str(i)+'.csv')
            change_edge_index.to_csv(save_csv) 

        print((len(list_index), len(unique_list)))
        print("----Re indexing----")
  

    ############################################# Make Edge attributes##########################################
    def make_edge_attr(self,i): 
        # print(ef.at[0,'2'])
        # edge_index: [2, num_edges], edge_attr: [num_edges, dim_edge_features]
        edge_feature_path = os.path.join(self.FILEPATH, self.problem, 'edge_features')
        prob_file_list = natsort.natsorted(os.listdir(edge_feature_path))
      
        # Call info from new edge_index 
        for prob in prob_file_list:
            inx_search_path = os.path.join(edge_feature_path, prob, 'edge_index')
            inx_file_list = natsort.natsorted(os.listdir(inx_search_path))
            inx_path = os.path.join(inx_search_path, inx_file_list[i])
            # print(inx_path)

            # # # Read csv file to tensor
            ef = pd.read_csv(inx_path, index_col=0)
            # print("\n[Edge index]\n",ef)
            ef_index = ef.index.to_list()
            
            # ####################### Recommend to change ################
            ## Edge index 정보로 부터 연결된 node 가져오기
            list_attr = []
            list_i = []
            list_c = []
        
            for ef_index in range(len(ef_index)):
                for column in range(len(ef.columns)):
                    if ef.at[ef_index, str(column)] == 1:    # Recommend to change '.iat' to speed up
                        list_i.append(ef_index)
                        list_c.append(column)
                        list_attr.append((ef_index, column))
                      
            # Original data
            ea_example = self.edge_attr
            # print("\n[Original]\n",ea_example)
            
            # Changed data
            ea_example.index = list_attr
            ea_example.index.name = "ID"
            edge_attr_csv = ea_example
            # print("\n[New]\n",edge_attr_csv)


            # SAVE PATH (edge_attr)
            save_attr_path = os.path.join(edge_feature_path, prob,'edge_attr')
            createFolder(save_attr_path)
            save_csv = os.path.join(save_attr_path,'ea' +str(i)+'.csv')
            edge_attr_csv.to_csv(save_csv) 
   

    ############################## Make graph ##################################

    def make_graph(self):
     
        # Weight 부여되면 굵어지게
        list_edge_index = []
        list_edge_attr = []

        # Make nodes
        nodes = self.node_feature.index.to_list()

        # Connect edge
        ea_inx = self.edge_attr.index.to_list()

        # edge_attr의 column 데이터 list로 가져오기
        col = self.edge_attr.columns.to_list()


        
        # edge_attr file에서 'rel'이 들어간 문자열 정보 가져오기 
        ea_col = [col[i] for i in range(len(col)) if col[i].find('rel') == 0]    
     
        
        #  Relation 보기 간편하게 바꿔줌
        for i in range(len(ea_inx)):
            ei = eval(ea_inx[i])
            list_edge_index.append(ei)
            for j in range(len(ea_col)):
                if self.edge_attr.at[ea_inx[i], ea_col[j]] == 1:
                    if ea_col[j] == 'rel_on_right':
                        attr = ea_col[j].replace('rel_on_right', 'On')
                    elif ea_col[j] == 'rel_on_left':
                        attr = ea_col[j].replace('rel_on_left', 'On')
                    elif ea_col[j] == 'rel_in_right':
                        attr = ea_col[j].replace('rel_in_right', 'In')
                    elif ea_col[j] == 'rel_in_left':
                        attr = ea_col[j].replace('rel_in_left', 'In')
                    elif ea_col[j] == 'rel_in_grasp':
                        attr = ea_col[j].replace('rel_in_grasp', 'Grasp')
                    elif ea_col[j] == 'rel_grasp':
                        attr = ea_col[j].replace('rel_grasp','Grasp')
                    elif ea_col[j] == 'rel_attach':
                        attr = ea_col[j].replace('rel_attach','Attach')
                    else:
                        print("----Re-check relations----")
                    list_edge_attr.append(attr)
        
           
        print("\n[List edge index]:", list_edge_index)
        print("\n[List edge attribute]:",list_edge_attr)

       

        ############################ Make graph ####################
        import matplotlib.pyplot as plt
        import networkx as nx
        import PIL

        
        # Image URLs for graph nodes
        icons = {
            "Robot0": "/home/jeni/Desktop/dataloader/seq_dataset/icons/robot_hand.jpeg",
            "Block1": "/home/jeni/Desktop/dataloader/seq_dataset/icons/block1.jpeg",
            "Block2": "/home/jeni/Desktop/dataloader/seq_dataset/icons/block1.jpeg",
            "Block3": "/home/jeni/Desktop/dataloader/seq_dataset/icons/block1.jpeg",
            "Block4": "/home/jeni/Desktop/dataloader/seq_dataset/icons/block1.jpeg",
            "Block5": "/home/jeni/Desktop/dataloader/seq_dataset/icons/block1.jpeg",
            "Bowl6": "/home/jeni/Desktop/dataloader/seq_dataset/icons/block1.jpeg",
            "Bowl7": "/home/jeni/Desktop/dataloader/seq_dataset/icons/block1.jpeg",
            "Table": "/home/jeni/Desktop/dataloader/seq_dataset/icons/table_icon.jpg",
        }
        # Load images
        images = {k: PIL.Image.open(fname) for k, fname in icons.items()}
        
        # Generate graph
        g = nx.Graph()

        # Add nodes
        # g.add_nodes_from(nodes, images = images["Block1"])
        g.add_node(0, images = images["Robot0"])
        g.add_node(1, images = images["Block1"])
        g.add_node(2, images = images["Block2"])
        g.add_node(3, images = images["Block3"])
        g.add_node(4, images = images["Block4"])
        g.add_node(5, images = images["Block5"])
        g.add_node(6, images = images["Bowl6"])
        g.add_node(7, images = images["Bowl7"])
        g.add_node(8, images = images["Table"])

    
        # Add edges
        for i in range(len(list_edge_attr)):
            g.add_edges_from([list_edge_index[i]], label = f'{list_edge_attr[i]}')

        # POS 1 사진으로 node image 가져오는 것 가능
        
        # pos 지정 => x,y 좌표 array값으로 받아서 사용할 수 있음
        # manually specify node position
        # pos = nx.spring_layout(g)
        # pos = nx.shell_layout(g)
        pos = {
            0: [0.4, 0.7],
            1: [0.5, 0.1],
            2: [0.5, 0.2],
            3: [0.5, 0.3],
            4: [0.5, 0.4],
            5: [0.5, 0.5],
            6: [0.3, 0.2],
            7: [0.7, 0.2],
            8: [0.5, 0]
        }



        # Get a repreducible layout and create figure
        
        fig, ax = plt.subplots() 
        # Transform from data coordinates
        tr_figure = ax.transData.transform
        # Transform from display to figure coordinates
        tr_axes = fig.transFigure.inverted().transform

        # Select the size of the image
        icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.05 # 0.05
        icon_center = icon_size / 2.0                          # 0.025
        edge_labels = nx.get_edge_attributes(g,'label')
        print("\n[Edge labels]:",edge_labels)

        
        for n in g.nodes:
            xf, yf = tr_figure(pos[n])
            xa, ya = tr_axes((xf, yf))
            # get overlapped axes and plot icon
            a = plt.axes([xa-icon_center, ya-icon_center , icon_size, icon_size])
            print(g.nodes[n])
            a.imshow(g.nodes[n]['images']) # print(g.nodes[n]) #dictionary에 'image' -> 'images'로 변경됨
            a.axis("off")
            
            # nx.draw_networkx_nodes(G=g, pos= pos, nodelist= nodes, cmap=plt.cm.Blues, alpha = 0.9, node_size = 500, node_shape='o')
            # nx.draw_networkx_nodes(G=g, pos= pos, nodelist= nodes, alpha = 0.9, node_size = 500, node_shape='o')
            # nx.draw_networkx_edges(G=g, pos= pos, edgelist= list_edge_index, edge_cmap = plt.cm.Greys)
            # nx.draw_networkx_labels(G=g, pos=pos, font_family='sans-serif', font_color='black', font_size = 12)
            nx.draw_networkx_edges(
                g,
                pos=pos,
                ax=ax,
                edgelist= list_edge_index
                # min_source_margin=10,
                # min_target_margin=10,
            )
            nx.draw_networkx_edge_labels(G= g, pos = pos, ax=ax, edge_labels = edge_labels, font_size = 12)
        

        
        # plt.title("Present state")
        plt.axis('off')
        # plt.figure(figsize=(10,6))
        plt.show()



    

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            pass
    except OSError:
        print ('Error: Creating directory.'  +  directory)









