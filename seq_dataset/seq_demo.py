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
    

    def create_node_features(self):
        pass




        # Ground = pd.read_csv("/home/jeni/Desktop/gcn/box_test1.csv")
        # print(Ground)

        # #### parsing relation

        # parse_r = Ground[["ID","Relation_In","Relation_On","Relation_Attach"]]

        # # print(parse_g)

        # splited1 = parse_r["Relation_In"].str.split(',',expand=True)
        # splited2 = parse_r["Relation_On"].str.split(',',expand=True)
        # splited3 = parse_r["Relation_Attach"].str.split(',',expand=True)

        # # print(splited1)

        # splited1.rename(columns= lambda x: "Relation_In_" + str(x), inplace=True) # Switch int to str
        # splited2.rename(columns= lambda x: "Relation_On_" + str(x), inplace=True)
        # splited3.rename(columns= lambda x: "Relation_Attach_" + str(x), inplace=True)

        # gt = pd.concat([Ground["ID"],splited1,splited2,splited3], axis=1)

        # # fill "NaN" data as "0"
        # df = gt.fillna(0)

        # # df.to_csv('/home/jeni/Desktop/gcn/gt_box_test.csv')

        # print("\nGT:",df)


        # ### parsing nodes

        # parse_n = Ground[["ID","Type","Property_V","Property_C","Property_G","Property_Color"]]
        # print(parse_n)


        # node_features = pd.get_dummies(parse_n)
        # print(node_features)


    def pick(self, save_file_num, obj1): # obj = ID number
        # Choose sample
        sample_inx_path = os.path.join(self.search_path,'edge_index')
        file_list = natsort.natsorted(os.listdir(sample_inx_path))

        # edge_path = os.path.join(self.search_path, 'test/edge_index',file_list[file_num])
        edge_path = os.path.join(self.search_path, 'edge_index',file_list[save_file_num])
        ef_csv = pd.read_csv(edge_path, index_col=0)

      
        # Data type) column :'object', index = 'int64'
        pick_csv = ef_csv


        # Preconditions
        if obj1 != 0 and obj1 != 8: # Not robot-hand and table (The very first:robot-hand, The very last:table)
            if pick_csv.loc[obj1,'0'] == 0 and pick_csv.loc[0,f'{obj1}'] == 0: # obj1 is has not relationship with robot-hand

                # Remove 'on' relation (Table) 
                pick_csv.loc[obj1,'8'] = 0
                pick_csv.loc[8,f'{obj1}'] = 0

                # Add 'in-grasp' relation (Robot-hand)
                pick_csv.loc[obj1,'0'] = 1
                pick_csv.loc[0,f'{obj1}'] = 1

          
            
                print(f'\n[ef_pick{str(obj1)}.csv] \n') 

        
                file_name = 'ei'+str(save_file_num+1)+'.csv'
                save_path = os.path.join(self.search_path,'edge_index')
                createFolder(save_path)
                pick_csv.to_csv(os.path.join(save_path,file_name))
                   

                self.save_file_num = save_file_num
                self.pick_csv = pick_csv
            
                return pick_csv
            
            else:
                print("\n----Check the '.csv' file again----\nFile lists:", file_list[save_file_num])
            
        else:
            print("\n----Cannot pick this object----\n")


        
    def place(self, load_file_num,save_file_num, obj1, obj2): 
        sample_inx_path = os.path.join(self.search_path,'edge_index')
        file_list = natsort.natsorted(os.listdir(sample_inx_path))

        # edge_path = os.path.join(self.search_path, 'test/edge_index',file_list[file_num])
        edge_path = os.path.join(self.search_path, 'edge_index',file_list[load_file_num])
        place_csv = pd.read_csv(edge_path, index_col=0)
        # place_csv = self.pick_csv
        
        # Check obj1 and obj2 range
        if obj1 != 0 and obj1 != 8 and obj2 != 0:
            # 'in-grasp' relation (Robot hand O -> X) , object are not equal
            if place_csv.loc[obj1,'0'] == 1 and place_csv.loc[0,f'{obj1}'] == 1:
                # Check obj1 and obj2 are equal
                if obj1 != obj2:

                    # Add 'on' relation with obj1 and obj2
                    place_csv.loc[obj1,f'{obj2}'] = 1
                    place_csv.loc[obj2,f'{obj1}'] = 1

                    # Remove 'in-grasp' relation (Robot hand)
                    place_csv.loc[obj1,'0'] = 0
                    place_csv.loc[0,f'{obj1}'] = 0

                    
                    print(f'\n[ef_place_{str(obj1)}_on_{str(obj2)}.csv] \n') 

                    ### Save files
                    file_name = 'ei'+str(save_file_num)+'.csv'
                    save_path = os.path.join(self.search_path,'edge_index')
                    createFolder(save_path)
                    place_csv.to_csv(os.path.join(save_path,file_name))

                    ###
                    self.place_csv = place_csv

                    return place_csv
                
                else:
                    print("----Object1 and object2 are equal----")
            else:
                print("\n----Robot hand does not hold obj1. Please check the '.csv' file again----\nFile lists:", self.file_list[self.save_file_num+1])
        else:
            print("----Cannot place this object----")

    def attach(self, obj1, obj2):
        attach_csv = self.place_csv
        # print(attach_csv)
        #### Simply attach object one by one 
        from_node_path = os.path.join(self.search_path, 'node_features', 'nf0.csv')
        nf = pd.read_csv(from_node_path, index_col=0)
        attached_boxes = nf[nf['Type_Attached_Boxes'] == 1].index.to_list()[0]
        self.attached_boxes = attached_boxes

        split_boxes = [x for x in f'{attached_boxes}']
        # print(split_boxes[0], type(split_boxes[1]))


        if obj1 != 0 and obj1 != 8 and obj2 != 0 and obj2 != 8:
            if attach_csv.loc[obj1,f'{obj2}'] == 1 and attach_csv.loc[obj2,f'{obj1}'] == 1:

                # Change relation with attached boxes) obj1 and obj2 
                # attach_csv.loc[obj1,f'{obj2}'] = 0
                # attach_csv.loc[obj2,f'{obj1}'] = 0

                # Attach Obj1 and Obj2 
                attach_csv.loc[attached_boxes, f'{split_boxes[0]}'] = 1 
                attach_csv.loc[int(split_boxes[0]), f'{attached_boxes}'] = 1
                attach_csv.loc[attached_boxes, f'{split_boxes[1]}'] = 1 
                attach_csv.loc[int(split_boxes[1]), f'{attached_boxes}'] = 1


                # Put on table
                attach_csv.loc[attached_boxes, '8'] = 1
                attach_csv.loc[8, f'{attached_boxes}'] = 1
               
                print(attach_csv)




                ### Save files
                file_name = 'ei'+str(self.file_num+3)+'.csv'
                save_path = os.path.join(self.search_path,'edge_index')
                createFolder(save_path)
                attach_csv.to_csv(os.path.join(save_path,file_name))

                return attach_csv

            else:
                print("----Obj1 is not <on> Obj2----")
        else:
            print("----Cannot attach those object----")

        


    def pour(self, file_num, obj1, obj2):
        file = 'ei'+str(file_num)+'.csv'
        edge_path = os.path.join(self.search_path, 'edge_index',file)
        pour_csv = pd.read_csv(edge_path, index_col=0)
        print("\n[Original]\n",pour_csv,'\n')
        
        # Node feature 따라서 만들고 obj1은 bowl이어야 함
        if obj1 == 6 or obj1 == 7 :
            for placed_obj in range(1,7):  # placed_obj 1 ~ 7

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
            print("\n[Change]\n",pour_csv)
               
            
            file_name = 'ei'+str(file_num+1)+'.csv'
            save_path = os.path.join(self.search_path,'edge_index')
            createFolder(save_path)
            pour_csv.to_csv(os.path.join(save_path,file_name))
    
        else:
            print("----Object is not a bowl----")

    
        
    # def save_file(self, action,i):
       
    #     if action == 'pick':
    #         action_pick = 'ef_'+ str(i) + '.csv'
    #         self.pick_csv.to_csv(os.path.join(self.search_path, action_pick))
    #         print("\n", action_pick,"is saved")

    #     elif action == 'place':
    #         action_place = 'ef_ '+ str(i) + '.csv'
    #         self.place_csv.to_csv(os.path.join(self.search_path, action_place))
    #         print("\n", action_place,"is saved")

    #     elif action == 'pour':
    #         action_pour = 'ef_ '+ str(i) + '.csv'
    #         self.pour_csv.to_csv(os.path.join(self.search_path, action_place))
    #         print("\n", action, "is saved")

    #     else:
    #         print("----Wrong action----")

    def init_edge_index(self):

        # # Save dataframe
     
        # edge_path = os.path.join(self.search_path, 'edge_index', 'ei0.csv')
        # ef = pd.read_csv(edge_path, index_col=0)
        from_node_path = os.path.join(self.search_path, 'node_features', 'nf0.csv')
        nf = pd.read_csv(from_node_path)
        node_index = nf['ID'].to_list()
        print(node_index)

        list_0 = [0 for i in range(len(node_index))]
        list_normal = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        list_table = [0, 1, 1, 1, 1, 1, 1, 1, 0, 0]

        edge_index0_csv = pd.DataFrame({'ID': node_index, '0': list_0, '1':list_normal ,'2':list_normal, '3': list_normal, '4': list_normal, \
                                        '5': list_normal, '6':list_normal, '7':list_normal, '8':list_table, '34':list_0})
        print(edge_index0_csv)


        edge_index0_csv = edge_index0_csv.set_index("ID")
        save_inx0_path = os.path.join(self.search_path, 'edge_index')
        createFolder(save_inx0_path)
        save_csv = os.path.join(save_inx0_path, 'ei0.csv')
        edge_index0_csv.to_csv(save_csv) 
        
        print("\n----ei0.csv file is saved----")
        # path = '/home/jeni/Desktop/dataloader/seq_dataset/stacking_v2/v2_ex_1_2_34_5/node_features/nf0.csv'






    def init_edge_attr(self, file_num):
        list_attr = []

        # Dataframe
        file = 'ei'+str(file_num)+'.csv'
        edge_path = os.path.join(self.search_path, 'edge_index',file)
        ef = pd.read_csv(edge_path, index_col=0)
           
        list_attr1 = []
        list_attr0 = []

        ID_list = list(map(int, ef.columns))
        for index in range(len(ID_list)):
            for column in range(len(ID_list)):
                if ef.iat[index, column] == 1:   
                    list_attr1.append((ID_list[index], ID_list[column]))
                elif ef.iat[index, column] == 0 and ID_list[index] != ID_list[column]:
                    list_attr0.append((ID_list[index], ID_list[column]))

        list_attr = list_attr1 + list_attr0

        print("[list1]",list_attr1,'\n')
        print(list_attr, "length", len(list_attr))

        list_0 = [0 for i in range(len(list_attr1))]

        edge_attr0_csv = pd.DataFrame({'ID': list_attr1, 'rel_on_right':list_0, 'rel_on_left': list_0, \
                                        'rel_in_right':list_0, 'rel_in_left': list_0, 'rel_attach':list_0, \
                                        'rel_in_grasp':list_0, 'rel_grasp': list_0, 'pos_x': list_0, 'pos_y': list_0, 'pos_z': list_0, \
                                        'pos_roll':list_0, 'pos_pitch':list_0, 'pos_yaw':list_0})

        edge_attr0_csv = edge_attr0_csv.set_index("ID")



        # SAVE PATH (edge_attr)
        save_attr_path = os.path.join(self.search_path, 'edge_attr')
        createFolder(save_attr_path)
        file = 'ea'+str(file_num)+'.csv'
        save_csv = os.path.join(save_attr_path, file)
        edge_attr0_csv.to_csv(save_csv) 
                    
        print("\n[init_ea0.csv]\n",edge_attr0_csv)        
        print("\n----Edge attribute is saved----")




    ##########################Call informations################################
    def sample_data(self, i): # i = range(0,8)
        # Node feature path
        # nf_path = os.path.join(self.FILEPATH, self.problem , 'node_features','nf0.csv')  # stacking_5, mixing_5
        nf_path = os.path.join(self.FILEPATH, self.problem, self.example, 'node_features', 'nf0.csv') # stacking_v2

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

        self.x = node_feature
        self.edge_index = edge_index
        self.edge_attr = edge_attr

        # print("\n[Node feature]:\n", node_feature)
        # print("\n[Edge index]:\n", edge_index)
        # print("\n[Edge_attribute]:\n", edge_attr)
        # print("\n[Index path]\n", index_path)
        # print("\n[ei file list]\n", ei_file_list)

        return self.x, self.edge_index, self.edge_attr
        



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
            save_csv = os.path.join(save_inx_path, str(folder_name)+'_ei' +str(i)+'.csv')
            change_edge_index.to_csv(save_csv) 

        print((len(list_index), len(unique_list)))
        print("----Re indexing----")
  

    ############################################# Make Edge attributes##########################################
    def make_edge_attr(self,i): 
        # edge_index: [2, num_edges], edge_attr: [num_edges, dim_edge_features]
        edge_feature_path = os.path.join(self.FILEPATH, self.problem, 'edge_features')
        order_file_list = natsort.natsorted(os.listdir(edge_feature_path))
      
        # Call info from new edge_index 
        for order in order_file_list:
            inx_search_path = os.path.join(edge_feature_path, order, 'edge_index')
            inx_file_list = natsort.natsorted(os.listdir(inx_search_path))
            inx_path = os.path.join(inx_search_path, inx_file_list[i])
            # print("\n[Order]", order)

            # # # Read csv file to tensor
            ef = pd.read_csv(inx_path, index_col=0)
            # print("\n[Edge index]\n",ef)
           
            list_attr1 = []
            list_attr0 = []

            ID_list = list(map(int, ef.columns))
            for index in range(len(ID_list)):
                for column in range(len(ID_list)):
                    if ef.iat[index, column] == 1:   
                        list_attr1.append((ID_list[index], ID_list[column]))
                    elif ef.iat[index, column] == 0 and ID_list[index] != ID_list[column]:
                        list_attr0.append((ID_list[index], ID_list[column]))

            list_attr = list_attr1 + list_attr0
            # print(list_attr, "length", len(list_attr))
                  

            # Original data
            ea_example = self.edge_attr
            print("\n[Original]\n",ea_example)
            

            # Changed data
            ea_example.index = list_attr
            ea_example.index.name = "ID"
            edge_attr_csv = ea_example
            print("\n[New]\n",edge_attr_csv)


            # SAVE PATH (edge_attr)
            save_attr_path = os.path.join(edge_feature_path, order,'edge_attr')
            createFolder(save_attr_path)
            save_csv = os.path.join(save_attr_path, str(order)+'_ea' +str(i)+'.csv')
            edge_attr_csv.to_csv(save_csv) 
   

    ############################## Make graph ##################################

    def make_graph(self, fig_num, pos):
        
        # Weight 부여되면 굵어지게
  
        list_edge_attr = []
        list_edge_on = []
        list_edge_grasp = []
        list_node_pair = []

        # Make nodes
        nodes = self.x.index.to_list()

        # Connect edge
        ea_inx = self.edge_attr.index.to_list()

        print("ea_inx", ea_inx)
        
        for tar in ea_inx:
            ret = [int(k) for k in re.split('[^0-9]', tar) if k]
            # print(ret)
            if ret != [0]:
                list_node_pair.append(tuple(ret))
            else:
                pass
        print("\n[List node pair]",list_node_pair)
        
        

        # edge_attr의 column 데이터 list로 가져오기
        col = self.edge_attr.columns.to_list()


        # edge_attr file에서 'rel'이 들어간 문자열 정보 가져오기 
        ea_col = [col[i] for i in range(len(col)) if col[i].find('rel') == 0]    

        print("\n[ea col]",ea_col)
     
        
        #  Relation 보기 간편하게 바꿔줌 string -> tuple
        for i in range(len(list_node_pair)):
            for j in range(len(ea_col)):
                if self.edge_attr.at[ea_inx[i], ea_col[j]] == 1:
                    if ea_col[j] == 'rel_on_right':
                        attr = ea_col[j].replace('rel_on_right', 'On')
                        list_edge_on.append(attr)
                    elif ea_col[j] == 'rel_on_left':
                        attr = ea_col[j].replace('rel_on_left', 'On')
                        list_edge_on.append(attr)
                    elif ea_col[j] == 'rel_in_right':
                        attr = ea_col[j].replace('rel_in_right', 'In')
                    elif ea_col[j] == 'rel_in_left':
                        attr = ea_col[j].replace('rel_in_left', 'In')
                    elif ea_col[j] == 'rel_in_grasp':
                        attr = ea_col[j].replace('rel_in_grasp', 'Grasp')
                        list_edge_grasp.append(attr)
                    elif ea_col[j] == 'rel_grasp':
                        attr = ea_col[j].replace('rel_grasp','Grasp')
                        list_edge_grasp.append(attr)
                    elif ea_col[j] == 'rel_attach':
                        attr = ea_col[j].replace('rel_attach','Attach')
                    else:
                        print("----Re-check relations----")
                    list_edge_attr.append(attr)
    
        
        print("\n[List edge attribute]:",list_edge_attr)
     

        ################### Make graph ####################
        import matplotlib.pyplot as plt
        import networkx as nx
        import PIL
        
        
        # Image URLs for graph nodes
        icons = {
            "Robot0": f"{self.FILEPATH}/icons/robot_hand.jpeg",
            "Block1": f"{self.FILEPATH}/icons/block1.jpg",
            "Block2": f"{self.FILEPATH}/icons/block2.jpg",
            "Block3": f"{self.FILEPATH}/icons/block3.jpg",
            "Block4": f"{self.FILEPATH}/icons/block4.jpg",
            "Block5": f"{self.FILEPATH}/icons/block5.jpg",
            "Bowl6": f"{self.FILEPATH}/icons/bowl6.jpeg",
            "Bowl7": f"{self.FILEPATH}/icons/bowl7.webp",
            "Table": f"{self.FILEPATH}/icons/table_icon.jpg",
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
            g.add_edges_from([list_node_pair[i]], label = f'{list_edge_attr[i]}')
         
               
                
        edge_labels = nx.get_edge_attributes(g,'label')
        print("\n[Edge labels]:",edge_labels)
      

        # POS 1 사진으로 node image 가져오는 것 가능
        
        # pos 지정 => x,y 좌표 array값으로 받아서 사용할 수 있음
        # manually specify node position
        # pos = nx.spring_layout(g)
        # pos = nx.shell_layout(g)
        
        # check the position

        # Get a repreducible layout and create figure
        fig, ax = plt.subplots() 
        
        # Transform from data coordinates
        tr_figure = ax.transData.transform
        # Transform from display to figure coordinates
        tr_axes = fig.transFigure.inverted().transform

       
         
        # Select the size of the image
        icon_size =  0.065 #(ax.get_xlim()[1] - ax.get_xlim()[0])*0.08 # 0.08
        icon_center = icon_size / 2.0                          # 0.025
      
        
        # Show title
        title_font = {'fontsize':14, 'fontweight':'bold'}
        plt.title("Present state", fontdict = title_font)  
        
        

        egrasp = [(u, v) for (u, v, d) in g.edges(data=True) if d["label"] == "Grasp"]
        eon = [(u, v) for (u, v, d) in g.edges(data=True) if d["label"] == "On"]
        ein = [(u, v) for (u, v, d) in g.edges(data=True) if d["label"] == "In"]
        eattach = [(u, v) for (u, v, d) in g.edges(data=True) if d["label"] == "Attach"]

        
        # Draw edges from edge attributes
        # styles = ['filled', 'rounded', 'rounded, filled', 'dashed', 'dotted, bold']
        nx.draw_networkx_edges(G=g, pos=pos, edgelist=egrasp, width=6, alpha=0.5, edge_color="b", style= "dotted")
        nx.draw_networkx_edges(G=g, pos=pos, edgelist=eon, width=3, alpha=0.5, edge_color="black")
        nx.draw_networkx_edges(G=g, pos=pos, edgelist=ein, width=4, alpha=0.5, edge_color="r")
        nx.draw_networkx_edges(G=g, pos=pos, edgelist=eattach, width=4, alpha=0.5, edge_color="r")

        # Draw edge labels from edge attributes
        nx.draw_networkx_edge_labels(G= g, pos = pos, ax=ax, edge_labels = edge_labels, font_size = 10)
      
        
                
        for n in g.nodes:
            xf, yf = tr_figure(pos[n])
            xa, ya = tr_axes((xf, yf))

            # get overlapped axes and plot icon
            a = plt.axes([xa-icon_center, ya-icon_center , icon_size, icon_size])
            a.set_aspect('equal')
            a.imshow(g.nodes[n]['images']) # print(g.nodes[n]) #dictionary에 'image' -> 'images'로 변경됨
            a.axis("off")
             
        
        # plt.figure(figsize=(10,8))  
        ### Check the graphs
        # plt.show() # 

        ### Save the graph files
        nx.draw(g) # 저장할 때
        graph_path = os.path.join(self.FILEPATH, self.problem, 'graph_image')
        createFolder(graph_path)
        task_name = 'task' + str(fig_num) + '.png'
        save_graph_path = os.path.join(graph_path, task_name)
        plt.savefig(save_graph_path)


    ############################################## With velcro objects ################################################
    def velcro_stack(self):
        pass





################################ Creating Folders ##############################################3

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            pass
    except OSError:
        print ('Error: Creating directory.'  +  directory)



stack_pos0 = {
    0: [0.33, 0.35],
    1: [0.33, 0.28],
    2: [0.40, 0.35],
    3: [0.67, 0.28],
    4: [0.60, 0.35],
    5: [0.5, 0.38],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}

stack_pos1 = {
    0: [0.60, 0.5],
    1: [0.33, 0.28],
    2: [0.40, 0.35],
    3: [0.67, 0.28],
    4: [0.60, 0.35],
    5: [0.5, 0.38],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}

stack_pos2 = {
    0: [0.65, 0.45],
    1: [0.33, 0.28],
    2: [0.40, 0.35],
    3: [0.6, 0.35],
    4: [0.5, 0.5],
    5: [0.5, 0.3],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}

stack_pos3 = {
    0: [0.6, 0.5],
    1: [0.33, 0.28],
    2: [0.40, 0.35],
    3: [0.6, 0.35],
    4: [0.5, 0.5],
    5: [0.5, 0.3],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}


stack_pos4 = {
    0: [0.6, 0.5],
    1: [0.33, 0.28],
    2: [0.40, 0.35],
    3: [0.5, 0.7],
    4: [0.5, 0.5],
    5: [0.5, 0.3],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}

stack_pos5 = {
    0: [0.4, 0.6],
    1: [0.33, 0.28],
    2: [0.40, 0.35],
    3: [0.5, 0.7],
    4: [0.5, 0.5],
    5: [0.5, 0.3],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}


stack_pos6 = {
    0: [0.4, 0.6],
    1: [0.35, 0.3],
    2: [0.5, 0.9],
    3: [0.5, 0.7],
    4: [0.5, 0.5],
    5: [0.5, 0.3],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}


stack_pos7 = {
    0: [0.35, 0.6],
    1: [0.35, 0.3],
    2: [0.5, 0.9],
    3: [0.5, 0.7],
    4: [0.5, 0.5],
    5: [0.5, 0.3],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}


stack_pos8 = {
    0: [0.35, 0.5],
    1: [0.5, 1.1],
    2: [0.5, 0.9],
    3: [0.5, 0.7],
    4: [0.5, 0.5],
    5: [0.5, 0.3],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}

stack_pos = [stack_pos0, stack_pos1, stack_pos2, stack_pos3, stack_pos4, stack_pos5, stack_pos6, stack_pos7, stack_pos8]


mix_pos0 = {     
    0: [0.33, 0.35],
    1: [0.67, 0.28],
    2: [0.60, 0.35],
    3: [0.5, 0.38],
    4: [0.40, 0.35],
    5: [0.33, 0.28],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}

mix_pos1 = {     
    0: [0.33, 0.4],
    1: [0.67, 0.28],
    2: [0.60, 0.35],
    3: [0.5, 0.38],
    4: [0.40, 0.35],
    5: [0.33, 0.28],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}

mix_pos2 = {     
    0: [0.33, 0.35],
    1: [0.67, 0.28],
    2: [0.60, 0.35],
    3: [0.5, 0.38],
    4: [0.40, 0.35],
    5: [0.21, 0.23],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}

mix_pos3 = {     
    0: [0.4, 0.47],
    1: [0.67, 0.28],
    2: [0.60, 0.35],
    3: [0.5, 0.38],
    4: [0.4, 0.35],
    5: [0.18, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}

mix_pos4 = {     
    0: [0.4, 0.47],
    1: [0.67, 0.28],
    2: [0.60, 0.35],
    3: [0.5, 0.38],
    4: [0.21, 0.25],
    5: [0.18, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}
mix_pos5 = {     
    0: [0.5, 0.5],
    1: [0.67, 0.28],
    2: [0.60, 0.35],
    3: [0.5, 0.38],
    4: [0.21, 0.25],
    5: [0.18, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}
mix_pos6 = {     
    0: [0.5, 0.5],
    1: [0.67, 0.28],
    2: [0.60, 0.35],
    3: [0.3, 0.3],
    4: [0.17, 0.27],
    5: [0.1, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}
mix_pos7 = {     
    0: [0.6, 0.47],
    1: [0.67, 0.28],
    2: [0.6, 0.35],
    3: [0.3, 0.3],
    4: [0.17, 0.27],
    5: [0.1, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}
mix_pos8 = {     
    0: [0.55, 0.28],
    1: [0.67, 0.28],
    2: [0.43, 0.27],
    3: [0.3, 0.3],
    4: [0.17, 0.27],
    5: [0.1, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}
mix_pos9 = {     
    0: [0.67, 0.4],
    1: [0.67, 0.28],
    2: [0.43, 0.27],
    3: [0.3, 0.3],
    4: [0.17, 0.27],
    5: [0.1, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}
mix_pos10 = {   
    0: [0.67, 0.4],
    1: [0.5, 0.22],
    2: [0.43, 0.27],
    3: [0.3, 0.3],
    4: [0.17, 0.27],
    5: [0.1, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}
mix_pos11 = {   
    0: [0.4, 0.31],
    1: [0.5, 0.22],
    2: [0.43, 0.27],
    3: [0.3, 0.3],
    4: [0.17, 0.27],
    5: [0.1, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}
mix_pos12 =  {   
    0: [0.4, 0.3],
    1: [0.9, 0.22],
    2: [0.83, 0.27],
    3: [0.7, 0.3],
    4: [0.57, 0.27],
    5: [0.5, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}
mix_pos13 =  {   
    0: [0.4, 0.3],
    1: [0.9, 0.22],
    2: [0.83, 0.27],
    3: [0.7, 0.3],
    4: [0.57, 0.27],
    5: [0.5, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}


mix_pos = [mix_pos0, mix_pos1,mix_pos2,mix_pos3,mix_pos4,mix_pos5,mix_pos6,mix_pos7,mix_pos8,mix_pos9,mix_pos10,mix_pos11, mix_pos12, mix_pos13]



