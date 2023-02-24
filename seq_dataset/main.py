from seq_demo import *




if __name__ == '__main__':
    # make_data = MakeDataset(problem = 'stacking_5', example = 'ex_1_2_3_4_5')
    

    ## Saving files
    # FILEPATH, _ = os.path.split(os.path.realpath(__file__))
    # print(FILEPATH)
    # save_path = os.path.join(FILEPATH,'stacking_velcro2',file_name, 'node_features')
    # save_path = os.path.join(FILEPATH,'stacking_velcro2',file_name, 'edge_index')
    # save_path = os.path.join(FILEPATH,'stacking_velcro2',file_name, 'edge_attr')
    # createFolder(save_path)
  

# Pick, place, pour 

    # print(make_data.pick(file_num = 0, obj1 = 5))
    # print(make_data.place(file_num = 0, obj1 = 5, obj2= 6))
    # print(make_data.pick(file_num = 0, obj1 = 4))
    # print(make_data.place(file_num = 0, obj1 = 4, obj2= 6))
    # print(make_data.pick(file_num = 0, obj1 = 3))
    # print(make_data.place(file_num = 4, obj1 = 3, obj2= 6))
   
    # print(make_data.pick(file_num = 0, obj1 = 5))
    # print(make_data.place(file_num = 0, obj1 = 5, obj2= 6))
    # print(make_data.pick(file_num = 1, obj1 = 6))
    # print(make_data.pour(file_num = 0, obj1 = 6 , obj2 = 7))

    pos0 = {
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

    pos1 = {
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
    
    pos2 = {
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

    pos3 = {
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


    pos4 = {
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

    pos5 = {
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


    pos6 = {
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


    pos7 = {
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


    pos8 = {
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
    
    
    position = [pos0, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8]

    make_data = MakeDataset(problem = 'stacking_5', example = 'ex_1_2_3_4_5')
  
    for a in range(0, 9):
    ################### Call sample data ####################
        # action_seq = ['pick','place','pick','place','pick','place','pick','place','pick','place']
        print(f"\n[[[[[Task{a}]]]]]")
        print(make_data.sample_data(i=a) )
        # # print(make_data.pick(file_num = 0, obj1 = 5))
        # # print(make_data.place(obj1 = 5, obj2=6)) #file_num = 1     
        # # print(make_data.pour(obj1= 6, obj2=7))
        # # print(make_data.place(obj1 = 6, obj2=2))
        # make_data.make_graph(fig_num=a, pos = position[a])
        # make_data.make_edge_index(i=a) 
        make_data.make_edge_attr(i=a) # def make_edge_index실행 후에 돌려

# plt.figure(figsize=)

### Checking paths


# print(make_data.pick(i=2, obj1=1))
# print(make_data.place(i=1, obj1=1, obj2=2))  # e.g.) obj1=3, obj2=4 -> obj1->obj2


# print(make_data.pick(i=0, obj1= 2))


# print(make_data.place(i=3,obj1=2, obj2=3))


# make_data.save_file(action='pick')
# make_data.save_file(action='place')





            
