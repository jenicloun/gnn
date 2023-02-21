from seq_demo import *




if __name__ == '__main__':
    make_data = MakeDataset(problem = 'stacking_5', example = 'ex_1_2_3_4_5')

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

  
  
    for a in range(1,2):
        ################### Call sample data ####################
        # action_seq = ['pick','place','pick','place','pick','place','pick','place','pick','place']
    
        make_data.sample_data(i=a) 
        # # print(make_data.pick(file_num = 0, obj1 = 5))
        # # print(make_data.place(obj1 = 5, obj2=6)) #file_num = 1

       
        # # print(make_data.pour(obj1= 6, obj2=7))
        # # print(make_data.place(obj1 = 6, obj2=2))
        make_data.make_graph(fig_num=a)
        # make_data.make_edge_index(i=a) 
        # make_data.make_edge_attr(i=a) # def make_edge_index실행 후에 돌려



### Checking paths


# print(make_data.pick(i=2, obj1=1))
# print(make_data.place(i=1, obj1=1, obj2=2))  # e.g.) obj1=3, obj2=4 -> obj1->obj2


# print(make_data.pick(i=0, obj1= 2))


# print(make_data.place(i=3,obj1=2, obj2=3))


# make_data.save_file(action='pick')
# make_data.save_file(action='place')





            
