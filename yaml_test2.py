import csv
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import count




# Parsing
import pandas as pd
Ground = pd.read_csv("/home/jeni/Desktop/gcn/box_test1.csv")
print(Ground)

parse_g = Ground[["ID","Relation_In","Relation_On","Relation_Attach"]]

print(parse_g)

splited1 = parse_g["Relation_In"].str.split(',',expand=True)
splited2 = parse_g["Relation_On"].str.split(',',expand=True)
splited3 = parse_g["Relation_Attach"].str.split(',',expand=True)

print(splited1)

splited1.rename(columns= lambda x: "Relation_In_" + str(x), inplace=True) # Switch int to str
splited2.rename(columns= lambda x: "Relation_On_" + str(x), inplace=True)
splited3.rename(columns= lambda x: "Relation_Attach_" + str(x), inplace=True)

gt = pd.concat([Ground["ID"],splited1,splited2,splited3], axis=1)

gt.to_csv('/home/jeni/Desktop/gcn/gt_box_test.csv')



# Creating_graph

G = nx.Graph()


with open('/home/jeni/Desktop/gcn/gt_box_test.csv','r') as f:
    data = csv.reader(f)
    headers = next(data)
    for row in tqdm(data):
        for i in range(0,len(row)-1):
            G.add_node(row[i])
            G.add_node(row[i+1])
            if G.has_edge(row[0],row[i+1]):
                G[row[0]][row[i+1]]['weight'] += 1
            else:
                G.add_edge(row[0],row[i+1], weight = 1)

        print(row)
       


G_nodes = G.number_of_nodes()
G_edges = G.number_of_edges()
print("Nodes =", G_nodes, "Edges =", G_edges)


# Figure size
plt.figure(figsize=(7,6))

# Node position (short for pos)
pos = nx.shell_layout(G)


# Node size
degree = nx.degree(G)


# nx.draw(G, pos, node_size = [50+v[1]*100 for v in degree], edge_color="black", with_labels = True)

nx.draw(G, pos, edge_color="black", with_labels = True)

plt.show()









