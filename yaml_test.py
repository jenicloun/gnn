import csv
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import count

G = nx.Graph()


with open('/home/jeni/Desktop/gcn/box_test1.csv','r') as f:
    data = csv.reader(f)
    print(data)
    print(type(data))

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
        print(type(row))



G_nodes = G.number_of_nodes()
G_edges = G.number_of_edges()
print("Nodes =", G_nodes, "Edges =", G_edges)


# Figure size
plt.figure(figsize=(7,6))

# # Node position (short for pos)
pos = nx.shell_layout(G)


# # Node size
degree = nx.degree(G)

# nx.draw(G, pos, edge_color="black", with_labels = True)
nx.draw(G, pos, node_size = [50+v[1]*100 for v in degree], edge_color="black", with_labels = True)

plt.show()

