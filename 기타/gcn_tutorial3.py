import numpy as np
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
import networkx as nx
import matplotlib.pyplot as plt
g = nx.karate_club_graph()

g.number_of_nodes(), g.number_of_edges()
# 34,78

communities = greedy_modularity_communities(g)
colors = np.zeros(g.number_of_nodes())

for i, com in enumerate(communities):
    # print(list(com))
    colors[list(com)] = i
    # print(colors[list(com)],i)
print(colors)

n_classes = np.unique(colors).shape[0]
labels = np.eye(n_classes)[colors.astype(int)]
# print(labels)
fig, ax = plt.subplots(figsize=(10,10))
pos = nx.spring_layout(g, k=5/np.sqrt(g.number_of_nodes()))
kwargs = {"cmap": 'gist_rainbow', "edge_color":'gray'}
nx.draw(
    g, with_labels=True, 
    node_color=colors, **kwargs)

# plt.show()