import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import operator

data = pd.read_csv('/home/jeni/Desktop/gcn/banana_pudding_600_predict_label.csv')
data.head()
data.tail()

print(len(data.isnull().any()))
data.isnull().any()
data['Object'].replace(np.nan, '0', inplace= True)

print("Number of object are:" + str(data['Task'].nunique()))
name = pd.DataFrame(data['Task'].unique().tolist(), columns= ['Task'])

object_total = data.copy()
print(object_total.columns)
object_total = pd.concat([object_total['Object'], data['State']])

# G = nx.DiGraph()
# G.add_nodes_from([1, 2, 3, 4, 5]) #same node can be received
# G.add_edges_from([(1, 2), (2, 1), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2,4), (4, 2),
#                  (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3)])
# degree = nx.degree(G)
# print(degree)

# # nx.draw(G,node_size=[500 + v[1]*500 for v in degree], with_labels=True)
# nx.draw(G, with_labels = True)

# plt.show()


#Network analysis
g = nx.Graph()
g = nx.from_pandas_edgelist(data, source = 'Object', target = 'State')
g = nx.from_pandas_edgelist(data, 'Object', 'Relation_on', create_using= nx.DiGraph())
nx.draw(g, with_labels =True)
print(nx.info(g))

plt.figure(figsize=(10,10))
pos = nx.spring_layout(g, k = 0.15)
nx.draw_networkx(g,pos, node_size = 18, node_color = 'blue')
plt.show()

# dgr = nx.degree_centrality(g)

# sorted_dgr = sorted(dgr.items(),key=operator.itemgetter(1), reverse=True)

# for i in range(len(sorted_dgr)):
#     g.add_edge(sorted_dgr[i][0], nodesize=sorted_dgr[i][1])

# nx.draw(g, pos= nx.spring_layout(g,k=3.5))
# ax = plt.gca()

# plt.show()