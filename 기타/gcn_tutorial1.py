import networkx as nx
import matplotlib.pyplot as plt

## tutorial1
g = nx.Graph()
g.add_edge('a','b', weight = 0.1)
g.add_edge('b','c', weight = 1.5)
g.add_edge('a','c', weight = 1.0)
g.add_edge('c','d', weight = 2.2)
print(nx.shortest_path(g,'b','d')) #weighted = False 
print(nx.shortest_path_length(g,'b','d')) #shortest path length
# print(nx.shortest_path(g,'b','d',weighted=True)) #??

## tutorial2
h = nx.path_graph(5) # Adding node from 0 to 4 
g.add_nodes_from(h)


## tutorial3
import math
g.add_node(math.cos) # cosine function
fh = open('tmp.txt','w')
g.add_node(fh)
print(g.nodes())
g.add_edges_from(h.edges()) # Container of edges



from collections import deque

def breadth_first_search(g, source):
    queue = deque([None, source])
    enqueued = set([source])
    while queue:
        parent, n = queue.popleft()
        yield parent,n
        new = set(g[n]) - enqueued
        enqueued |= new
        queue.extend([(n,child) for child in new])

nx.draw(g, with_labels =True)
plt.show()
