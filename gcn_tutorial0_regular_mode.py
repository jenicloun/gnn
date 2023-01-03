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
n_classes = np.unique(colors).shape[0]
labels = np.eye(n_classes)[colors.astype(int)]
fig, ax = plt.subplots(figsize=(10,10))
pos = nx.spring_layout(g, k=5/np.sqrt(g.number_of_nodes()))
kwargs = {"cmap": 'gist_rainbow', "edge_color":'gray'}
nx.draw(
    g, pos, with_labels=True, 
    node_color=colors, 
    ax=ax, **kwargs)
# plt.show()

# Get the Adjacency Matrix A and Node Features Matrix X as numpy array

A = np.array(nx.attr_matrix(g)[0])
X = np.array(nx.attr_matrix(g)[1])
X = np.expand_dims(X,axis=1)

# print('Shape of A: ', A.shape)
# print('\nShape of X:', X.shape)
# print('\nAdjacency Matrix A:\n', A)
# print('\nNode Features Matrix X:\n', X)


# Dot product Adjacency Matrix A and Node Features X
AX = np.dot(A,X)
# print("Cot product of A and X AX:\n", AX)

G_self_loops = g.copy()

self_loops = []
for i in range(g.number_of_nodes()):
    self_loops.append((i,i))
G_self_loops.add_edges_from(self_loops)

# Check the edges of G_self_loops after adding the self loops
# print('\nEdges of G with self-loops:\n', G_self_loops.edges)

# # Get the Adjacency Matrix A and Node Features Matirx X of added self-loops graph
A_hat = np.array(nx.attr_matrix(G_self_loops)[0])
# print('\nAdjacency Matrix of added self-loops G (A_hat):\n', A_hat)

# # Calculate the dot product of A_hat and X (AX)
AX = np.dot(A_hat, X)
# print('\nAX:\n', AX)

# #Get the Degree Matrix of the added self-loops graph
Deg_Mat = G_self_loops.degree()
# print('Degree Matrix of added self-loops G (D): ', Deg_Mat)

# #Convert the Degree Matrix to a N x N matrix where N is the number of nodes
D = np.diag([deg for (n,deg) in list(Deg_Mat)])
# print('Degree Matrix of added self-loops G as numpy array (D):\n', D)

# #Find the inverse of Degree Matrix (D)
D_inv = np.linalg.inv(D)
# print('Inverse of D:\n', D_inv)

# #Dot product of D and AX for normalization
DAX = np.dot(D_inv,AX)
# print('DAX:\n', DAX)

from scipy.linalg import fractional_matrix_power
#Symmetrically-normalization
D_half_norm = fractional_matrix_power(D, -0.5)
DADX = D_half_norm.dot(A_hat).dot(D_half_norm).dot(X)
# print('DADX:\n', DADX)

np.random.seed(77777)
n_h = 4 #number of neurons in the hidden layer
n_y = 2 #number of neurons in the output layer
W0 = np.random.randn(X.shape[1],n_h) * 0.01
W1 = np.random.randn(n_h,n_y) * 0.01

# #Implement ReLu as activation function
def relu(x):
    return np.maximum(0,x)

# #Build GCN layer
# #In this function, we implement numpy to simplify
def gcn(A,H,W):
    I = np.identity(A.shape[0]) #create Identity Matrix of A
    A_hat = A + I #add self-loop to A
    D = np.diag(np.sum(A_hat, axis=0)) #create Degree Matrix of A
    D_half_norm = fractional_matrix_power(D, -0.5) #calculate D to the power of -0.5
    eq = D_half_norm.dot(A_hat).dot(D_half_norm).dot(H).dot(W)
    return relu(eq)
    
# #Do forward propagation
H1 = gcn(A,X,W0)
H2 = gcn(A,H1,W1)
print('Features Representation from GCN output:\n', H2)

def plot_features(H2):
    #Plot the features representation
    x = H2[:,0]
    y = H2[:,1]

    size = 1000

    plt.scatter(x,y,size)
    plt.xlim([np.min(x)*0.9, np.max(x)*1.1])
    plt.ylim([-1, 1])
    plt.xlabel('Feature Representation Dimension 0')
    plt.ylabel('Feature Representation Dimension 1')
    plt.title('Feature Representation')

    for i,row in enumerate(H2):
        str = "{}".format(i)
        plt.annotate(str, (row[0],row[1]),fontsize=18, fontweight='bold')

    plt.show()


plot_features(H2)

