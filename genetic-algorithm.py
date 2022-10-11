
############################################################################################
#                                                                                          #
#                               Caden Perez and Celine Ogero                               #
#                               Genetic Algorithm Graph Miner                              #
#                                   CSCE-480 Intro to AI                                   #
#                                                                                          #
############################################################################################


import igraph as ig
import matplotlib.pyplot as plt

# Get user inputted values for connected nodes:
print('How many nodes would you like to connect?')
num_connected_nodes = 'x'
while not num_connected_nodes.isnumeric():
    num_connected_nodes = str(input())
print('Which nodes would you like to connect (0-18)?')
connecting_nodes = [None] * int(num_connected_nodes)
for i in range(int(num_connected_nodes)):
    connecting_nodes[i] = 'x'
    while not connecting_nodes[i].isnumeric() or int(connecting_nodes[i])>18 or int(connecting_nodes[i])<0:
        connecting_nodes[i] = str(input())

# Initialize graph
node_names = []
network_data = [[]]
for i in range(18):
    node_names.append(str(i))
file = open("node_data.txt", "r")
for l in file:
    if l[0] != '/' and l[0] != '\n':
        network_data.append( (int(l.split()[0]), int(l.split()[1]))  )
network_data.remove([])
n_vertices = 19
g = ig.Graph(n_vertices, network_data)
print(network_data)

# Set graph characteristics
g["title"] = "Genetic Network"
g.vs["names"] = node_names

# Plot graph in matplotlib
# fig, ax = plt.subplots(figsize=(5,5))
# ig.plot(
#     g,
#     target = ax
#     layout="circle", # may change later
#     vertex_size=0.1,
#     vertex_color=["steelblue" if gender == "M" else "salmon" for gender in g.vs["gender"]],
#     vertex_frame_width=4.0,
#     vertex_frame_color="white",
#     vertex_label=g.vs["name"],
#     vertex_label_size=7.0,
#     edge_width=[2 if married else 1 for married in g.es["married"]],
#     edge_color=["#7142cf" if married else "#AAA" for married in g.es["married"]]
# )

# plt.show()