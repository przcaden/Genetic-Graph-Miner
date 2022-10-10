
############################################################################################
#                                                                                          #
#                                       Caden Perez                                        #
#
#
#
############################################################################################


import igraph as ig
import matplotlib.pyplot as plt

n_vertices = 19
edges = [ (0,1), () ]
g = ig.Graph(n_vertices, edges)

# 
node_names = []
network_data = [[]]
for i in range(18):
    node_names.append(str(i))
file = open("node_data.txt", "r")
for l in file:
    if l[0] != '/' and l[0] != '\n':
        network_data.append( (int(l.split()[0]), int(l.split()[1])) )
network_data.remove([])
print(network_data)

# Set graph characteristics
g["title"] = "Genetic Network"
g.vs["names"] = node_names
g.vs["links"] = network_data
g.vs[""]