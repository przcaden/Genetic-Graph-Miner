
############################################################################################
#                                                                                          #
#                               Caden Perez and Celine Ogero                               #
#                               Genetic Algorithm Graph Miner                              #
#                                   CSCE-480 Intro to AI                                   #
#                                                                                          #
############################################################################################


import igraph as ig
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random
import queue

# Define algorithm constants
NUM_RANDOM_EDGES = 8
MUTATION_RATE = 0.01
NUM_GENERATIONS = 5
NUM_NODES = 19
NUM_EDGES = 37


# Function that determines the given fitness of a given generation.
# Pre: generation has been initialized with traversed edges.
#      all node connection data has been created and user has selected nodes.
# Post: fitness of the given state has been calculated.


#function to determine the distance between two nodes, ie the number of edges between nodes
# Get respective network data for all edges within a given population.
def getPopulationData(population, n_data):
    pop_data = []
    for i in range(len(population)):
        if population[i]:
            pop_data.append(n_data[i])
    return pop_data


# function to add edges to an adjacency list
def addEdges(edges, x, y):
    edges[x].append(y)
    edges[y].append(x)

 #function for finding minimum number of edges between any two nodes in the graph

def minimumEdgesBFS(edges, u, v):
     
    # visited[n] for keeping track
    # of visited node in BFS
    visited = [0] * NUM_NODES
 
    # Initialize distances as 0
    distance = [0] * NUM_NODES
 
    # queue to do BFS.
    Q = queue.Queue()
    distance[u] = 0
 
    Q.put(u)
    visited[u] = True
    while (not Q.empty()):
        x = Q.get()
         
        for i in range(len(edges[x])):
            if (visited[edges[x][i]]):
                continue
 
            # update distance for i
            distance[edges[x][i]] = distance[x] + 1
            Q.put(edges[x][i])
            visited[edges[x][i]] = 1
    return distance[v]
 
#function to determine edge fitness
#using the algorithm above we will determine the edge fitness by comparing the nunber of edges away from the selected nodes.
# Pre: generation has been initialized with traversed edges and user has selected nodes.
# Post: fitness of the given edge has been calculated.
def edgeFitness(edges, connection, h_nodes):
    fitness_score = 50 #initialize with 1
    number_of_edges_away = 0 #initialize this by 0
   
   #first find how far each of the nodes in the connection is from the nodes the user input.
   #The ones that are closest will be more fit ie  a connection that has 0 and 1 edges away will be more fit. 
    for i in h_nodes:
        if connection[0] in h_nodes:
            number_of_edges_away += minimumEdgesBFS(edges, connection[1], h_nodes[i])
            if number_of_edges_away == 1: # if the other adjacent node is only one edge away then increase the fitness by 1
                fitness_score += 1
            else: 
                #decrease the fitness by the number of edges away
                fitness_score -= number_of_edges_away

        if connection[1] in h_nodes:
            number_of_edges_away += minimumEdgesBFS(edges, connection[0], h_nodes[i])
            if number_of_edges_away == 1: # if the other adjacent node is only one edge away then increase the fitness by 1
                fitness_score += 1
            else: 
                #decrease the fitness by the number of edges away
                fitness_score -= number_of_edges_away

    return fitness_score


# Calculate fitnesses for each edge in a population.
# Pre: generation has been initialized with traversed edges and user has selected nodes.
# Post: a list containing a fitness for each edge has been generated.
def determineStateFitness(pop_data, h_nodes, network_data):
    fitnesses = []
    for edge in pop_data:
        fitnesses.append(edgeFitness(network_data, edge, h_nodes))
    return fitnesses


# Randomly generate a set of edges, which will be the initial population.
# Pre: none
# Post: a random set of [NUM_EDGES] edges are selected to be part of a poulation.
def random_population():
    new_path = [False] * NUM_EDGES
    generated_edges = []
    for i in range(NUM_RANDOM_EDGES):
        rand_edge = random.randint(0,NUM_EDGES-1) # random index
        while rand_edge in generated_edges:
            rand_edge = random.randint(0,NUM_EDGES-1)
        generated_edges.append(rand_edge)
        new_path[rand_edge] = True
    return new_path


# Generate a probability for each edge in a population to be selected
# Pre: a population has already been populated, along with a set of conencted edges.
# Post: determine a set containing a probability of selection for each edge in the given population.
def get_probabilities(pop_data, h_nodes, network_data):
    fitnesses = determineStateFitness(pop_data, h_nodes, network_data)
    total_fitness = sum(fitnesses)
    relative_fitnesses = [f/total_fitness for f in fitnesses]
    probabilities = [sum(relative_fitnesses[:i+1]) for i in range(len(relative_fitnesses))]
    return probabilities


# Get respective network data for all edges within a given population.
# Pre: network data has been retrieved and edges have been randomly selected.
# Post: a list containing network data corresponding to each edge has been created.
def getPopulationData(population, n_data):
    pop_data = []
    for i in range(len(population)):
        if population[i]:
            pop_data.append(n_data[i])
    return pop_data


# Select two parents for crossover based on generated probabilities
# Pre: a set of probabilities has been generated for the given population
# Post: two edges are chosen for crossover in the next population
def selection(pop_data, probabilities, n_data):
    chosen_edges = random.choices(pop_data, k=2)
    #cum_weight=probabilities, 
    return chosen_edges


# Choose respective traits from parents and derive offspring.
# Pre: selection has been performed and two parents were selected.
# Post: two offspring are created based off the traits of parents.
def parent_mating(parents):
    # this is a very basic start to a crossover function, we will likely have to change this later
    offspring1 = (parents[0][0], parents[1][1])
    offspring2 = (parents[0][1], parents[1][0])
    return offspring1, offspring2


# Determine if a given population is complete.
# Pre: population is generated.
# Post: a boolean value is determined, corresponding to if all highlighted nodes are accessed.
def isComplete(pop_data, c_nodes):
    # Traverse population to find possible highlighted nodes
    nodes_traversed = []
    for i in range(len(pop_data)):
        if pop_data[i][0] in c_nodes and pop_data[i][0] not in nodes_traversed:
            for edge in pop_data:
                if pop_data[i][1] in edge and edge != pop_data[i]:
                    nodes_traversed.append(pop_data[i][0])
                    break
        if pop_data[i][1] in c_nodes and pop_data[i][1] not in nodes_traversed:
            for edge in pop_data:
                if pop_data[i][0] in edge and edge != pop_data[i]:
                    nodes_traversed.append(pop_data[i][1])
                    break
    # Determine if every node was hit
    return len(nodes_traversed) == len(c_nodes) and sorted(nodes_traversed) == sorted(c_nodes)


# Function to be called when the plot's button is clicked.
def next(val):
    # Ends the infinite loop called when opening the plot
    plt.gcf().canvas.stop_event_loop()


def main():
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
            connecting_nodes[i] = str(input('Enter a node value: '))
    connecting_nodes = list(map(int, connecting_nodes))

    # Initialize graph data
    node_names = []
    highlighted_nodes = []
    network_data = [[]] # data of all

    # Create node properties
    for i in range(NUM_NODES):
        # Append a value for the node
        node_names.append(str(i))
        # Determine if node is part of the connected subgraph and highlight it if so
        if i in connecting_nodes:
            highlighted_nodes.append(True)
        else: highlighted_nodes.append(False)

    # Get graph path data
    file = open("node_data.txt", "r")
    for l in file:
        if l[0] != '/' and l[0] != '\n':
            network_data.append( (int(l.split()[0]), int(l.split()[1]))  )
    network_data.remove([])

    # Initialize graph with a random population
    g = ig.Graph(NUM_NODES, network_data)
    g["title"] = "Genetic Network"
    g.vs["names"] = node_names
    g.vs["nodes"] = highlighted_nodes
    g.es["connections"] = network_data
    population = random_population()
    g.es["population"] = population
    print('Initial population: ', population)

    #TESTING FOR WHETHER THE BFS ALGORITHM CAN FIND THE SHORTEST DISTANCE GIVEN TWO NDOES
    #test the number of edges between two nodes
    # first get adjacency list of graph using network data
    edges_adjacency_list = [[] for i in range(19)]
    for a, b in network_data:
        addEdges(edges_adjacency_list, a, b)
    
    print ("this is the adjacency list of the graph", edges_adjacency_list)
    print(" shortest distance between nodes 4 and 8 is:" , minimumEdgesBFS(edges_adjacency_list, 4, 8), "edges aways")

    # Get network data of the population's edges only
    pop_data = getPopulationData(population, network_data)
   
    # Begin genetic algorithm
    for current_generation_index in range(NUM_GENERATIONS):
        
        # Plot graph in matplotlib
        fig, ax = plt.subplots(figsize=(5,5))
        ig.plot(
            g,
            target = ax,
            vertex_size=0.25,
            vertex_color=["steelblue" if node_highlighted else "salmon" for node_highlighted in g.vs["nodes"]],
            vertex_frame_width=2.0,
            vertex_frame_color="white",
            vertex_label=g.vs["names"],
            vertex_label_size=9.0,
            edge_width=[2 if edge_traversed else 1 for edge_traversed in g.es["population"]],
            edge_color=["#7142cf" if edge_traversed else "#AAA" for edge_traversed in g.es["population"]]
        )
        axes = plt.axes([0.6, 0.001, 0.3, 0.075])
        bnext = Button(axes, 'Next Generation', color='yellow')
        bnext.on_clicked(next)
        plt.show()

        # Perform selection, crossover, mutation
        population_probabilities = get_probabilities(pop_data, highlighted_nodes, network_data)
        parents = selection(pop_data, population_probabilities, network_data)
        offspring1, offspring2 = parent_mating(parents)
        pop_data.append(offspring1)
        pop_data.append(offspring2)
        print('Offspring 1: ', offspring1)
        print('Offspring 2: ', offspring2)

        # Update graph with new population
        for i in range(len(population)):
            if sorted(offspring1) == sorted(network_data[i]) or sorted(offspring2) == sorted(network_data[i]):
                population[i] = True
        
        g.es["population"] = population
        
        print('Parents: ', parents)

        print(isComplete(pop_data, connecting_nodes))
        # Commenting the following line so we can compare generations
        # plt.close() # close window of previous generation

if __name__ == "__main__":
    main()