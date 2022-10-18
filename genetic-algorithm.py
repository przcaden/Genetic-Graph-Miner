
############################################################################################
#                                                                                          #
#                               Caden Perez and Celine Ogero                               #
#                               Genetic Algorithm Graph Miner                              #
#                                   CSCE-480 Intro to AI                                   #
#                                                                                          #
############################################################################################


import igraph as ig
import matplotlib.pyplot as plt
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
#h_nodes: nodes that the user would like to connect
#population data. #will contain all the edges in the current population. 
# a connection will basically be one edge which consists of two nodes. 
#edges adjacency list contains a list of the entire graph
def edgeFitness(edges_adjacency_list, connection, h_nodes):
    fitness_score = 50 #initialize with 50
    num_edges_away_from_first_node = [] #will hold the distances between the first node and the nodes that the user input
    num_edges_away_from_second_node = [] # wiwll hold the distances between the  2nd node and the nodes that the user input

   #first find how far each of the nodes in the connection is from the nodes the user input.
   #The ones that are closest will be more fit ie  a connection that has 0 and 1 edges away will be more fit. 
    for i in range(len(h_nodes)):
        num_edges_away_from_first_node.append(minimumEdgesBFS(edges_adjacency_list, h_nodes[i], connection[0]))
        num_edges_away_from_second_node.append(minimumEdgesBFS(edges_adjacency_list, h_nodes[i], connection[1]))
        # print("these are the distances of :", connection[0], "from", h_nodes[i], num_edges_away_from_first_node)
        if connection[0] == h_nodes[i]:
            fitness_score += 1
        
        elif connection[1] == h_nodes[i]:
            fitness_score += 1

        else: 
            #find which of the nodes in the edge is closest and decrease the fitness score by that.
            #the farther a node is from one of the selectiode
            for i in num_edges_away_from_first_node:
                for j in num_edges_away_from_second_node:
                    if i < j:
                        fitness_score -= i
                    else:
                        fitness_score -= j
        
    return fitness_score

def determineStateFitness(pop_data, edges_adjacency_list,  h_nodes):
    #do we need this function? 
    # what if we make it return a list of the fitness of the connected edges in these generation?
    fitnesses = []
    # If edge is highlighted (traversed), calculate fitness.
    # Fitness increases by 1 for each highlighted node connected by the edge.
    for edge in pop_data:
        fitnesses.append(edgeFitness(edges_adjacency_list, edge, h_nodes))
    print("these are the fitnesses", fitnesses)
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
    print("this is the initial population", new_path)
    return new_path


# Generate a probability for each edge in a population to be selected
# Pre: a population has already been populated, along with a set of conencted edges.
# Post: determine a set containing a probability of selection for each edge in the given population.
def get_probabilities(pop_data, edges_adjacency_list,  h_nodes):
    fitnesses = determineStateFitness(pop_data, edges_adjacency_list, h_nodes)
    total_fitness = sum(fitnesses)
    relative_fitnesses = [f/total_fitness for f in fitnesses]
    probabilities = [sum(relative_fitnesses[:i+1]) for i in range(len(relative_fitnesses))]
    print('Calculated probabilities: ', probabilities)
    return probabilities


# Get respective network data for all edges within a given population.
def getPopulationData(population, n_data):
    pop_data = []
    for i in range(len(population)):
        if population[i]:
            pop_data.append(n_data[i])
    print("the population data is: ", pop_data)
    return pop_data


# Select two parents for crossover based on generated probabilities
# Pre: a set of probabilities has been generated for the given population
# Post: two edges are chosen for crossover in the next population
def selection(pop_data, probabilities):
    # chosen_edges = random.choices(pop_data, cum_weight=probabilities, k=2)
    # return chosen_edges
    chosen = []
    for n in range(2): #we want to generate two of the fittest organisms to serve as parents.
        r = random.random() # generate a random number between 0 and 1
        for (i, individual) in enumerate(pop_data):
            if r <= probabilities[i]:
                chosen.append(list(individual))
                break
    print("the selected parents are:", chosen)
    return chosen


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
                if pop_data[i][1] in edge:
                    nodes_traversed.append(pop_data[i][0])
                    break
        if pop_data[i][1] in c_nodes and pop_data[i][1] not in nodes_traversed:
            for edge in pop_data:
                if pop_data[i][0] in edge:
                    nodes_traversed.append(pop_data[i][1])
                    break
    # Determine if every node was hit
    return len(nodes_traversed) == len(c_nodes) and sorted(nodes_traversed) == sorted(c_nodes)

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
            connecting_nodes[i] = str(input())
    connecting_nodes = list(map(int, connecting_nodes))
    print("connecting nodes", connecting_nodes)

    # Initialize graph data
    node_names = []
    highlighted_nodes = []
    base_population = [False]*NUM_EDGES # highlighted edges (bool)
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

    # Set initial graph characteristics
    g = ig.Graph(NUM_NODES, network_data)
    g["title"] = "Genetic Network"
    g.vs["names"] = node_names
    g.vs["nodes"] = highlighted_nodes
    g.es["connections"] = network_data
    g.es["population"] = base_population

    #TESTING FOR WHETHER THE BFS ALGORITHM CAN FIND THE SHORTEST DISTANCE GIVEN TWO NDOES
    #test the number of edges between two nodes
    # first get adjacency list of graph using network data
    edges_adjacency_list = [[] for i in range(19)]
    for a, b in network_data:
        addEdges(edges_adjacency_list, a, b)
    
    print ("this is the adjacency list of the graph", edges_adjacency_list)
    print(" shortest distance between nodes 4 and 8 is:" , minimumEdgesBFS(edges_adjacency_list, 4, 8), "edges aways")
   
    # Generate initial population
    population = random_population()
    g.es["population"] = population

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
        plt.show()

        pop_data = getPopulationData(population, network_data)

        # Perform selection, crossover, mutation
        population_probabilities = get_probabilities(pop_data, edges_adjacency_list, connecting_nodes)
        parents = selection(pop_data, population_probabilities)
        offspring1, offspring2 = parent_mating(parents)
        population.append(offspring1, offspring2)
        g.es["population"] = population
        print(parents)

        print(isComplete(pop_data, connecting_nodes))

        # Step up to next generation (temporary)
        user_input = ''
        while user_input == '':
            user_input = input('Enter any value to proceed to next generation: ')

if __name__ == "__main__":
    main()