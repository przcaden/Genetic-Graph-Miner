
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
def determineStateFitness(state, n_data, h_nodes):
    fitness = 0
    # If edge is highlighted (traversed), calculate fitness.
    # Fitness increases by 1 for each highlighted node connected by the edge.
    for i in range(state):
        if state[i] == True:
            fitness += edgeFitness(n_data[i], h_nodes)
    return fitness


# Determines the fitness of a specific edge.
# Pre: generation has been initialized with traversed edges and user has selected nodes.
# Post: fitness of the given edge has been calculated.
def edgeFitness(connection, h_nodes):
    fitness = 0
    # Check if edge connects highlighted node(s).
    # For each highlighted node connected, fitness increases by 1
    if connection[0] in h_nodes:
        fitness += 1
    if connection[1] in h_nodes:
        fitness += 1
    return fitness

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
def get_probabilities(population, n_data, h_nodes):
    fitnesses = []
    for i in range(NUM_EDGES):
        if population[i]:
            fitnesses.append(edgeFitness(n_data[i], h_nodes))
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
    return pop_data


# Select two parents for crossover based on generated probabilities
# Pre: a set of probabilities has been generated for the given population
# Post: two edges are chosen for crossover in the next population
def selection(pop_data, probabilities, n_data):
    chosen_edges = random.choices(pop_data, cum_weight=probabilities, k=2)

    # for i in range(2):
    #     r = random.random()
    #     for (i, edge) in enumerate(population):
    #         print('i: ',i,' edge: ', edge)
    #         if r <= probabilities[i] and population[i]:
    #             chosen_edges.append(n_data[i])
    #             break
    return chosen_edges


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

    # Initialize graph data
    node_names = []
    highlighted_nodes = []
    base_population = [False]*NUM_EDGES # highlighted edges (bool)
    network_data = [[]] # data of all

    base_population[17] = True
    base_population[16] = True
    base_population[13] = True
    base_population[4] = True

    # Create node properties
    for i in range(NUM_NODES):
        # Append a value for the node
        node_names.append(str(i))
        # Determine if node is part of the connected subgraph and highlight it if so
        if i in connecting_nodes:
            highlighted_nodes.append(True)
        else: highlighted_nodes.append(False)

    highlighted_nodes = [False] * NUM_NODES
    highlighted_nodes[3] = True
    highlighted_nodes[1] = True
    highlighted_nodes[13] = True
    connecting_nodes = (1,3,13)

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

    # Generate initial population
    # g.es["edges"] = random_population()

    # Begin genetic algorithm
    populations = [base_population]
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

        # Perform selection
        # population_probabilities = get_probabilities(g.es["edges"], network_data, highlighted_nodes)
        # parents = selection(g.es["edges"], population_probabilities, network_data)
        # print(parents)

        pop_data = getPopulationData(g.es["population"], network_data)
        print(isComplete(pop_data, connecting_nodes))

        # Step up to next generation (temporary)
        user_input = ''
        while user_input == '':
            user_input = input('Enter any value to proceed to next generation: ')

if __name__ == "__main__":
    main()