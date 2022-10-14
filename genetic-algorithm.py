
############################################################################################
#                                                                                          #
#                               Caden Perez and Celine Ogero                               #
#                               Genetic Algorithm Graph Miner                              #
#                                   CSCE-480 Intro to AI                                   #
#                                                                                          #
############################################################################################


from email.mime import base
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
def get_probabilities(population, n_data, h_nodes):
    fitnesses = []
    for i in range(NUM_EDGES):
        if population[i]:
            fitnesses.append(edgeFitness(n_data[i], h_nodes))
    total_fitness = sum(fitnesses)
    relative_fitnesses = [f/total_fitness for f in fitnesses]
    print('Fitnesses: ', fitnesses)
    print('Relative fitnesses: ', relative_fitnesses)
    probabilities = [sum(relative_fitnesses[:i+1]) for i in range(len(relative_fitnesses))]
    print('Edge probabilities: ', probabilities)
    return probabilities

# Select two parents for crossover based on generated probabilities
def selection(population, probabilities):
    chosen_edges = []
    for i in range(2):
        r = random.random()
        for (i, edge) in enumerate(population):
            print('i: ',i,' edge: ', edge)
            if r <= probabilities[i]:
                chosen_edges.append(edge)
                break
    return chosen_edges

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
        if str(i) in connecting_nodes:
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
    g.es["edges"] = [False] * NUM_EDGES

    # Generate initial population
    g.es["edges"] = random_population()

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
            edge_width=[2 if edge_traversed else 1 for edge_traversed in g.es["edges"]],
            edge_color=["#7142cf" if edge_traversed else "#AAA" for edge_traversed in g.es["edges"]]
        )
        plt.show()

        # Perform selection
        population_probabilities = get_probabilities(g.es["edges"], network_data, highlighted_nodes)
        parents = selection(g.es["edges"], population_probabilities)
        print(parents)

        # Step up to next generation (temporary)
        user_input = ''
        while user_input == '':
            user_input = input('Enter any value to proceed to next generation: ')

if __name__ == "__main__":
    main()