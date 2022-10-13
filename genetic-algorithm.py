
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

NUM_RANDOM_GENERATIONS = 8
NUM_GENERATIONS = 5
NUM_NODES = 19
NUM_EDGES = 37

#           Genetic algorithm pseudocode
# begin 
#     generation = 0 
#     while ( best_fitness != 0 ): 
#         selection(population) 
#         crossover(population) 
#         mutation(population) 
#         if ( Best(population) < best_fitness ): 
#             then best_fitness = Best(population) 
#         generation += 1 
#     end while 
#     return best_fitness 
# end 




#           Selection function pseudocode
#       Inputs: 
#       Returns: 
# begin
#     for each f in fitnesses:
#         for each parent of node:
#             select 2 parents with highest fitnesses?




#           Fitness-determining function pseudocode
# every time one of the target nodes appears in a state, increment fitness by 1
#       Inputs: list of traversed edges (a state)
#       Returns: an integer (rating of fitness).
# begin
#     fitness = 0
#     for each edge=True in state:
#         if edge is connected to a highlighted node:
#             fitness += 1
#         if edge connects two highlighted nodes:
#             fitness += 2
#     return fitness

# Function that determines the given fitness of a given generation.
# Pre: state has been initialized with traversed edges.
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
def random_population(path_traversed):
    new_path = [path_traversed]
    generated_edges = []
    for i in range(NUM_RANDOM_GENERATIONS):
        rand_edge = random.randint(0,NUM_EDGES-1) # random index
        while rand_edge in generated_edges:
            rand_edge = random.randint(0,NUM_EDGES-1)
        generated_edges.append(rand_edge)
        print(rand_edge)
        new_path[rand_edge] = True
    return new_path

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
    path_traversed = [] # highlighted edges (bool)
    network_data = [[]] # data of all

    print(NUM_NODES)
    print(connecting_nodes)
    for i in range(NUM_NODES):
        # Append a value for the node
        node_names.append(str(i))
        # Determine if node is part of the connected subgraph and highlight it if so
        if str(i) in connecting_nodes:
            highlighted_nodes.append(True)
        else: highlighted_nodes.append(False)
    print(highlighted_nodes)

    for i in range(NUM_EDGES):
        # Initialize path highlighting (False = not traversed yet)
        path_traversed.append(False)

    # Get graph path data
    file = open("node_data.txt", "r")
    for l in file:
        if l[0] != '/' and l[0] != '\n':
            network_data.append( (int(l.split()[0]), int(l.split()[1]))  )
    network_data.remove([])

    # Set graph characteristics
    g = ig.Graph(NUM_NODES, network_data)
    g["title"] = "Genetic Network"
    g.vs["names"] = node_names
    g.vs["nodes"] = highlighted_nodes
    g.es["connections"] = network_data
    g.es["edges"] = [False] * NUM_EDGES

    # Plot graph in matplotlib
    fig, ax = plt.subplots(figsize=(5,5))
    ig.plot(
        g,
        target = ax,
        vertex_size=0.3,
        vertex_color=["steelblue" if node_highlighted else "salmon" for node_highlighted in g.vs["nodes"]],
        vertex_frame_width=2.0,
        vertex_frame_color="white",
        vertex_label=g.vs["names"],
        vertex_label_size=7.0,
        edge_width=[2 if edge_traversed else 1 for edge_traversed in g.es["edges"]],
        edge_color=["#7142cf" if edge_traversed else "#AAA" for edge_traversed in g.es["edges"]]
    )
    plt.show()

    # Generate initial population
    g.es["edges"] = random_population(path_traversed)
    plt.draw()

    populations = [[]]
    fitnesses = []
    newPopulation = []
    populations.append(random_population())
    determineStateFitness(populations[0])

    # while i in range(NUM_GENERATIONS):
        # update_graph()
        # selection():
        #   - compare fitness of all edges: one that are more fit get selected
        #   - if more than one state in states, compare completeness of all states
        #   - goal is to find another generation

        # def propagate(self):
        #     newPopulation = []
        #     while len(newPopulation) < len(self.population):
        #         parents = self.chooseParents()
        #         if random() < self.crossover_chance:
        #             [child1, child2] = parents[0].crossover(parents[1])
        #             newPopulation.append(child1)
        #             newPopulation.append(child2)
        #         else:
        #             newPopulation.append(parents[0])
        #             newPopulation.append(parents[1])
        #     if len(newPopulation) > len(self.population):
        #         newPopulation.pop()
        #     for potentialSolution in newPopulation:
        #         if random() < self.mutation_chance:
        #             potentialSolution.mutate()
        #     self.population = newPopulation

if __name__ == "__main__":
    main()