import numpy as np
import matplotlib.pyplot as plt
import random

class Graph:
    def __init__(self, nb_client):
        """
        Generate a complete random graph with nb_client nodes
        Return the adjacency matrix and nodes coordinates and demand for each node
        """
        xc = np.random.rand(nb_client + 1)*10
        yc = np.random.rand(nb_client + 1)*10

        # Generate initial traffic matrix
        adjacency_matrix = np.zeros((nb_client + 1, nb_client + 1))
        for i in range(nb_client + 1):
            for j in range(nb_client + 1):
                if i != j:
                    adjacency_matrix[i, j] = np.sqrt((xc[i] - xc[j])**2 + (yc[i] - yc[j])**2)
                    adjacency_matrix[j, i] = adjacency_matrix[i, j]

        self.adjacency_matrix = adjacency_matrix
        self.xc = xc
        self.yc = yc
        self.demand = np.random.randint(1, 10, nb_client + 1)

    def extract_submatrix(self, points):
        """
        Extract a submatrix from a larger matrix given a list of points.
        """
        points = np.append([0], points)  # add depot to cluster
        submatrix = self.adjacency_matrix[np.ix_(points, points)]
        return submatrix

    def update_traffic_matrix(self, current_time):
        """
        Update the traffic matrix according to the current time
        """
        # Here we simulate traffic variation by adding a sinusoidal noise
        self.adjacency_matrix += np.sin(current_time / 10.0)
        return self.adjacency_matrix

class SweepAlgorithm:
    def __init__(self):
        pass
        

    def calculate_angle(self, starting_point_x, starting_point_y, point_x, point_y):
        """
        Calculate the angle between the starting point and the point
        """
        angle = np.arctan2(point_y - starting_point_y, point_x - starting_point_x)
        return angle
    
    def algorithm(self, starting_point, points, truck_capacity):
        """
        Calculate the clusters using the sweep algorithm
        """
        sorted_customers = sorted(points, key=lambda x: self.calculate_angle(starting_point[0], starting_point[1], x[0], x[1]))

        # Create the clusters
        clusters = [[]]
        current_route = 0
        current_capacity = truck_capacity

        for i in range(len(sorted_customers)):
            customer = sorted_customers[i]

            demand = customer[2]
            if demand <= current_capacity:
                clusters[current_route].append(i)
                current_capacity -= demand
            else:
                current_route += 1
                clusters.append([i])
                current_capacity = truck_capacity - demand

        return clusters

    def plot_clusters(self, clusters):
        """
        Plot the clusters
        """
        # Plot the graph and the clusters        
        plt.plot(xc[0], yc[0], "o")
        plt.scatter(xc, yc)
        # Plot the clusters
        for cluster_id, nodes in clusters.items():
            plt.scatter(xc[nodes], yc[nodes], label=f"Cluster {cluster_id}", marker="x")
        plt.show()

class GeneticAlgorithmTSP:
    def __init__(self, graph, population_size=100, starting_point=0, selection_number=10, mutation_rate=0.1):
        self.graph = graph
        self.starting_point = starting_point
        self.population_size = population_size
        self.selection_number = selection_number
        self.mutation_rate = mutation_rate
        self.population = []
        pass

    def create_individual(self):
        """
        Create a random individual
        """
        clients = list(range(1, len(self.graph)))
        np.random.shuffle(clients)
        return [self.starting_point] + clients + [self.starting_point]

    def create_population(self, population_size):
        """
        Create a population of individuals
        """
        self.population = [self.create_individual() for _ in range(population_size)]

    def calculate_fitness(self, individual):
        """
        Calculate the fitness of an individual
        """
        fitness = 0
        for i in range(len(individual) - 1):
            fitness += self.graph[individual[i], individual[i + 1]]
        return fitness

    def selection(self):
        """
        Select the best individuals
        """
        self.population = sorted(self.population, key=lambda x: self.calculate_fitness(x))[:self.selection_number]

    def crossover(self, parent1, parent2):
        """
        Create a new individual from two parents
        """
        # Supprimer la première et la dernière valeur de chaque parent
        p1 = parent1[1:-1]
        p2 = parent2[1:-1]
        
        # Générer un point de croisement aléatoire
        cross_point = random.randint(1, len(p1) - 1)
        
        # Créer un nouvel enfant en combinant les segments des deux parents
        child = p1[:cross_point] + p2[cross_point:]
        
        # Vérifier et supprimer les doublons dans la partie héritée du parent2
        for gene in child[:cross_point]:
            if gene in child[cross_point:]:
                child.remove(gene)
        
        # Si le croisement supprime certains gènes, compléter le nouvel enfant avec les gènes manquants
        for gene in p1:
            if gene not in child:
                child.append(gene)
        
        # Ajouter la première et la dernière valeur
        child.insert(0, parent1[0])
        child.append(parent1[-1])
        
        return child

    def mutation(self, individual):
        """
        Mutate an individual
        """
        # Sélectionner deux points de mutation aléatoires
        mutation_points = random.sample(range(1, len(individual) - 1), 2)
        
        # Inverser les deux points de mutation
        individual[mutation_points[0]], individual[mutation_points[1]] = individual[mutation_points[1]], individual[mutation_points[0]]
        
        return individual

    def renew_population(self):
        """
        Create a new population
        """
        population_size = len(self.population)

        # Sélectionner les meilleurs individus
        self.selection()

        while len(self.population) < population_size:
            # Sélectionner deux parents
            parent1, parent2 = random.sample(self.population, 2)
            
            # Créer un nouvel enfant
            child = self.crossover(parent1, parent2)
            
            # Muter l'enfant
            if random.random() < self.mutation_rate:
                child = self.mutation(child)
            
            # Ajouter l'enfant à la population
            self.population.append(child)

    def convergence(self):
        """
        Check if the algorithm has converged
        """
        fitness = [self.calculate_fitness(individual) for individual in self.population]
        return len(set(fitness)) == 1

    def run(self):
        """
        Run the genetic algorithm
        """
        self.create_population(self.population_size)
        while not self.convergence():
            self.renew_population()
        return self.population[0]

graph = Graph(nb_client=10)

starting_point = (graph.xc[0], graph.yc[0], graph.demand[0])
points = [(graph.xc[i], graph.yc[i], graph.demand[i]) for i in range(1, len(graph.xc))]
truck_capacity = 20

sweep = SweepAlgorithm()
clusters = sweep.algorithm(starting_point, points, truck_capacity)

routes = []

for cluster_id, nodes in enumerate(clusters):
    print(f"Cluster {cluster_id}:", nodes)

    # Get the subgraph of the cluster
    sub_matrix = graph.extract_submatrix(nodes)

    # Run the genetic algorithm
    ga = GeneticAlgorithmTSP(sub_matrix, population_size=20, starting_point=0, selection_number=10, mutation_rate=0.1)
    path = ga.run()

    # Readapt the path to the original graph
    path_final = [0] + [nodes[i-1] for i in path[1:-1]] + [0]
    routes.append(path_final)

    print("Path:", path)
    print("Cluster", cluster_id,"Path_final:", path_final, " for a total distance of", ga.calculate_fitness(path), "\n")

# Plot the graph and the routes
plt.plot(graph.xc[0], graph.yc[0], "o", label="Depot", markersize=10, color="green")
plt.scatter(graph.xc[1:], graph.yc[1:], label="Clients", color="red")
print(routes)
for i, route in enumerate(routes):
    plt.plot(graph.xc[route], graph.yc[route], label=f"Route {i}")

# Annotate the graph
print(points)

for i, (x, y, d) in enumerate(points):
    plt.annotate(f"{i+1} ({d})", (x, y))
plt.legend()
plt.show()