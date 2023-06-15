import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx

def generate_random_graph(n=10, weight_range=(1, 10)):
    # First thing is to create the map that we're trying to visit
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            weight = random.randint(*weight_range)
            matrix[i][j] = weight
            matrix[j][i] = weight
    return matrix

def create_individual(nb_client, starting_point=0):
    """
    Create a random individual
    """
    clients = list(range(1, nb_client))
    random.shuffle(clients)

    return [starting_point] + clients + [starting_point]


def create_population(nb_client, starting_point=0, size=100):
    """
    Create a population of individuals
    """
    return [create_individual(nb_client, starting_point) for _ in range(size)]

def fitness(individual, adjency_matrix):
    """
    Compute the fitness of an individual
    """
    fitness = 0
    for i in range(len(individual)-1):
        fitness += adjency_matrix[individual[i]][individual[i+1]]
    return fitness

def selection(population, adjency_matrix, k=10):
    """
    Select the best individuals in the population
    """
    # We need to sort the population by fitness
    # https://docs.python.org/3/howto/sorting.html
    # Select the k best individuals with the lowest fitness
    population = sorted(population, key=lambda x: fitness(x, adjency_matrix))
    return population[:k]

def crossover(parent1, parent2):
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

def mutation(individual, adjency_matrix):
    """
    Mutate an individual
    """
    # We select two random points in the individual
    # and we swap them
    i, j = random.sample(range(1, len(individual)-1), 2)
    individual[i], individual[j] = individual[j], individual[i]
    return individual

def renew_population(population, adjency_matrix, k=10, mutation_rate=0.15):
    """
    Renew the population
    """
    population_size = len(population)

    # Select the best individuals
    population = selection(population, adjency_matrix, k=k)
    
    while len(population) < population_size:
        # Create the children
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        child = crossover(parent1, parent2)
        if random.random() < mutation_rate:
            child = mutation(child, adjency_matrix)
        population.append(child)

    return population

def convergence(population, adjency_matrix):
    """
    Check if the population has converged
    """
    # We need to check if all the individuals have the same fitness
    fitnesses = [fitness(individual, adjency_matrix) for individual in population]
    return len(set(fitnesses)) == 1

def genetic_algorithm(adjency_matrix, starting_point=0, population_size=100, k=10, mutation_rate=0.15):
    """
    Run the genetic algorithm
    """
    # Create the initial population
    population = create_population(nb_client=len(adjency_matrix), starting_point=starting_point, size=population_size)
    # Loop until convergence
    while not convergence(population, adjency_matrix):
        print("Best fitness:", fitness(population[0], adjency_matrix), end="\r")
        population = renew_population(population, adjency_matrix, k=k, mutation_rate=mutation_rate)

    # Select the best individual
    best_individual = selection(population, adjency_matrix, k=1)[0]
    return best_individual

adjency_matrix = generate_random_graph(1000)
best_individual = genetic_algorithm(adjency_matrix, starting_point=0, population_size=50, k=25, mutation_rate=0.15)
print(best_individual)
print(fitness(best_individual, adjency_matrix))