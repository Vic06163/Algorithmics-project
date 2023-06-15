import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
import multiprocessing as mp

def generate_random_graph(n=10, weight_range=(1, 10)):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            weight = random.randint(*weight_range)
            matrix[i][j] = weight
            matrix[j][i] = weight
    return matrix

def create_individual(nb_client, starting_point=0):
    clients = list(range(1, nb_client))
    random.shuffle(clients)

    return [starting_point] + clients + [starting_point]

def create_population(nb_client, starting_point=0, size=100):
    return [create_individual(nb_client, starting_point) for _ in range(size)]

def fitness(individual, adjency_matrix):
    fitness = 0
    for i in range(len(individual)-1):
        fitness += adjency_matrix[individual[i]][individual[i+1]]
    return fitness

def selection_worker(population, adjency_matrix, k, output_queue):
    population = sorted(population, key=lambda x: fitness(x, adjency_matrix))
    output_queue.put(population[:k])

def crossover_worker(parent1, parent2, output_queue):
    p1 = parent1[1:-1]
    p2 = parent2[1:-1]
    cross_point = random.randint(1, len(p1) - 1)
    child = p1[:cross_point] + p2[cross_point:]

    for gene in child[:cross_point]:
        if gene in child[cross_point:]:
            child.remove(gene)

    for gene in p1:
        if gene not in child:
            child.append(gene)

    child.insert(0, parent1[0])
    child.append(parent1[-1])

    output_queue.put(child)

def mutation_worker(individual, adjency_matrix, output_queue):
    i, j = random.sample(range(1, len(individual)-1), 2)
    individual[i], individual[j] = individual[j], individual[i]
    output_queue.put(individual)

def renew_population_parallel(population, adjency_matrix, k, mutation_rate):
    selected_population = []
    crossover_output = mp.Queue(maxsize=20)
    mutation_output = mp.Queue(maxsize=20)

    # Selection
    selection_process = mp.Process(target=selection_worker, args=(population, adjency_matrix, k, crossover_output))
    selection_process.start()
    selected_population = crossover_output.get()
    selection_process.join()

    # Crossover
    crossover_processes = []
    for i in range(len(population)):
        parent1 = random.choice(selected_population)
        parent2 = random.choice(selected_population)
        crossover_process = mp.Process(target=crossover_worker, args=(parent1, parent2, mutation_output))
        crossover_process.start()
        crossover_processes.append(crossover_process)

    crossover_results = []
    for process in crossover_processes:
        process.join()
        crossover_results.append(mutation_output.get())

    # Mutation
    mutation_processes = []
    for result in crossover_results:
        mutation_process = mp.Process(target=mutation_worker, args=(result, adjency_matrix, mutation_output))
        mutation_process.start()
        mutation_processes.append(mutation_process)

    mutated_population = []
    for process in mutation_processes:
        process.join()
        mutated_population.append(mutation_output.get())

    return selected_population + mutated_population

def convergence(population, adjency_matrix):
    fitnesses = [fitness(individual, adjency_matrix) for individual in population]
    return len(set(fitnesses)) == 1

def genetic_algorithm(adjency_matrix, starting_point=0, population_size=100, k=10, mutation_rate=0.15):
    population = create_population(nb_client=len(adjency_matrix), starting_point=starting_point, size=population_size)

    while not convergence(population, adjency_matrix):
        print("Best fitness:", fitness(population[0], adjency_matrix), end="\r")
        population = renew_population_parallel(population, adjency_matrix, k=k, mutation_rate=mutation_rate)

    best_individual = sorted(population, key=lambda x: fitness(x, adjency_matrix))[0]
    return best_individual

if __name__ == "__main__":
    adjency_matrix = generate_random_graph(1000)
    best_individual = genetic_algorithm(adjency_matrix, starting_point=0, population_size=50, k=25, mutation_rate=0.15)
    print(best_individual)
    print(fitness(best_individual, adjency_matrix))
