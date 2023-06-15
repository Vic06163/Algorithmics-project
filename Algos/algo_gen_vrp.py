from collections import defaultdict
import random

def create_edge_table(parent1, parent2):
    edge_table = defaultdict(set)
    for parent in [parent1, parent2]:
        for route in parent.values():
            for i in range(1, len(route)-1):
                edge_table[route[i]].update([route[i-1], route[i+1]])
    return edge_table

def edge_recombination_crossover(parent1, parent2):
    edge_table = create_edge_table(parent1, parent2)
    
    current_node = random.choice(list(edge_table.keys()))  # Step 2
    offspring = {0: [0], 1: [0], 2: [0]}
    current_route = 0
    while edge_table:
        if current_node in edge_table:  # still has unvisited neighbors
            next_node = min(edge_table[current_node], key=lambda x: len(edge_table[x]))
            edge_table.pop(current_node)
            for node, edges in edge_table.items():
                edges.discard(current_node)
        else:  # if all neighbors have been visited
            unvisited = list(edge_table.keys())
            next_node = unvisited[random.randrange(len(unvisited))]
        
        offspring[current_route].append(next_node)
        current_node = next_node

        if len(offspring[current_route]) == len(parent1[current_route]):  # if route is completed
            current_route += 1
            offspring[current_route].append(0)
            
    for route in offspring.values():
        route.append(0)

    return offspring

parent1 = {0: [0, 3, 1, 5, 0], 1: [0, 6, 9, 8, 0], 2: [0, 7, 2, 4, 0]}
parent2 = {0: [0, 9, 7, 6, 0], 1: [0, 1, 2, 3, 0], 2: [0, 8, 5, 4, 0]}

offspring = edge_recombination_crossover(parent1, parent2)

print(offspring)
