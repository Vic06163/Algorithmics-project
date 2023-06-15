import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def generate_random_graph(nb_client=10):
    """
    Generate a complete random graph with nb_client nodes
    Return the adjacency matrix and nodes coordinates
    """
    xc = np.random.rand(nb_client + 1)*100
    yc = np.random.rand(nb_client + 1)*100

    # Generate initial traffic matrix
    adjacency_matrix = np.zeros((nb_client + 1, nb_client + 1))
    for i in range(nb_client + 1):
        for j in range(nb_client + 1):
            if i != j:
                adjacency_matrix[i, j] = np.sqrt((xc[i] - xc[j])**2 + (yc[i] - yc[j])**2)
                adjacency_matrix[j, i] = adjacency_matrix[i, j]

    return adjacency_matrix, xc, yc

def update_traffic_matrix(adjacency_matrix, current_time):
    """
    Update the traffic matrix according to the current time
    """
    # Here we simulate traffic variation by adding a sinusoidal noise
    adjacency_matrix += np.sin(current_time / 10.0)
    return adjacency_matrix

def extract_submatrix(matrix, points):
    """
    Extract a submatrix from a larger matrix given a list of points.
    """
    submatrix = matrix[np.ix_(points, points)]
    return submatrix

def nearest_neighbour_vrp(adjacency_matrix, clusters):
    """
    Solve the VRP for each cluster using the nearest neighbour heuristic
    """
    tours = []
    current_time = 0.0

    for cluster in clusters:
        cluster = np.append([0], cluster)  # add depot to cluster
        submatrix = extract_submatrix(adjacency_matrix, cluster)
        visited = np.zeros(submatrix.shape[0], dtype=bool)
        visited[0] = True  # start from depot
        tour = [0]  # start tour from depot

        while True:
            current_node = tour[-1]
            submatrix = update_traffic_matrix(submatrix, current_time)
            next_nodes = np.where(visited, np.inf, submatrix[current_node])
            if np.all(np.isinf(next_nodes)):
                break  # break if no unvisited nodes left
            next_node = np.argmin(next_nodes)
            current_time += submatrix[current_node, next_node]
            visited[next_node] = True
            tour.append(next_node)

        tour.append(0)  # return to depot
        tour = [cluster[i] for i in tour]  # translate back to original node indices
        tours.append(tour)

    return tours

def cluster_clients(xc, yc, nb_vehicles):
    """
    Cluster clients into nb_vehicles clusters using KMeans
    """
    coords = np.vstack((xc[1:], yc[1:])).T  # exclude depot
    kmeans = KMeans(n_clusters=nb_vehicles).fit(coords)
    clusters = [np.where(kmeans.labels_ == i)[0] + 1 for i in range(nb_vehicles)]  # +1 to exclude depot
    return clusters

adjacency_matrix, xc, yc = generate_random_graph(nb_client=100)
clusters = cluster_clients(xc, yc, nb_vehicles=5)
tours = nearest_neighbour_vrp(adjacency_matrix, clusters)

for i, tour in enumerate(tours):
    print(f"Tour for vehicle {i+1}: {tour}")


# PLot the graph and the clusters
import matplotlib.pyplot as plt
plt.plot(xc[0], yc[0], "o", markersize=10)
plt.scatter(xc, yc)
# Plot the clusters
for cluster_id, nodes in enumerate(clusters):
    plt.scatter(xc[nodes], yc[nodes], label=f"Cluster {cluster_id}", marker="o")

# Plot the tours
for i, tour in enumerate(tours):
    plt.plot(xc[tour], yc[tour], label=f"Tour {i+1}")

# plt.legend()
plt.show()