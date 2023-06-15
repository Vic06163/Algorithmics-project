import numpy as np
from sklearn.cluster import KMeans

def generate_random_graph(nb_client=10):
    """
    Generate a complete random graph with nb_client nodes
    Return the adjacency matrix and nodes coordinates
    """
    xc = np.random.rand(nb_client + 1)*100
    yc = np.random.rand(nb_client + 1)*100
    
    return xc, yc

def cluster_points(xc, yc, nb_clusters):
    """
    Cluster points using K-means
    """
    # Reshape the coordinates to a 2D array
    points = np.column_stack((xc, yc))

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=nb_clusters, random_state=0).fit(points)

    # Print the clusters
    clusters = {}
    for i, cluster_id in enumerate(kmeans.labels_):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(i)

    return clusters

xc, yc = generate_random_graph(nb_client=10)
clusters = cluster_points(xc, yc, nb_clusters=3)

for cluster_id, nodes in clusters.items():
    print(f"Cluster {cluster_id}:")
    print(nodes)

# PLot the graph and the clusters
import matplotlib.pyplot as plt
plt.plot(xc[0], yc[0], "o")
plt.scatter(xc, yc)
# Plot the clusters
for cluster_id, nodes in clusters.items():
    plt.scatter(xc[nodes], yc[nodes], label=f"Cluster {cluster_id}", marker="x")
plt.show()
