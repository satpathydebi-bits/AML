import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y_true = iris.target  # True class labels

# Elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow method diagram
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')  # Within cluster sum of squares
plt.show()

# Perform K-Means clustering with the optimal number of clusters
optimal_k = 3  # Chosen based on the elbow method plot
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
kmeans.fit(X)

# Map cluster labels to match the true class labels
cluster_labels = np.zeros_like(kmeans.labels_)
for i in range(optimal_k):
    mask = (kmeans.labels_ == i)
    cluster_labels[mask] = np.bincount(y_true[mask]).argmax()

# Assign cluster labels based on the mapping
y_pred = cluster_labels

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)
print("Accuracy %: {}".format(accuracy*100))

# Visualize the clusters
# Here, we visualize the first two features
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering with {} Clusters (Accuracy: {:.2f}%)'.format(optimal_k, accuracy * 100))
plt.show()
