# k-means clustering
from sklearn.cluster import kMeans

# Build three distinct blobs (each 50 points)
X_cluster = np.vstack([
# blob around [0, 0]
np.random.randn(50,2) + np.array([0, 0]), 
# blob around [5, 5]
np.random.randn(50, 2) + np.array([5, 5]),
# blob around [0, 5]
np.random.randn(50, 2) + np.array(0, 5)])

# k-Means with k = 3, deterministic init via random_state
kmeans = kMeans(n_clusters=3, random_state=0)
kmeans.fit(X_cluster)
print("cluster centers: ", kmeans.cluster_centers_)
print("labe; of first point: ", kmeans,labels_[0])
