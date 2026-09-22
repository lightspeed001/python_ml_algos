# Principal Component Analysis
from sklearn.decomposition import PCA

# 5-dimentional data
X_high = np.random.rand(200, 5)

pca = PCA(n_components=2) # keep 2 principal components
X-reduced = pca.fit_transform(X_high)
print("explained variance ratio:", pca.explained_variance_ratio_ )
print("shape after reduction: ", X-reduced.shape)

