from sklearn.tree import DecisionTreeClassifier # tree-based classifier
from sklearn.datasets import load_iris # classic 3-class dataset

# load the Iris data (150 samples, 4 features)
iris = load_iris()
X, y = iris.data, iris.target

# Build a shallow tree (max depth = 3) to keep it interoperable
tree = DecisionTreeClassifier(max_depth=3, random_state=0)

# fit the tree
tree.fit(X, y)

# show how much each feature contributed to splits
print("feature importances: ", tree.feature_importances_)
