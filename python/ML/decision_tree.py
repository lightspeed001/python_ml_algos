from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(X, y)
print("feature importances: ", tree.feature_importances_)
