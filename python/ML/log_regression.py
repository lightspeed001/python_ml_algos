from sklearn.linear_model import logisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=4,
n_informative=2, n_redundant=0,
random_state=0)

clf = LogisticRegression(max_iter=1000)
clf.fit(x, y)
print("probability of class 1 for first sample: ", 
clf.predict_proba(X[:1]))
