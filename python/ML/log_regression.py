from sklearn.linear_model import logisticRegression  # classification model
from sklearn.datasets import make_classification     # synthetic dataset generator

# Generate a binary classification dataset
X, y = make_classification(
n_samples=200, # total rows
n_features=4,  # total columns (all used)
n_informative=2, # only 2 actually predictive 
n_redundant=0, 
random_state=0)

# initialize logistic regression
clf = LogisticRegression(max_iter=1000) # increase max_tier for convergence safety
# train the classifier
clf.fit(x, y)  # fits weight vector + bias
# predict probabilities for the first sample
print("probability of class 1 for first sample: ", 
clf.predict_proba(X[:1]))
