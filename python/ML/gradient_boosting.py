# gradient boosting machine (GBM)
from sklearn.ensamble import GradientBoostingRegressor # boosting for regression

# Regression target (linear + noise)
y_reg = X @ np.array([1.2, -0.0, 0.5, 2.0]) + np.random.randn(150) * 0.3

# initialize GBM
# - 200 weak learners (trees)
#  small learning_rate = 0.1 (shrinkage)
# -  max_depth = 3 for each tree
gb = GradientBoostingRegressor(n_estimators=200,
learning_rate=0.1,
max_depth=3,
random_state=0)

# Fit the model
gb.fit(x, y_reg)

# R^2 score on training data (just for illustration)
print("GBM R^2: ", gb.score(X, y_reg))
