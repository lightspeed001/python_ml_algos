# gradient boosting machine (GBM)
from sklearn.ensamble import GradientBoostingRegressor # boosting for regression

# Regression target
y_reg = X @ np.array([1.2, -0.0, 0.5, 2.0]) + np.random.randn(150) * 0.3
gb = GradientBoostingRegressor(n_estimators=200,
learning_rate=0.1,
max_depth=3,
random_state=0)

gb.fit(x, y_reg)
print("GBM R^2: ", gb.score(X, y_reg))
