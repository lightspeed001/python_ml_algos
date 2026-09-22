# imports
from sklearn.linear_model import LinearRegression
import numpy as np

# Dummy data: 100 samples, 2 features
x = np.random.rand(100, 2)  # shape (100, 2)
y = 3.5 * X[:, 0] - 2.1 * X[:, 1] + 0.8 + np.random.randn(100) * 0.1 # True relationship: y = 3.5 x0 - 2.1 x1 + 0.8 + noise

# initialize the linear regression model

model = LinearRegression()  # no regularisation by default
model.fit(X, y) # fit (train) the model on the data
# inspect learned parameters
print("coeficients: ", model.coef_) # [3.5, -2.1] (approx.)
print("intercept: ", model.intercept_) # ~ 0.8
