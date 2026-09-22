from sklearn.linear_model import LinearRegression
import numpy as np

# Dummy data: 10 samples, 2 features
x = np.random.rand(100, 2)
y = 3.5 * X[:, 0] - 2.1 * X[:, 1] + 0.8 + np.random.randn(100) * 0.1

model = LinearRegression()
model.fit(X, y) # train
print("coeficients: ", model.coef_) # [3.5, -2.1] (approx.)
print("intercept: ", model.intercept_)
