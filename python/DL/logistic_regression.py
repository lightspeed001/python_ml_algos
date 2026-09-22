# For small tabular datasets, classical model is often a better starting point than deep learning
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import acccuracy_score

# Create a synthetic binary-class dataset
X, y = make_classification(
  n_samples=2000,
  n_features=20,
  n_informative=10,
  random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
  X, y,
  test_size=0.2,
  random_state=42,
  stratify=y
)

# Scaling is important for logistic regression
model = make_pipeline(
  StandardScaler(),
  LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train) 

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Test accuracy: {accuracy:.3f}")

