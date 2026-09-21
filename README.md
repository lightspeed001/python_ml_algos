# python_ml_algos
## ML Algorithms along with their PyTorch, Scikit Learn and JAX implementation

### 1. ML Models Math Families :people_hugging:

- **The Error Minimizers**: Linear Regression, Logistic Regression, and Neural Networks; Use Calculus to minimize a cost function.
- **The Space Splitters (Geometric)**: Support Vector Models (SVM) and Decision Trees look for ways to cut up high deminesional space to seperate data points. 
- **The Proximity Seekers (Distance Metrics)**: K-Nearest Neighbors (KNN) and K-Means Clustering rely heavily on linear algebra (vector distances like Euclidean or Cosine distance).
- **The Probability Counters (Statistical)**: Naive Bayes and Hidden Markov Models rely on counting frequencies and applying Bayes' Theorem.  

### 2. Core Principals :pen:

As an MLE you need to understand:
- **The Assumptions**: Does this algorithm assume my data is linear? (eg. Linear Regression)
- **The Constraints**: Does it handle missing data well? (eg. Random Forests do, SVMs do not)
- **The Trade-offs**: Is it fast to train but slow to predict? (eg. KNN)

### 3. The Essential 7 :rescue_helmet:

- Linear & Logistic Regression (The baseline constants)
- Decision Trees & Random Forests (The go to for tabular data)
- Gradient Boosting Machines (XGBoost/LightGBM)
- Support Vector Machines (Great for complex boundary lines)
- K-Means Clustering (The unsuervised standard)
- Principal COmponent Analysis (PCS for dimentionality reduction)
- Multi-Layer Perceptrons (The gateway to Deep Learning)

**Questions to ask?**

1. What **problem** is it trying to solve?
2. What is the **loss function** (how does it know what is wrong)?
3. How does it **optimize** that loss function (How does it get better)?


