# python_ml_algos
## ML, DL and RL Algorithms along with their PyTorch, Scikit Learn, Gymnasium and JAX implementation

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
- Multi-Layer Perceptrons (The gateway to Deep Learning. The Simple Neural Network)

### Deep Learning Algorithms :key:

- **Logistic Regression**: For small tabular datasets, a classical model is often a better starting point than deep learning.
- **Multilayer perception (MLP)**: An MLP is useful for tabular data and general purpose classification or regression.
- **Convolutional Neural Network (CNN)** - CNN's are commonly used for images because convolutions detect local patterns such as edges, textures and shapes.
- **LSTM for sequence classification**: LSTM are useful for sequences such as time series, sensor readings, and token sequences.
- **Transformer encoder (_for sequence classification_)**: Transformers are widely used for language, time series, and other sequence problems. Unlike an LSTM a Transformer can process sequence positions in parrallel.
- **Autoencoder (_for dimentionality reduction_)**: An autoencoder learns to reconstruct its input. It can be used for compression, denoising and anomaly detection.

### Reinforcement Learning Algorithms :robot:

- Q-Learning (Tabular): A basic algo for discrete action spaces.
- Deep Q-Network (DQN): Uses a neural network to approximate the Q-function.
- Policy Gradient (REINFORCE): A policy-based method that directly optimizes the policy.
- Proximal Policy Optimization (PPO): A modern policy gradient method with clipped objective.
- Soft Actor-Critic (SAC): A state-of-the-art RL algorithm that optimizes for entropy-regulated rewards.

**Questions to ask?**

1. What **problem** is it trying to solve?
2. What is the **loss function** (how does it know what is wrong)?
3. How does it **optimize** that loss function (How does it get better)?
