# python_ml_algos/math
## Mapping ML, DL and RL Jargon to Simple Math

Here's a mapping of common ML/DL/RL jargon into simple mathematical terms.

### Machine Learning (ML) Jargon :robot:

|  **Term**            |  **Math Equivalent**                 |  **Explanation**                                                    |
|----------------------|--------------------------------------|---------------------------------------------------------------------|
|  **Feature**         |  Vector (or scalar)                  |  Input variables (eg. `[x₁, x₂, ..., xₙ]`).                         |
|  **Label**           |  Scalar (or vector)                  |  Output/target (eg., `y` for regression, `c` for classification)    |
|  **Model**           |  Function `f: X → Y`                 |  Maps inputs (`X`) to outputs (`Y`).                                |
|  **Loss Function**   |  Scalar Function `L(y, ŷ)`           |  Measures error between true (`y`) and predicted (`ŷ`) values       |
|  **Gradient**        |  Vector of partial derivatives `∇L`  |  Direction of steepest ascent in loss.                              |
|  **Optimizer**       |  Update rule (eg. `θ = θ - α∇L`)     |  Adjusts model parameters (`θ`) to minimize loss.                   |
|  **Bias**            |  Scalar offset `b`                   |  Shifts output (e.g. `y = wx + b`)                                  |
|  **Weight**          |  Scalar (or matrix) `y`              |  Scales input contribution (eg. `y = wx`)                           |
|  **Hyperparameter**  |  Scalar (eg. learning rate `α`)      |  Configurable setting (not learned from data)                       |


### Deep Learning (DL) Jargon :ocean:

|  **Term**            |  **Math Equivalent**                    |  **Explanation**                                                    |
|----------------------|-----------------------------------------|---------------------------------------------------------------------|
|  **Tensor**          |  Multi-dimensional array (eg. `ℝⁿˣᵐˣᵏ`) |  Generalization of vectors/matrices (eg. 3D for RGB images)         |
|  **Layer**           |  Function composition `fₙ ∘ fₙ₋₁ ∘ ...` |  Stacked transformations (eg. `ReLU(Wx + b)`)                       |  
|  **Activation**      |  Nonlinear function `σ(z)`              |  Introduces nonlinearity (eg. `ReLU(z) = max(0, z)`)                |
|  **Batch**           |  Subset of data `X ∈ ℝᵇˣⁿ`              |  Mini batch of `b` samples, each with `n` features.                 |
|  **Epoch**           |  Full pass over dataset `D`             |  One iteration over all training data.                              |
|  **Backpropagation** |  Chain rule `∂L/∂θ = (∂L/∂ŷ)(∂ŷ/∂θ)`    |  Computes gradients via automatic differntiation.                   |
|  **Convolution**     |  Linear operator `*K`                   |  Applies kernel `K` to input (eg. `y = X * K`)                      |
|  **Pooling**         |  Downsampling function (eg., `max`)     |  Reduces spacial dimensions (eg. `max_pool(X)`)                     |


### Reinforcement Learning (RL) Jargon :mechanical_arm:

|  **Term**              |  **Math Equivalent**                    |  **Explanation**                                                          |
|------------------------|-----------------------------------------|---------------------------------------------------------------------------|
|  **State**             |  Vector `s ∈ S`                         |  Environment's current configuration                                      |
|  **Action**            |  Vector `a ∈ A`                         |  Agent's decision (eg. move left/right)                                   |
|  **Reward**            |  Scalar `r`                             |  Immediate feedback from environment                                      |
|  **Policy**            |  Function `π(a|s)`                      |  Maps states to actions (eg. `a = π(s)`)                                  |
|  **Value Function**    |  `Vπ(s) = ℝ[Σγᵗrₜ | s₀ = s]`            |  Expected return starting from `s` under policy `π`                       |
|  **Q-Function**        |  `Qπ(s,a) = ℝ[Σγᵗrₜ | s₀ = s, a₀ = a]`  |  Expected return for taking action `a` i state `s`.                       |
|  **Discount Factor**   |  Scalar  `γ ∈ [0,1]`                    |  Weighs future rewards (eg. `y=0.9` priorities immediate rewards)         |
|  **Bellman Equation**  |  Recursive update `V(s) = r + γV(s')`   |  Relate value of a state to its successor.                                |
|  **Exploration**       |  Stochastic policy `π(a|s)`             |  Balances trying new actions vs. exploiting known rewards (eg. ε-greedy)  |  


### General ML/DL/RL Jargon (Math Translations) :computer:

|  **Term**              |  **Math Equivalent**                    |  **Explanation**                                                          |
|------------------------|-----------------------------------------|---------------------------------------------------------------------------|
|

### **Key Differences**

| **Aspect**       | **Forward Pass**               	 | **Backward Pass**               |
|------------------|-------------------------------------|---------------------------------|
|	**Purpose**    | Compute output (`ŷ`)				 | Compute gradients (`∇θL`)	   |
|	**Direction**  | Input --> Output					 | Output --> Input				   |
|	**Operations** | Matrix multiplications, activations | Chain rule, derivatives		   |
|	**Used for**   | Inference, loss computation		 | Parameter updates (training)	   |

- Code example (Backwards Pass):

```python
import torch

# forward pass
x = torch.tensor([1.0, 2.0])
W = torch.tensor([0.5, -0.3], [0.1, 0.4])
b = torch.tensor([0.2, -0.1])
z = torch.matmul(x, W) + b # Linear tranformation
y_pred = torch.relu(z) # Activation

# Backward pass (automatic in PyTorch)
loss = (y_pred - torch.tensor([1.0, 0.5])) ** 2
loss.backward() # Computes ∇W, ∇b, etc.

print("Gradients:", W.grad, b.grad)
```

### Key Takeaway :spiral_notepad:

- **Forward**: Pure function evaluation (`x → ŷ`)
- **Backward Pass**: Gradient computation via chain rule (`∇θL`)
- **Optimization**: Parameter updates (`θ = θ - α∇L`)
- **Generalization**: Balancing bias-variance, regularization, and validation
- **Reinforcement Learning**: Maximizing cumulative reward (`Vπ(s)`, `Qπ(s,a)`)

### **Timeline: Autoencoders vs Transformers**

| **Year** | **Model**               	| **Key Contribution**                                                   	| **Relation to Autoencoders**                     | **1986** | Autoencoder (Hilton et al) | First formalization of the autoencoder as NNs for unsupervided learning   | Foundational work
| 1990s	   | Variational Autoencoders   | Introduced probabalistic latent spaces (`z ~ N(μ, σ²)`)					| Extentionof auto encoders
| 2012	   | AlexNet (CNN)				| Popularized DL: autoencoders were already widely used for feature learning| Autoencoders were a standard tool in DL
| 2014	   | GANs (Goodfellow et. al)   | AE like desgns for generative modelling									| Built on AE principles
| 2017	   | Transformers (Vaswani etc.)| Self attention mechs for sequence modelling.								| Not related to AE
| 2018	   | ViT						| Tranformers to image data (AE like patch embeddings)						| Hybrid approches emerged later.

__Key differences:__ :bulb:

| **Aspect**            | **Autoencoders**                          	| **Transformers**                          				|
|-----------------------|-----------------------------------------------|-----------------------------------------------------------|
| **Core Idea**			| Compress/ reconstruct data (`x → z → x̂`)  	| Model long-range dependencies via attention (`x → z → y`).|
| **Architecture**		| Encoder-decoder (eg. CNN/RNN)					| Self-attention + feed-forward layers.						|
| **Training Objective**| Minimize reconstruction error (`||x - x̂||²`)	| Maximize sequence likelihood (eg. `P(y|x)`)				|
| **Use Cases**			| DR, Anamoly detection, Gen models				| NLP, vision, time-series, multimodal tasks				|
| **First proposed**	| **1986** (Hitlton et al.)						| **2017**													|


__Summary__ :notebook:

- Autoencoders (1986): came first and are still widely used
- Transformers: area separate breakthrough for sequence modelling
- Modern Hybrids (eg. MAE): combine both for tasks like self supervised learning



