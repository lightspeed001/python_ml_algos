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
|  **Gradient**        |  Vector of partial derivatives `VL`  |  Direction of steepest ascent in loss.                              |
|  **Optimizer**       |  Update rule (eg. `θ = θ - αVL`)     |  Adjusts model parameters (`θ`) to minimize loss.                   |
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

### Key Takeaway :spiral_notepad:

