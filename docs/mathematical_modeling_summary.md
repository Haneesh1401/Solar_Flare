# Mathematical Modeling of Solar Flare Prediction Algorithms

## 1. Neural Network (Deep Learning)

### Input and Output Variables
- Input: Feature vector \( \mathbf{x} = [\text{flux}, \text{month}, \text{day}, \text{hour}, \text{day_of_year}] \in \mathbb{R}^5 \)
- Output: Probability vector \( \hat{\mathbf{y}} \in \mathbb{R}^5 \) representing flare class probabilities (NO FLARE, B, C, M, X)

### Core Mathematical Functions
- Layers: Fully connected (Dense) layers with ReLU activation
- Output layer: Softmax activation for multi-class classification
- Forward pass:
  \[
  \mathbf{h}^{(1)} = \text{ReLU}(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)})
  \]
  \[
  \mathbf{h}^{(2)} = \text{ReLU}(\mathbf{W}^{(2)} \mathbf{h}^{(1)} + \mathbf{b}^{(2)})
  \]
  \[
  \hat{\mathbf{y}} = \text{softmax}(\mathbf{W}^{(3)} \mathbf{h}^{(2)} + \mathbf{b}^{(3)})
  \]

### Loss / Objective Function
- Categorical Cross-Entropy Loss:
  \[
  \mathcal{L} = - \sum_{i=1}^N \sum_{c=1}^5 y_{i,c} \log(\hat{y}_{i,c})
  \]
  where \( y_{i,c} \) is the true label indicator for class \( c \) of sample \( i \).

### Optimization Method
- Adam optimizer with learning rate tuning
- Mini-batch gradient descent with early stopping and learning rate reduction callbacks

---

## 2. Regression Models (Linear and Ridge Regression)

### Input and Output Variables
- Input: Feature vector \( \mathbf{x} \in \mathbb{R}^d \) (engineered features including rise time, decay time, location, etc.)
- Output: Continuous target \( y \in \mathbb{R} \) representing flare intensity or numeric class

### Core Mathematical Functions
- Linear Regression:
  \[
  \hat{y} = \mathbf{w}^\top \mathbf{x} + b
  \]
- Ridge Regression (L2 regularization):
  \[
  \min_{\mathbf{w}, b} \sum_{i=1}^N (y_i - \hat{y}_i)^2 + \alpha \|\mathbf{w}\|_2^2
  \]

### Loss / Objective Function
- Mean Squared Error (MSE):
  \[
  \mathcal{L} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
  \]

### Optimization Method
- Closed-form solution or iterative solvers (e.g., gradient descent) for Ridge regression

---

## 3. Tree-Based Models (Random Forest, XGBoost, LightGBM, CatBoost)

### Input and Output Variables
- Input: Feature vector \( \mathbf{x} \in \mathbb{R}^d \)
- Output: Predicted class probabilities or labels

### Core Mathematical Functions
- Ensemble of decision trees
- Each tree partitions feature space via splits to minimize impurity (e.g., Gini impurity or entropy)
- Gradient boosting models (XGBoost, LightGBM, CatBoost) build trees sequentially to correct errors of previous trees

### Loss / Objective Function
- Multi-class logarithmic loss (mlogloss):
  \[
  \mathcal{L} = - \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(p_{i,c})
  \]

### Optimization Method
- Gradient boosting with tree-based learners
- Regularization parameters tuned via grid search or Bayesian optimization

---

## 4. Logistic Regression

### Input and Output Variables
- Input: Feature vector \( \mathbf{x} \in \mathbb{R}^d \)
- Output: Probability of class membership via sigmoid function

### Core Mathematical Functions
- Logistic function:
  \[
  p = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}
  \]

### Loss / Objective Function
- Log-loss (binary cross-entropy):
  \[
  \mathcal{L} = - \sum_{i=1}^N \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
  \]

### Optimization Method
- Maximum likelihood estimation via iterative solvers (e.g., Newton-Raphson, gradient descent)

---

## 5. Support Vector Machine (SVM)

### Input and Output Variables
- Input: Feature vector \( \mathbf{x} \in \mathbb{R}^d \)
- Output: Class label prediction

### Core Mathematical Functions
- Decision function:
  \[
  f(\mathbf{x}) = \text{sign} \left( \sum_{i=1}^N \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b \right)
  \]
  where \( K \) is the kernel function (e.g., RBF, linear)

### Loss / Objective Function
- Hinge loss with regularization:
  \[
  \min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \max(0, 1 - y_i f(\mathbf{x}_i))
  \]

### Optimization Method
- Quadratic programming solvers or SMO algorithm

---

This summary captures the key mathematical modeling aspects of the solar flare prediction algorithms implemented in the project.
