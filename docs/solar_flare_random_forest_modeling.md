# Mathematical Modeling for Solar Flare Classification using Random Forest

## Input Variables
X = [duration_minutes, rise_minutes, decay_minutes, flux] — engineered features from solar flare data including duration, rise time, decay time, and peak flux.

## Output Variable
y ∈ {A, B, C, M, X} — solar flare classes based on intensity.

## Equations / Model
For classification using Random Forest:
ŷ = "mode" (f1(X), f2(X), ..., fn(X))
where fi(X) are decision trees trained on random subsets of data.

## Loss Function
L = - ∑ yi log(ŷi) (Cross-Entropy Loss)

## Optimization Method
Gini Impurity or Entropy used to optimize splits in Decision Trees via the CART (Classification and Regression Trees) algorithm. This selects the feature and threshold that maximize the reduction in impurity at each node, leading to better separation of classes and lower cross-entropy loss.

## Flow of Equations Usage

### 1. Data Preprocessing
- Load the historical GOES solar flare dataset (e.g., from CSV file).
- Select input features X = [duration_minutes, rise_minutes, decay_minutes, flux], which are engineered from raw data like start time, peak time, end time, and peak flux.
- Select target variable y representing flare classes {A, B, C, M, X}.
- Standardize features to have zero mean and unit variance: X′ = (X - μ) / σ, where μ is the mean and σ is the standard deviation of each feature. This improves model convergence and performance.

### 2. Model Definition
- Define the Random Forest model as a mathematical mapping ŷ = f_θ(X), where θ represents the model parameters (tree structures).
- The model is an ensemble: ŷ = mode(f1(X), f2(X), ..., fn(X)), where each fi(X) is a decision tree trained on a random subset of the data and features.
- Each decision tree fi recursively splits the data based on feature thresholds to minimize impurity.

### 3. Training / Optimization
- Train the model by minimizing the loss function over the training data.
- For classification, use Cross-Entropy Loss: L = -∑_{i=1}^N ∑_{c=1}^C y_{i,c} log(ŷ_{i,c}), where N is the number of samples, C is the number of classes, y_{i,c} is the true label indicator, and ŷ_{i,c} is the predicted probability for class c.
- During training, optimize each decision tree by selecting splits that maximize the reduction in impurity (Gini or Entropy) using the CART algorithm. This involves evaluating all possible feature-threshold combinations at each node to find the best split.

### 4. Prediction
- For new test data X_test, compute the predicted class ŷ = f_θ(X_test) by passing X_test through each decision tree and taking the mode (most frequent) prediction across all trees.

### 5. Evaluation
- Assess model performance using metrics such as Accuracy (fraction of correct predictions), Precision (true positives / (true positives + false positives)), Recall (true positives / (true positives + false negatives)), and F1-Score (harmonic mean of precision and recall).
- Compare predicted ŷ with true y to quantify classification quality.

### 6. Visualization
- Display results such as a confusion matrix (showing true vs. predicted classes), classification report (detailed metrics per class), or feature importance plot (ranking features by their contribution to splits).
