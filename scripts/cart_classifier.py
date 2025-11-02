import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset (a classification problem)
iris = load_iris()
X = iris.data 
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- Implement CART for Classification ---
print("--- CART Classification (Decision Tree) ---")

# Create a Decision Tree Classifier (CART is the basis for this implementation)
# criterion='gini' is used to measure impurity (Gini Impurity, common for CART)
# max_depth limits the size of the tree to prevent overfitting
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.4f}")

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, 
          feature_names=feature_names,
          class_names=class_names,
          filled=True, 
          rounded=True)
plt.title("CART Decision Tree Classifier")
plt.show()