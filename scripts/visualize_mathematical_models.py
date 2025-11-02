import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set script directory
script_dir = Path(__file__).resolve().parent
output_dir = script_dir.parent / "docs" / "visualizations"
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Neural Network Visualization
def visualize_nn():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Input layer
    for i in range(5):
        ax.add_patch(plt.Rectangle((1, 5-i), 1, 0.8, fill=True, color='blue', alpha=0.7))
        ax.text(1.5, 5.4-i, f'x{i+1}', ha='center', va='center', color='white', fontsize=10)

    # Hidden layer 1
    for i in range(4):
        ax.add_patch(plt.Rectangle((4, 4.5-i*1.2), 1, 0.8, fill=True, color='green', alpha=0.7))
        ax.text(4.5, 4.9-i*1.2, 'ReLU', ha='center', va='center', color='white', fontsize=8)

    # Hidden layer 2
    for i in range(3):
        ax.add_patch(plt.Rectangle((7, 4-i*1.5), 1, 0.8, fill=True, color='green', alpha=0.7))
        ax.text(7.5, 4.4-i*1.5, 'ReLU', ha='center', va='center', color='white', fontsize=8)

    # Output layer
    for i in range(5):
        ax.add_patch(plt.Rectangle((10, 4.5-i*0.9), 1, 0.8, fill=True, color='red', alpha=0.7))
        ax.text(10.5, 4.9-i*0.9, 'Softmax', ha='center', va='center', color='white', fontsize=8)

    # Arrows
    ax.arrow(2.5, 3, 1.5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.arrow(5.5, 3, 1.5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.arrow(8.5, 3, 1.5, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')

    ax.set_title('Neural Network Architecture')
    plt.savefig(output_dir / 'neural_network.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Regression Model Visualization
def visualize_regression():
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.normal(0, 1, 100)
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5, label='Data')
    plt.plot(x, 2*x + 1, color='red', label='Linear Fit: ŷ = 2x + 1')
    plt.xlabel('Input Feature')
    plt.ylabel('Output')
    plt.title('Linear Regression Model')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'linear_regression.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Decision Tree Visualization (Simplified)
def visualize_tree():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Root
    ax.add_patch(plt.Circle((5, 8), 0.5, fill=True, color='lightblue'))
    ax.text(5, 8, 'Root\nFeature <= Threshold', ha='center', va='center', fontsize=10)

    # Left child
    ax.add_patch(plt.Circle((3, 6), 0.5, fill=True, color='lightgreen'))
    ax.text(3, 6, 'Left\nClass A', ha='center', va='center', fontsize=10)

    # Right child
    ax.add_patch(plt.Circle((7, 6), 0.5, fill=True, color='lightcoral'))
    ax.text(7, 6, 'Right\nClass B', ha='center', va='center', fontsize=10)

    # Arrows
    ax.arrow(5, 7.5, -1.5, -1, head_width=0.1, head_length=0.1, fc='k', ec='k')
    ax.arrow(5, 7.5, 1.5, -1, head_width=0.1, head_length=0.1, fc='k', ec='k')

    ax.set_title('Decision Tree Structure')
    plt.savefig(output_dir / 'decision_tree.png', dpi=300, bbox_inches='tight')
    plt.close()

# 4. Logistic Regression Visualization
def visualize_logistic():
    x = np.linspace(-5, 5, 100)
    y = 1 / (1 + np.exp(-x))
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='blue', linewidth=2, label='Sigmoid Function')
    plt.axhline(0.5, color='red', linestyle='--', label='Decision Boundary')
    plt.xlabel('Linear Combination (w·x + b)')
    plt.ylabel('Probability')
    plt.title('Logistic Regression: Sigmoid Function')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'logistic_regression.png', dpi=300, bbox_inches='tight')
    plt.close()

# 5. SVM Visualization
def visualize_svm():
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(20, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[y==0, 0], X[y==0, 1], color='blue', label='Class 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], color='red', label='Class 1')

    # Decision boundary (simplified)
    x1 = np.linspace(-3, 3, 100)
    x2 = -x1  # w·x + b = 0, assuming w=[1,1], b=0
    plt.plot(x1, x2, color='black', linestyle='-', label='Decision Boundary')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'svm.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    visualize_nn()
    visualize_regression()
    visualize_tree()
    visualize_logistic()
    visualize_svm()
    print(f"Visualizations saved to {output_dir}")
