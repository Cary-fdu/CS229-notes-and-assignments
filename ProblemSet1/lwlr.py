import numpy as np
from scipy.linalg import inv
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Logistic sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute the weights for each training example based on the query point
def get_weights(x_query, X_train, tau):
    diff = X_train - x_query
    squared_distances = np.sum(diff ** 2, axis=1)
    weights = np.exp(-squared_distances / (2 * tau ** 2))
    return weights

# Compute the gradient of the objective function
def compute_gradient(X, y, weights, theta, lambda_):
    predictions = sigmoid(X @ theta)
    z = weights * (y - predictions)  # Element-wise multiplication by weights
    grad = X.T @ z - lambda_ * theta  # Gradient calculation
    return grad

# Compute the Hessian matrix
def compute_hessian(X, weights, theta, lambda_):
    predictions = sigmoid(X @ theta)
    D = np.diag(-weights * predictions * (1 - predictions))  # Diagonal matrix
    H = X.T @ D @ X - lambda_ * np.eye(X.shape[1])  # Hessian calculation
    return H

# Newton's method for optimization
def newton_method(X, y, x_query, tau, lambda_=0.0001, max_iter=10, tol=1e-6):
    m, n = X.shape
    X_with_bias = np.hstack([np.ones((m, 1)), X])  # Add bias term to X
    theta = np.zeros(n + 1)  # Initialize theta

    x_query_with_bias = np.insert(x_query, 0, 1.0)  # Make sure query also has bias

    for iteration in range(max_iter):
        # Calculate weights based on query point
        weights = get_weights(x_query_with_bias, X_with_bias, tau)

        # Compute gradient and Hessian
        grad = compute_gradient(X_with_bias, y, weights, theta, lambda_)
        H = compute_hessian(X_with_bias, weights, theta, lambda_)

        # Newton update
        theta_new = theta - inv(H) @ grad

        if np.linalg.norm(theta_new - theta) < tol:
            break
        theta = theta_new

    return theta


#implementation
X = np.loadtxt(r"D:\Users\elain\Desktop\cs229\q2\data\x.dat")
y = np.loadtxt(r"D:\Users\elain\Desktop\cs229\q2\data\y.dat")
# readers can revise these
def plot_lwlr_decision_boundary(X_train, y_train, taus, lambda_=0.0001):
    # create grid
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    for i, tau in enumerate(taus):
        predictions = np.zeros_like(xx, dtype=int)
        
        for j in range(xx.shape[0]):
            for k in range(xx.shape[1]):
                x_query = np.array([xx[j, k], yy[j, k]])
                theta = newton_method(X_train, y_train, x_query, tau, lambda_)
                x_query_with_bias = np.insert(x_query, 0, 1.0)
                prob = sigmoid(x_query_with_bias @ theta)
                predictions[j, k] = 1 if prob >= 0.5 else 0
        
        ax = axes[i]
        ax.contourf(xx, yy, predictions, alpha=0.3, cmap=ListedColormap(['blue', 'red']))
        ax.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], c='blue', label='y=0')
        ax.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], c='red', label='y=1')
        ax.set_title(f"τ = {tau}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend()

    plt.suptitle("Locally Weighted Logistic Regression: Decision Boundaries")
    plt.tight_layout()
    plt.show()


taus = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
plot_lwlr_decision_boundary(X, y, taus)