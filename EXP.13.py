import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal

# Generate synthetic dataset
np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=1.0)

# Function to initialize the parameters
def initialize_parameters(X, n_clusters):
    n_samples, n_features = X.shape
    means = X[np.random.choice(n_samples, n_clusters, replace=False)]
    covariances = [np.eye(n_features)] * n_clusters
    weights = np.ones(n_clusters) / n_clusters
    return means, covariances, weights

# E-Step: Compute responsibilities
def e_step(X, means, covariances, weights):
    n_samples, n_clusters = X.shape[0], len(means)
    responsibilities = np.zeros((n_samples, n_clusters))
    for k in range(n_clusters):
        responsibilities[:, k] = weights[k] * multivariate_normal.pdf(X, means[k], covariances[k])
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities

# M-Step: Update parameters
def m_step(X, responsibilities):
    n_samples, n_features = X.shape
    n_clusters = responsibilities.shape[1]
    N_k = responsibilities.sum(axis=0)
    weights = N_k / n_samples
    means = np.dot(responsibilities.T, X) / N_k[:, np.newaxis]
    covariances = []
    for k in range(n_clusters):
        diff = X - means[k]
        covariances.append(np.dot(responsibilities[:, k] * diff.T, diff) / N_k[k])
    return means, covariances, weights

# Log likelihood
def log_likelihood(X, means, covariances, weights):
    n_samples = X.shape[0]
    log_likelihood = 0
    for k in range(len(means)):
        log_likelihood += weights[k] * multivariate_normal.pdf(X, means[k], covariances[k])
    return np.sum(np.log(log_likelihood))

# EM Algorithm
def em_algorithm(X, n_clusters, n_iterations=100, tol=1e-4):
    means, covariances, weights = initialize_parameters(X, n_clusters)
    log_likelihoods = []
    for _ in range(n_iterations):
        responsibilities = e_step(X, means, covariances, weights)
        means, covariances, weights = m_step(X, responsibilities)
        log_likelihoods.append(log_likelihood(X, means, covariances, weights))
        if len(log_likelihoods) > 1 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break
    return means, covariances, weights, responsibilities, log_likelihoods

# Parameters
n_clusters = 3
n_iterations = 100

# Run the EM algorithm
means, covariances, weights, responsibilities, log_likelihoods = em_algorithm(X, n_clusters, n_iterations)

# Plotting the results
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=responsibilities.argmax(axis=1), cmap='viridis', marker='o')
plt.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=100)
plt.title('EM Algorithm - Gaussian Mixture Model')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
