import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GMM:
    def __init__(self, n_components, n_iter=100, tol=1e-6):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol

    def fit(self, X):
        self.n_samples, self.n_features = X.shape
        self.weights = np.full(self.n_components, 1 / self.n_components)
        self.means = X[np.random.choice(self.n_samples, self.n_components, False)]
        self.covariances = np.array([np.cov(X.T) for _ in range(self.n_components)])

        log_likelihoods = []

        for _ in range(self.n_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)
            log_likelihood = self._log_likelihood(X)
            log_likelihoods.append(log_likelihood)

            if len(log_likelihoods) > 1 and np.abs(log_likelihood - log_likelihoods[-2]) < self.tol:
                break

        return self

    def _e_step(self, X):
        likelihood = np.array([multivariate_normal.pdf(X, mean=self.means[k], cov=self.covariances[k]) for k in range(self.n_components)]).T
        weighted_likelihood = likelihood * self.weights
        responsibilities = weighted_likelihood / weighted_likelihood.sum(axis=1)[:, np.newaxis]
        return responsibilities

    def _m_step(self, X, responsibilities):
        Nk = responsibilities.sum(axis=0)
        self.weights = Nk / self.n_samples
        self.means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
        self.covariances = np.array([
            np.dot((responsibilities[:, k] * (X - self.means[k]).T), (X - self.means[k])) / Nk[k]
            for k in range(self.n_components)
        ])

    def _log_likelihood(self, X):
        likelihood = np.array([multivariate_normal.pdf(X, mean=self.means[k], cov=self.covariances[k]) for k in range(self.n_components)]).T
        weighted_likelihood = likelihood * self.weights
        log_likelihood = np.log(weighted_likelihood.sum(axis=1)).sum()
        return log_likelihood

    def predict(self, X):
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    X1 = np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], 100)
    X2 = np.random.multivariate_normal([3, 3], [[1, -0.8], [-0.8, 1]], 100)
    X = np.vstack([X1, X2])

    gmm = GMM(n_components=2)
    gmm.fit(X)
    y_pred = gmm.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=40, cmap='viridis')
    plt.scatter(gmm.means[:, 0], gmm.means[:, 1], c='red', s=100, marker='x')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Gaussian Mixture Model Clustering")
    plt.show()
