import numpy as np
import stars

# Define a dummy estimator (returns the same estimate for all subsamples)
def estimator(subsamples, lmbda):
    p = subsamples.shape[2]
    A = np.triu(np.random.uniform(size=(p,p)), k=1)
    A += A.T
    A = A > 0.5
    return np.array([A] * len(subsamples))

# Generate data from a neighbourhood graph (page 10 of the paper)
true_precision = stars.neighbourhood_graph(100)
true_covariance = np.linalg.inv(true_precision)
X = np.random.multivariate_normal(np.zeros(100), true_covariance, size=400)

# Run StARS + Graphical lasso
stars.fit(X, estimator)
