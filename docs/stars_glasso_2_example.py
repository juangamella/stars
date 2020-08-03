import numpy as np
import stars, stars.glasso

# Generate data from a neighbourhood graph (page 10 of the paper)
true_precision = stars.neighbourhood_graph(100)
true_covariance = np.linalg.inv(true_precision)
X = np.random.multivariate_normal(np.zeros(100), true_covariance, size=400)

# Set additional parameters for the Graphical Lasso estimator
args = {'max_iter': 100, 'mode': 'lars'}

# Run StARS + Graphical lasso
estimate = stars.glasso.fit(X, glasso_params = args)
