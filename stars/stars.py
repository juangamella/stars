# Copyright 2020 Juan Luis Gamella Martin

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import scipy
from sklearn.covariance import GraphicalLasso


#---------------------------------------------------------------------
#

def fit_w_glasso(X, beta, N, start=0, step=0.05, tol=1e-5, max_iter=10, debug=False):
    """Wrapper function to run StARS using the Graphical Lasso from
    scikit.learn. The estimator is run with the default parameters.

    Parameters:
      - X (n x p np.array): n observations of p variables
      - beta (float): probability that the returned estimate contains
        the true graph
      - N (int): number of subsamples, must be divisor of n
      - start (float): initial lambda
      - step (float): initial step at which to increase lambda
      - tol (float): tolerance of the search procedure
      - max_iter (int, default=1000): max number of iterations to run
        the search procedure, that is, max number of times the estimator
        is run

    Returns:
      - the selected regularization parameter
      - the subsample estimates at that regularization value

    """
    estimator = glasso
    return fit(X, beta, estimator, N, start, step, tol, max_iter, debug)

def fit(X, beta, estimator, N, start=0, step=0.05, tol=1e-5, max_iter=10, debug=False):
    """
    Run the StARS algorithm to select the regularization parameter for the given estimator.

    Parameters:
      - X (n x p np.array): n observations of p variables
      - beta (float): probability that the returned estimate contains
        the true graph
      - estimator (function): estimator to be used*
      - N (int): number of subsamples, must be divisor of n
      - start (float): initial lambda
      - step (float): initial step at which to increase lambda
      - tol (float): tolerance of the search procedure
      - max_iter (int, default=1000): max number of iterations to run
        the search procedure, that is, max number of times the estimator
        is run

    Returns:
      - the selected regularization parameter
      - the subsample estimates at that regularization value

    *Note on the estimator: It must be a function that takes as arguments:
      - subsamples (N x p x p np.array)
      - lambda (float): the regularization parameter
    and returns the adjacency matrices of the graph estimates for each subsample.
    """
    (n,p) = X.shape
    # Subsample the data
    subsamples = subsample(X, N)
    # Solve the supremum alpha as in
    # (https://arxiv.org/pdf/1006.3316.pdf, page 6)
    # Set up functions for the search procedure
    search_fun = lambda lmbda: estimate_instability(subsamples, estimator, lmbda)
    # Run the search procedure
    opt_lambda, estimates = optimize(search_fun, beta, start, step, max_iter, tol, debug)
    return opt_lambda, estimates

def subsample(X, N):
    """
    Given observations of p variables X, return N subsamples.

    Parameters:
      - X (np.array): observations
      - N (int): number of subsamples

    Returns:
      - Subsamples (N x n/N x p np.array)
    """    
    if len(X) % N != 0 or N == 0:
        raise Exception("The number of samples must be a multiple of N")
    b = int(len(X) / N)
    rng = np.random.default_rng()
    return rng.choice(X, axis=0, replace=False, size=(N,b))

def estimate_instability(subsamples, estimator, lmbda, return_estimates=False):
    """Estimate the instability using a set of subsamples, as in
    (https://arxiv.org/pdf/1006.3316.pdf, page 6)

    Parameters:
      - subsamples (N x b x p np.array): the subsample array
      - estimator (function): the estimator to be used
      - lmbda (float): the regularization parameter at which to run the estimator

    Returns:
      - (float) The estimated total instability
      - estimates (N x p x p): The adjacency matrix of the graph
        estimated for each subsample
    """
    estimates = estimator(subsamples, lmbda)
    p = subsamples.shape[2]
    edge_average = np.mean(estimates, axis=0)
    edge_instability = 2 * edge_average * (1-edge_average)
    # In the following, the division by 2 is to account for counting
    # every edge twice (as the estimate matrix is symmetric)
    total_instability = np.sum(edge_instability, axis=(0,1)) / scipy.special.binom(p,2) / 2
    if return_estimates:
        return total_instability, estimates
    else:
        return total_instability
    
def glasso(subsamples, alpha, precision_tolerance = 1e-3, mode='cd', return_precisions = False):
    """Run the graphical lasso from scikit learn over the given
    subsamples, at the given regularization level.

    Parameters:
      - subsamples (N x b x p np.array): the subsample array
      - alpha (float): the regularization parameter at which to run
        the estimator, taken as 1/lambda, i.e, lower values mean
        sparser

    Returns:
      - estimates (N x p x p): The adjacency matrix of the graph
        estimated for each subsample

    """
    (N,_,p) = subsamples.shape
    precisions = np.zeros((len(subsamples),p,p))
    g = GraphicalLasso(alpha = 1 / alpha,
                       #tol = tol,
                       #max_iter = max_iter,
                       mode = mode)
    for j,sample in enumerate(subsamples):
        precision = g.fit(sample).precision_
        precisions[j,:,:] = precision - np.diag(np.diag(precision))
    estimates = (abs(precisions) > precision_tolerance).astype(int)
    if return_precisions:
        return estimates, precisions
    else:
        return estimates

def optimize(fun, thresh, start, step, max_iter, tol = 1e-5, debug=False):
    """Given a function fun:X -> R and a (float) threshold thresh,
    approximate the supremum \sup_x \{fun(x) \leq thresh\}. Adapted
    version of the bisection method.

    Additional parameters:
      - start (value in X): initial value for x
      - step (value in X): initial step at which to increase x
      - max_iter (int): maximum number of iterations of the procedure
      - tol (float): tolerance, i.e. if difference between thresh and current fun(x) is below tol, stop the procedure
      - debug (Bool): if True, print debug messages

    Returns:
      - x (value in X): the approximated supremum
      - val (value in V): the value of the function at x

    """
    x, val = start, fun(start)
    if val > thresh:
        raise Exception("Invalid starting value")
    i = 0    
    while i < max_iter and thresh - val > tol:        
        next_val = fun(x + step)
        if next_val > thresh:
            print("  ",i,": f(", x+ step, ")=",val,">", thresh)
            step /= 2
        else:
            print("JUMP ",i,": f(", x+ step, ")=",val," - delta: ", thresh - val) if debug else None
            x += step
            val = next_val
        i += 1
    return x, val
