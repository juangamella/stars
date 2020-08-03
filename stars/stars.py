# Copyright 2020 Juan L Gamella

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
import math
from sklearn.covariance import GraphicalLasso


def fit(X, estimator, beta=0.05, N=None, start=1, step=1, tol=1e-5, max_iter=20, debug=False):
    """Run the StARS algorithm to select the regularization parameter for the given estimator.

    Parameters:
      - X (n x p np.array): n observations of p variables
      - estimator (function): estimator to be used*
      - beta (float, optional): maximum allowed instability between subsample estimates
      - N (int, optional): number of subsamples, must be divisor of n. Defaults
        to the value recommended in the paper
        (https://arxiv.org/pdf/1006.3316.pdf, page 9): int(n / np.floor(10 * np.sqrt(n)))
      - start (float, optional): initial lambda
      - step (float, optional): initial step at which to increase lambda
      - tol (float, optional): tolerance of the search procedure
      - max_iter (int, optional): max number of iterations to run
        the search procedure, that is, max number of times the estimator
        is run
      - debug (bool, optiona): if debugging messages should be printed
        during execution

    Returns:
      - estimate (p x p np.array): the estimate at the regularization
        value selected by StARS

    *Note on the estimator: It must be a function that takes as arguments:
      - subsamples (N x p x p np.array)
      - lambda (float): the regularization parameter
    and returns the adjacency matrices of the graph estimates for each subsample.
    """
    (n,p) = X.shape
    # Standardize the data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Select the number of subsamples
    if N is None:
        N = int(n / np.floor(10 * np.sqrt(n)))
    subsamples = subsample(X, N) # Subsample the data
    # Solve the supremum alpha as in
    # (https://arxiv.org/pdf/1006.3316.pdf, page 6)
    # Set up functions for the search procedure
    search_fun = lambda lmbda: estimate_instability(subsamples, estimator, lmbda)
    opt = find_supremum(search_fun, beta, start, step, max_iter, tol, debug)
    # Fit over all data using the optimal regularization parameter
    return estimator(subsample(X, 1), opt)[0]

def subsample(X, N):
    """
    Given observations of p variables X, return N subsamples.

    Parameters:
      - X (np.array): observations
      - N (int): number of subsamples

    Returns:
      - Subsamples (N x n/N x p np.array)
    """
    # Check that N is appropriate
    if len(X) % N != 0 or N == 0:
        raise Exception("The number of samples must be a multiple of N")
    if N == 1:
        return np.array([X])
    b = int(len(X) / N)
    rng = np.random.default_rng()
    # Subsample without replacement
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
    total_instability = np.sum(edge_instability, axis=(0,1)) / comb(p, 2) / 2
    if return_estimates:
        return total_instability, estimates
    else:
        return total_instability
    
def find_supremum(fun, thresh, start, step, max_iter, tol = 1e-5, debug=False):
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
            print("  ",i,": f(", x+ step, ")=",val,">", thresh) if debug else None
            step /= 2
        else:
            print("JUMP ",i,": f(", x+ step, ")=",val," - delta: ", thresh - val) if debug else None
            x += step
            val = next_val
        i += 1
    return x

def comb(n,k):
    """Return the number of ways to choose k items from n items without
    repetition and without order."""
    return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

def neighbourhood_graph(p, max_nonzero=2, rho=0.245):
    """Generate a "neighborhood graph" o p variables as described in page 10 of the
    paper (https://arxiv.org/pdf/1006.3316.pdf). Return its precision matrix."""
    Y = np.random.uniform(size=(p,2))
    prob = np.zeros((p,p))
    precision = np.zeros((p,p))
    for i in range(p):
        for j in range(p):
            if i!=j:
                dist = (Y[i,:] - Y[j,:]) @ (Y[i,:] - Y[j,:]).T
                prob[i,j] = 1 / np.sqrt(2*np.pi) * np.exp(-16 * dist)
    for i in range(p):
        indices = np.argsort(prob[i,:])[-max_nonzero:]
        precision[i,indices] = rho
        precision[indices,i] = rho
    precision[range(p), range(p)] = 1
    return precision
