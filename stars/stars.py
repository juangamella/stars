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
      - X (np.array): Array containing n observations of p
          variables. Columns are the observations of a single variable
      - estimator (function): Wrapper function for your estimator, as
          described below.
      - beta (float, optional): Maximum allowed instability between
          subsample estimates. Defaults to 0.05, the value recommended in the
          paper.
      - N (int, optional): Number of subsamples, must be divisor of
          n. Defaults to the value recommended in the paper,
          i.e. approximately int(n / np.floor(10 np.sqrt(n))).
      - start (float, optional): Starting lambda in the search
          procedure. Defaults to 1.
      - step (float, optional): Initial step at which to increase
          lambda. Defaults to 1.
      - tol (float, optional): Tolerance of the search procedure,
          i.e. the search procedure stops when the instability at a given
          lambda is below tol of beta. Defaults to 1e-5.
      - max_iter (int, optional): Maximum number of iterations for which
          the search procedure is run, i.e. the maximum number of times
          the estimator is run. Defaults to 20.
      - debug (bool, optional): If debugging messages should be printed
          during execution. Defaults to False.

    Returns:
      - estimate (np.array): The adjacency matrix of the resulting
          graph estimate.

    Estimator function: it must take two arguments:

    - subsamples (np.array): An array containing the subsampled data,
        of dimension Nxbxp, where N is the number of subsamples,
        b=n/N and p is the number of variables.
    - lambda (float): The regularization value at which to run the estimator.
    
    and it must return a Nxpxp np.array containing the adjacency
    matrix (0s or 1s) of the estimate for each subsample.
    """
    (n,p) = X.shape
    # Standardize the data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Select the number of subsamples
    if N is None:
        N = find_N(n)
    subsamples = subsample(X, N) # Subsample the data
    # Solve the supremum alpha as in
    # (https://arxiv.org/pdf/1006.3316.pdf, page 6)
    # Set up functions for the search procedure
    search_fun = lambda lmbda: estimate_instability(subsamples, estimator, lmbda)
    opt = find_supremum(search_fun, beta, start, step, max_iter, tol, debug)
    # Fit over all data using the optimal regularization parameter
    return estimator(subsample(X, 1), opt)[0]

def subsample(X, N):
    """Given n observations of p variables X, return N subsamples.

    Parameters:
      - X (np.array): Observations. Columns correspond to variables.
      - N (int): Number of subsamples. Must be a divisor of n.

    Returns:
      - Subsamples (np.array): Array containing the subsampled data,
        of dimension Nxnxp.
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
      - subsamples (np.array): the subsample array
      - estimator (function): the estimator to be used. See
        documentation for stars.fit for more info.
      - lmbda (float): the regularization parameter at which to run
        the estimator
      - return_estimates (bool, optional): If the estimate for each
        subsample should also be returned. Defaults to False.

    Returns:
      - instability (float): The estimated total instability

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

    Parameters:

      - fun (function): f:X->R. The function for which we perform the search
      - thresh (float): The given threshold
      - start (value in X): Initial value for x
      - step (value in X): Initial step at which to increase x
      - max_iter (int, optional): Maximum number of iterations of the procedure
      - tol (float, optional): Tolerance, i.e. if difference between
        thresh and current fun(x) is below tol, stop the procedure
      - debug (bool, optional): If True, print debug messages

    Returns:
      - x (value in X): the approximated supremum \sup_x \{fun(x) \leq thresh\}
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
    repetition and without order.

    """
    return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

def neighbourhood_graph(p, max_nonzero=2, rho=0.245):
    """Generate a "neighborhood graph" o p variables as described in page
    10 of the paper (https://arxiv.org/pdf/1006.3316.pdf). Return its
    precision matrix.

    """
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

def find_N(n):
    """Find the value of N closest to the value specified in the paper, i.e.
          n / np.floor(10 * np.sqrt(n))
    which is a divisor of n"""
    N_lo = N_hi = int(n / np.floor(10 * np.sqrt(n)))
    while (n % N_lo) != 0 and (n % N_hi) != 0:
        N_lo -= 1
        N_hi += 1
    N = N_lo if (n % N_lo == 0) else N_hi
    return N
