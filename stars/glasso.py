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
import stars
from sklearn.covariance import GraphicalLasso

def fit(X, beta=0.05, N=None, start=1, step=1, tol=1e-5, precision_tol = 1e-4, max_iter=20, glasso_params={}, debug=False):
    """Function to run StARS using the Graphical Lasso from
    scikit.learn.

    Parameters:
      - X (np.array): Array containing n observations of p
          variables. Columns are the observations of a single variable
      - beta (float, optional): Maximum allowed instability between
          subsample estimates. Defaults to 0.05, the value recommended in the
          paper.
      - N (int, optional): Number of subsamples, must be divisor of
          n. Defaults to the value recommended in the paper,
          i.e. int(n / np.floor(10 np.sqrt(n))).
      - start (float, optional): Starting lambda in the search
          procedure. Defaults to 1.
      - step (float, optional): Initial step at which to increase
          lambda. Defaults to 1.
      - tol (float, optional): Tolerance of the search procedure,
          i.e. the search procedure stops when the instability at a given
          lambda is below `tol` of `beta`. Defaults to 1e-5.
      - precision_tol (float, optional): Cutoff value at which nonzero
          elements of the precision matrix returned by GLASSO are
          considered edges in the graph. Defaults to 1e-4.
      - max_iter (int, optional): Maximum number of iterations for which
          the search procedure is run, i.e. the maximum number of times
          the estimator is run. Defaults to 20.
      - glasso_params (dict, optional): Dictionary used to pass
          additional parameters to sklearn.covariance.GraphicalLasso. Defaults to `{}`.
      - debug (bool, optional): If debugging messages should be printed
          during execution. Defaults to `False`.

    Returns:
      - estimate (np.array): The adjacency matrix of the resulting
          graph estimate.

    """
    estimator = lambda subsamples, alpha: glasso(subsamples, alpha, precision_tol = precision_tol, glasso_params = glasso_params)
    return stars.fit(X, estimator, beta, N, start, step, tol, max_iter, debug)

def glasso(subsamples, alpha, precision_tol=1e-4, glasso_params = {}):
    """Run the graphical lasso from scikit learn over the given
    subsamples, at the given regularization level.

    Parameters:
      - subsamples (np.array): the subsample array
      - alpha (float): the regularization parameter at which to run
        the estimator, taken as 1/lambda, i.e, lower values mean
        sparser

    Returns:
      - estimates (np.array): The adjacency matrices of the graphs
        estimated for each subsample
    """
    (N,_,p) = subsamples.shape
    precisions = np.zeros((len(subsamples),p,p))
    g = GraphicalLasso(alpha = 1 / alpha,
                       **glasso_params)
    for j,sample in enumerate(subsamples):
        precision = g.fit(sample).precision_
        precisions[j,:,:] = precision - np.diag(np.diag(precision))
    estimates = (abs(precisions) > precision_tol).astype(int)
    return estimates
