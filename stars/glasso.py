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
    """Wrapper function to run StARS using the Graphical Lasso from
    scikit.learn. The estimator is run with the default parameters.

    Parameters:
      - X (n x p np.array): n observations of p variables
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
      - debug (bool, optional): if debugging messages should be printed
        during execution

    Returns:
      - estimate (p x p np.array): the GLASSO estimate at the
        regularization value selected by StARS

    """
    estimator = lambda subsamples, alpha: glasso(subsamples, alpha, precision_tol = precision_tol, glasso_params = glasso_params)
    return stars.fit(X, estimator, beta, N, start, step, tol, max_iter, debug)

def glasso(subsamples, alpha, precision_tol=1e-4, glasso_params = {}):
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
                       **glasso_params)
    for j,sample in enumerate(subsamples):
        precision = g.fit(sample).precision_
        precisions[j,:,:] = precision - np.diag(np.diag(precision))
    estimates = (abs(precisions) > precision_tol).astype(int)
    return estimates
