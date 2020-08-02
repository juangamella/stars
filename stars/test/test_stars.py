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

#---------------------------------------------------------------------
# Unit testing template module

import unittest
import stars
import numpy as np

class StarTests(unittest.TestCase):

    def test_subsample_1(self):
        n = 100
        p = 10
        X = np.random.uniform(size=(n, p, p))
        for N in [1, 2, 5, 10]:
            S = stars.subsample(X, N)
            # Check that all there are the right number of subsamples
            self.assertEqual(len(S), N)            
            # Check that the observations in each subsample are not
            # repeated in other subsamples (i.e. that there is no
            # replacement)
            for i,s1 in enumerate(S):
                for j,s2 in enumerate(S):
                    if j != i:
                        self.assertTrue(disjoint_rows(s1, s2))
            # Check that all subsamples are the appropriate length and
            # have the appropriate shape
            for s in S:
                self.assertEqual(n / N, len(s))
                self.assertEqual(s.shape, (n/N,p,p))

    def test_subsample_2(self):
        # Check that an exception is thrown when the number of requested
        # subsamples is not valid
        n = 100
        p = 10
        X = np.random.uniform(size=(n, p, p))
        for N in [0, 3, 6, 11, 21]:
            try:
                stars.subsample(X, N)
                fails = False
            except:
                fails = True
            finally:
                self.assertTrue(fails)

    def test_optimize(self):
        thresh =  -0.75
        f = lambda x: x**0.5 - 2**x
        # Check execution with and without debug options
        stars.optimize(f,
                       thresh,
                       0,
                       0.05,
                       10,
                       debug=True)
        opt, val = stars.optimize(f,
                                  thresh,
                                  0,
                                  0.05,
                                  10,
                                  debug=False)
        # Check that the approximation does not get worse with the
        # number of iterations, and that it is in fact a supremum
        prev = -np.infty
        for max_iter in [5, 10, 20, 30]:
            opt, val = stars.optimize(f,
                                      thresh,
                                      0,
                                      0.05,
                                      max_iter,
                                      tol = 1e-6,
                                      debug=False)
            self.assertLessEqual(val, thresh)
            self.assertLessEqual(f(opt), thresh)
            self.assertLessEqual(prev, val)
            self.assertLessEqual(prev, f(opt))
            prev = val

    def test_glasso(self):
        # Test that the Graphical Lasso wrapper actually returns
        # an adjacency matrix (only ones, zeros and zeros on the
        # diagonal), and the correct number of adjacency matrices
        true_cov = np.array([[0.8, 0.0, 0.2, 0.0],
                             [0.0, 0.4, 0.0, 0.0],
                             [0.2, 0.0, 0.3, 0.1],
                             [0.0, 0.0, 0.1, 0.7]])
        np.random.seed(0)
        X = np.random.multivariate_normal(mean=[0, 1, 2, 3],
                                          cov=true_cov,
                                          size=400)
        S = stars.subsample(X, 4)
        estimates = stars.glasso(S, 0.01)
        self.assertEqual(len(S), len(estimates))
        for estimate in estimates:
            self.assertTrue(np.sum(np.logical_and(estimate != 1, estimate != 0)) == 0)
            self.assertTrue((np.diag(estimate) == 0).all())
            self.assertEqual((4,4), estimate.shape)

    def test_instability_1(self):
        # Test the computation of the instability estimate
        # Check that, if all estimates are the same, instability should be zero
        N = 10
        estimates = np.array([np.array([[0,1,1,1],
                                        [0,0,1,1],
                                        [0,0,0,1],
                                        [0,0,0,0]])] * N)
        estimator = lambda subsamples, alpha: estimates
        X = np.random.uniform(size=(100, 4, 4))
        S = stars.subsample(X, N)
        instability = stars.estimate_instability(S, estimator, 0.01)
        self.assertEqual(instability, 0)
        instability, returned_estimates = stars.estimate_instability(S, estimator, 0.01, return_estimates = True)
        self.assertEqual(instability, 0)
        for estimate in returned_estimates:
            self.assertEqual((4,4), estimate.shape)
            self.assertTrue(np.sum(np.logical_and(estimate != 1, estimate != 0)) == 0)
            self.assertTrue((np.diag(estimate) == 0).all())

    def test_instability_2(self):
        # Test the computation of the instability estimate Check that,
        # if all half of the estimates have no edge in common with the
        # other half, instability should be 1/2
        N = 10
        estimate_a = np.array([[0,1,0,0],
                                 [1,0,1,0],
                                 [0,1,0,1],
                                 [0,0,1,0]])
        estimate_b = np.array([[0,0,1,1],
                                 [0,0,0,1],
                                 [1,0,0,0],
                                 [1,1,0,0]])
        estimates = np.array([estimate_a] * int(N / 2) + [estimate_b] * int(N / 2))
        assert(len(estimates) == N)
        estimator = lambda subsamples, alpha: estimates
        X = np.random.uniform(size=(100, 4, 4))
        S = stars.subsample(X, N)
        instability = stars.estimate_instability(S, estimator, 0.01)
        self.assertEqual(instability, 0.5)
        instability, returned_estimates = stars.estimate_instability(S, estimator, 0.01, return_estimates=True)
        self.assertEqual(instability, 0.5)
        for estimate in returned_estimates:
            self.assertEqual((4,4), estimate.shape)
            self.assertTrue(np.sum(np.logical_and(estimate != 1, estimate != 0)) == 0)
            self.assertTrue((np.diag(estimate) == 0).all())

    def test_comb(self):
        cases = np.array([
            [34, 30, 46376.0],
            [37, 37, 1.0],
            [47, 43, 178365.0],
            [49, 1, 49.0],
            [42, 32, 1471442973.0],
            [30, 26, 27405.0],
            [23, 20, 1771.0],
            [47, 7, 62891499.0],
            [46, 26, 5608233007145.998],
            [13, 4, 715.0],
            [24, 4, 10626.0],
            [13, 2, 78.0],
            [15, 6, 5005.0],
            [36, 35, 36.0],
            [41, 21, 269128937220.00006],
            [23, 3, 1771.0],
            [38, 7, 12620256.0],
            [48, 19, 11541847896480.004],
            [5, 1, 5.0],
            [42, 31, 4280561376.0],
            [28, 25, 3276.0],
            [49, 0, 1.0],
            [18, 10, 43758.0],
            [43, 38, 962598.0],
            [44, 22, 2104098963720.0],
            [39, 2, 741.0],
            [11, 5, 462.0],
            [40, 35, 658008.0],
            [45, 29, 646626422970.0001],
            [21, 18, 1330.0],
            [16, 12, 1820.0],
            [28, 25, 3276.0],
            [31, 0, 1.0],
            [29, 12, 51895935.0],
            [32, 9, 28048800.0],
            [33, 22, 193536720.0],
            [16, 6, 8008.0],
            [17, 9, 24310.0],
            [34, 3, 5984.0],
            [45, 44, 45.0],
            [19, 17, 171.0],
            [15, 8, 6435.0],
            [41, 39, 820.0],
            [20, 11, 167960.0],
            [33, 27, 1107568.0],
            [14, 8, 3003.0],
            [27, 24, 2925.0],
            [36, 10, 254186856.0],
            [48, 46, 1128.0],
            [40, 14, 23206929840.000004]])
        for (n,k,truth) in cases:
            self.assertTrue(np.isclose(truth, stars.comb(n,k)))

        
def disjoint_rows(A, B):
    """Check that two arrays have disjoint rows"""
    for a in A:
        if a in B:
            return False
    return True
