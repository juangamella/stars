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
import stars
import sempler
import matplotlib.pyplot as plt

#---------------------------------------------------------------------
#

p = 100
n = 10000

W = sempler.dag_avg_deg(p, 3, 0.1, 0.2)
sem = sempler.LGANM(W, (1,2))
true_cov = sem.sample(population=True).covariance
X = sem.sample(n)

alphas = np.linspace(0,1,10)
N = int(n / np.floor(10 * np.sqrt(n)))

print(N)

insts, estimates = stars.fit(X, 0.9, alphas, N, mode='cd', debug=True)

maximum, index = 0, 0
for i,instability in enumerate(insts):
    print(i,instability)
    if instability >= maximum:
        maximum, index = instability, i
    else:
        break

plt.subplot(131)
plt.imshow(estimates[index, 0, :, :])
plt.subplot(132)
plt.imshow(estimates[index, 1, :, :])
plt.subplot(133)
plt.imshow(sem.W + sem.W.T)
plt.show(block=False)

