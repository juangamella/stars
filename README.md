# Stability Approach to Regularization Selection (StARS) for High Dimensional Graphical Models

This repository contains an implementation of the StARS algorithm, from the paper *"Stability Approach to Regularization Selection (StARS) for High Dimensional Graphical Models"* by Liu, Kathryn Roeder, Larry Wasserman (https://arxiv.org/pdf/1006.3316.pdf).

## Requirements

Minimum requirements (see [setup.py](setup.py)):

- Standard library
- `numpy`

To run with the [Graphical Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html) from Scikit-learn:
- Standard library
- `numpy`
- `sklearn>=0.20`

## Using the Graphical Lasso implementation from sklearn

The function `stars.glasso.fit` selects the regularization parameter via StARS, and then runs Scikit-learn's [Graphical Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html) the data.

Parameters:

- **X** (n x p np.array): n observations of p variables.
- **beta** (float, optional): maximum allowed instability between subsample estimates.
- **N** (int, optional): number of subsamples, must be divisor of n. Defaults to the value recommended in the paper, i.e. `int(n / np.floor(10 * np.sqrt(n)))`.
- **start** (float, optional): starting lambda in the search procedure. Defaults to 1.
- **step** (float, optional): initial step at which to increase lambda. Defaults to 1.
- **tol** (float, optional): tolerance of the search procedure, i.e. the search procedure stops when the instability at a given lambda is below `tol` of `beta`. Defaults to 1e-5.
- **max_iter** (int, optional): max number of iterations for which the search procedure is run, i.e. the max number of times the estimator is run. Defaults to 20.
- debug (bool, optional): if debugging messages should be printed during execution. Defaults to False.

Returns:

- **estimate** (p x p np.array): The adjacency matrix of the resulting graph estimate.

## Example: Using an estimator of your choice

Parameters:

<dt> X: <i>np.array</i> of size nxp</dt>
  <dd>n observations of p variables.</dd>
<dt> beta: <i>float</i>, optional</dt>
   <dd>maximum allowed instability between subsample estimates.</dd>
<dt> N: <i>int</i>, optional</dt>
  <dd>number of subsamples, must be divisor of n. Defaults to the value recommended in the paper, i.e. <code>int(n / np.floor(10 * np.sqrt(n)))</code>.</dd>
<dt> start: <i>float</i>, optional</dt>
  <dd>starting lambda in the search procedure. Defaults to 1.</dd>
<dt> step: <i>float</i>, optional</dt>
  <dd>initial step at which to increase lambda. Defaults to 1.</dd>
<dt> tol: <i>float</i>, optional</dt>
  <dd>tolerance of the search procedure, i.e. the search procedure stops when the instability at a given lambda is below <code>tol</code> of <code>beta</code>. Defaults to 1e-5.</dd>
<dt> max_iter: <i>int</i>, optional</dt>
  <dd>max number of iterations for which the search procedure is run, i.e. the max number of times the estimator is run. Defaults to 20.</dd>
<dt> debug  <i>bool</i>, optional</dt>
  <dd>if debugging messages should be printed during execution. Defaults to False.

Returns:
</dd>
<dt> estimate: <i>np.array</i> if size pxp</dt>
  <dd>The adjacency matrix of the resulting graph estimate.</dd>