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

<dl>
  <dt>X: (n x p np.array)</dt>
  <dd>n observations of p variables</dd>

  <dt>Markdown in HTML</dt>
  <dd>Does *not* work **very** well. Use HTML <em>tags</em>.</dd>
</dl>

- **X: (n x p np.array)**:
 
   n observations of p variables

- beta (float, optional): maximum allowed instability between subsample estimates
- N (int, optional): number of subsamples, must be divisor of n. Defaults to the value recommended in the paper (https://arxiv.org/pdf/1006.3316.pdf, page 9): int(n / np.floor(10 * np.sqrt(n)))
- start (float, optional): initial lambda
- step (float, optional): initial step at which to increase lambda
- tol (float, optional): tolerance of the search procedure
- max_iter (int, optional): max number of iterations to run the search procedure, that is, max number of times the estimator is run
- debug (bool, optional): if debugging messages should be printed during execution

Returns:

## Example: Using an estimator of your choice

