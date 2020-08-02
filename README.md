# Stability Approach to Regularization Selection (StARS) for High Dimensional Graphical Models

This repository contains an implementation of the StARS algorithm, from the paper "Stability Approach to Regularization Selection (StARS) for High Dimensional Graphical Models" by Liu, Kathryn Roeder, Larry Wasserman (https://arxiv.org/pdf/1006.3316.pdf).

## A note about requirements

The package is built so the minimum requirements are needed. The minimum needed is `numpy` and the standard library. However, there is also the option to run StARS directly with the [Graphical Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html) implemented in scikit-learn; in this case, you will also need to install `sklearn >= 0.20`.
