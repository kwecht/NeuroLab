NeuroLab
========

This repository contains my modifications to the Neurolab neural network library for python (https://pypi.python.org/pypi/neurolab).

Modifications as of 21 December 2014: 
1. included support for regularization terms of network weights and biases for feed forward networks that make use of ff_grad.
2. added the cross-entropy error function (used for logistic regression) to the available error functions in error.py.

These changes have required some modification of the Neurolab source code. For example, error functions and their derivatives now require 3 arguments, not 1. I also introduced functions in error.py to initialize the regularization paramters and calculate the norms of network weights and biases.

Incorporation into Neurolab library as of 21 December 2014:
I have contacted the developers of Neurolab and hope that these modifications can be incorporated into the standard distribution of the Neurolab library.
