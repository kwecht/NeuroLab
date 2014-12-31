#def test_cee():

"""
Script to test newly implemented cross-entropy error function.

Logistic regression and a neural network binary classifier with
one output node, a logistic-sigmoid transfer function, and no
hidden layers should give the same results.

Part 1 - Set up problem. Define input data and targets
Part 2 - Perform logistic regression
Part 3 - Train neural net with CEE
Part 4 - Compare results
"""

import neurolab as nl
import numpy as np
import pandas as pd
import statsmodels.api as sm


# Part 1 - Define input data and targets
#    5 input units, 10 data points
nobs = 50
dim  = 5
np.random.seed(1)
train = np.random.uniform(-1,1,size=dim*nobs).reshape(nobs,dim)
target = 0.5*train[:,0]+0.5*train[:,1]+np.random.normal(loc=0,scale=10,size=nobs)
target = np.where(target<0,0,1).reshape(nobs,1)

# Part 2 - Perform logistic regression
y = pd.Series(target.reshape(nobs))
X = pd.DataFrame(train)
X = sm.add_constant(X,prepend=True)
results_regress = sm.Logit(y,X).fit(maxiter=1000,method='bfgs')

# Part 3a - Train neural network
np.random.seed(1)
net1 = nl.net.newff([[np.min(train),np.max(train)]]*dim,[1])
net1.layers[0].transf = nl.trans.LogSig()
net1.errorf = nl.error.CEE()
net1.trainf = nl.train.train_delta
print 'Training the network to fine precision. This takes a minute ...'
err1 = net1.train(train,target,show=1000,goal=1e-10,epochs=10000,lr=0.0008)

# Part 4 - Compare results
print 'Logistic Regression'
for ii in range(dim+1):
    print '   Parameter {0} = {1}'.format(results_regress.params.index[ii],
                                          results_regress.params.values[ii])
print ''
print 'Neural Networks CEE'
for ii in range(dim+1):
    if ii==0:
        print '   Parameter bias = {0}'.format(net1.layers[0].np['b'][0])
    else:
        print '   Parameter {0} = {1}'.format(ii-1,net1.layers[0].np['w'][0][ii-1])

