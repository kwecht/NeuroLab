# Test regularization implementation in neuralab.

import neurolab as nl
import numpy as np
import matplotlib.pyplot as plt



# Train the a network using regularization.
# Plot results of the training using different values
#     of the regularization parameter.


# ---- Create training and test data ----
np.random.seed(2)   # Set random seed to reproduce results
nobs = 60
train = np.random.uniform(-10,10,(nobs,2))
test = np.random.uniform(-10,10,(nobs,2))

# normalize to 0 mean, 1 standard deviation
train = (train - np.mean(train)) / np.std(train)   
test = (test - np.mean(test)) / np.std(test)

# Calculate target values for the train and test data. Sign of addition!
target = (train[:,0] + train[:,1])  # Addition of two input nodes
target = np.where( target < 0., -1, 1 ).reshape(nobs,1)   # Round to 1 or -1
testtarget = (test[:,0] + test[:,1])  # Addition of two input nodes
testtarget = np.where( testtarget < 0., -1, 1 ).reshape(nobs,1)   # Round to 1 or -1



# ---- Loop through different values of the regularization parameter, ----
#      recording final train error and test error for each simulation
powers = np.linspace(-10,2,10)  
results = {'trainerror':[], 'testerror':[], 'regparam':[], 'weights':[]}
for power in powers:

    # Set random seed to reproduce results
    np.random.seed(22)

    # Create neural network to train. 
    net = nl.net.newff([[np.min(train),np.max(train)]]*2,[nobs,1])

    # Set regularization parameter
    reg_param = 10**power
    net.errorf = nl.error.MSE(reg_param_w=reg_param)

    # Train network
    err = net.train(train,target,goal=1e-7)

    # Record results for plotting
    results['regparam'].append(reg_param)
    ws = []
    for layer in net.layers:
        ws.extend(layer.np['w'].reshape(layer.np['w'].size))
    results['weights'].append(np.mean(np.abs(ws))/10.)
    testsim = net.sim(test)
    trainsim = net.sim(train)
    errorf = nl.error.MSE(reg_param_w=0.)
    results['trainerror'].append(errorf(target,trainsim,net))
    results['testerror'].append(errorf(testtarget,testsim,net))


# ---- Visualize test vs. training error ----
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
l1 = plt.plot(results['regparam'],results['trainerror'],'-r',
              lw=3,label='Train Error')
l2 = plt.plot(results['regparam'],results['testerror'],'-b',
              lw=3,label='Test Error')
l3 = plt.plot(results['regparam'],results['weights'],'-g',
              lw=2,label='Magnitude of Weights / 10')
ax.set_xscale('log')
ax.set_ylim(-0.1,1.0)
plt.xlabel(r'Regularization Parameter  $\lambda$',fontsize=16)
plt.ylabel('Error',fontsize=16)
legend = plt.legend()
plt.show()
