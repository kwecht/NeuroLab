# -*- coding: utf-8 -*-
""" Train error functions with derivatives

    :Example:
        >>> msef = MSE()
        >>> x = np.array([[1.0, 0.0], [2.0, 0.0]])
        >>> msef(x)
        1.25
        >>> # calc derivative:
        >>> msef.deriv(x[0])
        array([ 1.,  0.])

"""

import numpy as np



def set_regparams(kwargs):
    """
    Set regularization parameters from keyword arguments

    """

    # Initialize regularization parameter values
    w,b = 0,0

    # Get new regularization parameters from keywords
    temp_kwargs = kwargs.copy()
    for name,value in temp_kwargs.iteritems():
        w = kwargs.pop("reg_param_w",0)
        b = kwargs.pop("reg_param_b",0)
        if kwargs:
            raise TypeError("Unknown keyword arguments to error function")

    # Return regularization parameters
    return (w,b)



def param_norms(net,ord=2):
    """
    Calculate norm of weights and and biases for calculating
    the regularization term.

    :Parameters:
        net: neurolab net object

    :Keywords:
        ord: int
            order of norm for regularization term. Usually in {1,2}

    """

    # Assemble weights and biases into 1D vectors
    w = []
    b = []
    for layer in net.layers:
        w.extend(layer.np['w'].reshape(layer.np['w'].size))
        b.extend(layer.np['b'].reshape(layer.np['b'].size))

    # Calculate norms 
    w = np.linalg.norm(w,ord=ord)
    b = np.linalg.norm(b,ord=ord)

    return (w,b)


class MSE:
    """
    Mean squared error function

    :Parameters:
        target: ndarray
            target values for network
        output: ndarray
            simulated output of network
        net: neurolab net object

    :Keywords:
        reg_param_w: float
            Magnitude of regularization parameter for network weights
        reg_param_b: float
            Magnitude of regularization parameter for network biases

    :Returns:
        v: float
            Error value
    :Example:
        >>> f = MSE()
        >>> tar = np.array([1,2,3])
        >>> out = np.array([0,1,2])
        >>> x = f(tar,out,net)
        1.0

        Train network with regularization
        >>> net = net.newff([[np.min(train_data),np.max(train_data)]]*2,[nobs,1])
        >>> net.errorf = error.MSE(reg_param_w=10e-5)
        >>> err = net.train(train,target)

    """

    # Set regularization parameters upon initialization
    def __init__(self,**kwargs):
        (self.reg_param_w,self.reg_param_b) = set_regparams(kwargs)

    def __call__(self, target, output, net):
        # Objective term in cost function
        N = target.size
        v = np.sum(np.square(target-output)) / N

        # Regularization term
        (w,b) = param_norms(net)

        # Add terms in cost function
        v = v + self.reg_param_w * w + self.reg_param_b * b
        
        return v

    def deriv(self,target, output):
        """
        Derivative of MSE error function

        :Parameters:
            e: ndarray
                current errors: target - output
        :Returns:
            d: ndarray
                Derivative: dE/d_out
        :Example:
            >>> f = MSE()
            >>> x = np.array([1.0, 0.0])
            >>> # calc derivative:
            >>> f.deriv(x)
            array([ 1.,  0.])

        """
        e = target - output
        N = float(len(e))
        d = -1. * e * (2 / N)

        return d


class SSE:
    """
    Sum squared error function

    :Parameters:
        target: ndarray
            target values for network
        output: ndarray
            simulated output of network
        net: neurolab net object

    :Keywords:
        reg_param_w: float
            Magnitude of regularization parameter for network weights
        reg_param_b: float
            Magnitude of regularization parameter for network biases

    :Returns:
        v: float
            Error value  (0.5 * sum of squared errors)

    :Example:
        >>> f = SSE()
        >>> tar = np.array([1,2,3])
        >>> out = np.array([0,1,2])
        >>> x = f(tar,out,net)
        1.5

        Train network with regularization
        >>> net = net.newff([[np.min(train_data),np.max(train_data)]]*2,[nobs,1])
        >>> net.errorf = error.SSE(reg_param_w=10e-5)
        >>> err = net.train(train,target)


    """

    # Set regularization parameters upon initialization
    def __init__(self,**kwargs):
        (self.reg_param_w,self.reg_param_b) = set_regparams(kwargs)

    def __call__(self, target, output, net):
        v = 0.5 * np.sum(np.square(target-output))

        # Regularization term
        (w,b) = param_norms(net)

        # Add terms in cost function
        v = v + self.reg_param_w * w + self.reg_param_b * b

        return v

    def deriv(self,target, output):
        """
        Derivative of SSE error function

        :Parameters:
            e: ndarray
                current errors: target - output
        :Returns:
            d: ndarray
                Derivative: dE/d_out

        """
        e = target-output
        return -1.*e


class SAE:
    """
    Sum absolute error function

    :Parameters:
        target: ndarray
            target values for network
        output: ndarray
            simulated output of network
        net: neurolab net object

    :Keywords:
        reg_param_w: float
            Magnitude of regularization parameter for network weights
        reg_param_b: float
            Magnitude of regularization parameter for network biases

    :Returns:
        v: float
            Error value
    :Example:
        >>> f = SAE()
        >>> tar = np.array([1,2,3])
        >>> out = np.array([0,1,2])
        >>> x = f(tar,out,net)
        3.0

        Train network with regularization
        >>> net = net.newff([[np.min(train_data),np.max(train_data)]]*2,[nobs,1])
        >>> net.errorf = error.SAE(reg_param_w=10e-5)
        >>> err = net.train(train,target)

    """

    # Overwrite default regularization parameters upon initialization (optional)
    def __init__(self,**kwargs):
        (self.reg_param_w,self.reg_param_b) = set_regparams(kwargs)

    def __call__(self, target, output, net):
        v = np.sum(np.abs(target-output))

        # Regularization term
        (w,b) = param_norms(net)

        # Add terms in cost function
        v = v + self.reg_param_w * w + self.reg_param_b * b

        return v

    def deriv(self, target, output):
        """
        Derivative of SAE error function

        :Parameters:
            e: ndarray
                current errors: target - output
        :Returns:
            d: ndarray
                Derivative: dE/d_out

        """
        e = target-output
        d = -np.sign(e)
        return d


class MAE:
    """
    Mean absolute error function

    :Parameters:
        target: ndarray
            target values for network
        output: ndarray
            simulated output of network
        net: neurolab net object

    :Keywords:
        reg_param_w: float
            Magnitude of regularization parameter for network weights
        reg_param_b: float
            Magnitude of regularization parameter for network biases

    :Returns:
        v: float
            Error value
    :Example:
        >>> f = MAE()
        >>> tar = np.array([1,2,3])
        >>> out = np.array([0,1,2])
        >>> x = f(tar,out,net)
        1.0

        Train network with regularization
        >>> net = net.newff([[np.min(train_data),np.max(train_data)]]*2,[nobs,1])
        >>> net.errorf = error.MAE(reg_param_w=10e-5)
        >>> err = net.train(train,target)

    """

    # Set regularization parameters upon initialization
    def __init__(self,**kwargs):
        (self.reg_param_w,self.reg_param_b) = set_regparams(kwargs)

    def __call__(self, target, output, net):
        e = target-output
        v = np.sum(np.abs(e)) / e.size

        # Regularization term
        (w,b) = param_norms(net)

        # Add terms in cost function
        v = v + self.reg_param_w * w + self.reg_param_b * b
        
        return v

    def deriv(self, target, output):
        """
        Derivative of SAE error function

        :Parameters:
            e: ndarray
                current errors: target - output
        :Returns:
            d: ndarray
                Derivative: dE/d_out

        """
        e = target-output
        d = -np.sign(e) / e.size
        return d


class CEE:
    """
    Cross-entropy error function.
    For use when targets in {0,1}

    :Parameters:
        target: ndarray
            target values for network
        output: ndarray
            simulated output of network
        net: neurolab net object

    :Keywords:
        reg_param_w: float
            Magnitude of regularization parameter for network weights
        reg_param_b: float
            Magnitude of regularization parameter for network biases

    :Returns:
        v: float
            Error value
    :Example:
        >>> f = CEE()
        >>> tar = np.array([1,0,1])
        >>> out = np.array([0,1,1])
        >>> x = f(tar,out,net)
        1.0

    """

    # Set regularization parameters upon initialization
    def __init__(self,**kwargs):
        (self.reg_param_w,self.reg_param_b) = set_regparams(kwargs)

    def __call__(self, target, output, net):
        # Objective term in cost function
        N = target.size
        v = -1.*np.sum(target*np.log(output) + (1-target)*np.log(1-output)) / N

        # Regularization term
        (w,b) = param_norms(net)

        # Add terms in cost function
        v = v + self.reg_param_w * w + self.reg_param_b * b
        
        return v

    def deriv(self, target, output):
        """
        Derivative of CEE error function

        :Parameters:
            target: ndarray
                target values
            output: ndarray
                network predictions

        :Returns:
            d: ndarray
                Derivative: dE/d_out
        
        """

        N = target.size
        e = -1.*(target - output) / N
        return e
