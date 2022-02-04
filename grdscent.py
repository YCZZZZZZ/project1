
import numpy as np
def grdescent(func,w0,stepsize,maxiter,tolerance=1e-02):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14 #minimum step size for gradient descent

    # YOUR CODE HERE

    w = w0
    loss = 100000
    for k in range(maxiter):
        lossold = loss
        loss, gradient = func(w)
        dk = -1 * gradient
        if np.linalg.norm(dk) < tolerance:
            break
        else:
            pass
        w = w + stepsize * dk

        if loss < lossold:
            stepsize = 1.01 * stepsize
        else:
            stepsize = 0.5 * stepsize

    return w
