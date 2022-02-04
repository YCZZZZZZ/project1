from numpy import maximum
import numpy as np


def hinge(w,xTr,yTr,lambdaa):
#
#
# INPUT:
# xTr dxn matrix (each column is an input vector)
# yTr 1xn matrix (each entry is a label)
# lambda: regularization constant
# w weight vector (default w=0)
#
# OUTPUTS:
#
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w


    # YOUR CODE HERE
    n, m = np.shape(xTr)
    loss = 0
    for i in range(0, m):
        loss = loss + np.max(1-yTr[:, i]*np.dot(w.T, xTr[:, i]), 0)
    loss = loss + lambdaa*np.square(np.linalg.norm(w))

    gradient = 0
    for i in range(0, m):
        if np.max(1-yTr[:, i]*np.dot(w.T, xTr[:, i]), 0) == 0:
            gradient = gradient+0
        else:
            gradient = gradient - yTr[:, i]*xTr[:, i]

    gradient = gradient + 2*lambdaa*w.T
    return loss,gradient.T
