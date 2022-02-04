
import numpy as np


def ridge(w,xTr,yTr,lambdaa):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);

    # YOUR CODE HERE
    loss = np.dot(np.dot(w.T, xTr)-yTr, (np.dot(w.T, xTr)-yTr).T) + lambdaa*np.dot(w.T, w)
    gradient = 2*np.dot(np.dot(xTr, xTr.T), w) - 2*np.dot(xTr, yTr.T) + 2*lambdaa*w

    return loss, gradient
