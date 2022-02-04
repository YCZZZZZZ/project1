import math
import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''
def logistic(w,xTr,yTr):

    # YOUR CODE HERE
    n, m = np.shape(xTr)
    loss = 0
    for i in range(m):
        loss = loss + np.log(1 + np.exp(-np.float(yTr[:, i]) * np.dot(w.T, xTr[:, i])))

    gradient = 0;
    a = np.zeros((len(w), 1))
    for i in range(m):
        gradient = gradient - yTr[:, i] * xTr[:, i] * (np.exp(-yTr[:, i] * np.dot(w.T, xTr[:, i]))) / (
                    1 + np.exp(-yTr[:, i] * np.dot(w.T, xTr[:, i]))) + a.T
    return loss, gradient.T
