import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def kernel(point, xmat, k):  # gives the value of W
    m, n = np.shape(xmat)
    weights = np.mat(np.eye((m)))  # identity matrix
    for j in range(m):
        diff = point - X[j]
        weights[j, j] = np.exp(diff*diff.T/(-2.0*k**2))
    return weights


def localWeight(point, xmat, ymat, k):  # beta = (X^T W X)^-1 X^T W y
    wei = kernel(point, xmat, k)
    W = (X.T*(wei*X)).I*(X.T*(wei*ymat.T))  # W = beta
    return W


def localweightregression(xmat, ymat, k):  # (bill_amount,tips,k)
    m, n = np.shape(xmat)  # 244 2
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i] * localWeight(xmat[i], xmat, ymat, k)
        # Prediction= xO * beta
    return ypred


def graphplot(X, ypred):  # (bill amount,predicted tips amount)
    sortindex = X[:, 1].argsort(0)
    # returns the index values in an order that would sort the input array
    xsort = X[sortindex][:, 0]
    # 0 - total_bill
    # takes the bill amount and sorts in ascending order
    # Based on values of sortindex select all rows 0th column(i.e Total bill)
    print('xsort', xsort)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(bill, tip, color='green')
    ax.plot(xsort[:, 1], ypred[sortindex], color='red', linewidth=4)

    plt.xlabel('Total Bill')
    plt.ylabel('Tip')
    # plt.show()


data = pd.read_csv('data10_tips.csv')

bill = np.array(data.total_bill)
tip = np.array(data.tip)

mbill = np.mat(bill)
mtip = np.mat(tip)

m = np.shape(mbill)[1]
one = np.mat(np.ones(m))
X = np.hstack((one.T, mbill.T))  # 244 rows 2 columns
print(X)
ypred = localweightregression(X, mtip, 0.5)  # increase to get smooth curve
graphplot(X, ypred)
