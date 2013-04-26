# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#Let's calculate the log-likelihood
from math import exp, log
from scipy.optimize import fmin_bfgs
from scipy.optimize import check_grad
import numpy as np
from numpy import array
from scipy.misc import logsumexp

# <codecell>

# Test for this fucntion
def calculate_probmatrix(Weights,X,b):
    """
    D = number of variables (including the bias)
    N = number of training samples
    C = number of classes

    Calculate probabily matrix using the definition in Sec. 8.3.7
    param: Weights - A Weight matrix shape =  DxC
    param: X: Training sample matrix with DxN
    returns: A matrix P CxN   have probability for each sample (=row) and for each class (=column) 
    as a result we have probaility for each sample (row) for each class (column)
    """
    N = X.shape[1]
    D =  X.shape[0]
    C = Y.shape[1]
    
    #print "D,C",D,C
    
    P = np.zeros((N,C))
    # print "W =", W
    # print "X =", X

    
    P1  = np.exp(W.transpose().dot(X)  +  b).transpose()
    
    for i in range(N):
        P1[i,:] = P1[i,:]/P1[i,:].sum()

    return P1

def newmygrad(W,X,Y,b):
    """
    calculate the gradient of the  loglikelihood
    N = num of samples
    D = num of variables including the bias
    C = number of classes 
    
    Wr = weight matrix dim (DxC) - needs to be reshaped from W - each column a Class
    X = matrix of data, stacked horizontally X = DxN (each colum a sample)
    Y = 1-of-C encoding of the matrix in (NxC) (Stacked horizontally)
    """
    

    N = X.shape[1]
    D =  X.shape[0]
    C = Y.shape[1]

    
    # calculate the probability matrix
   
    Wr = W.reshape((D,C))
    
    #get probability matrix and cut off probabilities for last class and transpose
    #as a result we have probaility for each sample (=column) for each class (=row)
    #we can directly modify probMatrix, it's each time new generated when calling getProbabilityMat

  
    P = calculate_probmatrix(Wr,X,b).transpose()

    #print "P = \n", P
    #print "Y.transpose \n", Y

    grad = np.zeros(W.size)
    for i in range(N):
        #print np.kron(P[:,i],X[:,i])
        grad += np.kron(P[:,i],X[:,i])
        #suma += np.kron((scores[:,i] - Y.transpose()[:,i]),X[:,i])
    
    return grad



def newloglikelihood(W,X,Y,bias_weights):
    """
    calculate the negative loglikelihood
    N = num of samples
    D = num of variables including the bias
    C = number of classes 
    
    Wr = weight matrix dim (DxC) - needs to be reshaped from W - each column a Class
    X = matrix of data, stacked horizontally X = DxN (each colum a sample)
    Y = 1-of-C encoding of the matrix in (NxC) (Stacked horizontally)
    b = bias vector  1xC
    """
    
    #let's do it sequentllay first

    #get the appropiate dimenssions 
    N = X.shape[1]
    D =  X.shape[0]
    C = Y.shape[1]

    Wr = W.reshape((D,C))
    #Wr[-2,:] = 0 # whatever they say

    # now bdotx contains Xi dot Bi for all classes (=row) and samples (= column)
    bdotx = Wr.transpose().dot(X) + bias_weights
  #  print "BdotX", bdotx 
    index = (Y.transpose() > 0)

    # calculate x_i dot b_j for each class (class = row) and then for each sample in batch (sample = column)
    # the result is a matrix with #Class rows and #Sample columns
    scores =  logsumexp( Wr.transpose().dot(X),0 )
    #scores = np.log( (np.exp( ( Wr.transpose().dot(X) ) ) ).sum(axis=0) )
    
    #with left we calculate the left side of the formula 8.35 of Murphy
    left = bdotx[index]
    loglike_or_something =  (left  - scores).sum()

    return -loglike_or_something


def my_check_the_gradient(W,X,T,b):
    def floglik(W):
        return  newloglikelihood (W,X,T,b)
    def gfloglik(W):
        return newmygrad(W,X,T,b)
    return    check_grad(floglik, gfloglik, W)


#Using b separately
X = np.asarray([[2, 4, 1],[1,1,2],[8,3,4],[2,3,5]]).transpose()
W = np.asarray([[0.1,0.3,0.5],[0.2,0.4,0.6],[0.3,0.5,0.7] ]).transpose()
Y = np.asarray([[1,0,0],[0,1,0],[0,0,1],[1,0,0] ])
b = np.asarray([[1,1,1]]).transpose()


#X = np.asarray([[2, 4, 1,1],[-1,-1,-2,-1],[8,3,4,1],[2,3,5,1]]).transpose()
#W = np.asarray([[-0.4,1,-0.1,-10]]).transpose()
#Y = np.asarray([[0],[1],[0],[1] ])

#calculate_probmatrix(W,X,b)
#newmygrad(W.reshape(W.size),X,Y)
#newloglikelihood(W.reshape(W.size),X,Y,b)
print "Difference :S" , my_check_the_gradient(W.reshape(W.size),X,Y,b) 

