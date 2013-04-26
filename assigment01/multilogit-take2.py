# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

#Let's calculate the log-likelihood
from math import exp, log
from scipy.optimize import fmin_bfgs
from scipy.optimize import check_grad
import numpy as np
from numpy import array

# <codecell>

# Test for this fucntion
def calculate_probmatrix(Weights,X):
    """
    D = number of variables (including the bias)
    N = number of training samples
    C = number of classes

    Calculate probabily matrix using the definition in Sec. 8.3.7
    param: Weights - A Weight matrix shape =  DxC
    param: X: Training sample matrix with DxN
    returns: A matrix P CxN   have probability for each sample (=column) and for each clas (=row) 
    as a result we have probaility for each sample (column) for each class (row)
    """
    N = X.shape[1]
    D =  X.shape[0]
    C = Y.shape[1]
        
    P = np.zeros((N,C))
    #print "W =", W
    #print "X =", X

    # TODO: optimize using matrix multiplication (Left like that now for clarity)
    
    for i in range(N):
        for c in range(C):
            P[i,c] = np.exp(Weights[:,c].transpose().dot(X[:,i]))
        sumi = P[i,:].sum()
        P[i,:] = P[i,:]/sumi
     
    #P[:,C-1] = 0
    return P

def example_prob_matrix():

    X = np.asarray([[2, 4,1],[1,1,1],[8,3,1],[2,3,1]]).transpose()
    W = np.asarray([[0.01,0.01,1],[-.03,0.05,1],[-0.4,0.1,1] ]).transpose()
    Y = np.asarray([[1,0,0],[0,1,0],[0,0,1],[1,0,0] ])

    X = np.asarray([[2, 4, 1,1],[1,1,2,1],[8,3,4,1],[2,3,5,1]]).transpose()
    W = np.asarray([[0.01,0.01,0.01,1],[-.03,0.05,-0.04,1],[-0.4,0.1,0.3,1] ]).transpose()
    Y = np.asarray([[1,0,0],[0,1,0],[0,0,1],[1,0,0] ])


    calculate_probmatrix(W,X)




# <codecell>

# First we define the log likelihood
# let's redefine the formulas based on
def newloglikelihood(W,X,Y):
    """
    calculate the negative loglikelihood
    N = num of samples
    D = num of variables including the bias
    C = number of classes 
    
    Wr = weight matrix dim (DxC) - needs to be reshaped from W - each column a Class
    X = matrix of data, stacked horizontally X = DxN (each colum a sample)
    Y = 1-of-C encoding of the matrix in (NxC) (Stacked horizontally)
    """
    #let's do it sequentllay first
    #get the appropiate dimenssions 

    N = X.shape[1]
    D =  X.shape[0]
    C = Y.shape[1]

    Wr = W.reshape((D,C))
    #Wr[-2,:] = 0 # whatever they say
    l = 0.
 
    # calculate x_i dot b_j for each class (class = row) and then for each sample in batch (sample = column)
    # the result is a matrix with #Class rows and #Sample columns
    scores = np.log( (np.exp( ( Wr.transpose().dot(X) ) ) ).sum(axis=0) )


    
# We can do this also using matrix but we do it  like this for clarity now 
# TODO: Uncomment and modifify the loop
#    scores = Y.dot(scores)
#    print scores

    for i in range(N):
        for c in range(C):
            l +=  Y[i,c]*(Wr[:,c].transpose().dot(X[:,i]))
        l = l - scores[i]

    return -l

def example_loglikelihood():
    X = np.asarray([[2, 4,1],[1,1,1],[8,3,1],[2,3,1]]).transpose()
    W = np.asarray([[0.01,0.01,1],[-.03,0.05,1],[-0.4,0.1,1] ])
    Y = np.asarray([[1,0,0],[0,1,0],[0,0,1],[1,0,0] ])

    newloglikelihood(W.reshape(W.size),X,Y)


# <codecell>


def newmygrad(W,X,Y):
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
   
    #get probability matrix and and transpose
    #as a result we have probaility for each sample (=column) for each class (=row)
    #we can directly modify probMatrix, it's each time new generated when calling getProbabilityMat

  
    P = calculate_probmatrix(Wr,X).transpose()

    
    P -= Y.transpose()
    
    suma = 0

    grad = np.zeros(W.size)
    for i in range(N):
        print np.kron(P[:,i],X[:,i])
        grad += np.kron(P[:,i],X[:,i])

    
    return grad


def example_grad():
    X = np.asarray([[2, 4, 1,1],[1,1,2,1],[8,3,4,1],[2,3,5,1]]).transpose()
    W = np.asarray([[0.3,0.5,1,1],[-.03,0.05,-0.04,1],[-0.4,0.1,0.1,1] ])
    Y = np.asarray([[1,0,0],[0,1,0],[0,0,1],[1,0,0] ])
    newmygrad(W.reshape(W.size),X,Y)

    
# <codecell> - Check gradient

def my_check_the_gradient(W,X,T):
    def floglik(W):
        return  newloglikelihood (W,X,T)
    def gfloglik(W):
        return newmygrad(W,X,T)
    return    check_grad(floglik, gfloglik, W)
        

# <codecell>


#my_check_the_gradient(W.reshape(W.size),X,Y)

# Check the gradient by hand

Xt = np.asarray([[1, 0, 0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0]]).transpose()
Wt = np.asarray([[-2,0.0,0.0,0],[0.,0,0,0],[0.,0,0.0,0] ]).transpose()
Yt = np.asarray([[1,0,0],[0,1,0],[0,1,0],[0,0,1] ])


X = np.asarray([[2, 4, 1,1],[1,1,2,1],[8,3,4,1],[2,3,5,1]]).transpose()
W = np.asarray([[0.01,0.01,0.01,1],[-.03,0.05,-0.04,1],[-0.4,0.100,-10,1] ])
Y = np.asarray([[1,0,0],[0,1,0],[0,0,1],[1,0,0] ])

def floglik(w):
        return  newloglikelihood (w,X,Y)
def gfloglik(w):
        return newmygrad(w,X,Y)



def cus_check_grad(f, fprime, x0):                                                            
    eps = 1e-5                                                                            
    approx = np.zeros(len(x0))                                                            
    for i in xrange(len(x0)):                                                             
        x0_ = x0.copy()                                                                   
        x0_[i] += eps                                                                     
        approx[i] = (f(x0_) - f(x0)) / eps
    print "Aprox:" , approx
    print "FPRIME:",  fprime(x0)
    return np.linalg.norm(approx.ravel() - fprime(x0).ravel())    


cus_check_grad(floglik,gfloglik,W.reshape(W.size))
my_check_the_gradient(W.reshape(W.size),X,Y)

    



