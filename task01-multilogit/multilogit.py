#!/usr/bin/env python
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
#for parsing the trianing data
import cPickle, gzip
import matplotlib.pyplot as plt
import argparse

# <codecell>

# Test for this fucntion
def calculate_probmatrix(Weights,X,b):
    """
    D = number of variables (including the bias)
    N = number of training samples
    C = number of classes

    Calculate probabily matrix using the definition in Sec. 8.3.7
    param: Weights - A Weight matrix shape =  DxC
    param: X: Training sample matrix with NXD
    param: b: Bias weight of 1xC vector
    param: Y: 1-of-C coding scheme  NxC matrix input
    returns: A matrix P NxC   have probability for each sample (=row) and for each class (=column) 
    as a result we have probaility for each sample (row) for each class (column)
    """
    N = X.shape[0]
    D =  X.shape[1]
    C = Weights.shape[1]
    
    #P = np.zeros((N,C))
    #print P.shape
    P1 = np.dot(X,Weights) + b
    P1  = np.exp(P1)
    
    for i in range(N):
        P1[i,:] = P1[i,:]/P1[i,:].sum()
   # print P1
    return P1

def grad_neg_loglikelihood(Weights,X,Y,b):
    """
    calculate the gradient of the  loglikelihood
    N = num of samples
    D = num of variables including the bias
    C = number of classes 
    
    Wr = weight matrix dim (DxC) - needs to be reshaped from W - each column a Class
    X = matrix of data, stacked vertically X = NxD (each row a sample)
    Y = 1-of-C encoding of the matrix in (NxC) (Stacked vertically)
    """
    N = X.shape[0]
    D =  X.shape[1]
    C = Y.shape[1]
    #print N,D,C

    #print "W shape", Weights.shape
    #print "D,C =" , D,C
    
  
    Wr = Weights.reshape((D,C))
    bias =  b
    
    # calculate the probability matrix
   
    #get probability matrix and cut off probabilities for last class and transpose
    #as a result we have probaility for each sample (=column) for each class (=row)
    #we can directly modify probMatrix, it's each time new generated when calling getProbabilityMat

  
    P = (calculate_probmatrix(Wr,X,bias) - Y).transpose()
    #print P
   

    grad = np.zeros((D,C))
   
    # Change to Kron product! - or leave it like that if not  too slow
    for c in range(C):
        for (i,x_i) in enumerate(X):
            #print x_i.shape      
            grad[:,c] += P[c,i]*x_i 
            
    
    # for i in range(N):
    #     #print np.kron(P[:,i],X[:,i])
    #     grad[:] += np.kron(P[:,i].transpose(),X[:,i])
    #     #suma += np.kron((scores[:,i] - Y.transpose()[:,i]),X[:,i])
    #print "Grads.shape ", grad.shape
    #print "Grad :", grad
    return  grad.reshape(Weights.size)
     



def neg_loglikelihood(Weights,X,Y,b):
    """
    calculate the negative loglikelihood
    N = num of samples
    D = num of variables including the bias
    C = number of classes 
    
    Wr = weight matrix dim (DxC) - needs to be reshaped from W - each column a Class
    X = matrix of data, stacked vertically X = NxD (each row a sample)
    Y = 1-of-C encoding of the matrix in (NxC) (Stacked horizontally)
    bias_weight = bias vector  1xC
    """
    
    #get the appropiate dimenssions 
    N = X.shape[0]
    D =  X.shape[1]
    C = Y.shape[1]

     
    Wr = Weights.reshape((D,C))
    bias =  b

    # now bdotx contains Xi dot Bi for all classes (=row) and samples (= column)
    bdotx = np.dot(X,Wr) + bias
    index = (Y > 0)
    left = bdotx[index]
    

  # calculate x_i dot b_j for each class (class = row) and then for each sample in batch (sample = column)
    # the result is a matrix with #Class rows and #Sample columns
    scores =  logsumexp(np.dot(X,Wr),1)
   
    #print scores.shape

    #with left we calculate the left side of the formula 8.35 of Murphy
    loglike_or_something =  (left  - scores).sum()

    return -loglike_or_something



def my_check_the_gradient(Wii,X,T):
    def floglik(ws):
        return  neg_loglikelihood (ws,X,T)
    def gfloglik(ws):
        return grad_neg_loglikelihood(ws,X,T)
    return    check_grad(floglik, gfloglik, Wii)

def my_check_the_gradient_b(Wii,X,T,b):
    def floglik(ws):
        return  neg_loglikelihood (ws,X,T,b)
    def gfloglik(ws):
        return grad_neg_loglikelihood(ws,X,T,b)
    return    check_grad(floglik, gfloglik, Wii)


# calculate_probmatrix(W,X,b)
# print   neg_loglikelihood (W.reshape(W.size),X,Y,b)
# print grad_neg_loglikelihood(W.reshape(W.size),X,Y,b)

#print newloglikelihood(W.reshape(W.size),X,Y,b)
#print neg_loglikelihood(Wall.reshape(Wall.size),X,Y)

#print "Difference:" , my_check_the_gradient_b(W.reshape(W.size),X,Y,b) 


# <codecell>





def estmlogit(W, X,T,  m=None, algo="bfgs", maxiter=50, epsilon = 1.0e-8, full_output=True, reflevel= 0, disp=False):
    """
    Returns estimates of the unknown parameters of the multivalued logistic regression
        initial estimates, the response Y vector and predictors X matrix. X must have
    a column of all 1s if a constant is in the model! 
    The estimates are determined by fmin_bfgs.
    """
 
    # if m is None:
    #    m = len(itable(Y))
 
    def floglik(W):
        return newloglikelihood(W,X,T)
    def gfloglik(W):
        return newmygrad(W,X,T)   
 
    if algo=="bfgs":
       output = fmin_bfgs(floglik , W, fprime=gfloglik, maxiter=maxiter, epsilon = epsilon, full_output = True, retall=True, disp=True)
    return output



    

# <codecell>

def load_data():
    
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set,valid_set,test_set
    
    



# <codecell>


class MultiLogReg(object):
    """
    """
    
    def __init__(self, training_set, testing_set, validation_set):
        """
        Init the parameters and set meaning ful constant based on the input parameters
        
        N = number tranining samples
        D = Dimension of each sample
        no_classes = number of classes
        
        Arguments:
        - `training_set`: the training data in a tuple of (data,labels) 
              
        """
                
        self._training_data = training_set[0]
        self._training_labels = training_set[1]
        
                 
        self._testing_data = testing_set[0]
        self._testing_labels = testing_set[1]
        
        self._validation_data = validation_set[0]
        self._validation_labels = validation_set[1]
                

        
        self._N = self._training_data.shape[0]
        self._D = self._training_data.shape[1]

        # We assume that the labels are encoded in an array of Nx1 wich each entry an integer
        # defining the class starting at 0

        self._no_classes = max(self._training_labels) + 1

        #other useful values
        self._weights_filename = 'current_weigths.pickle'

        
        

        

    def calculate_zero_one_loss(self,dataset="testing"):
        """
        
        Arguments:
        - `self`:
        """
        
        # A matrix P NxC   have probability for each sample (=row) and for each class (=column) 
        # as a result we have probaility for each sample (row) for each class (column)
    
        if(dataset=="testing"):
            data = self._testing_data
            labels = self._testing_labels
        elif(dataset=="training"):
            data = self._training_data
            labels = self._training_labels
        elif(dataset=="validation"):
            data = self._validation_data
            labels = self._validation_labels
        else:
            raise StandardError('invalid data set to compute one-0-zero loss')
        

        prob_mat  = calculate_probmatrix(self._betas[:-self._no_classes].reshape(self._D,self._no_classes),
                                         data,
                                         self._betas[-self._no_classes:])
                                     
        
        predicted_labels = np.argmax(prob_mat, axis=1) 
    
       
        testing_no = predicted_labels.size 

        error =  np.nonzero(predicted_labels - labels)[0].size
        
        error = error/float(testing_no)
        #print "Error in {0} set  = {1}".format(dataset, error)
        return  error


    def train(self,batch_number=10,epochs=25,learning_rate=0.1,momentum=0.9,optimizer=None):
        """
        Regardless of the optimizer we use a batch based approach

        -'method': how to train the logistic regresion. A function receiven the weights and...
        for training the parameter. It will return the &Weights used to  and blah blah blah
        """

        self._batch_number = batch_number
        self._batch_size = self._N / self._batch_number
        self._momentum = momentum
        
        self._epochs = epochs
        self._learning_rate = learning_rate

        #if method is None:
        #Then we use the 
        #initial weights
        self._betas = np.random.normal(0, 0.1, (self._D * self._no_classes  +  self._no_classes))
        # biases... don't really know why I just put them here - recommendation of the instructor
        # but whatever - he says that for the future... for neural nets or whatever.
        self._betas[-self._no_classes:] = 1

        y_total =  self.make_1_of_c_encoding(0, self._training_labels.size)
    
        total_err_func = self.make_negloglikelihood(self._training_data,y_total,self._betas[-self._no_classes:])

       

       

        print "Training with: batch_size={0},  number_of_batches={1}, epochs={2}, learning_rate = {3} and momentum = {4}".format(self._batch_size, self._batch_number,self._epochs,self._learning_rate, self._momentum)

        
        self._batch_size = self._N / self._batch_number
        
        # Just to make clear the notation below
        weights = self._betas[:-self._no_classes] 
        bias = self._betas[-self._no_classes:]
        
        #     self.calculate_zero_one_loss_testing()
        #print "initial grad:  ", wgrad[wgrad > 0]
        old_wgrad = np.zeros(self._betas[:-self._no_classes].size)
        momentum_grad  =  np.zeros(self._betas[:-self._no_classes].size)

        stop = False #for stopping criteria
        batches_to_stop = 10 # this paramter say how many batches should I wait until I don't get an improvement
                             # in the testing error to break
        old_batch_error = 1 #
        epsilon = 0.0001


        #Error reporting and graphs
        # where to store the errors for plotting
         
        self.error_testing = np.zeros(self._epochs*self._batch_number)
        self.error_validation = np.zeros(self._epochs*self._batch_number)
        self.error_training = np.zeros(self._epochs*self._batch_number)
        self.neglog_error = np.zeros(self._epochs*self._batch_number)


      
                           

        for epoch in range(self._epochs):
            if (stop):
                    break
            
            #mini-batch gradient descent
            
            for  batch in range(self._batch_number):
                print "Trainining: Epoch {0}/{1} - Batch {2}".format(epoch+1,self._epochs,batch+1)
                
                start = self._batch_size*batch
                end =   self._batch_size*batch + self._batch_size
                samples = self._training_data[start:end,:]
                labels = self.make_1_of_c_encoding(start,end)
                
                           
                old_weights = np.copy(self._betas)
                fprime = self.make_grad_loglikelihood(samples,labels,bias)
                #update the weights each batch
                wgrad = fprime(weights)
                #copy all the weights

               
                #Momentum
                
                momentum_grad  = self._momentum*old_wgrad - self._learning_rate*(wgrad/self._batch_size)

                old_wgrad = np.copy(momentum_grad)
                self._betas[:-self._no_classes] =  self._betas[:-self._no_classes] + momentum_grad
                batch_error = self.calculate_zero_one_loss("testing")
                           
                self.error_testing[epoch*self._batch_number + batch] = batch_error
                self.error_validation[epoch*self._batch_number + batch] = self.calculate_zero_one_loss("validation")
                self.error_training[epoch*self._batch_number + batch] =  self.calculate_zero_one_loss("training")
                self.neglog_error[epoch*self._batch_number + batch] =  total_err_func(self._betas[:-self._no_classes])
                           

                #stopping criteria

            if (old_batch_error - batch_error < epsilon  ):
                stop = True
                break
            old_batch_error = batch_error
            
            print "Last epoch error in testing dataset: {0} % ".format(batch_error*100)            
            

        #training is finish let's store the learned weights
        self.dump_weights()
    
    def dump_weights(self):
        f = open(self._weights_filename, 'w')
        cPickle.dump(self._betas[:-self._no_classes],f)
        f.close()
                
    def visualize_receptive_fields(self):
        beta_set = cPickle.load(open(self._weights_filename,"r"))

        beta_set = beta_set.reshape(self._D,self._no_classes).T
       
 
        for i, beta in enumerate(beta_set):
            print beta.shape
            d = beta.reshape(28, 28)
            gray_map = plt.get_cmap('gray')
            plt.imshow(d, gray_map)
            plt.savefig('receptive_field'+str(i)+'.png', dpi=150)
                        
        
                           

    def plot_error(self):
        """
                           
        Arguments:
        - `training`:
        - `testing `:
        - `validation`:
        """

        print "Plotting error in all datasets Alter"
        
        plt.figure(1)
        x = np.arange(self.error_testing.size)
        p1, = plt.plot(x,self.error_testing)
        p2, = plt.plot(x,self.error_validation)
        p3, = plt.plot(x,self.error_training)
        
        

        plt.legend([p2, p1,p3], ["Testing", "Validation","Training"])
        
        
        #plt.show()
        

        plt.savefig('error-plots.png')
        plt.close()
        
        
        plt.figure(2)
        p4, = plt.plot(x,self.neglog_error)
        plt.legend([p4], ["negloglikelihood"])
        
        plt.savefig('neglog.png')
        plt.close()
        
    
    def make_negloglikelihood(self,samples,labels,bias):
        def fneg(w):
            return neg_loglikelihood(w,samples,labels,bias)
        return fneg
        

    def make_grad_loglikelihood(self,samples,labels,bias):
        def fprime(w):
            return  grad_neg_loglikelihood(w,samples,labels,bias)
        return fprime
        
        
    def make_1_of_c_encoding(self,start,end):
        # get one of C encoding matrix for each class (=row) and sample (=column)
        Y = np.zeros(shape=(end - start,self._no_classes))
        labels = self._training_labels[start:end]
        
        for row, label in enumerate(labels):
            Y[row, label] = 1
        return Y
     
    def test_make_1_of_c_encoding(self):
        print self._training_labels[0:10]
        print self.make_1_of_c_encoding(0,10)
        


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description='Classify handwritten digits from the MNIST dataset using Multnomial logistic regression. Ouputs to the filesystem png files with the graphs of the errors and the receptive fields.')
    parser.add_argument('-e','--epochs', metavar='E', type=int,default=10,
                   help='number of epochs for the training,  E = 25 by default')

    parser.add_argument('-b','--number-of-batches', metavar='B', type=int,default=10,
                        help='number of batches, how many batches divide the training set. B = 5000 by default')

    parser.add_argument('-l','--learning-rate', metavar='L', type=float,default=0.1,
                        help='learning rate, default L = 0.1')
    parser.add_argument('-m','--momentum', metavar='M', type=float,default=0.9,
                        help='momentum, default M = 0.1')

    parser.add_argument('-o','--optimization-method',default="msgd",
                         help='Optimization technique,  on of  [msgd,gc,bfgs] - only implemented smgd (stochastic minibatch gradient descent)')
    
    

    args = parser.parse_args()
    #print args

    print "Starting Multinomial logistic regression training..."
    train_set,valid_set,testing_set = load_data()

    c = MultiLogReg(train_set,valid_set,testing_set)

    c.train(epochs=args.epochs,
            learning_rate=args.learning_rate,
            momentum=args.momentum,
            batch_number=args.number_of_batches)
     

    c.plot_error()
    c.visualize_receptive_fields()
    print "Training finished ... please check the png outputs for the graphsof the errors and receptive fields"

