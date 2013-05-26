# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

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

def sigmoid(z):
    return np.true_divide(1,1 + np.exp(-z) )

#not calculated really - this the fake version to make it faster.  using previously "a" calcultion
#so no need to calculate it again
def sigmoid_prime(a):
    return  (a)*(1 - a)

# <codecell>

class NeuralNet(object):
    """implementation of a neural net
    """

    def _extract_weights(self,W):
        """
        This will extract the weights from we big W array. in a 1-hidden layer network.
        this can be easily generalized.
        """
        wl1_size = self._D*self._hidden_layer_size
        bl1_size = self._hidden_layer_size
        
        wl2_size = self._hidden_layer_size*self._output_size
        bl2_size = self._output_size

        
        weights_L1 = W[0:wl1_size].reshape((self._D,self._hidden_layer_size))
        bias_L1  =   W[wl1_size:wl1_size+bl1_size]
        
        start_l2 = wl1_size+bl1_size

        weights_L2 = W[start_l2: start_l2  + wl2_size].reshape((self._hidden_layer_size,self._output_size))
        bias_L2  =  W[start_l2  + wl2_size : start_l2  + wl2_size + bl2_size]
        
    
        
        return weights_L1,bias_L1,weights_L2,bias_L2
        

    def _forward_prop(self,W,X,transfer_func=sigmoid):
        """
        Return the output of the net a Numsamples (N) x Outputs (No CLasses)
        # an array containing the size of the output of all of the laye of the neural net
        """
        
        # Hidden layer DxHLS
        weights_L1,bias_L1,weights_L2,bias_L2 = self._extract_weights(W)    
        
        # Output layer HLSxOUT
                
        # A_2 = N x HLS
        A_2 = transfer_func(np.dot(X,weights_L1) + bias_L1 )
        
        # A_3 = N x  Outputs
        A_3 = transfer_func(np.dot(A_2,weights_L2) + bias_L2)
        
        # output layer
        return [A_2,A_3]
        
    
    def __init__(self,training_set,testing_set,validation_set,no_neurons=300,transfer_func=None,optimization_func=None):
        """
        yup
        """

        self._transfer_func = transfer_func

                  
        self._training_data = training_set[0]
        self._training_labels = training_set[1]
        
                 
        self._testing_data = testing_set[0]
        self._testing_labels = testing_set[1]
        
        self._validation_data = validation_set[0]
        self._validation_labels = validation_set[1]
        self._hidden_layer_size = no_neurons

        self._weights_filename = "nn_receptive_fields.dump"
                

        #how mauch data do we have for training?
        self._N = self._training_data.shape[0]
        
        #Dimension of the data (basically, the number of inputs in the input layer)
        self._D = self._training_data.shape[1]

        # We assume that the labels are encoded in an array of Nx1 wich each entry an integer
        # defining the class starting at 0

        #number of classes or basically, how many outputs are we going to have.
        self._output_size = max(self._training_labels) + 1

        #initialize the weights for the layers: - we are going to work with one 1 hidden layer

        #layer 1: input * number neuros in the hidden layer + hidde_layer_size for  the biases

        # first layer
        network_weight_no =  self._D * self._hidden_layer_size  +  self._hidden_layer_size 
        # second layer
        network_weight_no  +=  self._hidden_layer_size * self._output_size  +  self._output_size

        self._betas = np.random.normal(0, 0.1, network_weight_no)
        
        #layer 2: hidden layer * no_classes + no_classes for  the biases


      
    
    def _traing_mini_sgd(self,learning_rate=0.2,batch_number=10,epochs=1):
        """
        Training miniSGD
        """
                
        self._batch_number = batch_number
        self._batch_size = self._N / self._batch_number
        
        self._epochs = epochs
        self._learning_rate = learning_rate
        
        print "Training with learning_rate = {0} :: #Batches = {1} :: #Epochs = {2}".format(learning_rate,
                                                                                            batch_number,
                                                                                             epochs)
        # Erro reporting: 
        self.error_testing = np.zeros(self._epochs*self._batch_number)
        self.error_validation = np.zeros(self._epochs*self._batch_number)
        self.error_training = np.zeros(self._epochs*self._batch_number)
        self.cost_output= np.zeros(self._epochs*self._batch_number)
         
       
        patience = 5000  # look as this many examples regardless
        patience_increase = 2     # wait this much longer when a new best is found
        improvement_threshold = 0.995  # a relative improvement of this much is # considered significant
        validation_frequency = min(self._batch_number, patience/2)
        # go through this many
        # minibatches before checking the network
        # on the validation set; in this case we
        # check every epoch





        #Mean Square weight for rmsprop
        means_sqrt_w = np.zeros(self._betas.size)
        
        early_stopping=0
        done_looping = False

        for epoch in range(self._epochs):

            #early stopping
            if(done_looping):
                break
                
       
            for  batch in range(self._batch_number):
                print "Trainining: Epoch {0}/{1} - Batch {2}".format(epoch+1,self._epochs,batch+1)
                
                start = self._batch_size*batch
                end =   self._batch_size*batch + self._batch_size
                
                Xs = self._training_data[start:end,:]
                Ls = self._training_labels[start:end]
                
                
                delta_ws = self._back_prop(self._betas,Xs,Ls)
                means_sqrt_w = 0.9*means_sqrt_w + 0.1*(delta_ws**2)
                            
              
                rmsdelta = np.divide(delta_ws,np.sqrt(means_sqrt_w))
                
                self._betas = self._betas - self._learning_rate*rmsdelta
                
              
                iter = (epoch - 1) * self._batch_number + batch


                if(early_stopping > 0):
                    if (iter + 1) % validation_frequency == 0:
                        this_validation_loss = self.calculate_zero_one_loss("validation")
                        if this_validation_loss < best_validation_loss:

                            # improve patience if loss improvement is good enough
                            if this_validation_loss < best_validation_loss * improvement_threshold:
                                patience = max(patience, iter * patience_increase)
                                best_params = np.copy(self._betas)
                                best_validation_loss = this_validation_loss

                    if patience <= iter:
                        done_looping = True
                    break


                    

                
                # for the graph
                self.error_testing[epoch*self._batch_number + batch] = self.calculate_zero_one_loss()
                print "Error in the testing datatest: {0}%".format(self.error_testing[epoch*self._batch_number + batch]*100)
                self.error_validation[epoch*self._batch_number + batch] = self.calculate_zero_one_loss("validation")
                self.error_training[epoch*self._batch_number + batch] =  self.calculate_zero_one_loss("training")
                self.cost_output[epoch*self._batch_number + batch] = self.cost_function(self._betas,
                                                                                        self._training_data,
                                                                                        self._training_labels)
        # after traing dump the weights
        self.dump_weights()
                
    def dump_weights(self):
        #dumping only the hidden layer  weights
        
        f = open(self._weights_filename, 'w')
        cPickle.dump(self._betas[0:self._D*self._hidden_layer_size],f)
        f.close()

    def visualize_receptive_fields(self):
        import sys
        
        beta_set = cPickle.load(open(self._weights_filename,"r"))

        beta_set = beta_set.reshape(self._D,self._hidden_layer_size).T
        #print beta_set.shape
        plt.figure(9)

        sys.stdout.write('Writing  visualizations of receptive field to disk...')
        sys.stdout.flush()
        for i, beta in enumerate(beta_set):
           
            d = beta.reshape(28, 28)
            gray_map = plt.get_cmap('gray')
            plt.imshow(d, gray_map)
            plt.savefig('nn_receptive_fields/receptive_field'+str(i)+'.png', dpi=150)
            sys.stdout.write('.')
            sys.stdout.flush()

        plt.close()
        sys.stdout.write('.DONE - check directory nn_receptive_fields/ \n')
        sys.stdout.flush()


    def train(self,learning_rate=0.1,algorithm="msgd",**kwargs):
        """
        Do some nifty training here :)
        """
        
        
        e = self.cost_function(self._betas,self._training_data,self._training_labels)
        print "Inital total  cost: ", e     
      


        
        if(algorithm=="msgd"):
            #There should be a more elegant way to do this
            self._traing_mini_sgd(learning_rate,**kwargs)
        else:
            raise Exception("Algorithm not yet implemented, check later ;)")
                            
        loss = self.calculate_zero_one_loss()
        print "After training error: ", loss
        

        e = self.cost_function(self._betas,self._training_data,self._training_labels)
        print "Final total cost: ", e     
        

                
        
        
    def make_1_of_c_encoding(self,labels):
        # get one of C encoding matrix for each entry (=row) and class (=column)

        Y = np.zeros(shape=(labels.shape[0],self._output_size))
        #labels = self._training_labels[start:end]
        
        for row, label in enumerate(labels):
            Y[row, label] = 1
        return Y


    def cost_function(self,W,X,labels,reg=0.001):
        """
        reg: regularization term
        No weight decay term - lets leave it for later
        """
        
        outputs = self._forward_prop(W,X,sigmoid)[-1] #take the last layer out
        sample_size = X.shape[0]

        y = self.make_1_of_c_encoding(labels)
      
        e1 = (np.sum((outputs - y), axis=1))
        
        #error = e1.sum(axis=1)
        error = e1.sum()/sample_size + 0.5*reg*(np.square(W)).sum()

        return error
        
    def _back_prop(self,W,X,labels,f=sigmoid,fprime=sigmoid_prime,lam=0.001):
    
        """
        Calculate the partial derivates of the cost function using backpropagation.
        Using a closure,can be used with more advanced methods of optimization
        lam: regularization term / weight decay
        
        """
        
        Wl1,bl1,Wl2,bl2  = self._extract_weights(W)
        
        
    
        layers_outputs = self._forward_prop(W,X,f)
        
        
        y = self.make_1_of_c_encoding(labels)
        num_samples = X.shape[0] # layers_outputs[-1].shape[0]
        
        # Dot product return  Numsamples (N) x Outputs (No CLasses)
        # Y is NxNo Clases
        # Layers output to
        # small_delta_nl = NxNo_Outputs

        big_delta = np.zeros(Wl2.size + bl2.size + Wl1.size + bl1.size)
        big_delta_wl1, big_delta_bl1, big_delta_wl2, big_delta_bl2 = self._extract_weights(big_delta)
                                                                    
        # print big_delta_wl1.shape
        # print big_delta_wl2.shape
        # print big_delta_bl1.shape
        # print big_delta_bl2.shape  
        
        #  print big_delta_wl2.shape
        
        for i,x in enumerate(X):
            #print layers_outputs[-1][i,:]
            #1xNclasses vector - each per class

            dE_dy = layers_outputs[-1][i,:] -  y[i,:] 
            
            big_delta_bl2 +=   dE_dy

            dE_dz_out  = dE_dy * fprime(layers_outputs[-1][i,:])
            

            dE_dhl = dE_dy.dot(Wl2.T)
  

            big_delta_bl1 += dE_dhl
            
            small_delta_hl = dE_dhl*fprime(layers_outputs[-2][i,:])


           
            
            big_delta_wl2 += np.outer(layers_outputs[-2][i,:],dE_dz_out)
            big_delta_wl1 +=   np.outer(x,small_delta_hl)
           
            

            
            
        #TODO Regularization
        #print num_samples
        #num_samples = 1
        
        big_delta_wl2 = np.true_divide(big_delta_wl2,num_samples) + lam*Wl2*2
        big_delta_bl2 = np.true_divide(big_delta_bl2,num_samples)
        big_delta_wl1 = np.true_divide(big_delta_wl1,num_samples) + lam*Wl1*2
        big_delta_bl1 = np.true_divide(big_delta_bl1,num_samples)
        
        #return big_delta

        return np.concatenate([big_delta_wl1.ravel(),
                               big_delta_bl1,
                               big_delta_wl2.ravel(),
                               big_delta_bl2])

    
    

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
        

        prob_mat  =  self._forward_prop(self._betas,data,sigmoid)[-1]
        
        predicted_labels = np.argmax(prob_mat, axis=1) 
        
        
        testing_no = predicted_labels.size 

        error =  np.nonzero(predicted_labels - labels)[0].size
        
        error = error/float(testing_no)
        #print "Error in {0} set  = {1}".format(dataset, error)
        return  error
        
    def plot_error(self):
        """
                           
        Arguments:
        - `training`:
        - `testing `:
        - `validation`:
        """

        print "Plotting error..."
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
        p4, = plt.plot(x,self.cost_output)
        plt.legend([p4], ["negloglikelihood"])
      
        plt.savefig('neglog.png')

        print "DONE. Check error-plots.png and neglog.png"


    def test_cost_function(self):
        
        e = self.cost_function(self._betas,self._training_data[0:100,:],self._training_labels[0:100],sigmoid)

        print e 

    def test_backprop(self):
        self._back_prop(self._betas,self._training_data[0:100,:],self._training_labels[0:100])

    def check_gradient(self):
        def cost(ws):
            return   self.cost_function(ws,self._training_data[0:100,:],self._training_labels[0:100])
                                        
        def gradcost(ws):
            return self._back_prop(ws,self._training_data[0:100,:],self._training_labels[0:100])
        
        print check_grad(cost, gradcost,self._betas)



def load_data():
    import cPickle,gzip
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set,valid_set,test_set


if __name__ == "__main__":

    #train_set,test_set,valid_set = load_data()
    nn = NeuralNet(train_set,test_set,valid_set,300)
    #nn.check_gradient()
    nn.train(0.001,epochs=30,batch_number=100)
    
#     parser = argparse.ArgumentParser(description='Classify handwritten digits from the MNIST dataset using a neural network with a hidden layer with rmsprop and mini-batch stochastic gradient descent.')
#     parser.add_argument('-e','--epochs', metavar='E', type=int,default=25,
#                         help='number of epochs for the training,  E = 25 by default')

#     parser.add_argument('-b','--number-of-batches', metavar='B', type=int,default=200,
#                         help='number of batches, how many batches divide the training set. B = 200 by default')

#     parser.add_argument('-l','--learning-rate', metavar='L', type=float,default=0.001,
#                         help='learning rate, default L = 0.001')

#     parser.add_argument('-j','--hidden-layer-size', type=int,default=300,
#                          help='numbers of neurons in the hidden layer, default = 300')

#     parser.add_argument('-o','--optimization-method',default="msgd",
#                         help='Optimization technique,  on of  [msgd,gc,bfgs] - only implemented msgd - minibatch stochastic gradient descent')
    
#     parser.add_argument('-s','--early-stoping',action='store_true',
#                         help='Use early stopping - currently disabled by problems with the gradient, although implemented.')
        
    
#     args = parser.parse_args()

#     train_set,test_set,valid_set = load_data()
    
#     nn = NeuralNet(train_set,test_set,valid_set,no_neurons=args.hidden_layer_size)
#     print "Training the neural network...."
#     nn.train(learning_rate=args.learning_rate,
#              batch_number=args.number_of_batches,
#              epochs=args.epochs,
#              algorithm=args.optimization_method)

    
#     #nn.dump_weights()
#     nn.visualize_receptive_fields()
#     nn.plot_error()

