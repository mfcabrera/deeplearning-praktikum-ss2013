## Neural Network with one Hidden Layer for Classifying the MNIST dataset

Author: Miguel Cabrera <miguel.cabrera@tum.de>

 python multilogit.py -h

     usage: neuralnet.py [-h] [-e E] [-b B] [-l L] [-j HIDDEN_LAYER_SIZE]
                         [-o OPTIMIZATION_METHOD] [-s]

     Classify handwritten digits from the MNIST dataset using a neural network with
     a hidden layer with rmsprop and mini-batch stochastic gradient descent.

     optional arguments:
     -h, --help            show this help message and exit
     -e E, --epochs E      number of epochs for the training, E = 25 by default
     -b B, --number-of-batches B   number of batches, how many batches divide the
                                                 training set. B = 200 by default
     -l L, --learning-rate L  learning rate, default L = 0.001
     -j HIDDEN_LAYER_SIZE, --hidden-layer-size HIDDEN_LAYER_SIZE
                        numbers of neurons in the hidden layer, default = 300
     -o OPTIMIZATION_METHOD, --optimization-method OPTIMIZATION_METHOD
                        Optimization technique, on of [msgd,gc,bfgs] - only
                        implemented msgd - minibatch stochastic gradient
                        descent
     -s, --early-stoping   Use early stopping - currently disabled by problems
                         with the gradient, although implemented.



### Description

Hopefully the parameters are self-explanatory. This code contain and
implementation of a neural network classification for the MNIST dataset. In
order to run this it is necessary that the file  *mnist.pkl.gz* be in the
working directory. This is implemented in Numpy/SciPy  purely. The Math is
taken mainly from UFDL Tutorial and Andrw NG Ml course. It uses rmsprop as
explained by Prof. Hinton in the Neural Network course on Coursera. 
For now only normal sigmoid is used (with the grad being calculated with a
trick to speedup the implmentation.)
9-10% of error is reached due to a problem with the gradient that *I am still trying to solve*
Early stopping is implemented but deactivated due the problem with the
gradient. The "patience" is implemented is described as explained in the Deep
learning turorial.


It uses a full vector (un-rolled) implementation. This would allow to use
anyother optimization method easily. However, only  mini-batch sgd with
rmsprop is used. Momentum is not used, as Prof. Hinton hints (:P) that is not
of much help.
 
