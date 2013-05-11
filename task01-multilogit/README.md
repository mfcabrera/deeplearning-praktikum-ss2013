## Multinomial Logistic regression implementation

Author: Miguel Cabrera <miguel.cabrera@tum.de>

Usage: python multilogit.py -h


    Classify handwritten digits from the MNIST dataset using Multnomial logistic
    regression. Ouputs to the filesystem png files with the graphs of the errors
    and the receptive fields.

    optional arguments:
    -h, --help            show this help message and exit
    
    -e E, --epochs E      number of epochs for the training, E = 25 by
     default
    
    -b B, --number-of-batches B number of batches, how many batches divide the
                              training set. B = 10 by default
                              
     -l L, --learning-rate learning rate, default L = 0.1
     
    -m M, --momentum M    momentum, default M = 0.1
    
     -o OPTIMIZATION_METHOD, --optimization-method OPTIMIZATION_METHOD
                        Optimization technique, on of [msgd,gc,bfgs] - only
                        implemented smgd (stochastic minibatch gradient
                        descent)


### Description

Hopefully the parameters are self-explanatory. This code contain and
implementation of multinomial logistic regression for the MNIST dataset. In
order to run this it is necessary that the file  mnist.pkl.gz be in the
working directory. This is implemented in Numpy/SciPy  purely. The Math is
taken mainly from Murphy book.  8-9% is reached. I had issues witht this code
at the beginning when I based my implementation on Bishop's book. After that
I had an issue with my implementation due to a missusage of the kron
operator.

