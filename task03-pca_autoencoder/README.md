## Neural Network with one Hidden Layer for Classifying the MNIST dataset

Author: Miguel Cabrera <miguel.cabrera@tum.de>

 pca-autoencoder.py -h

     usage: pca-autoencoder.py [-h] [-n N] [--cifar] [--mnist]

     Generate scatter plot from CIFAR-10 and MNIST.

     optional arguments:
     -h, --help            show this help message and exit
     -n N, --no-samples N  Number of samples N to use, N = 1000 by default
     --cifar               generate from CIFAR-10 dataset
     --mnist               generate from MNIST dataset

    
### Description


Generates scatter plots using PCA of CIFAR-10 and MNIST dataset. The auto-encoder is not ready yet, therefore
not included. At least one of  --cifar or --mnist options should be
indicated. For that is necessary that   *mnist.pkl.gz* for MNIST; and
*batches.meta* and *data_batch_1* be in the working directory.
