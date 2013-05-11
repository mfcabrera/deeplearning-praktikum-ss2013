#!/usr/bin/env python

from __future__ import division
import numpy as np
import cPickle,gzip
import matplotlib.pyplot as plt
from itertools import product
from sklearn.utils import shuffle
import argparse
dot = np.dot

def load_mnist():
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    return train_set
    f.close()

def rgb2gray(RGBimgs):
    GRAYimgs = np.empty(( len(RGBimgs), 1024) )
    for i in xrange(len(RGBimgs)):
        rgb = RGBimgs[i, :].reshape(3, 32, 32)
        r, g, b = np.rollaxis(rgb[:3, ...], axis = 0)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        GRAYimgs[i, :] = gray.flatten()
    return GRAYimgs



class PCA:
    def __init__( self,npc = 2,  fraction=0.90 ):
        assert 0 <= fraction <= 1
        
        
        self.fraction = fraction
        self.npc = npc
       

    def pc( self ):
        """ e.g. 1000 x 2 U[:, :npc] * d[:npc], to plot etc. """
        print self.npc
        n = self.npc
        return self.U[:, :n] * self.d[:n]


    def fit_transform(self,data):
        
        
        centered = data.T - np.mean(data.T, axis = 1)[:, np.newaxis]
        sigma = np.cov(centered, bias = 1)

        U,s,V = np.linalg.svd(sigma)
        
        return dot(data, U[:,0:self.npc])  

    
    def center( self, A, axis=0, scale=True, verbose=1 ):
        self.mean = A.mean(axis=axis)
        if verbose:
            print "Center -= A.mean:", self.mean
        A -= self.mean
        if scale:
            std = A.std(axis=axis)
            self.std = np.where( std, std, 1. )
            if verbose:
                print "Center /= A.std:", self.std
            A /= self.std
        else:
            self.std = np.ones( A.shape[-1] )
        self.A = A

    def uncenter( self, x ):
        return np.dot( self.A, x * self.std ) + np.dot( x, self.mean )


def gen_scatter_mnist(no_samples=1000):
    """
    Generate a scatter plot of mnist
    """
    print "Reading MNIST ..." ,
    training = load_mnist()
   
    print "MNIST read!" ,
  
    pca = PCA(npc=2)

    #x_transformed = p.fit_transform()
    #print pc.shape
    
    
    
    X_train, y_train = training[0][:no_samples] / 255., training[1][:no_samples]
    #print X_train.shape
    #print y_train.shape
 
    X_train, y_train = shuffle(X_train, y_train)
    
    fig, plots = plt.subplots(10, 10)
    fig.set_size_inches(50, 50)
    plt.prism()
    for i, j in product(xrange(10), repeat=2):
        if i > j:
            continue
      
        X_ = X_train[(y_train == i) + (y_train == j)]
        y_ = y_train[(y_train == i) + (y_train == j)]

        #print X_.shape
        
        X_transformed = pca.fit_transform(X_)
        plots[i, j].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
        plots[i, j].set_xticks(())
        plots[i, j].set_yticks(())
  
        plots[j, i].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
        plots[j, i].set_xticks(())
        plots[j, i].set_yticks(())
        if i == 0:
            plots[i, j].set_title(j)
            plots[j, i].set_ylabel(j)
            
    
        #plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
    plt.tight_layout()
    plt.savefig("mnist_pairs.png")
    print "Please check mnist_pairs.png. Vielen Dank und Aufwiedersehen."


def gen_scatter_cifar ( no_samples = 50, gray_images = True, ch = 0 ): 
    import sys
    assert (ch == 0) or (ch == 1) or (ch == 2), 'Selected RGB not valid! (0, 1 or 2)'
    print 'Reading CIFAR dataset...'
    f = open('data_batch_1')
    l = open('batches.meta')
    dict = cPickle.load(f)
    labels_names = cPickle.load(l)
    f.close()
    l.close()
    print 'CIFAR ...dataset read!'
    
    imgs = dict['data']
    labels = np.asarray( dict['labels'] )
    assert no_samples<len(imgs), 'CIFAR: too many samples selected!'
    imgs, labels = shuffle(imgs, labels)
    imgs, labels = imgs[:no_samples], labels[:no_samples]
    
    labels_names = labels_names['label_names']
    
    if  (gray_images): 
        imgs = rgb2gray(imgs)
    
    pca = PCA(npc=2)
    fig, plots = plt.subplots(10, 10)
    
    count = 0


    sys.stdout.write('The output is going to be AWE...wait for it...')
    for i, j in product(xrange(10), repeat = 2):
        if i > j:
            continue
        count += 1
        sys.stdout.write('.')
        sys.stdout.flush()
        #print 'CIFAR:', (count*100)/55, '% complete...'

        if not(gray_images):
            x = imgs[(labels == i) + (labels == j), (1024*ch):(1024*(ch+1))]
        else:
            x = imgs[(labels == i) + (labels == j)]
        y = labels[(labels == i) + (labels == j)]
        
        x_rotated = pca.fit_transform(x)
        
        plots[i, j].scatter(x_rotated[:, 0], x_rotated[:, 1], c = y)
        plots[i, j].set_xticks(())
        plots[i, j].set_yticks(())
      
        plots[j, i].scatter(x_rotated[:, 0], x_rotated[:, 1], c = y)
        plots[j, i].set_xticks(())
        plots[j, i].set_yticks(())
        
        if i == 0:
            plots[i, j].set_title(labels_names[j])
            plots[j, i].set_ylabel(labels_names[j])
    
    plt.suptitle('CIFAR Dataset')
    plt.savefig("cifar_plot.png")
    #plt.show()    
    sys.stdout.write('SOOOOOME: please check cifar_plot.png!')


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description='Generate scatter plot from CIFAR-10 and MNIST.')
    parser.add_argument('-n','--no-samples', metavar='N', type=int,default=1000,
                   help='Number of samples N to use, N = 1000 by default')

    parser.add_argument('--cifar',help='generate from CIFAR-10 dataset',action='store_true')
    parser.add_argument('--mnist',help='generate from MNIST dataset',action='store_true')

    args = parser.parse_args()
    #print args

    if not (args.cifar or args.mnist):
        parser.error('At least one dataset shoud be selected, selected, add --cifar or --mnist')

    if(args.cifar):
        gen_scatter_cifar(args.no_samples)
    if(args.mnist):
        gen_scatter_mnist(args.no_samples)
    
        
