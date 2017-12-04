# -*- coding: utf-8 -*-
"""
Created on Sun Oct 1 14:41:27 2017

@author: Lisa Tostrams
Class for building a MultiLayer Perceptron with one hidden layer, that mimimizes the MSE

"""


import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, X,y,binary=True):
        """MLP(X,y), builds 3 layer MLP on data X using labels y, minimizing the Mean Square Error:
            MSE(X,y,W) = 1/N * sum_i (W_o*(W_h*X_i) - y)^2
        optional argument:
            binary -> whether or not output should be binary instead of continuous"""
        # add bias term
        self.X = np.vstack([np.ones([1,X.shape[1]]),X])
        self.y=y
        self.binary=binary
        self.done=False
        
    def sigmoid(self,x):
        """ Sigmoid(x), returns value and gradient; only implemented sigmoid activation due to convenient grad"""
        s = 1.0 / (1 + np.exp(-x))
        grad_s = s * (1 - s)
        return s, grad_s
    
    def class_error(self,o):
        """ class_error(y_hat), returns the misclassification rate using output o to label data points compared to y """
        if(self.binary):
            y_hat = o>np.mean(o)
        else:
            y_hat=o
        return 1-np.mean((self.y==y_hat).flatten())
    
    def forwardprop(self,W_h, W_o, X):
        """forwardprop(Wh,Wo,X), forward propagation of X through hidden layer (W_h) and output layer (W_o)
        output:
            y_hat -> label estimate when done
        else:
            h -> output of hidden units
            o -> output of output units
            grad_h -> gradient of the hidden units wrt weights 
            grad_o -> gradient of the output units wrt weights  """
        activation_h = np.dot(W_h, X)
        h, grad_h = self.sigmoid(activation_h) 
        activation_o = np.dot(W_o, h)    
        o, grad_o = self.sigmoid(activation_o)
        if(self.binary and self.done):
            return o>0.5
        return h, o, grad_h, grad_o
    
    def backprop(self,h,o, grad_h, grad_o,W_o,X):
        """ backprop(h,o,grad_h,grad_o,W_o,X), propagates error of output o compared to labels y back through layers,
                    computes gradient in objective function MSE(X,y,W) wrt weights in both layers
        output:
            grad_E_h -> gradient of MSE(X,y,W) wrt W_h
            grad_E_o -> gradient of MSE(X,y,W) wrt W_o   """
        error_output = (o - self.y) * grad_o     
        error_hidden = grad_h *(np.dot(W_o.T, error_output))
        grad_E_o = np.dot(error_output, h.T)     
        grad_E_h = np.dot(error_hidden, X.T)
        return grad_E_h, grad_E_o
    
    def learn_weights(self,nhidden=1,nepochs=2000,eta=0.01):
        """ learn_weights(), learns the weights in each layer
        optional arguments:
            nhidden -> number of hidden units
            nepochs -> number of learning steps
            eta -> learning rate 
        """  
        self.done=False
        ninput = self.X.shape[0]
        noutput = self.y.shape[0] 
        W_h = np.random.uniform(-3, 3, [nhidden,ninput])
        W_o = np.random.uniform(-3, 3, [noutput,nhidden])
        for epoch in xrange(0,nepochs):
            h,o,grad_h,grad_o = self.forwardprop(W_h, W_o, self.X)
            if(epoch%500==0):
                print('Iteration: {} / {} ; misclassication rate: {:.4f}'.format(epoch,nepochs,self.class_error(o))) 
            grad_E_h, grad_E_o = self.backprop(h, o, grad_h, grad_o,W_o, self.X)
            W_h = W_h - eta * grad_E_h                        
            W_o = W_o - eta * grad_E_o                                                                          

        print('Final misclassification rate: {:.4f}'.format(self.class_error(self.forwardprop(W_h, W_o, self.X)[1])))
        self.forwardprop(W_h, W_o, self.X)[1]
        self.done=True
        return W_h,W_o,o       
        
    
    def plot_boundaries(self,W1,W2,Data):
        """ plot_boundries(Wh,Wo,X), plots the decision boundries of the MLP with weights Wh and Wo and how data X is placed
            kind of a weird function to have user input instead of using internal variables but want to leave something to do for users"""
        self.done=False
        tmp = np.vstack([np.ones([1,Data.shape[1]]),Data])
        o = self.forwardprop(W1,W2,tmp)[1]
        x0 = np.arange(min(Data[0,:])-0.2, max(Data[0,:])+0.2, 0.1)
        x1 = np.arange(min(Data[1,:])-0.2, max(Data[1,:])+0.2, 0.1)
        xx, yy = np.meshgrid(x0, x1, sparse=False)
        space = np.asarray([xx.flatten(),yy.flatten()])
        space = np.vstack([np.ones([1,space.shape[1]]),space])
        z = self.forwardprop(W1,W2,space)[1]
        self.done=True
        y_hat = o>np.mean(o)
        plt.scatter(Data[0,y_hat.flatten()],Data[1,y_hat.flatten()])
        plt.scatter(Data[0,(y_hat==False).flatten()],Data[1,(y_hat==False).flatten()])
        h = plt.contourf(x0,x1,np.reshape(z,[len(x0),len(x0)]),levels=[0,np.mean(o),1],colors=('orange','b'), alpha=0.1)

