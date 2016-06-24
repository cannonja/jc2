import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pdb


class ff_net:
    '''Class represents a generalized, fully-connected neural network using
       the frame work discussed in Rojas Chpt. 7'''

    def __init__(self, layers):
        self.layers = layers   #List of layer sizes
        self.input = None      #Input layer - 'o' vector in Rojas
        self.output = None     #Output layer
        self.e = None          #Error for current data point
        self.activations = []  #list of activation values throughout each forward prop
        self.W_bar = []        #list of connection matrices (num layers - 1)
        self.D = []            #list of gradient matrices
        self.d = []            #list of backpropagated erros
        self.rmse = np.array([])  #variable to hold data for plotting rmse during training
        self.learn_rate = 1.0   #Learning rate
        self.decay_rate = 1.0  #Learning rate decay
        self.lr_decay = np.array([])  #variable to hold data for plotting learning rate decay
    
        for i in range(len(layers) - 1):
            self.W_bar.append(np.random.randn(self.layers[i] + 1, self.layers[i+1]))




    def clear_rmse(self):
        self.rmse = np.array([])

    def set_lr_stats(self, learn_rate, decay_rate = 1.0):
        self.learn_rate = learn_rate
        self.decay_rate = decay_rate

    def decay(self):
        self.learn_rate *= self.decay_rate

    def plot_lr_decay(self, path = None):
        if self.lr_decay.shape == (0,):
            print ('ERROR: No Learning-rate decay data')
            return

        plt.figure()
        plt.plot(self.lr_decay)
        plt.xlabel('Number of training examples seen')
        plt.title('Learning-rate Decay During Backprop')
        if path is None:
            plt.show()
        else:
            plt.savefig(path)


    def plot_rmse(self, win1 = 100, win2 = 500, path = None):
        if self.rmse.shape == (0,):
            print ('ERROR: No RMSE data')
            return
        df = pd.DataFrame(self.rmse)
        ma1 = df.iloc[:,0].rolling(window = win1).mean().values
        ma2 = df.iloc[:,0].rolling(window = win2).mean().values

        plt.figure()
        plt.plot(self.rmse, color = 'gray', alpha = 0.6, label = 'Raw')
        plt.plot(ma1,  color = 'red', label = 'MA - ' + str(win1) + ' periods')
        plt.plot(ma2,  color = 'blue', label = 'MA - ' + str(win2) + ' periods')
        plt.xlabel('Number of examples seen')
        plt.ylabel('RMSE')
        #plt.title('RMSE During Backprop')
        plt.legend()
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
            
            



    def set_input(self, data_in):
        self.activations = []
        # If input is a row vector, assign to class and add bias term for activation
        if data_in.shape == (1, self.layers[0]):
            self.input = data_in
        elif data_in.shape == (self.layers[0], 1):
            self.input = data_in.T
        else:
            print ('Input data dim doesn\'t match network input layer dim')
            return
        self.activations.append(np.hstack((self.input, np.array([[1]]))))  #Add bias term



    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))



    def d_sig(self, x):
        sig = self.sigmoid(x)
        return np.multiply(sig, (1 - sig))



    def forward_prop(self, label):
        if self.input is None:
            print ('No input data')
            return
        ## Loop through all but last activation as last activation
        ## excludes bias term
        self.D = []
        for i in range(len(self.W_bar) - 1):
            z = np.dot(self.activations[i], self.W_bar[i])
            self.activations.append(np.hstack((self.sigmoid(z), np.array([[1]]))))  #Add bias term
            self.D.append(np.diagflat(self.d_sig(z)))  #Exclude bias term in gradients
        z = np.dot(self.activations[-1], self.W_bar[-1])
        self.activations.append(self.sigmoid(z))  #Exclude bias term for output activation
        self.D.append(np.diagflat(self.d_sig(z))) #Exclude bias term in gradients

        self.output = self.activations[-1].copy()
        self.e = self.output.T - label    # e is supposed to be a column vector
        self.rmse = np.append(self.rmse, np.sqrt(np.dot(self.e.T, self.e)))



    def back_prop(self):
        if self.output is None:
            print ('No output data')
            return

        self.d = []
        self.d.append(np.dot(self.D[-1], self.e))
        if len(self.D) > 1:
            for i in range(len(self.D) - 2, -1, -1):
                self.d.insert(0, np.dot(np.dot(self.D[i], self.W_bar[i+1][:-1,:]), self.d[0]))
        for i in range(len(self.d)):
            self.W_bar[i] += (-self.learn_rate * np.dot(self.d[i], self.activations[i])).T

        self.lr_decay = np.append(self.lr_decay, self.learn_rate)




    

    
