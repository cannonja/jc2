import numpy as np
import pandas
import matplotlib.pyplot as plt
import random
import pdb


class ff_net:
    'Class represents a generalized, fully-connected neural network'

    def __init__(self, layers):
        self.layers = layers   #List of layer sizes
        self.input = None      #Input layer
        self.output = None     #Output layer
        self.error = None      #Error for current data point
        self.activations = []  #list of activation values throughout each forward prop
        self.connections = []  #list of connection matrices (num layers - 1)
        self.D = []            #list of gradient matrices
        self.d = []            #list of 
        

        for i in range(len(layers) - 1):
            ## Extra column is added to "from" layer to account for bias weights 
            self.connections.append(np.random.rand(self.layers[i+1], self.layers[i] + 1))



    def set_input(self, data_in):
        if data_in.shape[0] != self.layers[0]:
            print ('Input data dim doesn\'t match network input layer dim')
            return
        self.input = np.array(data_in).reshape(data_in.shape[0], 1)
        self.activations.append(np.vstack((self.input, 1)))  #Add bias term



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
        for i in range(len(self.connections) - 1):
            z = np.dot(self.connections[i], self.activations[i])
            self.activations.append(np.vstack((self.sigmoid(z), 1)))  #Add bias term
            self.D.append(np.diagflat(self.d_sig(z)))  #Exclude bias term in gradients
        z = np.dot(self.connections[-1], self.activations[-1])
        self.activations.append(self.sigmoid(z))  #Exclude bias term for output activation
        self.D.append(np.diagflat(self.d_sig(z))) #Exclude bias term in gradients
        
        self.output = self.activations[-1].copy()
        self.error = self.output - label




    def back_prop(self, learn_rate):
        if self.output is None:
            print ('No output data')
            return

        pdb.set_trace()
        self.d = []
        self.d.append(np.dot(self.D[-1], self.error))
        if len(self.D) > 1:
            for i in range(len(self.D) - 2, -1, -1):
                self.d.insert(0, np.dot(np.dot(self.D[i], self.connections[i][:, :-1]), self.d[0]))
        for i in range(len(self.d)):
            self.connections[i] += -learn_rate * np.dot(self.d[i], self.activations[i][:-1].T)
            
        
        

    

    
