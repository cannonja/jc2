import numpy as np
import pandas
import matplotlib.pyplot as plt
import random


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
            self.connections.append(np.random.rand(self.layers[i+1], self.layers[i]))

    def set_input(self, data_in):
        if data_in.shape[0] != self.layers[0]:
            print ('Input data dim doesn\'t match network input layer dim')
            return
        self.input = np.array(data_in).reshape(data_in.shape[0], 1)
        self.input = np.append(self.input, 1)  #Add bias term
        self.activations.append(self.input.copy())

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sig(self, x):
        sig = sigmoid(x)
        return np.multiply(sig, (1 - sig))

    def forward_prop(self, label):
        self.D = []
        for i in range(len(self.connections)):
            z = np.dot(self.connections[i], self.activations[i])
            self.activations.append(self.sigmoid(z))
            self.D.append(np.diagflat(self.d_sig(z)))
        self.output = self.activations[-1].copy()
        self.error = self.output - label

    def back_prop(learn_rate):
        self.d = []
        self.d.append(np.dot(self.D[-1], self.error))
        for i in range(len(self.D) - 2, -1, -1):
            self.d.insert(0, np.dot(np.dot(self.D[i], self.connections[i]), self.d[0]))
        for i in range(len(self.d)):
            self.connections += -learn_rate * np.dot(d)
            
        
        

    

    
