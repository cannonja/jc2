import numpy as np
import pandas as pd
import r_network_class as lca


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sig(x):
    sig = sigmoid(x)

    return np.multiply(sig, (1 - sig))

def forward_prop(weights, input_data, label_data):
    #Integrate weights and sparse code, feed to sigmoid
    #Calc gradient matrix D - only one given the architecture
    z = np.dot(weights, input_data)
    activation = sigmoid(z)[:, np.newaxis]
    gradients = d_sig(activation)
    D = np.diagflat(gradients)

    #Convert ouptut vector to binary - max value set to one
    #All others set to zero
    prediction = np.where(activation == activation.max())[0][0]
    label = np.zeros((10,1))
    label[label_data] = 1

    error = activation - label

    return (prediction, error, D)


def back_prop(D, sparse, error, learn_rate):
    d = np.dot(D, error)
    update = -learn_rate * np.dot(d, sparse[np.newaxis, :])

    return update





