import os
import struct
import sys
from PIL import Image
import numpy as np
import functools as ft

##Big laptop
sys.path.append('C:\\Users\\Jack2\\Google Drive\\URMP\\jc2\\MNIST_Load')
##Little laptop
#sys.path.append('C:\\Users\\Jack\\Google Drive\\URMP\\jc2\\MNIST_Load')
import mnist_load as mnist

#Function takes a 2-D numpy array and returns a tuple with a normalized 2-D numpy array
#and the associated max and min used to normalize
#Normalization method is (im_data[i,j] - min) / (max - min)
def normalize(im_data):
    normalized = np.zeros(im_data.shape)
    dmax = np.amax(im_data)
    dmin = np.amin(im_data)
    
    for i,j in zip(np.nditer(im_data), np.nditer(normalized, op_flags = ['readwrite'])):
        j[...] = (i - dmin) / (dmax - dmin)

    return (normalized, dmax, dmin)


'''
#Function takes 2-D numpy array and associated max/min, then returns un-normalized array
def denormalize(norm, dmax, dmin):
'''

#Function takes list of 2-D numpy arrays representing images
#Then flattens each image (converts to atom) and adds to dictionary
#Function returns a dictionary where each atom is an image
def build_dictionary(im_data):
    rows, cols = im_data[0].shape
    dictionary = np.zeros(((rows * cols), len(im_data)))
    
    for i in range(len(im_data)):
        dictionary[:, i] = im_data[i].flatten()

    return dictionary
        
#Function takes a 2-D numpy array (dictionary subset) and a 1-D numpy array (signal)
#It returns a 2-tuple containing the coefficient vector and the index vector
def least_squares(X, y):
    X_t = np.transpose(X) #Calculate X transpose
    X_t_X = np.dot(X_t, X)  #Calculate Xt * X
    X_t_X.astype(float)
    inv_XtX = np.linalg.inv(X_t_X)
    print ("X: ", X, X.shape)
    print ("y: ", y, y.shape)
    print ("X_t: ", X_t)
    print ("X_t_X: ", X_t_X)
    print ("inv(X_t_X): ", inv_XtX)
    print ("inv(X_t_X) * X_t: ", np.dot(inv_XtX, X_t))
    beta = ft.reduce(np.dot, [inv_XtX, X_t, y]) #Calculate solution
    #beta = np.linalg.inv(X_t_X).dot(X_t).dot(y)

    return beta

    

    
    
    
    
    
