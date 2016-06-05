import os
import struct
import sys
from PIL import Image
import numpy as np
import functools as ft
#import scipy.optimize as opt
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
    rows, cols, _ = im_data[0].shape
    dictionary = np.zeros(((rows * cols), len(im_data)))
    
    for i in range(len(im_data)):
        dictionary[:, i] = im_data[i].flatten()

    return dictionary
        
#Function takes a 2-D numpy array (dictionary subset) and a 1-D numpy array (signal)
#It returns a 2-tuple containing the coefficient vector and the index vector
def least_squares(X, y):
    if X.ndim == 1:
        s1 = np.dot(X,np.transpose(X))
        s2 = np.dot(y, np.transpose(X))
        beta = s2 / s1
    else:
        X_t = np.transpose(X) #Calculate X transpose
        X_t_X = np.dot(X_t, X)  #Calculate Xt * X
        X_t_X.astype(float)
        inv_XtX = np.linalg.inv(X_t_X)
        '''
        print ("In least_squares,,,")
        print ("X: ", X, X.shape)
        print ("y: ", y, y.shape)
        print ("X_t: ", X_t)
        print ("X_t_X: ", X_t_X)
        print ("inv(X_t_X): ", inv_XtX)
        print ("inv(X_t_X) * X_t: ", np.dot(inv_XtX, X_t))
        '''
        
        beta = ft.reduce(np.dot, [inv_XtX, X_t, y]) #Calculate solution
        #beta = np.linalg.inv(X_t_X).dot(X_t).dot(y)

    return beta

#Function takes full dictionary (2-D numpy array), a list of indices,
#and signal to approximate (1-D numpy array)
def choose_atoms(D, y, index = None):
    dmin = None #Initialize minimum variable
    
    if index is None:
        for i in range(D.shape[1]):
            #print (D[:,i])
            #beta = least_squares(D[:,i], y)


            ##Using nnls - constrained betas
            A = np.transpose(np.matrix(D[:, i]))
            y = np.transpose(y)
            print(A.shape, y.shape, type(y))
            beta = opt.nnls(A, y)[0]
            error = np.dot(A, beta) - y
            


            #Calc RMSE
            #error = np.dot(D[:,i], beta) - y
            MSE = np.mean(np.square(error))
            RMSE = np.sqrt(MSE)

            if (dmin is None) or (RMSE < dmin):
                dmin = RMSE
                min_index = [i]
    else:
        for i in range(D.shape[1]):
            print (i not in index)
            if i not in index:
                mod_index = index + [i]
                #beta = least_squares(D[:,mod_index], y)


                ##Using nnls - constrained betas
                A = np.transpose(np.matrix(D[:, i]))
                y = np.transpose(y)
                print(A.shape, y.shape, type(y))
                beta = opt.nnls(A, y)[0]
                error = np.dot(A, beta) - y


                #Calc RMSE
                #error = np.dot(D[:,mod_index], beta) - y
                MSE = np.mean(np.square(error))
                RMSE = np.sqrt(MSE)

                if (dmin is None) or (RMSE < dmin):
                    dmin = RMSE
                    min_index = mod_index

    if (beta is None) or (min_index is None):
        return (None, None)
    else:
        return (beta, min_index)
    
    

    
    
    
    
    
