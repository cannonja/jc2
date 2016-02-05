import os
import struct
import sys
from PIL import Image
import numpy as np

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
        
    
    
    
    
    
        
    
    
    
    
