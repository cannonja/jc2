import numpy as np


'''
This function takes the spatial dimensions of the input image, the sqaure width of the filter,
the stride length, and the amount of zero-padding.

It calculates the spatial dimemsions of the convolved output and returns a nx4 numpy array where
each row represents the index of the output and contains the indices of the corners of the shape
of the input related to the output index in question.

The variable names come from the Stanford 231n course.

Args:
    w1 (int): input width
    h1 (int): input height
    f (int): filter square length
    s (int): stride length
    p (int): amount of zero-padding
    
Returns:
    shapes (np.array): 2-D array of shapes indices mapping to the input image (tl, tr, bl, br)

'''

def get_shapes(w1, h1, f, s = 1, p = 0):
    
    #Calulate spatial dimensions of output volume
    w2 = (w1 - f + 2 * p) / s + 1
    h2 = (h1 - f + 2 * p) / s + 1
         
    #Generate shapes array
    num_outputs = int(w2 * h2)
    shapes = np.empty((num_outputs, 4), dtype = int)
    
    for i in range(num_outputs):
        shapes[i, 0] = (i % w2) + (i // w2) * w1 * s
        shapes[i, 1] = shapes[i, 0] + f - 1
        shapes[i, 2] = shapes[i, 0] + w1 * (f - 1)
        shapes[i, 3] = shapes[i, 2] + f - 1
              
    return shapes




s1 = get_shapes(7, 7, 3)
s2 = get_shapes(7, 7, 3, 2)
              
    