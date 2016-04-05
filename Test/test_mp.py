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
import sparse_algo as sp

##############################Built list of files to iterate through####################################################

##Big laptop
os.chdir("C:\\Users\\Jack2\\Google Drive\\URMP\\jc2\\MNIST_Load")
##Little laptop
#os.chdir("C:\\Users\\Jack\\Google Drive\\URMP\\jc2\\MNIST_Load")

files = os.listdir()
file_list = []
for i in files:    
    if (i.find("idx") != -1):
        file_list.append(i)

########################################################################################################################

file_path = file_list[0]
dictionary_data = mnist.load_images(file_path,50)
y = mnist.load_images(file_list[0], 1, 100)[0].flatten()
D = sp.build_dictionary(dictionary_data)
#print ("D.shape: ", D.shape)
approx = None

##Generate sparse code
current_beta = None
next_beta = None
indices = None
condition = True

while condition:
    current_beta, indices = sp.choose_atoms(D, y, indices)
    next_beta, indices = sp.choose_atoms(D, y, indices)
    print (current_beta.shape, next_beta.shape)
    #print ("current_beta: ", current_beta, type(current_beta))
    #print ("next_beta: ", next_beta, type(next_beta))
    #print (type(current_beta == next_beta))
    #print (current_beta == next_beta)
    #print ((current_beta == next_beta).all())
    if isinstance(current_beta == next_beta, bool):
        condition = not(current_beta == next_beta)
    else:
        condition = not((current_beta == next_beta).all())
    #print (condition)
    
   # print (condition, condition.all(), condition.any())

beta = current_beta
#print ("beta: ", beta, ", indices: ", indices)
#print ("D[:, indices].shape: ", D[:, indices].shape)
approx = np.dot(D[:, indices], beta)
#print ("type(approx): ", approx)

approx = approx.reshape(28,28)
y = y.reshape(28,28)

im.show(y, approx)


