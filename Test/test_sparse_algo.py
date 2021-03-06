import imp
import os
import sys
from PIL import Image
import numpy as np

##Big laptop
sys.path.append('C:\\Users\\Jack2\\Google Drive\\URMP\\jc2\\MNIST_Load')
##Little laptop
#sys.path.append('C:\\Users\\Jack\\Google Drive\\URMP\\jc2\\MNIST_Load')
import mnist_load as mnist
import sparse_algo as sp


##############################Test normalize and build_dictionary############################################################################

image_data = []
image_data.append(np.arange(9).reshape(3,3))
image_data.append(np.arange(9, 18).reshape(3,3))
image_data.append(np.arange(18, 27).reshape(3,3))
image_data.append(np.arange(27, 36).reshape(3,3))
image_data.append(np.arange(36, 45).reshape(3,3))
image_data.append(np.arange(45, 54).reshape(3,3))
image_data.append(np.arange(54, 63).reshape(3,3))
#image_data.append(np.array([[0,1,2],[3,4,5],[6,7,8]], dtype = 'int32'))
#image_data.append(np.array([[9,10,11],[12,13,14],[15,16,17]], dtype = 'int32'))
'''
#normalize
print(type(image_data))
print (image_data[0])
print (image_data[1])
normal, maximum, minimum  = sp.normalize(image_data[0])
print (maximum, minimum)
print (normal)

#build_dictionary
D = sp.build_dictionary(image_data)
print (D)
print (D.shape)


#least_squares
y = np.array([1,2,3])
D = image_data[0][:,1]
print (D.ndim)
print (D)
beta = sp.least_squares(D, y)
print (beta)
'''


##############################

D = sp.build_dictionary(image_data)
y = np.random.randint(1, 71, size = 10)
#print (D)
#print (y)
beta, indices = sp.choose_atoms(D, y)
print (beta)
print (indices)



##############################Built list of files to iterate through####################################################
'''
##Big laptop
#os.chdir("C:\\Users\\Jack2\\Google Drive\\URMP\\jc2\\MNIST_Load")
##Little laptop
os.chdir("C:\\Users\\Jack\\Google Drive\\URMP\\jc2\\MNIST_Load")

files = os.listdir()
file_list = []
for i in files:    
    if (i.find("idx") != -1):
        file_list.append(i)
'''

##############################Check load_images and save_images#########################################################

##Big laptop   
#output_path = "C:\\Users\\Jack2\\Google Drive\\URMP\\jc2\\MNIST_Load\\Images\\test_image_"
##Little laptop   
#output_path = "C:\\Users\\Jack\\Google Drive\\URMP\\jc2\\MNIST_Load\\Images\\test_image_"

#image_data = mnist.load_images(file_list[0], 5, 5)





