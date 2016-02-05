import imp
import os
import sys
from PIL import Image
import numpy as np
<<<<<<< HEAD

##Big laptop
sys.path.append('C:\\Users\\Jack2\\Google Drive\\URMP\\jc2\\MNIST_Load')
##Little laptop
#sys.path.append('C:\\Users\\Jack\\Google Drive\\URMP\\jc2\\MNIST_Load')
import mnist_load as mnist
import sparse_algo as sp


##############################Built list of files to iterate through####################################################
'''
##Big laptop
os.chdir("C:\\Users\\Jack2\\Google Drive\\URMP\\jc2\\MNIST_Load")
##Little laptop
#os.chdir("C:\\Users\\Jack\\Google Drive\\URMP\\jc2\\MNIST_Load")

files = os.listdir()
file_list = []
for i in files:
    if (i.find("idx") != -1):
        file_list.append(i)
'''


##############################Test normalize############################################################################

image_data = []
image_data.append(np.array([[0,1,2],[3,4,5],[6,7,8]], dtype = 'int32'))
image_data.append(np.array([[9,10,11],[12,13,14],[15,16,17]], dtype = 'int32'))

print(type(image_data))
print (image_data[0])
print (image_data[1])
normal, maximum, minimum  = sp.normalize(image_data[0])
print (maximum, minimum)
print (normal)



=======
##Big laptop
#sys.path.append('C:\\Users\\Jack2\\Google Drive\\URMP\\jc2\\MNIST_Load')

##Little laptop
sys.path.append('C:\\Users\\Jack\\Google Drive\\URMP\\jc2\\MNIST_Load')

import mnist_load as mnist

##############################Import module in module object############################################################

'''
##Big laptop import
module = imp.load_source('mnist_load',
                         'C:\\Users\\Jack2\\Google Drive\\URMP\\jc2\\MNIST_Load\\mnist_load.py')
'''
'''
##Little laptop import
module = imp.load_source('mnist_load',
                         'C:\\Users\\Jack\\Google Drive\\URMP\\jc2\\MNIST_Load\\mnist_load.py')
'''
##tlab machine import
#module = imp.load_source('mnist_load', '/u/jc2/dev/jc2/Image_Class/image_class.py')

##############################Built list of files to iterate through####################################################

##Big laptop
#os.chdir("C:\\Users\\Jack2\\Google Drive\\URMP\\jc2\\MNIST_Load")
##Little laptop
os.chdir("C:\\Users\\Jack\\Google Drive\\URMP\\jc2\\MNIST_Load")

files = os.listdir()
file_list = []
for i in files:    
    if (i.find("idx") != -1):
        file_list.append(i)
    

##############################Check print_meta function#################################################################

for name in (file_list):
    mnist.print_meta(name)



##############################Check load_images and save_images#########################################################
>>>>>>> 5b041cb3cc41196187a74da63c6ff80b2a3bbb7d

##Big laptop   
#output_path = "C:\\Users\\Jack2\\Google Drive\\URMP\\jc2\\MNIST_Load\\Images\\test_image_"
##Little laptop   
<<<<<<< HEAD
#output_path = "C:\\Users\\Jack\\Google Drive\\URMP\\jc2\\MNIST_Load\\Images\\test_image_"

#image_data = mnist.load_images(file_list[0], 5, 5)




=======
output_path = "C:\\Users\\Jack\\Google Drive\\URMP\\jc2\\MNIST_Load\\Images\\test_image_"

image_data = mnist.load_images(file_list[0], 5, 5)
mnist.save_images(image_data, output_path)
>>>>>>> 5b041cb3cc41196187a74da63c6ff80b2a3bbb7d



