import imp
import os
import sys
from PIL import Image
import numpy as np

#Big laptop
sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
#Little laptop
#sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
import mnist_load as mnist
import sparse_algo as sp



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

##############################Load MNIST images#########################################################################

##Big laptop   
#output_path = "C:\\Users\\Jack2\\Google Drive\\URMP\\jc2\\MNIST_Load\\Images\\test_image_"
##Little laptop   
#output_path = "C:\\Users\\Jack\\Google Drive\\URMP\\jc2\\MNIST_Load\\Images\\test_image_"

#image_data = mnist.load_images(file_list[0], 5, 5)





