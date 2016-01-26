import imp
import os
from PIL import Image
import numpy as np

##############################Import module in module object############################################################

'''
##Big laptop import
module = imp.load_source('mnist_load',
                         'C:\\Users\\Jack2\\Google Drive\\URMP\\jc2\\MNIST_Load\\mnist_load.py')
'''
##Little laptop import
module = imp.load_source('mnist_load',
                         'C:\\Users\\Jack\\Google Drive\\URMP\\jc2\\MNIST_Load\\mnist_load.py')
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
'''
for name in (file_list):
    module.print_meta(name)
'''


##############################Check load_images and save_images#########################################################

##Big laptop   
#output_path = "C:\\Users\\Jack2\\Google Drive\\URMP\\jc2\\MNIST_Load\\Images\\test_image_"
##Little laptop   
output_path = "C:\\Users\\Jack\\Google Drive\\URMP\\jc2\\MNIST_Load\\Images\\test_image_"

image_data = module.load_images(file_list[0], 5, 5)
module.save_images(image_data, output_path)



