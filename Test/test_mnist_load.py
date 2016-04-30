import imp
import os
import sys
from PIL import Image
import numpy as np
import socket

machine = socket.gethostname()
if (machine == 'Jack-PC'):
    #Big laptop
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Rozell')
    os.chdir('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    file_path = 'C:\\Users\\Jack2\\Desktop'
elif (machine == 'Tab'):
    #Little laptop
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\Rozell')
    os.chdir('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
else:
    #PSU machines (linux lab)
    base1 = os.path.expanduser('~/dev/jc2')
    base2 = os.path.expanduser('~/Desktop')
    sys.path.append(os.path.join(base1, 'MNIST_Load'))
    sys.path.append(os.path.join(base1, 'Rozell'))
    os.chdir(os.path.join(base1, 'MNIST_Load'))
    file_path = base2

import mnist_load as mnist


##############################Built list of files to iterate through####################################################
'''
files = os.listdir()
file_list = []
for i in files:    
    if (i.find("idx") != -1):
        file_list.append(i)
'''   

##############################Check print_meta function#################################################################
'''
for name in (file_list):
    mnist.print_meta(name)
'''


##############################Check load_images, load_labels, and save_images#########################################################

num_images = 20
start_pos = 0
image_file = 'train-images.idx3-ubyte'
label_file = 'train-labels.idx1-ubyte'
output_path = file_path + '\\Git_Repos\\jc2\\MNIST_Load\\Images\\test_image_'
mnist.print_meta(image_file)

image_data = mnist.load_images(image_file, num_images, start_pos)
label_data = mnist.load_labels(label_file, num_images, start_pos)
mnist.save_images(image_data, output_path)
print (label_data)



