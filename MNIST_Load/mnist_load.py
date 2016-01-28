import os
import struct
from PIL import Image
import numpy as np

os.chdir("C:\\Users\\Jack2\\Google Drive\\Grad School\\URMP\\jc2\\MNIST_Load")
files = os.listdir()[2:]

file_name = files[0]

'''
def print_meta(file_name):
    in_file = open(file_name, 'r+b')
    
    if (file_name.find("image") != -1):
        meta_data = in_file.read(16)
        meta_data = struct.unpack('>4i', meta_data)
        meta_labels = ("Magic number", "Number of images", "Rows", "Columns")
    else:
        meta_data = in_file.read(8)
        meta_data = struct.unpack('>2i', meta_data)
        meta_labels = ("Magic number", "Number of items")        

    print ("File name:", in_file.name)
    for (i,j) in zip(meta_labels, meta_data):
        print (i, ": ", j, sep="")
    print ("\n")

    in_file.close()
'''

in_file = open(file_name, 'r+b')
in_file.seek(16)
array = np.empty((28,28), 'uint8')
images = np.zeros((28,28,10), 'uint8')
path = "C:\\Users\\Jack2\\Google Drive\\Grad School\\URMP\\jc2\\MNIST_Load\\Images\\test_image_"

for i in range(0,10):

    for j in range(0, 28):
        row_data = in_file.read(28)
        row_data = struct.unpack('>28B', row_data)
        array[j] = row_data

    images[:,:,i] = array

    
    im = Image.fromarray(images[:,:,i], mode='L')
    fp = path + str(i) + ".png"
    #fp = "test_image_" + str(i) + ".png"
    im.save(fp)
    


in_file.close()
