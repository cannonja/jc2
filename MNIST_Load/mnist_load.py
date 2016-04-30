import os
import struct
from PIL import Image
import numpy as np


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


def load_images(file_name, num_images, start=0):
    images = [] #Empty list to hold 2-D numpy arrays containing image data
    in_file = open(file_name, 'r+b') #input stream variable

    #Start at appropirate offset - 16 bytes for meta data
    #Each image is (28 * 28) bytes
    in_file.seek(16 + (start * 28 * 28)) 

    #Build image list element-wise where each element
    #is a 2-D array of image data
    for i in range(num_images):
        array = np.empty((28,28), 'uint8') #Array to hold one image's data
        for j in range(28):
            row_data = in_file.read(28)
            row_data = struct.unpack('>28B', row_data)
            array[j] = row_data
        images.append(array)

    in_file.close()        
    return images


def load_labels(file_name, num_images, start=0):
    labels = [] #Empty list to hold integers corresponding to image labels
    in_file = open(file_name, 'r+b') #input stream variable

    #Start at appropirate offset - 8 bytes for meta data
    #Each image is (28 * 28) bytes
    in_file.seek(8 + start) 

    #Build label list element-wise where each element
    #is an integer cooresponding to a label of an image
    for i in range(num_images):
        labels.append(struct.unpack('>1B', in_file.read(1)))
    in_file.close()        
    return labels


def save_images(image_list, output_path):
    for i in range(len(image_list)):
        im = Image.fromarray(image_list[i], mode='L')
        fp = output_path + str(i + 1) + ".png"
        im.save(fp)

    
