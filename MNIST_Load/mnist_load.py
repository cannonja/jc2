import os
import struct

os.chdir("C:\\Users\\Jack2\\Google Drive\\Grad School\\URMP\\jc2\\MNIST_Load")
files = os.listdir()[1:]

file_name = files[0]


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

