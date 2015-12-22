import os
import struct

os.chdir("C:\\Users\\Jack2\\Google Drive\\Grad School\\URMP\\MNIST")
files = os.listdir()[2:]

file = open(files[0], 'r+b')
'''
magic_num = file.read(4)
magic_num = struct.unpack('>i', magic_num)
num_images = file.read(4)
num_images = struct.unpack('>i', num_images)
num_rows = file.read(4)
num_rows = struct.unpack('>i', num_rows)
num_col = file.read(4)
num_col = struct.unpack('>i', num_col)
'''

meta = file.read(16)
meta = struct.unpack('>4i', meta)

'''
print ("File name:", file.name)
print ("Magic number:", magic_num[0])
print ("Number of images:", num_images[0])
print ("Rows:", num_rows[0])
print ("Columns:", num_col[0])
'''

'''
print ("\nFirst Image:")
for i in range(0,28):
    pixel = file.read(28)
    print ("Row ", i + 1, ": ", pixel, sep="")
'''

file.close()

