import imp
import os
from PIL import Image
import numpy as np

##Import module in module object

##Home machine import
module = imp.load_source('mnist_load',
                         'C:\\Users\\Jack2\\Google Drive\\Grad School\\URMP\\jc2\\MNIST_Load\\mnist_load.py')
##tlab machine import
#module = imp.load_source('mnist_load', '/u/jc2/dev/jc2/Image_Class/image_class.py')

##Built list of files to iterate through
os.chdir("C:\\Users\\Jack2\\Google Drive\\Grad School\\URMP\\jc2\\MNIST_Load")
files = os.listdir()
file_list = []
for i in files:    
    if (i.find("idx") != -1):
        file_list.append(i)
    
'''
##Check print_meta function
for name in (file_list):
    module.print_meta(name)
'''

##Check save_images
output_path = "C:\\Users\\Jack2\\Google Drive\\Grad School\\URMP\\jc2\\MNIST_Load\\Images\\test_image_"
image_data = module.load_images(file_list[0], 10)
print ("Instead, the function returns this:\n")
for i in range(len(image_data)):
    print ("images[", i, "]: ", image_data[i][15][:15])
    


#im = Image.fromarray(image_data[1], mode='L')
#im.show()




#module.save_images(image_data, output_path)



