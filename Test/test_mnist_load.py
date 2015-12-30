import imp
import os

##Import module in module object
'''
##Home machine import
in_module = "C:\\Users\\Jack2\\Google Drive\\Grad School\\URMP\\jc2\\
                MNIST_Load\\mnist_load.py"
module = imp.load_source('mnist_load', in_module)
'''


##tlab machine import
in_module = "../MNIST_Load/mnist_load.py"
module = imp.load_source('mnist_load', in_module)


##Built list of files to iterate through
os.chdir("../MNIST_Load")
files = os.listdir("../MNIST_Load")
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
'''
##Home machine
output_path = "C:\\Users\\Jack2\\Google Drive\\Grad School\\URMP\\jc2\\
                  MNIST_Load\\Images\\test_image_"
'''

##tlab machine
output_path = "/u/jc2/Desktop/test_image_"
module.save_images(file_list[0], output_path, 10)



