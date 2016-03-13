import imp
import os
import sys
from PIL import Image
import numpy as np
'''
#Big laptop
sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Rozell')
os.chdir('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
'''

#Little laptop
sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\Rozell')
os.chdir('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\MNIST_Load')

import mnist_load as mnist
import sparse_algo as sp
import r_network_class as lca

#############################Test thresh################################################################################
'''
u = np.zeros((5,1))
for i in range (5):
    u[i] = i
    
print (u)
network = lca.r_network(1)
for i in range(len(u)):
    u[i] = network.thresh(u[i], 2.5, 'H')
print (u)
'''


##############################Test return_sparse##########################################################################################

lamb = 1
tau = 10
delta = 0.05
u_stop = 0.002756
t_type = 'H'

#Match spreadsheet dictionary (single node case, lamb = 0)
'''
D = np.array([[3],[4]])
signal = np.array([[100],[15]])
'''

#Match spreadsheet dictionary (Multiple node case, lamb = 1)

signal = np.array(range(1,6))
D = np.ones((5,3))
D[:,0] = signal
D[:,1] = (7,0,32,5,7)
D[:,2] = (21,5,8,9,23)

'''
#Build list of files to iterate through - choose first file (t10k-images.idx3-ubyte)

files = os.listdir()
file_list = []
for i in files:    
    if (i.find("idx") != -1):
        file_list.append(i)


#Load MNIST dictionary and signal
signal_data = mnist.load_images(file_list[0], 1)
im = Image.fromarray(signal_data[0])
dict_data = mnist.load_images(file_list[0], 20, 20)

signal = signal_data[0].flatten()
D = sp.build_dictionary(dict_data)
'''


#Run Rozell and generate sparse code
network = lca.r_network(D)
network.set_stimulus(signal)
code = network.return_sparse(lamb, tau, delta, u_stop, t_type)
print (code)













##############################Built list of files to iterate through####################################################


##############################Load MNIST images#########################################################################

##Big laptop   
#output_path = "C:\\Users\\Jack2\\Google Drive\\URMP\\jc2\\MNIST_Load\\Images\\test_image_"
##Little laptop   
#output_path = "C:\\Users\\Jack\\Google Drive\\URMP\\jc2\\MNIST_Load\\Images\\test_image_"

#image_data = mnist.load_images(file_list[0], 5, 5)





