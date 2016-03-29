import imp
import os
import sys
from PIL import Image
import numpy as np
import socket
import pandas

machine = socket.gethostname()
if (machine == 'Jack-PC'):
    #Big laptop
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Rozell')
    os.chdir('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
elif (machine == 'Tab'):
    #Little laptop
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\Rozell')
    os.chdir('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
else:
    #PSU machines (linux lab)
    base = os.path.expanduser('~/dev/jc2')
    sys.path.append(os.path.join(base, 'MNIST_Load'))
    sys.path.append(os.path.join(base, 'Rozell'))
    os.chdir(os.path.join(base, 'MNIST_Load'))


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

lamb = .1
tau = 10
delta = 0.001
u_stop = 10
t_type = 'S'
num_images = 1

#Match spreadsheet dictionary (single node case, lamb = 0)
'''
D = np.array([[3],[4]])
signal = np.array([[100],[15]])
'''

#Match spreadsheet dictionary (Multiple node case, lamb = 1)
'''
signal = np.array(range(1,6))
D = np.ones((5,3))
D[:,0] = signal
D[:,1] = (7,0,32,5,7)
D[:,2] = (21,5,8,9,23)

#Test scaling
network = lca.r_network(D)
network.set_stimulus(signal)
network.set_scale(255)
print(signal)
print(network.dictionary)
'''

#Build list of files to iterate through - choose first file (t10k-images.idx3-ubyte)

files = os.listdir()
file_list = []
for i in files:    
    if (i.find("idx") != -1):
        file_list.append(i)


#Load MNIST dictionary and signal
signal_data = mnist.load_images(file_list[0], num_images, 500)
#Insert stimulus in dictionary
#dict_data = mnist.load_images(file_list[0], 49, 20)
#D = np.append(signal_data[0].flatten().reshape(784,1), sp.build_dictionary(dict_data), axis = 1)
##Use regular dictionary
dict_data = mnist.load_images(file_list[0], 50, 20)
D = sp.build_dictionary(dict_data)

#Run Rozell and generate sparse code
network = lca.r_network(D)
network.set_parameters(lamb, tau, delta, u_stop, t_type)
error_names = ['Lambda', 'E(t)', 'Resid', 'Cost', 'Sparsity']
lambdas = np.arange(0.1, 2.1, 0.1)
#lambdas = [0.1]

#For each image, run Rozell and generate sparse code
for i in range(num_images):
    signal = signal_data[i].flatten()
    network.set_stimulus(signal)
    error = pandas.DataFrame() #DataFrame used for error table
    display = []  #List to hold rows of image data for grid
    #For each value of lambda, set lambda and run Rozell on the given image
    for j in lambdas:
        display_row = []   #List to hold one row of image data
        network.set_lambda(j)
        network.generate_sparse()  #Calculate sparse code        
        ##Add row of error data to error table
        row = pandas.DataFrame(network.return_error())
        error = error.append(row)
        ##Add list of dictionary elements scaled by coefficients to list
        indices = np.where(network.a > 0)
        coeff = network.a[indices]
        rfields = network.dictionary[:, indices]
        for k in range(len(coeff)):
            display_row.append(coeff[k] * rfields[k])
        display.append(display_row)
        


    error.columns = error_names
    print (error)
    
    ##Get max number of components for display grid dimensions
    biggest = 0
    for j in display:
        if (len(j) > biggest):
            biggest = len(j)
    print (biggest)

    ##Allocate pixels for display grid
    grid = np.zeros((28 * len(lambdas), 28 * (biggest + 2)))
    print(grid.shape)
    
    

'''
orig = network.s.reshape((28,28))
recon = np.dot(network.dictionary, network.a).reshape((28,28))
im_orig = Image.fromarray(orig)
im_recon = Image.fromarray(recon)

im_orig.show()
im_recon.show()
'''













##############################Built list of files to iterate through####################################################


##############################Load MNIST images#########################################################################

##Big laptop   
#output_path = "C:\\Users\\Jack2\\Google Drive\\URMP\\jc2\\MNIST_Load\\Images\\test_image_"
##Little laptop   
#output_path = "C:\\Users\\Jack\\Google Drive\\URMP\\jc2\\MNIST_Load\\Images\\test_image_"

#image_data = mnist.load_images(file_list[0], 5, 5)





