import imp
import os
import sys
from PIL import Image
import numpy as np
import socket
import pandas
import matplotlib.pyplot as plt
import time
import datetime


machine = socket.gethostname()
if (machine == 'Jack-PC'):
    #Big laptop
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Rozell')
    os.chdir('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    file_path = 'C:\\Users\\Jack2\\Desktop'
    dict1_path = file_path + '/orig_dict.png'
    dict2_path = file_path + '/trained_dict.png'
    dict3_path = file_path + '/trained_data.csv'
    write_path = file_path + '/resid_data.csv'
    plot_path = file_path + '/resid_plot.png'
elif (machine == 'Tab'):
    #Little laptop
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\Rozell')
    os.chdir('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    file_path = 'C:\\Users\\Jack\\Desktop'
    dict1_path = file_path + '/orig_dict.png'
    dict2_path = file_path + '/trained_dict.png'
    dict3_path = file_path + '/trained_data.csv'
    write_path = file_path + '/resid_data.csv'
    plot_path = file_path + '/resid_plot.png'
else:
    #PSU machines (linux lab)
    base1 = os.path.expanduser('~/dev/jc2')
    base2 = os.path.expanduser('~/Desktop')
    sys.path.append(os.path.join(base1, 'MNIST_Load'))
    sys.path.append(os.path.join(base1, 'Rozell'))
    os.chdir(os.path.join(base1, 'MNIST_Load'))
    file_path = base2
    dict1_path = file_path + '/orig_dict.png'
    dict2_path = file_path + '/trained_dict.png'
    dict3_path = file_path + '/trained_data.csv'
    write_path = file_path + '/resid_data.csv'
    plot_path = file_path + '/resid_plot.png'


import mnist_load as mnist
import sparse_algo as sp
import r_network_class as lca


################### Set parameters ##############################################################
lamb = 1.
tau = 10.0
delta = 0.01
u_stop = 0.001
t_type = 'S'
alpha = 0.001
num_stims = 10
num_nodes = 50
num_dict_rows = 5
num_dict_cols = int(num_nodes / num_dict_rows)
num_stim_rows = 1
num_stim_cols = int(num_stims / num_stim_rows)
num_iterations =  3000
image_file = 't10k-images.idx3-ubyte'  #'train-images.idx3-ubyte'

##Load initial dictionary, either random or image
##Save out initial dictionary as image
#dict_data = np.random.rand(28,28)  #Use random init
training_data = mnist.load_images(image_file, num_stims)
dict_data =  mnist.load_images(image_file, num_nodes, num_stims)
for i in range(len(dict_data)):
    dict_data[i] = dict_data[i].astype(float)
    dict_data[i] /= 255.
for i in range(len(training_data)):
    training_data[i] = training_data[i].astype(float)
    training_data[i] /= 255.

D = sp.build_dictionary(dict_data)
T = sp.build_dictionary(training_data)

##For single stimulus case
'''
training_data = mnist.load_images(image_file, 1)[0]
og_im_data = training_data.copy()
og_im = Image.fromarray(og_im_data)
og_im.save(file_path + '/stim.png')
'''

##For multiple stims - use r_network methods to save out
stim_path = file_path + '/stim.png'
network_stim = lca.r_network(T)
network_stim.save_dictionary(num_stim_rows, num_stim_cols, stim_path)


#Initialize network dictionary, set parameters, then save orig dict
network = lca.r_network(D)
network.save_dictionary(num_dict_rows, num_dict_cols, dict1_path)
network.set_parameters(lamb, tau, delta, u_stop, t_type)

#Initiate x values and residual array for residual plot
x = range(num_iterations)
resid_plot = np.zeros((num_iterations))

#Train dictionary as each image is run through network
#Store length of residual vector in resid_plot array
for i in range(num_iterations):
    if (((i + 1) % 100) == 0):
        print("Iteration ", i + 1)
    for j in range(len(training_data)):
        network.set_stimulus(training_data[j].flatten(), True)
        network.generate_sparse(True)
        y = network.update_trained(alpha)
        resid_plot[i] = np.sqrt(np.dot(y,y))

#Save out trained dictionary as image
network.save_dictionary(num_dict_rows, num_dict_cols, dict2_path, True)

#Write residual data to csv file and plot
'''
df = pandas.DataFrame(np.column_stack((x, resid_plot)), columns = ['Iteration #', "Resid"])
df.to_csv(write_path, index = False)
'''
plt.plot(x, resid_plot, label = 'Raw')
plt.xlabel('Iteration')
plt.title('Reconstruction Error - ' + str(num_stims) + ' stims, '\
        + str(num_nodes) + ' nodes, alpha = ' + str(alpha))
plt.savefig(plot_path)
plt.show()
