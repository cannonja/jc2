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
import pdb


machine = socket.gethostname()
if (machine == 'Jack-PC'):
    #Big laptop
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Rozell')
    os.chdir('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    file_path = 'C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Test\\DB Classifier\\Overnight run'
    dict1_path = file_path + '/orig_dict.png'
    dict2_path = file_path + '/trained_dict.png'
    dict3_path = file_path + '/trained_data.csv'
    write_path = file_path + '/resid_data.csv'
    plot_path = file_path + '/resid_plot2.png'
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
    file_path = base1 + '/Test/DB Classifier/Overnight run'
    dict1_path = file_path + '/orig_dict.png'
    dict2_path = file_path + '/trained_dict.png'
    dict3_path = file_path + '/trained_data.csv'
    write_path = file_path + '/resid_data.csv'
    plot_path = file_path + '/resid_plot.png'

import mnist_load as mnist
import sparse_algo as sp
import r_network_class as lca


#################################### Set Rozell network parameters ##############################################################
lamb = 1.0
tau = 10.0
delta = 0.01
u_stop = 0.001
t_type = 'S'
alpha = 0.85
alpha_decay = 1
alpha_decay_rate = 0.98
alpha_decay_iters = 100
num_rfields = 50
num_images =  3000      #60000 - num_rfields
image_file = 'train-images.idx3-ubyte'    #'t10k-images.idx3-ubyte'

#Plotting parameters
win1 = 100  #Window for mov avg 1
win2 = 500 #Window for mov avg 2


######################## Preprocess data - dictionary gets loaded with first num_rfields MNIST images ############################
#################################### Training data starts loading after dictionary images ########################################
dict_data = mnist.load_images(image_file, num_rfields)
training_data = mnist.load_images(image_file, num_images, num_rfields)
dict_data = [np.array(i, dtype=float) / 255 for i in dict_data]
training_data = [np.array(i, dtype=float) / 255 for i in training_data]



######################################### Initialize network dictionary and parameters ###########################################
D = sp.build_dictionary(dict_data)
network = lca.r_network(D)
network.load_ims(training_data)
network.set_parameters(lamb, tau, delta, u_stop, t_type)
network.set_dim(dict_data[0].shape)
#Save out the original dictionary
network.save_dictionary(5, 10, dict1_path, line_color = 255)




#################################################### Train dictionary ############################################################
network.set_alpha(alpha)
network.train(alpha_decay, alpha_decay_rate, alpha_decay_iters, clamp_proc=False)
network.plot_rmse(win1, win2)
network.plot_decay()
#Save out trained dictionary
network.save_dictionary(5, 10, dict2_path, line_color = 255, train=True)

