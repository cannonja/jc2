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
import random
import pdb


machine = socket.gethostname()
if (machine == 'Jack-PC'):
    #Big laptop
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Rozell')
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Image_Class')
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Projects\\net')
    os.chdir('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    file_path = 'C:\\Users\\Jack2\\Desktop'
elif (machine == 'Tab'):
    #Little laptop
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\Rozell')
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\Image_Class')
    os.chdir('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    file_path = 'C:\\Users\\Jack\\Desktop'
else:
    #PSU machines (linux lab)
    base1 = os.path.expanduser('~/dev/jc2')
    base2 = os.path.expanduser('~/Desktop')
    sys.path.append(os.path.join(base1, 'MNIST_Load'))
    sys.path.append(os.path.join(base1, 'Rozell'))
    sys.path.append(os.path.join(base1, 'Image_Class'))
    sys.path.append(os.path.join(base1, 'Projects/net'))
    os.chdir(os.path.join(base1, 'MNIST_Load'))
    file_path = base1 + '/Test/DB Classifier/Overnight run'
    

import mnist_load as mnist
import sparse_algo as sp
import r_network_class as lca
import image_class as ic
from feed_forward import ff_net


net = ff_net([3, 3, 3])
#print ("layers = {}\nnum connections = {}\n".format(net.layers, len(net.connections)))
in_data = np.repeat(1, 3).reshape(3,1)
net.set_input(in_data)
net.forward_prop(np.array([[1],[0],[0]]))
'''
print ("Dimensions of all NN variables after one forward prop:")
print ("input = {}, with bias = {}".format(net.input.shape, net.activations[0].shape))
print ("W1 = {}, W1 bias = {}, W2 = {}, W2 bias = {}"\
        .format(net.connections[0][:,:-1].shape, net.connections[0].shape,\
        net.connections[1][:,:-1].shape, net.connections[1].shape))
print ("D1 = {}, D2 = {}".format(net.D[0].shape, net.D[1].shape))
print ("activations = {}, {}, {}".format(net.activations[0].shape,\
        net.activations[1].shape, net.activations[2].shape))
print ("output = {}".format(net.output.shape))
'''

pre = [np.array(i) for i in net.W_bar]
#print ("{}\t{}\n\n".format(net.W_bar[0], net.W_bar[1]))
net.back_prop(1.0)
#print ("{}\t{}\n\n".format(net.W_bar[0], net.W_bar[1]))
post = [np.array(i) for i in net.W_bar]
change = [np.subtract(j, i) for i,j in zip(pre, post)]
print ("{}\n\n{}\n\n".format(pre[0], post[0]))
print ("{}\n\n{}\n\n".format(pre[1], post[1]))
print ("{}\n\n{}".format(change[0], change[1]))




























