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
from feed_forward import ff_net as nn


net = nn([3, 3, 3])
#print ("layers = {}\nnum connections = {}\n".format(net.layers, len(net.connections)))
in_data = np.repeat(0, 3).reshape(3,1)
net.set_input(in_data)
net.forward_prop(np.array([[1],[0],[0]]))
print (net.connections)
net.back_prop(0.01)
print (net.connections)




























