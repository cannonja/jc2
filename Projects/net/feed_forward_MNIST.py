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

############## Load MNIST images and labels
image_file = 'train-images.idx3-ubyte'
label_file = 'train-labels.idx1-ubyte'
num_images = 5000
num_timages = 1000
image_data = mnist.load_images(image_file, num_images)
label_data = mnist.load_labels(label_file, num_images)
timage_data = mnist.load_images(image_file, num_timages, num_images)
tlabel_data = mnist.load_labels(label_file, num_timages, num_images)



############# Build training set

# Images
images = []
for i in range(len(image_data)):
    image_data[i] = image_data[i].astype(float)
    image_data[i] /= 255.
    images.append(image_data[i].flatten()[np.newaxis, :])

# Labels
labels = []
for i in range(len(label_data)):
    label = np.zeros((10, 1))
    label[label_data[i]] = 1
    labels.append(label)

training_set = [(images[i], labels[i]) for i in range(len(images))]


############# Build test set

# Images
images = []
for i in range(len(timage_data)):
    timage_data[i] = timage_data[i].astype(float)
    timage_data[i] /= 255.
    images.append(timage_data[i].flatten()[np.newaxis, :])

# Labels
labels = []
for i in range(len(tlabel_data)):
    label = np.zeros((10, 1))
    label[tlabel_data[i]] = 1
    labels.append(label)

test_set = [(images[i], labels[i]) for i in range(len(images))]



########### Train network
layers = [784, 50, 10]
learn_rate = 0.01
net = ff_net(layers)

for i in range(len(training_set)):
    if i % 100 == 0:
        print ("Image {}".format(i))
    net.set_input(training_set[i][0])
    net.forward_prop(training_set[i][1])
    net.back_prop(learn_rate)


########### Test network
confusion = np.zeros((10, 10), dtype='int32')
for i in range(len(test_set)):
    if i % 100 == 0:
        print ("Image {}".format(i))
    net.set_input(test_set[i][0])
    net.forward_prop(test_set[i][1])
    prediction = np.where(net.output == net.output.max())
    confusion[tlabel_data[i], prediction] += 1
    if prediction == tlabel_data[i] == prediction:
        correct += 1

accuracy = correct / num_timages
print ("Accuracy = {}\n\nConfusion:\n{}".format(accuracy, confusion))






















'''
net = ff_net([3, 3, 3])
in_data = np.repeat(1, 3).reshape(3,1)
net.set_input(in_data)
net.forward_prop(np.array([[1],[0],[0]]))
net.back_prop(1.0)
'''



























