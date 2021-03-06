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

################################## Set model params ####################################################

layers = [784, 30, 10]   #Specify number of neurons per layer
learn_rate = 1.5         #Set learning rate
shuffle_data = 1         #Randomize data? (1 = yes, 0 = no)
show_imnums = 0          #Print image numbers during training and testing? (1 = yes, 0 = no)
decay = 1                #Flag to decay learning rate (1 = yes, 0 = no)
decay_rate = 0.99        #Set learning rate decay
decay_iters = 500        #Set learning rate to decay every number of specified iterations
image_file = 'train-images.idx3-ubyte'    #Training images
timage_file = 't10k-images.idx3-ubyte'    #Test images
label_file = 'train-labels.idx1-ubyte'    #Training labels
tlabel_file = 't10k-labels.idx1-ubyte'    #Test labels
num_images = 50000   #Number of training images
num_timages = 10000  #Number of test images


############################ Load all MNIST images and labels #########################################

image_data = mnist.load_images(image_file, num_images)
label_data = mnist.load_labels(label_file, num_images)
timage_data = mnist.load_images(timage_file, num_timages)
tlabel_data = mnist.load_labels(tlabel_file, num_timages)
if len(image_data) != len(label_data):
    print ('TRAINING DATA ERROR: Num of images doesn\'t match num of labels!!!!!')
if len(timage_data) != len(tlabel_data):
    print ('TEST DATA ERROR: Num of images doesn\'t match num of labels!!!!!')

############################### Build training data ###################################

images = []
onehot_labels = []
numeric_labels = []
for i in range(len(image_data)):
    image_data[i] = image_data[i].astype(float)
    image_data[i] /= 255.
    images.append(image_data[i].flatten()[np.newaxis, :])
    onehot_label = np.zeros((10, 1))
    onehot_label[label_data[i]] = 1
    onehot_labels.append(onehot_label)
    numeric_labels.append(label_data[i])

training_data = [(images[i], onehot_labels[i], numeric_labels[i]) for i in range(len(images))]

################################ Build test data ########################################

images = []
onehot_labels = []
numeric_labels = []
for i in range(len(timage_data)):
    timage_data[i] = timage_data[i].astype(float)
    timage_data[i] /= 255.
    images.append(timage_data[i].flatten()[np.newaxis, :])
    onehot_label = np.zeros((10, 1))
    onehot_label[tlabel_data[i]] = 1
    onehot_labels.append(onehot_label)
    numeric_labels.append(tlabel_data[i])

test_data = [(images[i], onehot_labels[i], numeric_labels[i]) for i in range(len(images))]

########################### Select training and test sets ###############################

## If shuffled, data is randomly selected, otherwise
## selected data is the first num_images/num_timages in the set
if shuffle_data:
    random.shuffle(training_data)
    random.shuffle(test_data)

training_set = training_data[:num_images]
test_set = test_data[:num_timages]

############################### Train network ###########################################

net = ff_net(layers)
if decay:
    net.set_lr_stats(learn_rate, decay_rate)
else:
    net.set_lr_stats(learn_rate)

for i in range(len(training_set)):
    if (i+1) % 10000 == 0 and show_imnums:
        print ("Training Image {}".format(i+1))
    if decay and (i+1) % decay_iters == 0:
        net.decay()
    net.set_input(training_set[i][0])
    net.forward_prop(training_set[i][1])
    net.back_prop()

net.plot_rmse()
net.plot_lr_decay()

################################ Test network #########################################

confusion = np.zeros((10, 10), dtype='int32')
correct = 0
for i in range(len(test_set)):
    if (i+1) % 1000 == 0 and show_imnums:
        print ("Test Image {}".format(i+1))
    net.set_input(test_set[i][0])
    net.forward_prop(test_set[i][1])
    prediction = np.where(net.output == net.output.max())[1][0]
    confusion[test_set[i][2], prediction] += 1   #rows = actual, cols = predictions
    if prediction == test_set[i][2]:
        correct += 1

digit_accuracy = pandas.DataFrame(np.diag(confusion) / np.sum(confusion, axis=1),\
                                    columns=['Digit Accuracies'])
accuracy = correct / num_timages
#print ("Learn rate = {}".format(learn_rate))
print ("Confusion:\n\n{}".format(confusion))
print ("Accuracy = {}\n\n{}".format(accuracy, digit_accuracy))














































