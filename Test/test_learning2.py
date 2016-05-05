import os
import sys
from PIL import Image
import numpy as np
import socket
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import pdb


machine = socket.gethostname()
if (machine == 'Jack-PC'):
    #Big laptop
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Rozell')
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Classify')
    os.chdir('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    file_path = 'C:\\Users\\Jack2\\Desktop'
    dict_path = file_path + '\\Git_Repos\\jc2\\Classify\\trained_data'
elif (machine == 'Tab'):
    #Little laptop
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\Rozell')
    os.chdir('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
else:
    #PSU machines (linux lab)
    base1 = os.path.expanduser('~/dev/jc2')
    base2 = os.path.expanduser('~/Desktop')
    sys.path.append(os.path.join(base1, 'MNIST_Load'))
    sys.path.append(os.path.join(base1, 'Rozell'))
    sys.path.append(os.path.join(base1, 'Classify'))
    os.chdir(os.path.join(base1, 'MNIST_Load'))
    file_path = base2
    dict_path = base1 + '/Classify/trained_data'
    plot_path = base1 + '/Classify/RMSE_plot'
    accuracy_path = base1 + '/Classify/Accuracy_plot'
    weight_path = base1 + '/Classify/weights.csv'
    confusion_path = base1 + '/Classify/confusion.csv'

import mnist_load as mnist
import sparse_algo as sp
import r_network_class as lca
import classify


################### Set parameters ##############################################################
lamb = 1
tau = 10
delta = 0.001
u_stop = .01
t_type = 'S'
alpha = 0.1

################### Load MNIST image and label data #############################################
num_images  = 5000
start_pos = 50000
image_file = 'train-images.idx3-ubyte'
label_file = 'train-labels.idx1-ubyte'
image_data = mnist.load_images(image_file, num_images, start_pos)
label_data = mnist.load_labels(label_file, num_images, start_pos)

################### Initialize Rozell network and set parameters ################################
dict_data = pd.read_csv(dict_path, header = None)
rozell = lca.r_network(dict_data.values)
rozell.set_parameters(lamb, tau, delta, u_stop, t_type)

'''
################### Initialize random  matrix of weights for mapping ############################
################### sparse code to output layer.  Then train network #### #######################
weights = np.random.rand(10, 51)    #10 nodes in layer j+1 and 50 nodes in layer j
learn_rate = 0.01
error_plot = np.array([])
#pdb.set_trace()
for i, j in zip(image_data, label_data):
    #Run Rozell and forwardprop
    rozell.set_stimulus(i.flatten())
    sparse_code = np.append(rozell.generate_sparse(), 1) #Add bias term
    _, error, D = classify.forward_prop(weights, sparse_code, j)

    #Store error - represented as quadratic error: 0.5*(error)^2
    MSE = np.dot(np.transpose(error), 0.5 * error) / error.shape[0]
    RMSE = float(np.sqrt(MSE))
    error_plot = np.append(error_plot, RMSE)

    #Run backprop
    update = classify.back_prop(D, sparse_code, error, learn_rate)
    weights += update

#Save weights and plot RMSE
df = pd.DataFrame(weights)
df.to_csv(weight_path, header=False, index=False)
plt.plot(error_plot)
plt.xlabel('Image Number')
plt.title('RMSE During Backprop')
plt.savefig(plot_path)
plt.show()

'''
#################### Read in weights and test network #########################################
df = pd.read_csv(weight_path, header= None, names=None)
weights = df.values
correct = 0
count = 0
error_plot = np.array([])
accuracy_plot = np.array([])
confusion = np.zeros((10,10), dtype='int32')  #rows = actual, columns = predicted
#pdb.set_trace()
for i, j in zip(image_data, label_data):
    count += 1
    #Run Rozell and forwardprop
    rozell.set_stimulus(i.flatten())
    sparse_code = np.append(rozell.generate_sparse(), 1) #Add bias term
    prediction, error, D = classify.forward_prop(weights, sparse_code, j)
    confusion[j, prediction] += 1
    if (prediction == j):
        correct += 1

    #Store error - represented as quadratic error: 0.5*(error)^2
    MSE = np.dot(np.transpose(error), 0.5 * error) / error.shape[0]
    RMSE = float(np.sqrt(MSE))
    error_plot = np.append(error_plot, RMSE)
    accuracy_plot = np.append(accuracy_plot, (correct / count))

accuracy = correct / num_images
cf = pd.DataFrame(confusion)
print (cf)
cf.to_csv(confusion_path, header=range(10), index=range(10))

#Plot RMSE
plt.figure()
plt.plot(error_plot)
plt.xlabel('Image Number')
plt.title('RMSE During Testing - accuracy = ' + str(accuracy))
plt.savefig(plot_path + '2')

#Plot accuracy
plt.figure()
plt.plot(accuracy_plot)
plt.xlabel('Image Number')
plt.title('Accuracy During Testing')
plt.savefig(accuracy_path)
plt.show()




