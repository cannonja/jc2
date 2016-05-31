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
    file_path = 'C:\\Users\\Jack2\\Desktop'
    dict1_path = file_path + '/orig_dict.png'
    dict2_path = file_path + '/trained_dict.png'
    dict3_path = file_path + '/trained_data.csv'
    write_path = file_path + '/resid_data.csv'
    plot_path = file_path + '/resid_plot2.eps'
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


################### Set parameters ##############################################################
lamb = 1.0
tau = 10.0
delta = 0.001
u_stop = 0.1
t_type = 'S'
alpha = 0.85

#Plotting parameters
win1 = 100  #Window for mov avg 1
win2 = 500 #Window for mov avg 2

################### Load dictionary from first 50 MNIST images ##################################
################### Load training set from last 59950 MNIST images ##############################
num_rfields = 50
num_images =  20000      #60000 - num_rfields
image_file = 'train-images.idx3-ubyte'    #'t10k-images.idx3-ubyte'
dict_data = mnist.load_images(image_file, num_rfields)
training_data = mnist.load_images(image_file, num_images, 50)

for i in range(len(dict_data)):
    dict_data[i] = dict_data[i].astype(float)
    dict_data[i] /= 255.

for i in range(len(training_data)):
    training_data[i] = training_data[i].astype(float)
    training_data[i] /= 255.

#Initialize network dictionary and parameters
D = sp.build_dictionary(dict_data)
network = lca.r_network(D)
network.set_parameters(lamb, tau, delta, u_stop, t_type)


################### Run each training image through network #######################################
################### For each image, generate sparse code then update trained ######################

#Print out the time and start the training process
#Save out the original dictionary
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print("Start time: ", st)
network.save_dictionary(5, 10, dict1_path)

#Initiate x values and residual array for residual plot
x = range(num_images)
resid_plot = np.zeros((num_images))

#Train dictionary as each image is run through network
#Store length of residual vector in resid_plot array
for i in range(num_images):
    if (((i + 1) % 100) == 0):
        print("Image ",i + 1)
    stimulus = training_data[i].flatten()
    network.set_stimulus(stimulus, True)
    network.generate_sparse(True)
    if ((i + 1) % 100 == 0):
        alpha *= 0.92
        print (alpha)    
    y = network.update_trained(alpha)
    resid_plot[i] = np.sqrt(np.dot(y,y))

#Save out trained dictionary as image and csv, then print out time
network.save_dictionary(5, 10, dict2_path, True)
data = pandas.DataFrame(network.trained * 255.)
data.to_csv(dict3_path, index = False, header = False)
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print("End time: ", st)

#Write residual data to csv file and plot
df = pandas.DataFrame(np.column_stack((x, resid_plot)), columns = ['Image #', "Resid"])
df.to_csv(write_path, index = False)



#Plot and save out both raw and smoothed residuals
ma1 = df.iloc[:,1].rolling(window = win1).mean().values
ma2 = df.iloc[:,1].rolling(window = win2).mean().values

plt.plot(x, df.values[:,1],  color = 'gray', alpha = 0.6, label = 'Raw')
plt.plot(x, ma1,  color = 'red', label = 'MA - ' + str(win1) + ' periods')
plt.plot(x, ma2,  color = 'blue', label = 'MA - ' + str(win2) + ' periods')
plt.xlabel('Image Number')
plt.title('Reconstruction Error')
plt.legend()
plt.savefig(plot_path, format='eps', dpi=250)
plt.show()


