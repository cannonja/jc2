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
import r_network_class_single as lca


################### Set parameters ##############################################################
lamb = 3.
tau = 10.0
delta = 0.01
u_stop = 0.001
t_type = 'S'
alpha = 0.01
num_iterations =  10000

##Load initial dictionary, either random or image
##Save out initial dictionary as image
image_file = 't10k-images.idx3-ubyte'  #'train-images.idx3-ubyte'
#dict_data = np.random.rand(28,28)  #Use random init
dict_data =  mnist.load_images(image_file, 1, 1)[0]
og_dict = Image.fromarray(dict_data)
og_dict.save(dict1_path)
training_data = mnist.load_images(image_file, 1, 15)[0]
og_im_data = training_data.copy()
training_data = training_data.astype(float)
training_data /= 255.

#Initialize network dictionary, then save stimulus and orig dict
dict_data = dict_data.astype(float)
dict_data /= 255.
network = lca.r_network(dict_data.flatten())
network.set_parameters(lamb, tau, delta, u_stop, t_type)
network.set_stimulus(training_data.flatten(), True)
og_im = Image.fromarray(og_im_data)
og_im.save(file_path + '/stim.png')

#Initiate x values and residual array for residual plot
x = range(num_iterations)
resid_plot = np.zeros((num_iterations))

#Train dictionary as each image is run through network
#Store length of residual vector in resid_plot array
for i in range(num_iterations):
    if (((i + 1) % 100) == 0):
        print("Image ", i + 1)
    network.generate_sparse(True)
    y = network.update_trained(alpha)
    resid_plot[i] = np.sqrt(np.dot(y,y))

#Save out trained dictionary as image
tdict_data = network.trained.copy()
tdict_data *= 255.
tdict_im = Image.fromarray(tdict_data.reshape((28,28)))
tdict_im = tdict_im.convert('L')
tdict_im.save(dict2_path)

#Write residual data to csv file and plot
df = pandas.DataFrame(np.column_stack((x, resid_plot)), columns = ['Image #', "Resid"])
df.to_csv(write_path, index = False)
plt.plot(x, df.values[:,1], label = 'Raw')
#plt.plot(x, ma1,  color = 'red', label = 'MA - ' + str(win1) + ' periods')
#plt.plot(x, ma2,  color = 'blue', label = 'MA - ' + str(win2) + ' periods')
plt.xlabel('Image Number')
plt.title('Reconstruction Error')
#plt.legend()
plt.savefig(plot_path)
plt.show()

