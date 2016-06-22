from __future__ import print_function
import numpy as np
import pandas as pd
import socket
import os
import sys
import mr
import random
import matplotlib.pyplot as plt
from PIL import Image
from mr.unsupervised import Lca


machine = socket.gethostname()
if (machine == 'Jack-PC'):
    #Big laptop
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Rozell')
    sys.path.append('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Image_Class')
    os.chdir('C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    file_path = 'C:\\Users\\Jack2\\Desktop'
    dict1_path = file_path + '/orig_dict.png'
    dict2_path = file_path + '/trained_dict.png'
    dict3_path = file_path + '/trained_data.csv'
    write_path = file_path + '/resid_data.csv'
    plot_path = file_path + '/resid_plot2.png'
    nat_path = 'C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Rozell\\Natural_images'
elif (machine == 'Tab'):
    #Little laptop
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\MNIST_Load')
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\Rozell')
    sys.path.append('C:\\Users\\Jack\\Desktop\\Git_Repos\\jc2\\Image_Class')
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
    sys.path.append(os.path.join(base1, 'Image_Class'))
    os.chdir(os.path.join(base1, 'MNIST_Load'))
    file_path = base1 + '/Test/DB Classifier/Nat Images/Walt Run'
    dict1_path = file_path + '/orig_dict.png'
    dict2_path = file_path + '/trained_dict.png'
    dict3_path = file_path + '/trained_data.csv'
    write_path = file_path + '/resid_data.csv'
    plot_path = file_path + '/resid_plot.png'
    nat_path = base1 + '/Rozell/Natural_images'


import mnist_load as mnist
import r_network_class as Lca_jack
import image_class as ic

num_rfields = 50
num_patches = 3000
im_dims = (8,8,3)  #Patch shape
nat_image = ic.image_class(nat_path + '/' + 'city2.jpg')

## Get patches and set Lca variables
training_data = nat_image.slice_patches()[:num_patches]
random.shuffle(training_data)
X = np.zeros((num_patches, np.product(im_dims)))
for i in range(len(training_data)):
    X[i, :] = training_data[i].flatten()
net = Lca(num_rfields, tAlpha=0.8, tLambda=1.0)
net.init(np.product(im_dims),num_rfields)

## Use my Lca class to save pre dictionary
before = np.array(np.array(net._crossbar.copy()))
d1 = Lca_jack.r_network(np.array(net._crossbar))
d1.set_dim(im_dims)
d1.save_dictionary(5, 10, dict1_path)

## Run patches through Lca and train dictionary
MSEs = np.zeros((num_patches,))
for i in range(num_patches):
    patch = X[i, :].reshape(1, np.product(im_dims))
    recon = net.reconstruct(patch)
    resid = recon - patch
    MSEs[i] = np.mean(np.square(resid), axis=1)
    net.partial_fit(patch)
RMSEs = np.sqrt(MSEs)

## Use my Lca class to save post dictionary
after = np.array(net._crossbar.copy())
d2 = Lca_jack.r_network(np.array(net._crossbar))
d2.set_dim(im_dims)
d2.save_dictionary(5, 10, dict2_path)

'''
print (MSEs[0], RMSEs[0])
MSE = net.score(X)
RMSE = np.sqrt(MSE)
print (MSE, RMSE)
'''






#Plot both raw and smoothed residuals
x = range(num_patches)
win1 = 100
win2 = 300
df = pd.DataFrame(RMSEs, index=x)
ma1 = df.rolling(window = win1).mean().values
ma2 = df.rolling(window = win2).mean().values

plt.figure()
plt.plot(x, RMSEs,  color = 'gray', alpha = 0.6, label = 'Raw')
plt.plot(x, ma1,  color = 'red', label = 'MA - ' + str(win1) + ' periods')
plt.plot(x, ma2,  color = 'blue', label = 'MA - ' + str(win2) + ' periods')
plt.xlabel('Patch Number')
plt.title('Reconstruction Error (RMSE)')
plt.legend()
plt.savefig(plot_path)
'''
plt.figure()
hmap_data = np.sum(after - before, axis=2)
plt.colormesh(hmap_data, cmap='RdBu')
plt.show()
'''
