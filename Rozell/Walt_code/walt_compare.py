from __future__ import print_function
import numpy as np
import socket
import os
import sys
import mr
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
    file_path = base1 + '/Test/DB Classifier/Overnight run'
    dict1_path = file_path + '/orig_dict.png'
    dict2_path = file_path + '/trained_dict.png'
    dict3_path = file_path + '/trained_data.csv'
    write_path = file_path + '/resid_data.csv'
    plot_path = file_path + '/resid_plot.png'
    nat_path = base1 + '/Rozell/Natural_images'


import mnist_load as mnist
import image_class as ic

num_rfields = 50
num_patches = 3000
im_dims = (8,8,3)
nat_image = ic.image_class(nat_path + '/' + 'city2.jpg')
training_data = nat_image.slice_patches()[:num_patches]
#random.shuffle(training_data)

X = np.zeros((num_patches, np.product(im_dims)))
for i in range(len(training_data)):
    X[i, :] = training_data[i].flatten()

net = Lca(num_rfields)
net.fit(X)

MSE = net.score(X)
RMSE = np.sqrt(MSE)
print (MSE, RMSE)










'''
#Plot and save out both raw and smoothed residuals
ma1 = df.iloc[:,1].rolling(window = win1).mean().values
ma2 = df.iloc[:,1].rolling(window = win2).mean().values

plt.figure(1)
plt.plot(x, df.values[:,1],  color = 'gray', alpha = 0.6, label = 'Raw')
plt.plot(x, ma1,  color = 'red', label = 'MA - ' + str(win1) + ' periods')
plt.plot(x, ma2,  color = 'blue', label = 'MA - ' + str(win2) + ' periods')
plt.xlabel('Image Number')
plt.title('Reconstruction Error')
plt.legend()
plt.savefig(plot_path)
plt.show()
'''

