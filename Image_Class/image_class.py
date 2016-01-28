from PIL import Image
import numpy as np

class image_class:
    'Class for manipulating an image'

    def __init__(self, file_path):
        self.im = Image.open(file_path)
        self.data = np.asarray(self.im)
        self.data.flags.writeable = True        

    def get_wh(self):
        return (self.data.shape[1], self.data.shape[0])

    def get_1d(self):
        return self.data.flatten()

    def convert_gray(self):
        'Use luminosity formula: 0.21R + 0.72G + 0.07B'
        self.data = self.data.astype(float)
        self.data[:,:,0] *= 0.21
        self.data[:,:,1] *= 0.72
        self.data[:,:,2] *= 0.07
        self.data = self.data.astype('uint8')        
        self.im = Image.fromarray(self.data)

    def save_image(self, new_path):
        #self.im = Image.fromarray(self.data)
        self.im.save(new_path)
        
    















'''
##Read image and convert to numpy array
im = Image.open("Lenna.png")
array = np.asarray(im)
array.flags.writeable = True

##Set RGB to zero
array[:,:,0] = 0   #R
#array[:,:,1] = 0  #G
array[:,:,2] = 0   #B

#Convert to floating point bounded by [0,1]
#Will not output as floating point
#array = array / array.max()

im2 = Image.fromarray(array)
im2.save("Lenna2.png")
'''

