from PIL import Image
import numpy as np

class image_class:
    'Class for manipulating an image'

    def __init__(self, file_path):
        self.im = Image.open(file_path)
        self.data = np.asarray(self.im)
        self.data.flags.writeable = True
        self.data_orig = self.data.copy()
        self.data_gray = np.zeros((self.data.shape[0], self.data.shape[1]))

    def get_wh(self):
        return (self.data.shape[1], self.data.shape[0])

    def get_1d(self):
        return self.data.flatten()

    def convert_gray(self):
        'Use luminosity formula: 0.21R + 0.72G + 0.07B'
        self.data = self.data.astype(float)
        self.data_gray = (self.data[:,:,0] * 0.21 + self.data[:,:,1] * 0.72
                + self.data[:,:,2] * 0.07)
        self.data_gray = self.data_gray.astype('uint8')
        self.im = Image.fromarray(self.data_gray)

    #This method takes a 2-tuple indicating the shape of the patches (l and w)
    #It returns the image as a list of (patch_shape[0] x patch_shape[1] x 3) patches
    def slice_patches(self, patch_shape = (8.,8.)):
        #Make sure elements are floats
        patch_shape = list(patch_shape)
        patch_shape = [float(i) for i in patch_shape]
        #Verify compatible dimensions
        if self.data.shape[0] % patch_shape[0] or self.data.shape[1] % patch_shape[1]:
            print ('Image dimension not compatible with specified patch size')
            return
        #Set number of rows and columns to slice image into
        #Iterate columns over rows to build patch_list
        num_rows = int(self.data.shape[0] / patch_shape[0])
        num_cols = int(self.data.shape[1] / patch_shape[1])
        patch_list = []
        for i in range(num_rows):
            for j in range(num_cols):
                patch_list.append(self.data[i:patch_shape[0], j:patch_shape[0], :])

        return patch_list       
        
        
        
        

    #This method takes three tuples as arguments
    #coord is a 2-tuple of coordinates for he top left corner of the section
    #size is a 2-tuple specifying the length and width of the section
    #rgb is a 3-tuple indicating the r, g, and b settings for the pixels
    def set_block(self, coord, size, rgb):
        'Add exception if arguments are out of range?'
        length = coord[0] + size[0]
        width = coord[1] + size[1]
        self.data[coord[0]:length, coord[1]:width, 0] = rgb[0]
        self.data[coord[0]:length, coord[1]:width, 1] = rgb[1]
        self.data[coord[0]:length, coord[1]:width, 2] = rgb[2]
        self.data = self.data.astype('uint8')
        self.im = Image.fromarray(self.data)

    def reset_image(self):
        self.im = Image.fromarray(self.data_orig)

    def save_image(self, new_path):
        #self.im = Image.fromarray(self.data)
        self.im.save(new_path)


