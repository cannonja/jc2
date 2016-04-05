from PIL import Image
import numpy as np

class mnist_mp:
    'Class for running Matching Pursuit on MNIST data'

    #Takes dictionary in proper form (each column/atom) is a single signal
    def __init__(self, D):
        self.dictionary = D
        self.image_orig = None  #The original image to approximate
        self.sparse_code = None  #Vector of beta coefficients from running MP
        self.image_approx = None  #The reconstructed image (sparse code + dictionary atoms)
        self.approx_components = None  #Array of components used to reconstruct

    #Takes
    def load_orig(self, y):
        self.image_orig = y


    def save_image(self, new_path):
        #self.im = Image.fromarray(self.data)
        self.im.save(new_path)


