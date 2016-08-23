from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


class Video:
    
    def __init__(self, folder):
        prior_dir = os.getcwd()
        os.chdir(folder)        
        frames = os.listdir(folder)
        self.data = []
        
        for i in range(len(frames)):
            im = Image.open(frames[i])
            self.data.append(np.array(im.getdata(), np.uint8).reshape(im.size[0], 
                             im.size[1], 3))
        os.chdir(prior_dir)
        
        
        
        
    


