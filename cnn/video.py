from PIL import Image
import numpy as np
import os


class Video:
    
    def __init__(self, folder):
        prior_dir = os.getcwd()
        os.chdir(folder)        
        frames = os.listdir(folder)
        frame_list = []
        
        ## Put frames in list first, then flatten and store in ndarray
        for i in range(len(frames)):
            im = Image.open(frames[i])
            frame_list.append(np.array(im.getdata(), np.uint8).reshape(im.size[0], 
                             im.size[1], 3))
                             
        ## Store in ndarray as an attribute
        tot_pixels = np.prod(frame_list[0].shape)
        self.data = np.array(frame_list[0].flatten(), np.uint8).reshape(tot_pixels, 1)
        for i in range(1, len(frame_list)):
            self.data = np.hstack((self.data, np.array(frame_list[i].flatten(), 
                                    np.uint8).reshape(tot_pixels, 1)))
        os.chdir(prior_dir)
        
        
        
        
    


