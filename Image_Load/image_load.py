##Didn't flatten the array to 1D

from PIL import Image
import numpy as np

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



