
from PIL import Image
import numpy as np

im = Image.open("Lenna.png")
array = np.asarray(im)
array.flags.writeable = True

print ("Flattened array[0:11]:")
print (array.flatten()[0:11])

#Set RGB to zero
array[:,:,0] = 0
#array[:,:,1] = 0
array[:,:,2] = 0

#array = array / array.max()

im2 = Image.fromarray(array)
im2.save("Lenna2.png")



'''
print ("Num reds:")
print (red.shape)

print ("Num greens:")
print (green.shape)

print ("Num blues:")
print (blue.shape)
'''


'''
r, g, b = im.split()
red = np.asarray(r)
green = np.asarray(g)
blue = np.asarray(b)

red = np.zeros((red.shape[0], red.shape[1]))
red = Image.fromarray(red)

#green = np.zeros((green.shape[0], green.shape[1]))
green = Image.fromarray(green)

#blue = np.zeros((blue.shape[0], blue.shape[1]))
blue = Image.fromarray(blue)

im2 = Image.merge("RGB", (red, g, b))

im2.save("new.png")

#print array.shape
#print array.size

#print array.shape

#subset first ten red values
#print array[:10, 0, 0]
'''



'''
for i in array:
    for j in i:
        j[0] = 0
        j[1] = 0
'''



