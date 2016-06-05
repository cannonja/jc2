import imp
import numpy as np

##Import module in ic object

##Home machine import
ic = imp.load_source('image_class', 'C:\\Users\\Jack2\\Desktop\\Git_Repos\\jc2\\Image_Class\\image_class.py')

##tlab machine import
#ic = imp.load_source('image_class', '/u/jc2/dev/jc2/Image_Class/image_class.py')

##Check image object created correctly
image1 = ic.image_class("Bears.jpg")
#image2 = ic.image_class("Bears.jpg")
image1.im.show()

#print (image1.data.shape)
#print (image1.data_gray.shape)


##Check that get_flattened returns 1D array
##Should duplicate calling np.flatten() method
#print (image1.get_1d().shape)
#print (all(image1.get_1d() == image1.data.flatten()))


##Check gray scale conversion
#image1.convert_gray()
#image2.im = image1.im.convert('LA')
#image1.im.show()  #My method
#image2.im.show()  #Pillow method
#image1.reset_image()
#image1.im.show()


##Check set_block
coordinates = (356, 456)
size = (100, 100)
rgb = (200, 135, 67)
image1.set_block(coordinates, size, rgb)
image1.im.show()
image1.reset_image()
image1.im.show()

#image1.save_image("BearsGray.jpg")



####Resolve reset_image issue - works after convert_gray,
####but not after set_block
