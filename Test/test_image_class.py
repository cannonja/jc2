import imp
import numpy as np

##Import module in ic object
ic = imp.load_source('image_class', 'C:\\Users\\Jack2\\Google Drive\\Grad School\\URMP\\jc2\\Image_Class\\image_class.py')

##Check image object created correctly
image1 = ic.image_class("Bears.jpg")
image2 = ic.image_class("Bears.jpg")
#image1.im.show()

##Check that get_flattened returns 1D array
##Should duplicate calling np.flatten() method
print (image1.get_1d().shape)
print (all(image1.get_1d() == image1.data.flatten()))

##Check gray scale conversion
image1.convert_gray()
#image1.im.show()
#image1.save_image("BearsGray.jpg")
