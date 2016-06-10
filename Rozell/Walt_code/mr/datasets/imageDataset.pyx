
from common cimport ImageSet

import numpy as np
import os
from PIL import Image

class ImageDataset(ImageSet):
    def __init__(self, folder, ims):
        """Loads each file listed in ims from folder.  Defaults to RGB, call
        .grayscale() to get grayscale."""
        if not os.path.lexists(folder):
            raise ValueError("Folder {} not found".format(folder))

        cdef double[:] imdata, label
        cdef double scale = 1.0 / 255
        cdef int x, y, c, imW

        reset = False
        for im in ims:
            image = Image.open(os.path.join(folder, im))
            if not reset:
                reset = True
                self._reset(len(ims), image.size[1], image.size[0], 3, 0)
                imW = image.size[1]
            else:
                if image.size[1] != self.imWidth:
                    raise ValueError("Not all images have same width! {}"
                            .format(image.size[1]))
                if image.size[0] != self.imHeight:
                    raise ValueError("Not all images have same height {}"
                            .format(image.size[0]))

            arr = np.asarray(image.convert("RGB"))
            imdata, label = self._getNext()
            for x in range(arr.shape[0]):
                for y in range(arr.shape[1]):
                    for c in range(3):
                        imdata[c + 3 * (x + imW * y)] = arr[y, x, c] * scale
