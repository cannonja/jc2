
from common cimport float_t, uint8_t, uint32_t, CFile, ImageSet
cimport numpy as np

import os
import numpy as np

class MnistDataset(ImageSet):
    def __init__(self, path, nLoad = 1e10):
        """Dataset for the MNIST handwritten digit database.

        path - Path to train-images-idx3-ubyte and train-labels-idx1-ubyte as
                well as t10k-images-idx3-ubyte and t10k-labels-idx1-ubyte.
                Note that these files must be extracted (not in gzipped form),
                and for consistency, any dots in the file names must have been
                changed to dashes.

        nLoad - Number of images to load, total.
        """
        nLoad = int(min(nLoad, 70000))

        # Training resources
        ri = os.path.join(path, "train-images-idx3-ubyte")
        rl = os.path.join(path, "train-labels-idx1-ubyte")
        # Testing
        ei = os.path.join(path, "t10k-images-idx3-ubyte")
        el = os.path.join(path, "t10k-labels-idx1-ubyte")

        for p in [ ri, rl, ei, el ]:
            if not os.path.lexists(p):
                raise ValueError("Could not open: {}".format(p))

        self._reset(nLoad, 28, 28, 1, 10)
        nr = min(nLoad, 60000)
        ne = max(0, nLoad - 60000)

        self._read(nr, ri, rl)
        self._read(ne, ei, el)


    def _read(self, nr, imageFile, labelFile):
        cdef int i, j
        cdef uint32_t w, p1, p2

        with CFile(imageFile) as im, CFile(labelFile) as la:
            ### Read headers for images
            w = im.readUint32()
            if w != 2051:
                raise Exception("File {} has wrong magic number".format(
                        imageFile))

            # Read in some BS
            w = im.readUint32()
            p1 = im.readUint32()
            p2 = im.readUint32()

            ### Read headers for labels
            w = la.readUint32()
            if w != 2049:
                raise Exception("File {} has wrong magic number".format(
                    labelFile))

            [ la.readUint32() for _ in range(1) ]

            for _ in range(nr):
                image, labels = self._getNext()
                for j in range(p1 * p2):
                    image[j] = im.readUint8() / 255.0
                labels[la.readUint8()] = 1.0
