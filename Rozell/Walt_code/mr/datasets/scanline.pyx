
from common cimport ImageSet
cimport cython

import numpy as np

class ScanlineDataset(ImageSet):
    """Generates a dataset where each image falls into one of three
    categories:

    1. A horizontal line on an even-indexed row.
    2. A vertical line on an even-indexed column.
    3. A cross at the intersection of an even-indexed row and an
       even-indexed column.

    Pixels not related to the main form will be assigned a value between
    noiseFloor and noiseCeiling, inclusive.

    All told, there will be 1.5 * size * noiseCopies samples.
    """
    def __init__(self, size = 3, noiseCeiling = 0.0, noiseFloor = 0.0,
            noiseCopies = 1):
        """
        :param size: Single dimension of the image.  Final images are size x size.

        :param noiseCeiling: The highest value for uniformly distributed noise.

        :param noiseFloor: The lowest value for uniformly distributed noise.

        :param noiseCopies: The number of images to generate for each configuration.
                Useful for diversifying the noise portfolio."""

        lines = size // 2
        self._reset(noiseCopies * 3 * lines, size, size, 1, 3)

        rng = np.random.RandomState()

        cdef double[:] image, labels,tmpd
        for _ in range(noiseCopies):
            # Horizontal lines
            for l in range(1, size, 2):
                image, labels = self._getNext()
                labels[0] = 1.
                tmpd = rng.uniform(noiseFloor, noiseCeiling,
                        size=(size*size,))
                image[:] = tmpd
                image[size * l:size * l + size] = 1.

            # Vertical lines
            for l in range(1, size, 2):
                image, labels = self._getNext()
                labels[1] = 1.
                tmpd = rng.uniform(noiseFloor, noiseCeiling,
                        size=(size*size,))
                image[:] = tmpd
                image[l::size] = 1.

            # Cross
            for l in range(1, size, 2):
                image, labels = self._getNext()
                labels[2] = 1.
                tmpd = rng.uniform(noiseFloor, noiseCeiling,
                        size=(size*size,))
                image[:] = tmpd
                image[size*l:size*l + size] = 1.
                image[l::size] = 1.

