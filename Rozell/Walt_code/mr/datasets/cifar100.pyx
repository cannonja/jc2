
from common cimport float_t, uint8_t, uint32_t, CFile, ImageSet
cimport cython
cimport numpy as np

import os
import numpy as np

class Cifar100Dataset(ImageSet):
    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    #@cython.cdivision(True)
    def __init__(self, path, nLoad = 1e10):
        self._path = path
        nLoad = int(min(nLoad, 60000))

        for r in range(0, 2):
            f = self._getBatchFile(r)
            if not os.path.lexists(f):
                raise ValueError("Missing {}".format(f))

        # Allocate stuff
        self._reset(nLoad, 32, 32, 3, 100)
        cdef double[:] image, labels

        cdef int loaded = 0
        cdef int fileLoaded = 10000
        cdef int file = -1
        cdef int nRead = 0
        cdef CFile cFile = CFile("")
        cdef unicode fname
        cdef uint8_t label
        cdef uint8_t[:] data
        cdef int i, j
        recsInFile = 0
        while loaded < nLoad:
            if fileLoaded >= recsInFile:
                fileLoaded = 0
                file += 1
                cFile.close()
                fname = self._getBatchFile(file)
                cFile.open(<const char*>fname)
                if file == 0:
                    recsInFile = 50000
                elif file == 1:
                    recsInFile = 10000
                else:
                    raise NotImplementedError(file)

            labelCoarse = cFile.readUint8()
            labelFine = cFile.readUint8()

            nRead, data = cFile.read(32*32*3)
            if nRead != 3072:
                raise ValueError("Bad CIFAR file?")

            image, labels = self._getNext()
            # CIFAR files are RRR...GGG...BBB, not RGBRGBRGB
            labels[labelFine] = 1.0
            for i in range(3072):
                j = (i % (32*32)) * 3 + (i // (32*32))
                image[j] = data[i] / 255.0

            loaded += 1
            fileLoaded += 1


    def _getBatchFile(self, n):
        cdef unicode f
        if n == 0:
            f = u"train.bin"
        elif n == 1:
            f = u"test.bin"
        else:
            raise NotImplementedError(n)
        return os.path.join(self._path, f)
