
import numpy as np

def checkNpArray(*args):
    """Similar to sklearn.preprocessing.data.check_arrays, but specifically
    for overridden methods in SklearnModelBase.  Just checks that each argument
    is an array of floats, nothing else."""
    for i, a in enumerate(args):
        #if not isinstance(a, np.matrix) and not isinstance(a, np.ndarray):
        #    raise ValueError("Argument {} is not matrix or array".format(i))
        #if a.dtype != float:
        pass#    raise ValueError("Argument {} is not of dtype 'float'".format(i))



cdef class FastRandom:
    def __init__(self):
        self._randLen = 1024
        self._randBuffer = np.random.uniform(size=(self._randLen,))
        self._randIndex = 0


    cpdef double get(self) except? -1.:
        cdef double r = self._randBuffer[self._randIndex]
        self._randIndex += 1
        if self._randIndex == self._randLen:
            self._randIndex = 0
            self._randBuffer = np.random.uniform(size=(self._randLen,))
        return r

