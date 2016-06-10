
cimport cython
from libc.math cimport sqrt

import modelBaseUnpickler as mbu
import numpy as np

cdef class AdaDelta:

    PICKLE_INIT = [ 'w', 'h', 'rho', 'initial', 'momentum' ]
    PICKLE_STATE = [ '_edX', '_eG2' ]
    __reduce_ex__ = mbu.__reduce_ex__
    __setstate__ = mbu.__setstate__


    def __init__(self, int w, int h, double rho = 0.94, double initial = 1e-6,
            double momentum = 0.5):
        """Initializes a 2-d matrix of individually optimized parameters
        controlled by ADADELTA (Zeiler 2012).  A gradient of the overall score
        for each parameter must be estimable.

        rho - Controls the window size for ADADELTA.  Weight for moving average
            stability.

        initial - Controls divide-by-zero and provides for good initial
                learning.  Epsilon from the paper is always initial**2.
        """
        self.w = w
        self.h = h
        self.rho = rho
        self.initial = initial
        self.epsilon = initial
        self.momentum = momentum
        self._edX = np.zeros((w, h))
        self._eG2 = np.zeros((w, h))
        self._last = np.zeros((w, h))

        #self._edX[:, :] = initial
        #self._eG2[:, :] = initial


    cpdef convergerProps(self):
        return [ self._edX, self._eG2, self._last ]


    cpdef double getDelta(self, int i, int j, double gradient) except? -1:
        """Given a gradient (increasing this parameter by 1.0 is expected to
        increase the overall score by this much), adjust the parameter in such
        a way that score is minimized.

        Returns the recommended delta for the parameter."""
        cdef double edX = self._edX[i, j]
        cdef double eG2 = self._eG2[i, j]
        cdef double rho = self.rho, urho = 1. - rho, e = self.epsilon
        cdef double gScalar, delta
        # See if the variable is changing significantly enough to count
        gScalar = sqrt((edX + e) / (eG2 + e))
        if abs(gScalar * gradient) < abs(self._last[i, j]) * 1e-1 and True or False:
            #print("LALA GSCALE: {} / {} < {}*1e-1".format(gScalar, gradient,
            #        self._last[i, j] / gScalar))
            return -gScalar * gradient

        eG2 = rho * eG2 + urho * gradient * gradient
        gScalar = sqrt((edX + e) / (eG2 + e))
        delta = -gScalar * gradient
        self._edX[i, j] = rho * edX + urho * delta * delta
        self._eG2[i, j] = eG2

        delta = self._last[i, j] * self.momentum + (1. - self.momentum) * delta
        self._last[i, j] = delta
        return delta
