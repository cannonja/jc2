
#cython: profile=True

cimport cython
cimport numpy as np

from mr.modelBase cimport SklearnModelBase

from mr.adadelta import AdaDelta

import mr.util as util

import math
import numpy as np

cdef class LcaShapero(SklearnModelBase):
    UNSUPERVISED = True
    PARAMS = SklearnModelBase.PARAMS + [ 'adaDeltaRho', 'avgInput', 'tLambda',
            'nSpikes' ]
    PICKLE_VARS = SklearnModelBase.PICKLE_VARS + [ '_ada', '_crossbar' ]

    cdef public double adaDeltaRho, avgInput, tLambda, nSpikes

    cdef public object _ada
    cdef public double[:, :] _crossbar

    def __init__(self, nOutputs = 10, **kwargs):
        opts = { 'adaDeltaRho': AdaDelta(1, 1).rho, 'avgInput': 0.1,
                'tLambda': 0.1, 'nOutputs': nOutputs, 'nSpikes': 10 }
        opts.update(kwargs)
        super(LcaShapero, self).__init__(**opts)


    cpdef _init(self, int nInputs, int nOutputs):
        self._crossbar = np.matrix(np.random.uniform(size=(nInputs, nOutputs)),
                dtype = float)
        self._ada = AdaDelta(nInputs, nOutputs, rho = self.adaDeltaRho)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef _partial_fit(self, double[:] x, double[:] y):
        cdef int i, j
        cdef double[:] bo = self._bufferOut

        self._predict(x, bo)
        self._reconstruct(bo, self._bufferIn)

        cdef np.ndarray[np.double_t, ndim=2] r = np.asmatrix(x) - self._bufferIn
        for i in range(self._crossbar.shape[0]):
            for j in range(self._crossbar.shape[1]):
                self._crossbar[i, j] = max(0.0, min(1.0, self._crossbar[i, j]
                        + self._ada.getDelta(i, j, -2. * bo[j] * r[0, i])))


    cpdef _predict(self, double[:] x, double[:] y):
        cdef double t, maxT, dt, xAvg, nSpikes, aHatFade, thresh
        cdef np.ndarray[np.double_t, ndim=2] corr, my, dots, states, ahat

        lamb = self.tLambda
        t = 0.0
        maxT = 1.0
        nSpikes = self.nSpikes
        dt = 0.1 / nSpikes
        xAvg = self.avgInput
        aHatFade = math.exp(math.log(0.7) / (0.25 / nSpikes)) ** dt
        thresh = x.shape[0] * xAvg * xAvg * 0.15 / nSpikes

        corr = self._crossbar.base.T * self._crossbar - np.identity(self.nOutputs)
        my = np.asmatrix(y)
        my[0, :] = 0.0

        dots = np.asmatrix(x) * self._crossbar
        states = np.asmatrix(np.zeros(self.nOutputs))
        ahat = np.asmatrix(np.zeros(self.nOutputs))

        Q1 = 1
        while t < maxT:
            ahat *= aHatFade
            states += (dots - states*0-lamb - ahat * corr) * dt
            for idx, val in np.ndenumerate(states):
                if val >= thresh:
                    if Q1 is None:
                        Q1 = 1
                        print("{} at {}: {} / {}".format(idx, t, val, thresh))
                    states[idx] = 0.
                    ahat[idx] += 1.
                    my[idx] += 1

            t += dt

        my[0, :] /= nSpikes


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef _reconstruct(self, double[:] y, double[:] r):
        cdef int i, j
        r[:] = 0.0
        for j in range(self.nOutputs):
            for i in range(self.nInputs):
                r[i] += self._crossbar[i, j] * y[j]
