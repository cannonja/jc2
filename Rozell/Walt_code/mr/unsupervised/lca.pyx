
#cython: profile=True

from mr.modelBase cimport SklearnModelBase
cimport cython
cimport numpy as np

cdef extern from "math.h":
    double sqrt(double)


import mr.util as util

import math
import numpy as np

cdef class Lca(SklearnModelBase):
    """Rozell et al's LCA algorithm with Schultz et al's training."""

    UNSUPERVISED = True
    PARAMS = SklearnModelBase.PARAMS + [ 'tAlpha', 'tGamma',
            'tLambda', 'tLambdaTarget', 'simSteps', 'simLambda', 'homeoMin',
            'homeoDown', 'schultzDt', 'schultzDtScalePerDouble',
            'tLambdaTargetRate', 'learnMomentum', 'adaDeltaRho' ]
    PICKLE_VARS = SklearnModelBase.PICKLE_VARS + [ '_crossbar', '_dots',
            '_homeostasis', '_lambdaMod', '_lambdaBounds',
            '_eG2', '_edX', '_learnMoments']

    # Model parameters
    cdef public double tAlpha, tGamma, tLambda, tLambdaTarget
    cdef public double tLambdaTargetRate
    cdef public int simSteps
    cdef public double simLambda, homeoMin, homeoDown, schultzDt
    cdef public double schultzDtScalePerDouble

    # Adadelta stuff
    cdef public double learnMomentum
    cdef public double adaDeltaRho
    cdef public double[:, :] _eG2, _edX, _learnMoments

    # Model temporaries
    cdef public double[:, :] _crossbar
    cdef public double[:] _dots, _homeostasis
    cdef public double _lambdaMod
    cdef public object _lambdaBounds

    cpdef convergerProps(self):
        return [ self._crossbar, self._homeostasis, self._eG2, self._edX ]

    def __init__(self, nOutputs = 10, **kwargs):
        defaults = {
                'nOutputs': nOutputs,
                'tAlpha': 0.21737,
                'tGamma': 10000,
                'tLambda': 0.121,
                'tLambdaTarget': 0,
                'tLambdaTargetRate': 0.0001,
                'simSteps': 34,
                'simLambda': 0.2,
                'homeoMin': 1.+0*0.419,
                'homeoDown': 0.3,
                'schultzDt': 0.811111,
                'schultzDtScalePerDouble': 0.8,

                'learnMomentum': -1,
                'adaDeltaRho': 0.9,
        }
        defaults.update(kwargs)
        super(Lca, self).__init__(**defaults)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _fillDots(self, double[:] x):
        """Given inputs x, fill self._dots with the dot product for each
        column in the crossbar."""
        cdef int j
        cdef np.ndarray[np.double_t, ndim=2] ins = np.asmatrix(x)

        for j in range(self.nOutputs):
            self._dots[j] = ins * self._crossbar[:, j]


    cpdef _init(self, int nInputs, int nOutputs):
        self._crossbar = np.matrix(np.random.uniform(size=(nInputs, nOutputs)),
                dtype = float)
        self._learnMoments = np.asmatrix(np.zeros((nInputs, nOutputs), dtype = float))
        self._eG2 = np.asmatrix(np.zeros((nInputs, nOutputs), dtype = float))
        self._edX = np.asmatrix(np.zeros((nInputs, nOutputs), dtype = float))
        self._eG2[:, :] = 1e-3
        self._edX[:, :] = 1e-3
        self._dots = np.zeros(nOutputs, dtype = float)
        self._homeostasis = np.ones(nOutputs, dtype = float)
        self._lambdaMod = 1.0
        self._lambdaBounds = [ 1e300, -1e300 ]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _integrate(self, double[:] x, double[:] y):
        """Integrate x into y, given self._dots has already been calculated."""
        cdef int i, j

        # Correlation calculation between output receptive fields
        cdef np.ndarray[np.double_t, ndim=2] cb = self._crossbar.base
        cdef np.ndarray[np.double_t, ndim=2] corr = (cb.T * cb
                - np.identity(self.nOutputs))

        # Makes a 1xN matrix for the actual output, which maps to the y array
        cdef np.ndarray[np.double_t, ndim=2] my = np.asmatrix(y)
        my[0,:] = 0.0

        cdef np.ndarray[np.double_t, ndim=1] states = np.zeros(self.nOutputs,
                dtype = float)

        lastChange = 1e30
        cdef np.ndarray[np.double_t, ndim=1] changes = np.zeros(self.nOutputs,
                dtype = float)

        cdef double expVal, u
        cdef int nActive

        step = 0
        while True:
            nActive = 0
            for j in range(self.nOutputs):
                u = states[j]
                # math.exp() has overflow beyond 700
                expVal = -self.tGamma * (u - self.tLambda * self._lambdaMod)
                expVal = min(expVal, 700)
                my[0,j] = (u - self.tAlpha * self.tLambda * self._lambdaMod) / (1.0
                        + math.exp(expVal))
                if my[0, j] >= self.tLambda * self._lambdaMod:
                    nActive += 1
                else:
                    my[0, j] = 0.0

            if self.tLambdaTarget > 0:
                self._lambdaMod *= 1.0 + self.tLambdaTargetRate * (
                        1.0 * nActive / self.nOutputs - self.tLambdaTarget)
                if self._lambdaMod < self._lambdaBounds[0]:
                    self._lambdaBounds[0] = self._lambdaMod
                elif self._lambdaMod > self._lambdaBounds[1]:
                    self._lambdaBounds[1] = self._lambdaMod

            if lastChange < 1e-4 or step >= self.simSteps:
                break

            step += 1
            lastChange = 0.0
            scale = my * corr

            for j in range(self.nOutputs):
                changes[j] = (self._homeostasis[j] * self._dots[j] - states[j]
                        - scale[0,j])
                lastChange += changes[j] * changes[j]

            lastChange = math.sqrt(max(1e-100, lastChange))
            stepLambda = self.simLambda / lastChange

            for j in range(self.nOutputs):
                old = states[j]
                states[j] += stepLambda * changes[j]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef _partial_fit(self, double[:] x, double[:] y):
        # Predict, and then update according to schultz equation
        cdef int i, j
        cdef double cbDelta, e = 1e-6
        cdef double g, gScalar
        cdef double adRho = self.adaDeltaRho

        self._predict(x, self._bufferOut)
        self._reconstruct(self._bufferOut, self._bufferIn)

        cdef np.ndarray[np.double_t, ndim=2] r = np.asmatrix(x) - self._bufferIn
        # dt(n) = 1/t^n
        # dt(2n) = 1/(2^n * t^n)
        # dt(2n) / dt(n) = 2^-n = self.schultzDtScalePerDouble
        # n = -log(schultzScale) / log(2)
        dt_n = -np.log(self.schultzDtScalePerDouble) / np.log(2)
        dt = self.schultzDt / math.pow(self.t_, dt_n)

        for j in range(self.nOutputs):
            # For each output, we multiply the residual by schultz dt by the
            # output magnitude, and call it good
            if abs(self._bufferOut[j]) < 1e-3:
                continue

            for i in range(self.nInputs):
                if adRho > 0:
                    # Adadelta
                    # Score function: R^2 = (y - wx)^2
                    # dR/dW = 2*(-x)*(y - wx)
                    g = -2 * self._bufferOut[j] * r[0, i]
                    self._eG2[i, j] = adRho * self._eG2[i, j] + (1 - adRho) *g*g
                    gScalar = sqrt((self._edX[i, j] + e) / (self._eG2[i, j] + e))
                    cbDelta = -gScalar * g
                    if False and j == self.nOutputs // 2 and i == self.nInputs // 2:
                        print("Change: {}, (scale {}, {} / {})".format(cbDelta, gScalar, self._edX[i, j], self._eG2[i, j]))
                    self._edX[i, j] = adRho * self._edX[i, j] + (1 - adRho) * cbDelta * cbDelta
                else:
                    # Schultz dt with 1/n^alpha scaling
                    cbDelta = r[0,i] * self._bufferOut[j] * dt

                if self.learnMomentum > 0:
                    # Traditional momentum
                    cbDelta += self._learnMoments[i, j] * self.learnMomentum
                    self._learnMoments[i, j] = cbDelta

                self._crossbar[i, j] = max(0.0, min(1.0, self._crossbar[i, j]
                        + cbDelta))

        # Update homeostasis (only for fit, though, since homeostasis has no
        # place in evaluation after learning)
        for j in range(self.nOutputs):
            self._homeostasis[j] += ((1.0 - self._homeostasis[j])
                    / max(1, self.nOutputs - 1)
                    - self.homeoDown * self._bufferOut[j])
            self._homeostasis[j] = min(1.0, max(self.homeoMin,
                    self._homeostasis[j]))


    cpdef _predict(self, double[:] x, double[:] y):
        """Turns inputs x into outputs y."""
        cdef int i, j, dbgI
        cdef double rMin, rMax, e, v, r, w

        self._fillDots(x)
        self._integrate(x, y)
        if self._debug:
            dbgI = self.debugInfo_.index

            # Calculate crossbar drain using virtual ground.
            rMin = 52279.
            rMax = 206969.
            e = 0.
            self.debugInfo_.simTime[dbgI] = 1.
            for i in range(self._crossbar.shape[0]):
                v = 0.7 * x[i]
                # Bias column
                e += v*v / rMax
                for j in range(self._crossbar.shape[1]):
                    # From memristor survey paper
                    w = self._crossbar[i, j]
                    r = rMin * rMax / (w * rMax + (1. - w) * rMin)
                    e += v*v / r
            self.debugInfo_.energy[dbgI] = e


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef _reconstruct(self, double[:] y, double[:] r):
        cdef int i, j
        r[:] = 0.0
        for j in range(self.nOutputs):
            for i in range(self.nInputs):
                r[i] += self._crossbar[i, j] * y[j]
