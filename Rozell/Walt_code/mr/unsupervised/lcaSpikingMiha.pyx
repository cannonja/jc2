
cimport cython

cdef extern from "math.h":
    double fabs(double)

from mr.modelBase import SklearnModelBase

import math
import numpy as np
import sys

cdef class SpikingExciter:
    cdef public SpikingLcaLayer layer
    cdef public double[:, :] weights

    def __init__(self, a, b):
        self.layer = a
        self.weights = b


cdef class SpikingLcaLayer:
    cdef public object network
    cdef public int nStates

    cdef public object excites
    cdef public double Vr, Ve, Vi, dVth, tau

    cdef public double[:] states, pstates, nstates, threshes, counts

    def __init__(self, network, nStates):
        self.network = network
        self.nStates = nStates


    def reset(self, excites):
        self.excites = [ SpikingExciter(a, b) for a, b in excites ]
        self.Vr = self.network.Vr
        self.Ve = self.network.Ve
        self.Vi = self.network.Vi
        self.dVth = self.network.dVth
        self.tau = self.network.tau

        self.states = np.zeros(self.nStates, dtype = float)
        self.pstates = np.zeros(self.nStates, dtype = float)
        self.nstates = np.zeros(self.nStates, dtype = float)
        self.threshes = np.zeros(self.nStates, dtype = float)
        self.threshes[:] = self.dVth
        self.counts = np.zeros(self.nStates, dtype = float)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void step(self, double dt):
        cdef int i, o
        cdef SpikingLcaLayer layer
        cdef SpikingExciter exciter
        cdef double w
        cdef int idt
        cdef double sdt

        # Integrate!
        for o in range(self.nStates):
            if o == 0 and False:
                print("At {}, state {} d {}, 1 {}, 2 {}, 3 {}".format(self.network.t_, self.states[o],
                        dt * self.tau * (
                                (self.Vr - self.states[o])
                                + (self.Ve - self.states[o]) * self.pstates[o]
                                + (self.Vi - self.states[o]) * self.nstates[o]),
                                (self.Vr - self.states[o]),
                                (self.Ve - self.states[o]) * self.pstates[o],
                                (self.Vi - self.states[o]) * self.nstates[o]))

            w = self.tau * (
                    (self.Vr - self.states[o])
                    + (self.Ve - self.states[o]) * self.pstates[o]
                    + (self.Vi - self.states[o]) * self.nstates[o])
            idt = int(max(1.0, fabs(w) * dt / 0.1))
            sdt = dt / idt
            for i in range(idt):
                self.states[o] += sdt * self.tau * (
                        (self.Vr - self.states[o])
                        + (self.Ve - self.states[o]) * self.pstates[o]
                        + (self.Vi - self.states[o]) * self.nstates[o])
                self.pstates[o] += sdt * self.tau * -self.pstates[o]
                self.nstates[o] += sdt * self.tau * -self.nstates[o]
                self.threshes[o] += sdt * self.tau * -self.threshes[o]

        # Fire!
        for o in range(self.nStates):
            if self.states[o] > self.threshes[o]:
                if o == 0 and False:
                    print("At {}".format(self.network.t_))
                    print("dState 0: {}".format(dt * self.tau * (
                            (self.Vr - self.states[o])
                            + (self.Ve - self.states[o]) * self.pstates[o]
                            + (self.Vi - self.states[o]) * self.nstates[o])))
                    print("Firing 0: {} / {}, {}, {}".format(self.states[o], self.threshes[o],
                            self.pstates[o], self.nstates[o]))
                self.counts[o] += 1
                self.states[o] = self.Vr
                self.threshes[o] += self.dVth

                for exciter in self.excites:
                    layer = exciter.layer
                    for i in range(layer.nStates):
                        w = exciter.weights[o, i]
                        if w > 0:
                            layer.pstates[i] += w
                        else:
                            # w < 0, so this adds
                            layer.nstates[i] -= w


class LcaSpikingMiha(SklearnModelBase):
    UNSUPERVISED = True
    PARAMS = SklearnModelBase.PARAMS + [ 'Vr', 'Ve', 'Vi', 'dVth', 'tau',
            'debug' ]

    def __init__(self, nOutputs = 10, **kwargs):
        defaults = {
                'Vr': 0.0,
                'Ve': 1.0,
                'Vi': -1.0,
                'dVth': 0.20,
                'tau': 1.0,
                'nOutputs': nOutputs,
                'debug': False,
        }
        defaults.update(kwargs)
        super(LcaSpikingMiha, self).__init__(**defaults)


    def _init(self, nInputs, nOutputs):
        self._crossbar = np.asmatrix(np.random.uniform(
                size = (nInputs, nOutputs)))
        self._eG2 = np.asmatrix(np.zeros((nInputs, nOutputs), dtype = float))
        self._edX = np.asmatrix(np.zeros((nInputs, nOutputs), dtype = float))

        self._eG2[:, :] = 1e-3
        self._edX[:, :] = 1e-3


    def _partial_fit(self, x, y):
        e = 1e-6
        adRho = 0.9

        self._predict(x, self._bufferOut)
        self._reconstruct(self._bufferOut, self._bufferIn)

        r = np.asmatrix(x) - self._bufferIn

        for j in range(self.nOutputs):
            if abs(self._bufferOut[j]) < 1e-2:
                continue

            for i in range(self.nInputs):
                g = -2 * self._bufferOut[j] * r[0, i]
                self._eG2[i, j] = adRho * self._eG2[i, j] + (1 - adRho) * g * g
                gScalar = math.sqrt((self._edX[i, j] + e) / (self._eG2[i, j] + e))
                cbDelta = -gScalar * g
                self._edX[i, j] = adRho * self._edX[i, j] + (1 - adRho) * cbDelta * cbDelta

                self._crossbar[i, j] = max(0.0, min(1.0, self._crossbar[i, j]
                        + cbDelta))


    def _predict(self, double[:] x, y):
        cdef int loop, i
        cdef SpikingLcaLayer resPos = SpikingLcaLayer(self, self.nInputs)
        cdef SpikingLcaLayer resNeg = SpikingLcaLayer(self, self.nInputs)
        cdef SpikingLcaLayer v1 = SpikingLcaLayer(self, self.nOutputs)
        cdef SpikingLcaLayer outputs = SpikingLcaLayer(self, self.nOutputs)

        v1.reset([ (v1, -np.eye(self.nOutputs)),
                (outputs, np.eye(self.nOutputs)),
                (resPos, -self._crossbar.T), (resNeg, self._crossbar.T) ])
        resPos.reset([ (v1, self._crossbar) ])
        resNeg.reset([ (v1, -self._crossbar) ])
        outputs.reset([])

        cdef double dt = 0.01
        cdef double maxInputRate = (1. / dt)
        # Don't start loop at 1, so that it's easy to skip "everything fires"
        # first step
        self.t_ = 0.0
        for loop in range(1, 1+int(100.0 / dt)):
            # Stimulate!
            for i in range(self.nInputs):
                if loop % int(maxInputRate / max(1e-6, x[i])) == 0:
                #if np.random.uniform() < x[i] / 10.0:
                    #for o in range(self.nOutputs):
                    #    f = self._crossbar[i, o]
                    #    pstates[o] += f
                    resPos.pstates[i] += 1.0
                    #resNeg.nstates[i] += 1.0

            resPos.step(dt)
            resNeg.step(dt)
            v1.step(dt)
            # do NOT step outputs, as we just use pstates as a counter.
            self.t_ += dt

        if self.debug:
            sys.stderr.write(
                    "Total spikes:\n    V1: {}\n    Pos: {}\n    Neg: {}\n".format(
                    np.asarray(v1.counts), np.asarray(resPos.counts),
                    np.asarray(resNeg.counts)))
        y[:] = np.asarray(outputs.pstates) / 15.0


    def _reconstruct(self, y, r):
        np.asmatrix(r)[:] = np.asmatrix(y) * self._crossbar.T
