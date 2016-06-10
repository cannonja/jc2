
#cython: profile=True

cimport cython
cimport numpy as np
from mr.modelBase cimport SklearnModelBase

from mr.adadelta import AdaDelta

import numpy as np


cdef class Perceptron(SklearnModelBase):
    UNSUPERVISED = False
    PARAMS = SklearnModelBase.PARAMS + [ 'adaDeltaRho', 'hidden' ]
    PICKLE_VARS = SklearnModelBase.PICKLE_VARS + [ '_ada', '_bias',
            '_weights' ]

    # Defaults
    cdef public double adaDeltaRho
    cdef public object hidden

    # Internal data
    cdef public object _ada, _bias, _weights

    def __init__(self, hidden = [], **kwargs):
        """A basic, sigmoidal perceptron."""
        defaults = {
                'adaDeltaRho': AdaDelta(1, 1).rho,
                'hidden': hidden,
        }
        defaults.update(kwargs)
        super(Perceptron, self).__init__(**defaults)


    cpdef _init(self, int nInputs, int nOutputs):
        # Scale weights so that with full-intensity input, the total value
        # will be on [ -1, 1 ].
        ins = nInputs
        self._weights = []
        self._bias = []
        self._ada = []
        for outs in self.hidden + [ nOutputs ]:
            n = 2. / ins
            self._weights.append(np.asmatrix(np.random.uniform(-n, n,
                    size = (ins, outs))))
            self._bias.append(np.random.uniform(-1., 1., size = (outs,)))
            self._ada.append(AdaDelta(ins + 1, outs, rho = self.adaDeltaRho))
            ins = outs


    def convergerProps(self):
        return [ self._weights, self._bias, self._ada ]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef _partial_fit(self, double[:] x, double[:] y):
        cdef int nout, i, j
        cdef double[:, :] ix = np.asmatrix(x)

        # Forward pass
        ins = []
        outs = []
        for w, b in zip(self._weights, self._bias):
            ins.append(np.asmatrix(x))
            sums = x * w + b
            ix = 1. / (1. + np.exp(-sums))
            outs.append(np.asarray(ix[0, :]))

        # Derivative of our activation function:
        # y = 1 / (1 + e^-t))
        # dy/dt = (1 + e^-t)^-2 * e^-t
        # t = -ln(1 / y - 1)
        # dy/dt = (1 + 1 / y - 1)^-2 * (1 / y - 1)
        # dy/dt = y * (1 - y)

        # Back-propagate the error to derivatives in terms of each neuron's
        # weighted sum
        error = np.asarray(y) - outs[len(outs) - 1]
        #print error

        for nout, o in reversed(list(enumerate(outs))):
            w = self._weights[nout]
            b = self._bias[nout]
            ada = self._ada[nout]
            nerror = error * np.maximum(1e-2, o * (1. - o))

            # Update weights
            allUps = np.asmatrix(ins[nout]).T * nerror
            for i in range(w.shape[1]):
                for j in range(w.shape[0]):
                    w[j, i] += ada.getDelta(j, i, -allUps[j, i])
                    #w[j, i] = min(1., max(-1., w[j, i]))
                b[i] += ada.getDelta(w.shape[0], i,
                        -nerror[i])
                b[i] = min(1., max(-1., b[i]))

            # To pass back more, nerror2 = nerror * self._weights.T
            error = np.asarray(nerror * w.T)[0, :]
            #print error
        #print("W {}\nB {}".format(self._weights, self._bias))


    cpdef _predict(self, double[:] x, double[:] y):
        cdef double[:, :] ix
        ix = np.asmatrix(x)
        for w, b in zip(self._weights, self._bias):
            sums = np.asmatrix(ix) * w + b
            ix = 1. / (1. + np.exp(-sums))
        y[:] = ix[0, :]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef _reconstruct(self, double[:] y, double[:] r):
        cdef int i, j

        cdef double[:, :] iy
        iy = np.asmatrix(y)

        for w, b in reversed(zip(self._weights, self._bias)):
            ybound = np.maximum(1e-4, np.minimum(0.9999, np.asmatrix(iy).T))
            yn = -np.log(1. / ybound - 1.)
            yn -= np.asmatrix(b).T
            ma = np.zeros((w.shape[1], w.shape[0]), dtype = float)
            for i in range(w.shape[1]):
                wt = w[:, i]
                maxCoef = wt[wt > 0].sum() / max(1, (wt > 0).sum())
                maxCoef = 0.
                for j in range(self.nInputs):
                    ma[i, j] = wt[j] if wt[j] >= maxCoef else 0.
            iy, _residuals, _rank, _s = np.linalg.lstsq(ma, yn)
        rn = np.asmatrix(r).transpose()
        rn[:] = iy
