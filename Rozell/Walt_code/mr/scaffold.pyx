
# cython: profile=True

from mr.modelBase cimport SklearnModelBase

cimport cython
cimport numpy as np

import math
import numpy as np
import scipy.misc
import sklearn

class Scaffold(SklearnModelBase):
    """Exposes an interface consistent with sklearn, that distributes data
    to an internal network of other models by some logic.  Capable of supervised
    and unsupervised communication.

    PS, great neural network resource: http://www.bogotobogo.com/python/python_Neural_Networks_Backpropagation_for_XOR_using_one_hidden_layer.php
    """

    class Score(object):
        DEFAULT = "MSE for unsupervised, r2 for supervised"
        CLASS = ("Supervised; 1 point if highest output from network is "
                "highest output from labels.  0 points otherwise.")

    PARAMS = SklearnModelBase.PARAMS + [ '_layerMakers', '_transforms',
            '_passes', 'scoreType' ]

    def __init__(self, **kwargs):
        self.layers = []
        self._layerMakers = []
        self._transforms = []
        self._passes = []
        kwargs.setdefault('nOutputs', 0)
        kwargs.setdefault('scoreType', self.Score.DEFAULT)
        super(Scaffold, self).__init__(**kwargs)


    def layer(self, estimator, transform = None):
        self._layerMakers.append(estimator)
        self._transforms.append(transform)
        self._passes.append(False)


    def passthroughLayer(self, estimator, transform = None):
        if len(self._layerMakers) == 0:
            raise ValueError("A passthrough cannot be the first layer, "
                    "as it depends on a prior layer's output for activation")
        self._layerMakers.append(estimator)
        self._transforms.append(transform)
        self._passes.append(True)


    def predictAllLayers(self, X, debug = False):
        """Given X, return a Y of same dimensionality as the number of layers
        in this scaffold."""
        self._checkDatasets(X, None, True)

        if not self._isInit:
            raise ValueError("{} must be initialized before "
                    "predictAllLayers()".format(self))

        Y = []
        cdef double[:, :] Yinner
        cdef double[:] x, y
        cdef double t
        cdef int i, j, k, w

        cdef double noutInv

        self._resetDebug(debug, len(X))
        nLayers = len(self.layers)
        for j in range(nLayers):
            Yinner = np.zeros((len(X), self.layers[j].nOutputs), dtype = float)
            Y.append(Yinner)
            for i in range(len(X)):
                # We're scanning the layers in example order, rather than doing
                # a single layer to completion.  Therefore, our index changes
                # each time, and is the sample index.
                self.debugInfo_.index = i
                self.layers[j].debugInfo_.index = i
                self._predict(X[i], Yinner[i], toLayer = j, fromLayer = j)

                # Each layer needs its own debug info
                if self._debug:
                    # No matter what layer, aggregate energy and simTime info
                    self.debugInfo_.energy[i] += (
                            self.layers[j].debugInfo_.energy[i])
                    self.debugInfo_.simTime[i] += (
                            self.layers[j].debugInfo_.simTime[i])

                    # Fill in layer's activityCount and outputCount data
                    lDbg = self.layers[j].debugInfo_
                    noutInv = 1. / self.layers[j].nOutputs
                    for k in range(self.layers[j].nOutputs):
                        t = Yinner[i, k]
                        if abs(t) >= 0.01:
                            lDbg.activityCount[i] += noutInv
                            lDbg.outputCount[k] += 1
                        lDbg.activitySqr[i] += t*t
                        lDbg.outputSqr[k] += t*t

                    if j == nLayers - 1:
                        # Last layer describes our overall activity / output
                        self.debugInfo_.activityCount[i] = lDbg.activityCount[i]
                        self.debugInfo_.activitySqr[i] = lDbg.activitySqr[i]
                        for k in range(self.nOutputs):
                            self.debugInfo_.outputCount[k] = lDbg.outputCount[k]
                            self.debugInfo_.outputSqr[k] = lDbg.outputSqr[k]

            # Input to next row is output of this row
            X = Yinner


        return Y


    def reconstructFromLayer(self, Y, layer):
        """Given layer's output, produce X, the original inputs."""
        self._checkDatasets(Y, None, True)

        if not self._isInit:
            raise ValueError("{} must be initialized before "
                    "reconstructFromLayer()".format(self))
        if layer >= len(self.layers):
            raise ValueError("No such layer? {} > {}".format(layer,
                    len(self.layers)))

        cdef int i
        cdef double[:, :] X = np.zeros((len(Y), self.nInputs), dtype=float)

        for i in range(len(Y)):
            self._reconstruct(Y[i], X[i], fromLayer=layer)
        return X


    def visualize(self, params, path, inputs = None, activity = None):
        """Dumps an image at path, broken down into layers according to
        scaffold's conventions.  params is the input shape

        input - If specified, an input to the network to visualize.

        activity - If specified, only the model is presented, but unsupervised
                layers are ordered according to the aggregate activity from
                processing this list of inputs.

                Incompatible with input.
        """
        w, h, channels = params
        if w * h * channels != self.nInputs:
            raise ValueError("Bad visualization params")

        if inputs is not None and activity is not None:
            raise ValueError("inputs and activity cannot both be specified")

        # We have each layer...
        imw = w
        for i, l in reversed(list(enumerate(self.layers))):
            if not self._passes[i]:
                imw = max(imw, w * l.nOutputsConvolved)
            else:
                imw = max(imw, w * l[0].nOutputs * len(l))
        imh = h * len(self.layers)
        if inputs is not None:
            imh += h
        imdata = np.zeros((imh, imw, channels), dtype = np.uint8)

        def split(layerX, layerY, j):
            """returns x, y, c for x coord, y coord, and color channel of
            input j."""
            c = j % channels
            j //= channels
            ox = layerX + (j % w)
            oy = layerY + (j // w)
            return (ox, oy, c)

        if inputs is not None:
            # First image is source, then we head to the right as we get
            # further and further down our layers
            for j in range(self.nInputs):
                ox, oy, c = split(0, h*len(self.layers), j)
                imdata[oy, ox, c] = int(
                        255 * max(0.0, min(1.0, inputs[j])))

            layerIns = np.ones(self.nInputs)
            for i, l in enumerate(self.layers):
                layerOuts = np.ones(l.nOutputs, dtype = float)
                self._predict(inputs, layerOuts, toLayer = i)
                self._reconstruct(layerOuts, layerIns, fromLayer = i)
                for j in range(self.nInputs):
                    ox, oy, c = split(w*(i+1), h*len(self.layers), j)
                    imdata[oy, ox, c] = int(
                            255 * max(0.0, min(1.0, layerIns[j])))

        recBuffer = np.zeros(len(self._buffer), dtype = float)
        for i, l in enumerate(self.layers):
            layerOutCount = 0
            if not self._passes[i]:
                layerOutCount = l.nOutputs
            else:
                layerOutCount = len(l) * l[0].nOutputs

            layerOuts = np.arange(-layerOutCount, 0, dtype=float)
            if inputs is not None:
                self._predict(inputs, layerOuts, toLayer = i)
            elif activity is not None:
                layerOuts[:] = 0.
                for a in activity:
                    layerOutsAdd = np.zeros(l.nOutputs, dtype=float)
                    self._predict(a, layerOutsAdd, toLayer = i)
                    layerOuts += abs(layerOutsAdd)

            if not self._passes[i]:
                bufOut = recBuffer[:l.nOutputs]
                xw = imw / l.nOutputsConvolved
                # Most active first!
                if l.nOutputsConvolved != layerOuts.shape[0]:
                    raise ValueError("Bad assumption: {} != {}".format(
                            l.nOutputsConvolved, layerOuts.shape[0]))
                xorder = [ (xi, act) for xi, act in enumerate(layerOuts) ]
                if l.UNSUPERVISED:
                    xorder = sorted(xorder, key = lambda m: -abs(m[1]))
                for xi, liP in enumerate(xorder):
                    li = liP[0]
                    bufOut[:] = 0.0
                    for lii in range(li, l.nOutputs, l.nOutputsConvolved):
                        bufOut[lii] = 1.0 * layerOuts[lii]
                        if inputs is None:
                            bufOut[lii] = 1.
                            break
                    self._reconstruct(bufOut, self._bufferIn, fromLayer = i)
                    for j in range(self.nInputs):
                        ox, oy, c = split(xi*xw, i*h, j)
                        imdata[oy, ox, c] = int(
                                255 * max(0.0, min(1.0, self._bufferIn[j])))
            else:
                raise NotImplementedError("This code no longer applies since "
                        "integrating sorting by activity")
                no = l[0].nOutputs
                bufOut = recBuffer[:no * len(l)]
                xw = imw / len(l)
                for ii, ll in enumerate(l):
                    for li in range(no):
                        bufOut[:] = 0.0
                        bufOut[ii * no + li] = 1.0 * layerOuts[ii * no + li]
                        self._reconstruct(bufOut, self._bufferIn, fromLayer = i)
                        for j in range(self.nInputs):
                            ox, oy, c = split(ii * xw + li * w, i * h, j)
                            imdata[oy, ox, c] = int(
                                    255 * max(0.0, min(1.0, self._bufferIn[j])))

        if channels == 1:
            imdata = imdata[:, :, 0]
        scipy.misc.imsave(path, imdata)


    def _init(self, nInputs, nOutputs):
        # Make our layers
        if not self._layerMakers:
            raise ValueError("No layers specified")
        self.layers = []
        lastOuts = nInputs
        nextOuts = nOutputs
        lastPassed = nInputs
        maxOuts = max(nInputs, nOutputs)
        unsupervised = True
        def make(typ, nInputs, nOutputs):
            r = sklearn.clone(typ)
            r.init(nInputs, nOutputs)
            return r
        for i in range(len(self._layerMakers)):
            lastUnsupervised = unsupervised

            if not self._passes[i]:
                lastPassed = lastOuts
                if self._transforms[i]:
                    newIns = self._transforms[i][0](np.zeros(lastOuts))
                    lastOuts = len(newIns)
                self.layers.append(make(self._layerMakers[i],
                        lastOuts or self._layerMakers[i].nOutputs,
                        nextOuts))
                lastOuts = self.layers[-1].nOutputs
                unsupervised = self.layers[-1].UNSUPERVISED
            else:
                self.layers.append([ make(self._layerMakers[i], lastPassed,
                        nextOuts) for _ in range(lastOuts) ])
                lastOuts = lastOuts * self.layers[-1][0].nOutputs
                unsupervised = self.layers[-1][0].UNSUPERVISED

            if not lastUnsupervised and unsupervised:
                raise ValueError("Supervised must be stacked at end, for now.")
            maxOuts = max(maxOuts, lastOuts)

        self._buffer = np.zeros(maxOuts, dtype = float)
        self._buffer2 = np.zeros(maxOuts, dtype = float)
        self.nOutputs = lastOuts
        self.UNSUPERVISED = unsupervised


    def score(self, X, y=None, debug = False):
        # Configure our scoring function
        if self.scoreType == self.Score.DEFAULT:
            return super(Scaffold, self).score(X, y, debug = debug)
        elif self.scoreType == self.Score.CLASS:
            if self.UNSUPERVISED:
                raise ValueError("Cannot use Score.CLASS with unsupervised")
            elif y is None:
                raise ValueError("Class score requires expected output, y")
            return self.score_class(X, y, debug = debug)

        raise ValueError("Unrecognized scoreType: {}".format(
                self.scoreType))


    def score_class(self, X, y, debug = False):
        """Returns the % correct."""
        cdef int i, j
        self._checkDatasets(X, y)

        pX = self.predict(X, debug = debug)

        cdef double score = 0.0
        cdef double scorePer = 1. / len(X)
        for i in range(pX.shape[0]):
            if (np.asarray(pX[i]).argmax()
                    == np.asarray(y[i]).argmax()):
                score += scorePer

        return score


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def _partial_fit(self, double[:] x, double[:] y):
        cdef int i, j
        cdef double[:] ins = x
        cdef double[:] passed = x
        cdef double[:] outs = None
        cdef double activeDiv
        cdef double[:] _buffer
        cdef SklearnModelBase layer, llayer
        for i in range(len(self.layers)):
            # ins is output from last layer, passed is passed from however
            # long ago
            layer = self.layers[i]
            # Refresh buffer, may have changed since _swapBuffers() call
            _buffer = self._buffer
            if not self._passes[i]:
                passed = ins
                if self._transforms[i] is not None:
                    ins = self._transforms[i][0](ins)
                outs = _buffer[:layer.nOutputs]
                layer.t_ = self.t_
                layer._partial_fit(ins, None if layer.UNSUPERVISED else y)
                layer._predict(ins, outs)
            else:
                # Pass-through layer.  We output zeroes for anything not
                # active from the prior outputs
                actives = ins.copy()
                ins = passed
                if self._transforms[i]:
                    ins = self._transforms[i][0](ins)
                n = len(layer)
                nPer = layer[0].nOutputs
                outs = _buffer[:n * nPer]
                # Since we reconstruct additively, we want to preserve energy
                # in passthrough
                activeDiv = 0.0
                for j in range(len(layer)):
                    activeDiv += abs(actives[j])
                activeDiv = 1.0 / max(activeDiv, 1e-6)
                #activeDiv = 1.0
                for j in range(len(layer)):
                    llayer = layer[j]
                    layerOuts = outs[j * nPer:j * nPer + nPer]
                    if abs(actives[j]) >= 1e-10:
                        insFit = ins
                        if False:
                            # Dot product RF of activating neuron with input
                            # data
                            insFit = ins.copy()
                            y2 = np.zeros(self.layers[i - 1].nOutputs,
                                    dtype = float)
                            y2[:] = 0.0
                            y2[j] = 1.0
                            insDot = np.zeros(self.layers[i - 1].nInputs,
                                    dtype = float)
                            self.layers[i - 1]._reconstruct(y2, insDot)
                            if self._transforms[i - 1]:
                                insDot = self._transforms[i - 1][1](insDot)
                            for k in range(llayer.nInputs):
                                insFit[k] *= insDot[k]
                        llayer._partial_fit(insFit,
                                None if llayer.UNSUPERVISED else y)
                        llayer._predict(insFit,
                                layerOuts)
                        #layerOuts *= actives[j] * activeDiv
                    else:
                        layerOuts[:] = 0.0
            self._swapBuffers()
            ins = outs


    def _predict(self, double[:] x, double[:] y, toLayer = None,
            fromLayer = None):
        """Predict a slice of this scaffold.

        toLayer - If set, process up to and including this layer
        fromLayer - If set, process only from this layer onwards.  The input
                set x is assumed to be the appropriate format.
        """
        cdef double[:] ins = x
        cdef double[:] passed = x
        cdef double[:] outs = None
        cdef double[:] _buffer
        cdef int minLayer = 0
        cdef int maxLayer = len(self.layers)
        cdef SklearnModelBase layer
        if toLayer is not None:
            maxLayer = toLayer + 1
        if fromLayer is not None:
            minLayer = fromLayer
        for i in range(minLayer, maxLayer):
            # Refresh buffer after _swapBuffers
            _buffer = self._buffer
            if not self._passes[i]:
                layer = self.layers[i]
                passed = ins
                if self._transforms[i] is not None:
                    ins = self._transforms[i][0](ins)
                outs = _buffer[:layer.nOutputs]
                layer._predict(ins, outs)
            else:
                # Pass-through layer
                actives = ins
                ins = passed
                if self._transforms[i]:
                    ins = self._transforms[i][0](ins)
                n = len(self.layers[i])
                nPer = self.layers[i][0].nOutputs
                outs = _buffer[:n * nPer]
                activeDiv = 0.0
                for j in range(len(self.layers[i])):
                    activeDiv += abs(actives[j])
                activeDiv = 1.0 / max(activeDiv, 1e-6)
                activeDiv = 1.0
                for j in range(len(self.layers[i])):
                    layer = self.layers[i][j]
                    layerOuts = outs[j * nPer:j * nPer + nPer]
                    if abs(actives[j]) >= 1e-10:
                        layer._predict(ins, layerOuts)
                        #layerOuts *= actives[j] * activeDiv
                    else:
                        layerOuts[:] = 0.0
            self._swapBuffers()
            ins = outs
        if len(ins) != len(y):
            raise ValueError("Ahhh! {} vs {}".format(len(ins), len(y)))
        y[:] = ins


    def _reconstruct(self, double[:] y, double[:] r, fromLayer = None):
        cdef int i
        cdef double[:] lastR = None
        cdef SklearnModelBase layer
        passSkip = False
        if fromLayer is None:
            fromLayer = len(self.layers) - 1
        for i in range(fromLayer, -1, -1):
            if not self._passes[i]:
                if passSkip:
                    passSkip = False
                    continue
                layer = self.layers[i]
                lastR = self._buffer[:layer.nInputs]
                layer._reconstruct(y, lastR)
                if self._transforms[i] is not None:
                    lastR = self._transforms[i][1](lastR)
            else:
                passSkip = True
                # Average across passthrough
                lastR = self._buffer[:self.layers[i][0].nInputs]
                lastR[:] = 0.0
                tmp = np.zeros(self.layers[i][0].nInputs, dtype = float)
                no = self.layers[i][0].nOutputs
                for j in range(len(self.layers[i])):
                    yp = y[j*no:j*no+no]
                    self.layers[i][j]._reconstruct(yp, tmp)
                    lastR += tmp
                passSkip = True
            y = lastR
            self._swapBuffers()

        if len(lastR) != len(r):
            raise ValueError("Something wrong, {} != {}".format(
                    len(lastR), len(r)))
        r[:] = lastR


    def _resetDebug(self, debug, int lenX):
        SklearnModelBase._resetDebug(self, debug, lenX)
        for l in self.layers:
            l._resetDebug(debug, lenX)


    def _swapBuffers(self):
        """Swaps self._buffer and self._buffer2.  Useful to prevent overwriting
        data between layers."""
        t = self._buffer
        self._buffer = self._buffer2
        self._buffer2 = t
