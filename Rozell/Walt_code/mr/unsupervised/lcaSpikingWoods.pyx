
#cython: profile=True

cimport cython
from mr.electrical.odeCircuit cimport OdeCircuit

cdef extern from "math.h":
    double log(double)
    double sqrt(double)

from mr.electrical import odeCircuit as circuit
from mr.modelBase import SklearnModelBase
from mr import modelBaseUnpickler

import math
import numpy as np
import scipy.optimize
import sys

class LcaSpikingWoods(SklearnModelBase):
    UNSUPERVISED = True
    PARAMS = SklearnModelBase.PARAMS + [ 'avgInput', 'simTime', 'nHomeo',
            'nSpikes', 'spikeDensity', 'circuit_rMin', 'circuit_rMax',
            'untrainedK', 'vEventSet', 'tolerance_vFire' ]

    def __init__(self, nOutputs = 10, **kwargs):
        defaults = {
                'nOutputs': nOutputs,
                'avgInput': 0.1,
                'simTime': 0.1,
                'nHomeo': 1,
                'nSpikes': 10.0,
                'spikeDensity': 0.1,
                'circuit_rMin': 35e3,
                'circuit_rMax': 200e3,
                'untrainedK': 2.0,
                'vEventSet': 1.0,

                # +/- for various properties
                'tolerance_vFire': 0.001,
        }
        defaults.update(**kwargs)
        super(LcaSpikingWoods, self).__init__(**defaults)


    def _init(self, nInputs, nOutputs):
        self._eG2 = np.zeros((nInputs, nOutputs), dtype = float)
        self._edX = np.zeros((nInputs, nOutputs), dtype = float)
        self._eG2[:, :] = 1e-3
        self._edX[:, :] = 1e-3
        self._circuit = SubCircuit(nInputs, nOutputs, self.nHomeo,
                avgInput = self.avgInput,
                # Often a smaller step is taken.  However, for control reasons,
                # we need to do spiking calculations at least once per spike
                # duration.
                dtMax = self.simTime * 0.2e-0 * self.spikeDensity / self.nSpikes,
                dtMin = self.simTime * 0.2e-1 * self.spikeDensity / self.nSpikes,
                # The width of a single spike
                dtSpike = self.simTime / self.nSpikes,
                spikeWidth = self.simTime / self.nSpikes * self.spikeDensity,
                spikeDensity = self.spikeDensity,
                untrainedK = self.untrainedK,
                rMax = self.circuit_rMax,
                rMin = self.circuit_rMin,
                vEventSet = self.vEventSet,
                tolerance_vFire = self.tolerance_vFire)

        # Fix homeostasis to max
        self._homeostasis = np.zeros((nOutputs,), dtype = float)
        self.cNeuron_ = self._circuit.cNeuron
        self.vFire_ = self._circuit.vFire


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def _partial_fit(self, double[:] x, double[:] y):
        cdef int i, j

        cdef double adRho = 0.9, e = 1e-6
        cdef double g, gScalar, cbDelta

        cdef double[:] bufIn = self._bufferIn, bufOut = self._bufferOut
        cdef double[:,:] _eG2 = self._eG2, _edX = self._edX, _crossbar = self._circuit.crossbar

        self._predict(x, bufOut)
        self._reconstruct(bufOut, bufIn)

        for i in range(bufIn.shape[0]):
            bufIn[i] = x[i] - bufIn[i]

        for j in range(self.nOutputs):
            if abs(bufOut[j]) < 1e-2:
                continue

            for i in range(self.nInputs):
                g = -2 * bufOut[j] * bufIn[i]
                _eG2[i, j] = adRho * _eG2[i, j] + (1 - adRho) * g * g
                gScalar = sqrt((_edX[i, j] + e) / (_eG2[i, j] + e))
                cbDelta = -gScalar * g
                _edX[i, j] = adRho * _edX[i, j] + (1 - adRho) * cbDelta * cbDelta

                _crossbar[i, j] = max(0.0, min(1.0,
                        _crossbar[i, j] + cbDelta))

        self._updateHomeostasis(bufOut)


    def _predict(self, double[:] x, double[:] y):
        cdef int i
        cdef double invSpikes = 1.0 / self.nSpikes
        cdef double[:] r = self._circuit.run(self.simTime, x, self._homeostasis)
        for i in range(self.nOutputs):
            y[i] = r[i] * invSpikes

        if self._debug:
            self.debugInfo_.energy[self.debugInfo_.index] += (
                    self._circuit.energy)
            self.debugInfo_.simTime[self.debugInfo_.index] += (
                    self._circuit.simTime)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _reconstruct(self, double[:] y, double[:] r):
        cdef int i, j
        cdef double[:, :] _crossbar = self._circuit.crossbar
        r[:] = 0.0
        for j in range(self.nOutputs):
            for i in range(self.nInputs):
                r[i] += _crossbar[i, j] * y[j]


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def _updateHomeostasis(self, double[:] y):
        cdef int i
        cdef int nOut = self.nOutputs
        cdef double dh, a, b
        cdef double[:] _homeostasis = self._homeostasis
        # Normalize output and drop / recharge homeostasis
        for i in range(nOut):
            #dh = (1.0 - self._homeostasis[i]) * 0.1 - y[i] * 1.0
            #dh = 0.1 - y[i]
            #dh = -0.2 * y[i] + (1.0 - self._homeostasis[i]) / max(1, self.nOutputs - 1)
            #self._homeostasis[i] = max(0.0, self._homeostasis[i] + dh)
            dh = ((1.0 - _homeostasis[i]) / max(1, nOut - 1)
                    - 0.1 * y[i])
            # dh = a*(1-h) - b*y = 0
            # h = (a - by) / a = 1 - b/a * y
            # b/a = (1 - h) / y
            # If not firing, dh + ah - a = 0 -> sL - h0 + aL - a/s = 0
            #     L = (h0 + a/s) / (s + a)
            #     (sh0 + a)/s(s+a) = A/s + B/(s+a)
            #     A = 1, B = h0 - 1
            #     h = 1 + (h0 - 1) * e^(-at)
            # a = time constant = X / nOutputs
            # b = a * (1 - h) / y
            a = 1.0 / nOut
            b = a * (1 - 0.5) / (1.0 / nOut)
            dh = ((1. - _homeostasis[i]) * a - b * y[i])
            _homeostasis[i] = min(1.0, max(0.0, _homeostasis[i] + dh))



cdef class SubCircuit(OdeCircuit):
    cdef public int nInputs, nOutputs, nHomeo
    cdef public double avgInput, vEventSet, vFire, cNeuron
    cdef public double rMin, rMax
    cdef public double dtSpike, spikeWidth, untrainedK
    cdef public double tolerance_vFire, spikeDensity

    cdef double[:] memristorTable

    cdef public double[:, :] crossbar
    cdef public double[:] counts, ecounts, _inputFires, _inputs
    cdef public object[:] events
    cdef public object[:] neurons
    cdef public object tHomeo
    # nInputs+1(homeostasis) by nOutputs array of devices
    cdef public object[:, :] resistors

    # Calculated stuff
    cdef double _condMin, _condMax
    cdef int _randIndex, _randLen
    cdef double[:] _randBuffer

    PICKLE_INIT = [ 'nInputs', 'nOutputs', 'nHomeo', 'avgInput',
            'dtSpike', 'vEventSet', 'vFire', 'cNeuron', 'dtMax', 'dtMin',
            'stableScale', 'rMin', 'rMax', 'tolerance_vFire', 'spikeWidth',
            'untrainedK' ]
    PICKLE_STATE = [ 'crossbar' ]
    __reduce_ex__ = modelBaseUnpickler.__reduce_ex__
    __setstate__ = modelBaseUnpickler.__setstate__

    def __init__(self, nInputs, nOutputs, nHomeo, **kwargs):
        cdef int i, o

        super(SubCircuit, self).__init__()

        if 'vFire' in kwargs or 'cNeuron' in kwargs:
            raise ValueError("Cannot specify these anymore!")

        # Values of None here are parameters that can be overridden via
        # the circuit_ parameters in the LcaSpikingWoods model.
        defaults = {
                'nInputs': nInputs,
                'nOutputs': nOutputs,
                'nHomeo': nHomeo,
                'avgInput': None,
                'untrainedK': None,

                # Average time between spikes with input strength 1.0
                'dtSpike': None,
                # Width of a spike
                'spikeWidth': None,
                'spikeDensity': None,

                # Voltages
                'vEventSet': None,

                # Stability and simulation constants
                'dtMax': None,
                'dtMin': None,

                # Convergence is important, keep an eye on it.  Loosely, this
                # is the max voltage change for a node in a single time step.
                'stableScale': 0.01,

                # Epsilons
                'rMin': None,
                'rMax': None,
                'tolerance_vFire': None,
        }
        for k, w in kwargs.iteritems():
            if k not in defaults:
                raise ValueError("Unknown attribute {}".format(k))
            defaults[k] = w
        for k, v in defaults.iteritems():
            try:
                setattr(self, k, v)
            except Exception, e:
                raise ValueError("Could not set {}: {}".format(k, str(e)))

        self._condMax = 1.0 / self.rMin
        self._condMin = 1.0 / self.rMax
        self._randIndex = 0
        self._randLen = 1024
        self._randBuffer = np.random.uniform(size = (self._randLen,))

        self._populateMemristorTable()

        self.vFire = self._calcVFire()
        self.cNeuron = self._calcCNeuron(self.vFire)


        '''tNeuron journal:

        I'm trying to normalize device differences by calculating both a
        threshold weight for each output and a set voltage based on the
        application.

        Connecting the threshold neuron as an event is enticing.  This would
        allow homeostasis to be naturally implemented (increase weight for
        neurons that aren't firing).

        Firing seems to happen WAYYY too close to threshold.  This is maybe
        a timing / capacitance thing.  Note that right now I'm working with
        a peak frequency of input events of 100Hz.  This is SLOOWWW compared
        to silicon, of course.  Capacitance will probably jam down when I try
        e.g. 500MHz.

        Capacitance of threshold can be asymmetric!  May help firing threshold
        to fall quicker than e.g. event and neuron data.  Of course, may be
        bad as it removes stability.

        Goal is to have vFire be a smooth function, ideally with only one zero.

        0.035 reconstruction (way better than LCA!) with 2e-8 threshold capacitor!
        Linearizing the circuit across different levels of activity makes a huge
        difference.

        vFire 0.1 w/ 0.001 memristor ratio failed.  Try 0.1 with 0.1

        Adding momentum to learning the firing threshold was good.

        Trying homeostasis.  Works kinda well, but worse than I would have
        thought.

        Going back to targeting threshold voltage.  It changes a little too
        much, even with smoothing.  Additionally, changing the number of inputs
        greatly affects the linearity a _TON_.

        So, it seems that threshold needs to depend on input events.  Ergo,
        should drain to events, not neurons.  > >

        Stable point analysis.  Issue seems to be nonlinear with respect to
        number of active inputs and inactive inputs.
        Stable point: VDactive * Nactive / Ractive = VDinactive * Ninactive / Rinactive

        With that, let's track a single event's change in Vinactive, the floor
        voltage.

        We know that I = V / R, and V = Q / C, where Q = integral(I).

        When event happens, assuming all voltages are Vinactive except for the
        main event.  Now, we add Q = N * Ce * (Vset - Vinactive) into the system.  Divide
        this evenly across the network's storage, where
        Ctotal = Ce * Ninputs + Cn * Noutputs
        and we get VDtotal = N * (Vset - Vinactive) * Ce / (Ce * Ninputs + Cn * Noutputs).

        Ideally, Noutputs won't matter at all.  Voltage should, however, depend
        on events.  Ergo, Ce = Cn * Noutputs -> VDtotal = Cn / (Cn * Ninputs + Cn)
        = 1.0 / (Ninputs + 1).  Seems like a winning equation...
        More generally,

        Ce = XCn -> VDtotal = N * (Vset - Vinactive) / (Ninputs + Noutputs / X)

        In other words, assuming Vinactive << Vset, we have a linear change in
        voltage with respect to N, the number of set events.  Therefore, extra
        events would trigger a linear response in terms of overall floor.

        Now, let's find the maximal point for neurons...  We already know:
        stable -> (Vactive - Vstate) * N / Ractive
                = (Vstate - Vinactive) * (T - N) / Rinactive

        (... Sub equation, integral of Vactive
            Note that all of this is documented in testPlot.py

        dV = ((Vactive - V) * N / Ractive - (V - Vinactive) * (T - N) / Rinactive) / C
        CdV + V * (N / Ractive + (T - N) / Rinactive)
                - Vactive * N / Ractive - Vinactive * (T - N) / Rinactive = 0

        CsL{V} - CV(0) + L{V} * Q1 - Q2/s = 0
        L{V} = (Q2/s + CV(0)) / (Cs + Q1)
             = (Q2 + CV(0)s) / s(Cs + Q1)
             = A / s + B / C(s + Q1/C)
             A(Q1) = Q2
             A = Q2 / Q1
             B(-Q1 / C) = Q2 + CV(0) * -Q1/C
             B = -CQ2 / Q1 + CV(0)

        Q1 = N / Ractive + (T - N) / Rinactive
        Q2 = Vactive * N / Ractive + Vinactive * (T - N) / Rinactive
        V = Q2 / Q1 + (-Q2 / Q1 + V0) * math.exp(-t*Q1/C)
        ===================
        V = Q2 / Q1 * (1 - math.exp(-t*Q1/C)) + V0 * math.exp(-t * Q1 / C)
        ===================

        Ractive = 200e4
        Rinactive = 200e6
        C = 2e-7
        T = 784
        N = 80
        Vactive = 1.0
        Vinactive = 0.0
        t = 0.0
        V0 = 0.0
        def Q(T, N, t, V0 = 0.0):
            Q1 = N / Ractive + (T - N) / Rinactive
            Q2 = Vactive * N / Ractive + Vinactive * (T - N) / Rinactive
            V = Q2 / Q1 + (-Q2 / Q1 + V0) * math.exp(-t*Q1/C)
            return V

        # Deriving WHEN firing happens based in V (neuron) and V' (threshold)
        # So, find t when V == V'
        Q2 / Q1 + (-Q2 / Q1 + V0) * math.exp(-t*Q1/C)
                == Q2' / Q1' + (-Q2' / Q1' + V0') * math.exp(-t*Q1'/C')

        # Actual python now...
        Ractive = 200e4
        Rinactive = 200e6
        Rthresh = Rinactive
        Vactive = 1.0
        Vinactive = 0.0
        def error(t, Cn = 2e-7, Ct = 2e-6, T = 784, N = 100):
            Q1 = N / Ractive + (T - N) / Rinactive
            Q2 = Vactive * N / Ractive + Vinactive * (T - N) / Rinactive
            Q1p = T / Rthresh
            Q2p = (Vactive * N + Vinactive * (T - N)) / Rthresh
            V0 = 0.0
            V0p = 1.0
            return (
                    (Q2 / Q1 + (-Q2 / Q1 + V0) * math.exp(-t * Q1 / Cn))
                    - (Q2p / Q1p + (-Q2p / Q1p + V0p) * math.exp(-t * Q1p / Ct)))


        Solve for t, analytically so that we can relate C to C' to t

        NOW!  May be too obsessed with non-linearizing.  p-scaling works, but
        effectively it scales N.  Since that affects the threshold, you do NOT
        get scaling for different values of p or N.  Also, note that a 400 vs
        100 will not favor the 400 if the 100 is a better match, due to negative
        product.  So, let's just calculate the ideal threshold to get a given
        N to trigger at a given rate.  Also, note that scaling will take care
        of some of the non-linearities - over-representation will be scaled back
        by the learning function.

        V = Q2 / Q1 + (-Q2 / Q1 + V0) * math.exp(-t*Q1/C)
                == Vthresh

        Or, simpler, dropping V0:
            Q1 = N / Ractive + (T - N) / Rinactive
            Q2 = Vactive * N / Ractive + Vinactive * (T - N) / Rinactive
            Vthresh = (Q2 / Q1) * (1 - math.exp(-t * Q1 / C))


        Great!  So we can establish the charge max.  Q2 / Q1
            = (Vactive * N / Ractive + Vinactive * (T - N) / Rinactive)
                    / (N / Ractive + (T - N) / Rinactive)
            = (Vactive * N * Rinactive + Vinactive * (T - N) * Ractive)
                    / (N * Rinactive + (T - N) * Ractive)
            = Vactive * (N * Rinactive)
                    / (N * Rinactive + (T - N) * Ractive)

        ...)

        Or, Vstate = (Vactive * N / Ractive + Vinactive * (T - N) / Rinactive)
                / (N / Ractive + (T - N) / Rinactive)

        Since Vinactive starts at zero, we can assume that the initial target
        for Vstate is (Vactive * N / Ractive) / (N / Ractive + (T - N) / Rinactive),
        or

        Vstate = Vactive / (1 + (T - N) * Ractive / (N * Rinactive))
                + Vinactive / (1 + N * Rinactive / ((T - N) * Ractive))

        Now, we want Vstate to be independent of or linear to N.  In other
        words, (T-N) / N should be linear.  Which it most certainly is not.
        Since it's not exponential, we also can't correct it with a draining
        component.  Hmmm.

        T = 2, N = 2 -> Vactive
        T = 2, N = 1 -> Vactive / (1 + Ractive / Rinactive) + Vinactive / (1 + Rinactive / Ractive)
        T = 2, N = 0 -> Vinactive

        T = 3, N = 3 -> Vactive
        T = 3, N = 2 -> Vactive / (1 + Ractive / 2Rinactive) + Vinactive / (1 + 2Rinactive / Ractive)
        T = 3, N = 1 -> Vactive / (1 + 2Ractive / Rinactive) + Vinactive / (1 + Rinactive / 2Ractive)
        T = 3, N = 0 -> Vinactive

        Now, clearly that's (1-exp)ish gain, based on the ratio of Ractive / Rinactive.
        Let's plot this though:

        def show(T, rActiveOverRInactive, compare=10.0):
            data = []
            Ra = 1.0
            Rn = 1.0 / rActiveOverRInactive
            k = 0.11
            for i in range(T+1):
                r = {}
                data.append(r)
                r["N"] = i
                if i == 0:
                    r["Vstate"] = 0.0
                    r["VstateNmatch"] = 0.0
                else:
                    r["Vstate"] = 1.0 / (1.0 + (T - i) * Ra / (i * Rn))
                    r["VstateNmatch"] = 1.0 / (1.0 + (T - i) * 1.0 / i)
                r["compare"] = 1.0 - math.exp(-i * compare / T)

                # Let's be smarter about compare... start at 1, use Q drain
                # from above
                r["compare"] = 1e-2
                for j in range(i):
                    # Have a "zero-er" bonded to our threshold, with relative
                    # capacitance Cz and Ct
                    Cz = 1.0
                    Ct = compare
                    r["compare"] += (1.0-r["compare"]) * Cz / (Cz + Ct)

                r["compare"] /= k * i
            from figureMaker import FigureMaker
            fm = FigureMaker(data, imageDestFolder="tmp")
            fm.bind([ "Vstate", "VstateNmatch", "compare" ])
            with fm.new("hm") as f:
                f.plotValue("Vstate", "N", "Vstate")
                f.plotValue("VstateNmatch", "N", "VstateNmatch")
                f.plotValue("compare", "N", "compare")

        ^^^ Kind of works!  Calibrating works, but we actually want threshold
        linear with kN, where k is the average.  Though, maybe the problem is
        something else here.

        (T - N) * Ractive / (N * Rinactive) = Constant

        Here, Ractive and Rinactive represent that relative dot products of
        the receptive field of the inactive part vs the receptive field of the
        active part.  E.g., Ractive = R(1.0) and Rinactive = R(0.0) represents
        the charge on a perfect match.  Ractive = R(0.5) and Rinactive = R(0.5)
        represents a perfectly balanced wrongness... Or does it?  Suppose N/T is
        small.  This raises correlation of Ractive... and decreases Rinactive.
        Let's find the actual dot product in terms of Ractive and Rinactive,
        assuming initial Vstate = Vinactive = 0, Vactive = Vset = 1.0

        RTotalActive = Ractive / N
        RTotalInactive = Rinactive / (T - N)

        In conductance land,
        Cactive = NCactive
        Cinactive = (T - N)Cinactive

        We want a factor A that dictates the dot product.  E.g., A = 1.0 ->
        Cactive = argmax(Cactive), Cinactive = argmax(Cinactive),
        A = 0.0 -> opposite, and in betweens are sensible.

        In math land, the pure, frequency-unaware dot product would be...
        Positive part:
        \sum_i{V_i * C_i} = sqrt(\sum_i{V_i^2})sqrt(\sum_i{C_i^2})cos(\theta)

        And the negative part:
        \sum_i{V_i * C_i} ..., where V_i is all negative.  HM.  Actually, let's
        write it out.

        \sum_i_n{Vn_i_n * C_i_n} = sqrt(\sum_i_n(Vn_i_n^2))sqrt(\sum_i_n{C_i_n^2})cos(\theta_n)

        In other words, it stops charging when:

        cos(\theta) / cos(\theta_n)
                = sqrt(\sum_i_n(Vn_i_n^2))sqrt(\sum_i_n{C_i_n^2})
                    / sqrt(\sum_i{V_i^2})sqrt(\sum_i{C_i^2})

        If we ignore the thetas for now (assume, incorrectly, that they're
        constant over time), then we can use a constant ratio.

        We know that max(C) = 1.0 / Rmin, min(C) = 1.0 / Rmax.
        sqrt(sum(C_i_n^2))
        '''

        self.crossbar = np.asmatrix(np.random.uniform(
                size = (nInputs, nOutputs)))
        self.counts = np.zeros((self.nOutputs,), dtype = float)
        self.ecounts = np.zeros((self.nInputs,), dtype = float)

        self.events = np.zeros((self.nInputs,), dtype = object)
        self.neurons = np.zeros((self.nOutputs,), dtype = object)
        self.tHomeo = circuit.FixedNode(self, -10.0)
        self.resistors = np.zeros((self.nInputs+self.nHomeo, self.nOutputs),
                dtype = object)
        for i in range(self.nInputs):
            self.events[i] = circuit.FixedNode(self, 0.0)
        for o in range(self.nOutputs):
            self.neurons[o] = circuit.Node(self, self.cNeuron)
            for i in range(self.nInputs):
                self.resistors[i, o] = circuit.Memristor(self, self.events[i],
                        self.neurons[o], 1.0)
            for i in range(self.nHomeo):
                self.resistors[self.nInputs+i, o] = circuit.Memristor(self,
                        self.tHomeo, self.neurons[o], 1.0)


    cdef double _getCond(self, double weight):
        cdef double r = self._condMax * weight
        if r < self._condMin:
            return self._condMin
        return r


    cdef void _populateMemristorTable(self):
        """populates self.memristorTable as state -> resistance."""
        self.memristorTable = np.zeros(1001, dtype = float)
        cdef int i
        cdef double[:] table = self.memristorTable
        cdef double s

        c = circuit.OdeCircuit()
        p = circuit.FixedNode(c, self.vEventSet)
        m = circuit.Memristor(c, p, c.ground, 0.0)
        for i, s in enumerate(np.linspace(0.0, 1.0, table.shape[0])):
            m.state = s
            c.simulate(1e-10)
            table[i] = 1.0 / (m.getValue(u"lastVoltage")
                    / m.getValue(u"lastCurrent"))
        self._condMin = min(table[0], table[-1])
        self._condMax = max(table[0], table[-1])


    cdef double _getState(self, double weight):
        """Returns the memristive state corresponding to a given weight."""
        cdef double[:] table = self.memristorTable
        cdef double tableStep = 1.0 / (table.shape[0] - 1)
        cdef double targetC = self._condMax * weight
        cdef double diff = abs(targetC - table[0])
        cdef double diff2
        cdef int i
        for i in range(1, table.shape[0]):
            diff2 = abs(targetC - table[i])
            if diff2 > diff:
                # Moving away, use i - 1
                i -= 1
                break
            diff = diff2
        return i * tableStep


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline double _fastRand(self):
        """Returns a uniform random on [0,1) MUCH faster than just using
        np.random.uniform() directly.

        This shaved about 30% off of runtime, in my case.
        """
        cdef double R = self._randBuffer[self._randIndex]
        self._randIndex += 1
        if self._randIndex == self._randLen:
            self._randIndex = 0
            self._randBuffer = np.random.uniform(size = (self._randLen,))
        return R


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double _calcVFire_error(self, vFire, double uQ1, double uQ2,
            double tQ1, double tQ2) except -1:
        cdef double log1 = 1 - abs(vFire[0]) * uQ1 / uQ2
        cdef double log2 = 1 - abs(vFire[0]) * tQ1 / tQ2
        if log2 < log1:
            sys.stderr.write("Unexpected vFire: {}, {} / {}\n".format(
                    vFire[0], log1, log2))
        if log2 < 1e-100:
            log2 = 1e-100
        if log1 < 1e-100:
            log1 = 1e-100
        cdef double r =  tQ1 * log(log1) / (uQ1 * log(log2)) - self.untrainedK
        return r * r


    cdef double _calcVFire(self) except -1:
        """Returns vFire for given network params."""
        # Untrained (independent) and trained conductances
        cdef double cAvg = self._getCond(self.avgInput)
        cdef double cMax = self._getCond(self.avgInput)
        cdef double cHomeo = self.nHomeo * self._condMin

        # Number active inputs, on average
        cdef double T = self.nInputs

        # Untrained vars
        # An untrained network has the same average as a trained, but
        # its RF is as orthogonal as can be.  Since Q2 relies on
        # \sum_n{k_n c_n}, this looks like \sum_n{k_n c(k_n)} for a perfect
        # fit.  Ideally (no lower bound), c(k_n) == k_n c_{max}, this means
        # that Q2 relies on c_{max} \sum_n{k_n^2}.  k_n can be treated as an
        # array of i.i.d. random variables with mean self.avgInput.  Calculating
        # E(k_n^2) will give us the conductance to use for Q2.
        cdef double optimal = np.asarray([
                np.random.beta(self.avgInput, 1.0 - self.avgInput)**2
                for _ in range(10000) ]).mean() * self.spikeDensity
        cdef double suboptimal = np.asarray([
                np.random.beta(self.avgInput, 1.0 - self.avgInput)
                    * np.random.beta(self.avgInput, 1.0 - self.avgInput)
                for _ in range(10000) ]).mean() * self.spikeDensity
        cdef double uQ1 = cHomeo + T * cAvg
        cdef double uQ2 = self.vEventSet * (cHomeo + T * suboptimal
                * self._condMax)

        # Trained vars
        cdef double tQ1 = cHomeo + T * cAvg
        cdef double tQ2 = self.vEventSet * (cHomeo + T * optimal
                * self._condMax)

        # tU / tT
        #O = scipy.optimize.leastsq(self._calcVFire_error, [ 1e-3 ],
        #        args = (uQ1, uQ2, tQ1, tQ2))
        #cdef double vFire = O[0][0]
        cdef double vFire = np.asarray([
                self._calcVFire_error([ q ], uQ1, uQ2, tQ1, tQ2)
                for q in np.linspace(1e-4, 5e-1, 1001) ]).argmin() * (5e-1 - 1e-4) * 1e-3 + 1e-4

        cdef double vFireError = self._calcVFire_error([vFire], uQ1, uQ2, tQ1,
                tQ2)
        if vFireError > 1.:
            raise ValueError("Bad vFire? {}.  Error {}".format(vFire,
                    vFireError))

        if False:
            print("FOUND vFIRE {}, ERROR {} / {}".format(vFire,
                    self._calcVFire_error([vFire], uQ1, uQ2, tQ1, tQ2)
                        + self.untrainedK,
                    self.untrainedK))
            print("Others: {} / {}".format(
                    self._calcVFire_error([vFire + 0.01], uQ1, uQ2, tQ1, tQ2)
                        + self.untrainedK,
                    self._calcVFire_error([vFire - 0.01], uQ1, uQ2, tQ1, tQ2)
                        + self.untrainedK,))
        return abs(vFire)


    cdef double _calcCNeuron(self, double vFire) except -1:
        # Trained vars only
        cdef double cAvg = self._getCond(self.avgInput)
        cdef double cMax = self._getCond(self.avgInput)
        cdef double cHomeo = self.nHomeo * self._condMin

        cdef double T = self.nInputs

        cdef double optimal = np.asarray([
                np.random.beta(self.avgInput, 1.0 - self.avgInput)**2
                for _ in range(10000) ]).mean() * self.spikeDensity
        cdef double Q1 = cHomeo + T * cAvg
        cdef double Q2 = self.vEventSet * (cHomeo + T * optimal * self._condMax)

        cdef double t = self.dtSpike
        return -t * Q1 / log(1. - vFire * Q1 / Q2)


    cdef void preAdvance(self, double lastDt) except *:
        cdef int i, o
        cdef double lastVal, tmp, tGap
        cdef double[:] _inputFires = self._inputFires
        cdef double spikeWidth = self.spikeWidth
        cdef double dtSpike = self.dtSpike

        # Check if any inputs are spiking
        for i in range(self.nInputs):
            lastVal = _inputFires[i]
            _inputFires[i] -= lastDt
            if _inputFires[i] <= 0:
                if _inputFires[i] < -spikeWidth:
                    # End-of-life spike, calculate beginning of next spike
                    # before setting voltage
                    tGap = dtSpike / max(1e-10, self._inputs[i]) - spikeWidth
                    if lastVal < -1e3:
                        # Initial condition; find an _inputFires >= 0
                        _inputFires[i] = -spikeWidth - 10.0 * tGap
                        while True:
                            tmp = self._fastRand() * 2.0
                            _inputFires[i] += spikeWidth + tmp * tGap
                            if _inputFires[i] >= -spikeWidth:
                                break
                    else:
                        tmp = self._fastRand() * 2.0
                        # Next spike happens at this rate (see _sampleK in
                        # analytical model)
                        _inputFires[i] += spikeWidth + tmp * tGap

                    # Mark that the spike was reset
                    lastVal = 1.0

                if _inputFires[i] <= 0:
                    # Spike is active
                    if lastVal > 0.0:
                        self.ecounts[i] += 1.0
                    self.events[i].setValue(u"voltage", self.vEventSet)
                else:
                    self.events[i].setValue(u"voltage", 0.0)

        # Check firing logic
        cdef int zeroAll = False
        cdef double tVoltage = self.vFire
        cdef double avgActive = 0.0, avgInactive = 0.0
        cdef int nActive = 0, nInactive = 0
        for o in range(self.nOutputs):
            if self.neurons[o].getValue(u"voltage") >= tVoltage:
                if False and not zeroAll:
                    for i in range(self.nInputs):
                        v = self.events[i].getValue(u"voltage")
                        if v >= tVoltage:
                            nActive += 1
                            avgActive += v
                        else:
                            nInactive += 1
                            avgInactive += v
                    print("Firing at {} / {}, ({} / {}, {} / {})".format(
                            self.neurons[o].getValue(u"voltage"), tVoltage,
                            avgActive / max(nActive, 1), nActive,
                            avgInactive / max(nInactive, 1), nInactive))
                if False and zeroAll:
                    print("FIRE MULTI! {}: {}".format(o, self.neurons[o].getValue(u"voltage")))
                zeroAll = True
                self.counts[o] += 1.0
        if zeroAll:
            #print("Firing at {}".format(tVoltage))
            #for i in range(self.nInputs):
            #    self.events[i].setValue(u"voltage", 0.0)
            for o in range(self.nOutputs):
                tmp = self.neurons[o].getValue(u"voltage")
                self.energy += self.cNeuron * tmp * tmp * 0.5
                self.neurons[o].setValue(u"voltage", 0.0)


    cpdef double[:] run(self, double simTime, double[:] inputs, double[:] homeostasis):
        cdef int i, o
        cdef double thresholdW, thresholdCond

        # Set up resistances (we usually target a conductance)
        for i in range(self.nInputs):
            for o in range(self.nOutputs):
                self.resistors[i, o].state = self._getState(self.crossbar[i, o])

        # Firing threshold / homeostasis terms
        for o in range(self.nOutputs):
            for i in range(self.nHomeo):
                self.resistors[self.nInputs+i, o].state = self._getState(
                        homeostasis[o])

        # Initialize voltages
        self.tHomeo.setValue(u"voltage", self.vEventSet)
        for i in range(self.nInputs):
            self.events[i].setValue(u"voltage", 0.0)
        for o in range(self.nOutputs):
            self.neurons[o].setValue(u"voltage", 0.0)

        # Simulate the network for one second
        self.ecounts[:] = 0.0
        self.counts[:] = 0.0
        self._inputs = inputs
        self._inputFires = np.zeros((self.nInputs,), dtype = float)
        self._inputFires[:] = -1e30
        self.reset()
        self.simulate(simTime)

        # Return the raw number of output spikes
        return self.counts
