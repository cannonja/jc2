
# cython: profile=True
# notcython: linetrace=True, binding=True

r""":mod:`lcaSpikingWoodsAnalytical` provides an analytical (non-SPICE) implementation of the simple spiking architecture.

.. image:: /../mr/unsupervised/docExtra/lcaWoodsUninhibited.svg

The *Simple, Spiking, Locally Competitive Algorithm* (SSLCA) consists of rows with a set voltage (either :math:`V_{high}` or :math:`V_{low}` depending on input spike activity) and columns that contain a capacitor which is directly connected to the crossbar.

The design can be broken down into two versions:

* :ref:`sec-non-inhibit`
* :ref:`sec-inhibit`

.. _sec-non-inhibit:

Non-Inhibiting Version
======================

When inhibition is not a factor, the behavior is exactly as described above: a capacitor directly on the crossbar either charges or discharges through the rows.  When that capacitor reaches a threshold voltage, all capacitors are reset, and the process continues.  Mathematically, this can be viewed as the sum of each row's current into the node:

.. math::

    C\frac{\partial V_{neuron}}{\partial t} &= \sum_i{(V_i - V_{neuron})G_i}.

Then, assuming an input row :math:`i` spikes to voltage :math:`V_{set}` with a mean of :math:`K_i` activity (on for :math:`K_i`, off for :math:`1 - K_i`), and is grounded the rest of the time, this becomes:

.. math::

    C\frac{\partial V_{neuron}}{\partial t} &= \sum_i \biggl( K_i(V_{set} - V_{neuron})G_i\\
            &\hspace{5em}+ (1 - K_i)(0 - V_{neuron})G_i \biggr),\\
    &= \sum_i{(K_iV_{set} - V_{neuron})G_i}, \\
    &= V_{set}\sum_i{K_iG_i} - V_{neuron}\sum_i{G_i}, \\
    Q_1 &= \sum_i{G_i}, \\
    Q_2 &= V_{set}\sum_i{K_iG_i}, \\
    \mathcal{L}\{V_{neuron}\}s(Cs + Q_1) &= CsV_{neuron, t=0} + Q_2, \\
    V_{neuron}(t) &= \frac{Q_2}{Q_1}(1 - e^\frac{-tQ_1}{C}) + V_{neuron, t=0} e^\frac{-tQ_1}{C}.

Solving that equation for :math:`t`, letting :math:`G_i = G_{max}g_i, K_i = K_{max}k_i`, assuming the network has :math:`M` inputs, and that there is a distribution :math:`\chi` that matches the empirical data set that the network is exposed to, then the network may be specified by assuming that the network is fully trained with a matching element for each input:

.. math::

    Q_1 &= MG_{max}\overline{\chi}, \\
    Q_2 &= MV_{set}K_{max}G_{max}\overline{\chi^2}, \\
    t &= \frac{-C}{Q_1}ln\left(\frac{\frac{Q_2}{Q_1} - V_{neuron}(t)}{\frac{Q_2}{Q_1} - V_{neuron,t=0}}\right).

Thus, the anticipated fire time (desired spikes per second) can be calibrated based on the desired capacitance and threshold voltage.  Alternatively, by using the product of two different distributions instead of the square of a single for :math:`Q2`, :math:`\overline{\chi\chi}`, a ratio of firing times may be calibrated based on :math:`V_{neuron}`.


.. _sec-inhibit:

Inhibiting Version
==================

The inhibiting version has an identical layout to the non-inhibiting version.  However, it has more complicated row and column headers.  The CMOS layout is as follows:

.. image:: /../mr/unsupervised/docExtra/lcaWoodsInhibited.svg

If, in the *Inhibition Logic Module* (ILM), :math:`CBLOW = \overline{SPIKE}`, then the non-inhibiting architecture is realized.  However, if the ILM is implemented as above, then:

Derivation
----------

.. math::

    \text{As with uninhibited, a row has $K_i$ spike activity.} \span\\
    \text{Tracking the inhibition voltage $V_{i}$ (pre/post is relative to spike):} \span\\
    A &= \frac{1}{R_{cb}C}, \\
    B &= \frac{1}{RC}, \\
    V_{i,max} &= V_{cc}\frac{R}{R + R_{cb}}, \\
    V_{i,pre} &= V_{i,0} e^{-TB}, \\
    V_{i,post} &= V_{i,max} + (V_{i,pre} - V_{i,max})e^{-T_{spike}A}. \\
    \text{Making the assumption that spike activity is nullified when:} \span\\
    V_i &> V_{i,thresh}, \\
    \text{Then the time that spike activity is nullified is:} \span\\
    V_{i,thresh} &= V_{i,0} e^{-T_{run}B}, \\
    T_{inhib} &= \frac{-ln(\frac{V_{i,thresh}}{V_{i,0}})}{B}\text{, bounded on $[0, \infty)$}. \\
    \text{Since $V_{i,post}$ can be calculated, the effective block time added by } \span\\
    \text{a spiking event can be calculated by taking the difference of $T$ values.} \span\\
    \text{If we want constant inhibition, then $V_{post}$ after $V_{pre}$ needs to be $V_{i,0}$:} \span\\
    V_{i,0} &= V_{i,max} + (V_{i,0}e^{-T_{run}B} - V_{i,max})e^{-T_{spike}A}, \\
    T_{run} &= \frac{-ln\left( \frac{V_{i,0} - V_{i,max} + V_{i,max}e^{-T_{spike}A}}{V_{i,0}e^{-T_{spike}A}} \right)}{B}. \\
    \text{Back to the $K_i$ business, an inhibited network will have $K_i = 0$ for $T < T_{inhib}$.} \span\\
    \text{In a $1\times 1$ network, this means that the fire time will be delayed by $T_{inhib}$:} \span\\
    T_{run} &= \frac{-C}{Q_1}ln\left( 1 - V_{neuron}(t)\frac{Q_1}{Q_2} \right) + max\left(0, \frac{-ln(\frac{V_{i,thresh}}{V_{i,0}})}{B}\right), \\
    \text{Subsituting in the original design parameters: } \span\\
    T_{run} &= T_{planned} + max\left( 0, \frac{-ln(\frac{V_{i,thresh}}{V_{i,0}})}{B} \right). \\
    \text{$T_{run}$ is now the actual time between spikes; the frequency is thus $\frac{1}{T_{run} + T_{spike}}$.} \span\\
    \text{Substituting the $V_{i,0}$ stability equation into the $1\times 1$ equation yields: }\span\\
    \frac{V_{i,0} - V_{i,max} + V_{i,max}e^{-T_{spike}A}}{V_{i,0}e^{-T_{spike}A}} &= e^{-B\left[ ln(e^{T_{planned}}) + min\left(0, ln\left( \frac{V_{i,thresh}}{V_{i,0}} \right) \right) \right]} \\
            &= \begin{cases}
                e^{-BT_{planned}} & \text{if } V_{i,thresh} \ge V_{i,0} \\
                \left(\frac{V_{i,thresh}}{V_{i,0}}\right)^{-B}e^{-BT_{planned}} & \text{otherwise}
                \end{cases}


Consider now a niche problem for investigating this architecture:

* There are two dictionary elements with M inputs
* The first element has M-1 elements at max weight (other at K_{min})
* The second element has only the last element at K_B weight (others at K_{min})
* The input is a solid bar of K_{in} weight

Thus the ratio of firing should work out to 1:K/B between the two elements if inhibition is doing its job.  Essentially, we will run the analytical algorithm to determine the actual firing ratio.

.. math::

    \text{Column parameters denoted as $A\{Q_1\}$ and $B\{Q_1\}$, for instance.} \span\\
    A\{Q_1\} &= G_{max}(M - 1 + K_{min}), \\
    A\{Q_2\} &= V_{set}K_{max}G_{max}K_{in}(M - 1 + K_{min}), \\
    B\{Q_1\} &= G_{max}((M - 1)K_{min} + K_B), \\
    B\{Q_2\} &= V_{set}K_{max}G_{max}K_{in}((M - 1)K_{min} + K_B), \\
    \text{For sensitivity, $V_{fire} = 0.9\frac{B\{Q_1\}}{B\{Q_2\}}$}, \span\\
    Q_1 &= MG_{max}K_{in}, \\
    Q_2 &= MV_{set}K_{max}G_{max}K_{in}^2, \\
    C &= \frac{-T_{planned}MG_{max}K_{in}}{ln\left(1 - V_{fire}\frac{Q_1}{Q_2}\right)}. \\
    \text{We have two distinct input inhibition states: $V_{i,A}$ and $V_{i,B}$.} \span\\
    \text{At each step, re-compute all $Q_2$ according to inhibition terms.} \span\\
    \text{Measure times: $T_{inhib,A}$, $T_{inhib,B}$, $t_{fire,A}$, $t_{fire,B}$.} \span\\
    \text{Take the smallest time, re-up $V_{i,A}$, $V_{i,B}$, $V_A$, $V_B$.} \span\\
    \text{Rinse and repeat to 20 spike events.} \span

Now, inhibition works best when it's linear: the charge through the crossbar is exponential between the current inhibition and VCC.  The drain is from a max value of VCC down to ground.  Therefore, the optimal inhibition threshold is :math:`\frac{Vcc}{2}`, as the midpoint is the most linear range of an exponential for both towards VCC and towards ground.

The most important design decisions for inhibition are R and C.  Too large of an R produces good results, but makes the network run for far too long.  Too large of a C can cause performance issues as well.  The goal is to balance these for accuracy against speed; it is imperative that the network still process quickly.

New design setting: rather than :math:`\chi` based on the data set, using :math:`\chi = E\left[ \text{smallest RF} \right]` seems to work better.

Thus, design for uninhibited network:

#. Choose the minimal RF of interest, calculate :math:`Q_1\text{ and }Q_2`.
#. Calculate :math:`V_{fire}` based on that RF.
#. Choose the desired run-time length of algorithm, spike density, and spike resolution.
#. Calculate :math:`C` based on these parameters.

For an inhibited network:

#. Choose the minimal RF of interest, calculate :math:`Q_1\text{ and }Q_2`.
#. Calculate :math:`V_{fire}` based on that RF.
#. Choose the desired run-time length of algorithm, input spike density, output spike density, and output spike resolution for minimal RF.
#. Calculate :math:`C_{max}`, the maximum neuron capacitance (does not account for inhibition).
    #. For the minimal RF, calculate what C should be with a desired fire time of the desired length / spike resolution * (1. - outDensity) and the minimal RF's :math:`Q_1\text{ and }Q_2`.
#. TODO WHAT WE ACTUALLY WANT
    #. Want asymptotic, stable fire time :math:`F(g_i, k_i) = \frac{g_iT_{outSpikePeriod}}{k_i} - T_{outSpike}`.

        .. math::

            F(g_i, k_i) &= T_{outSpikeGap} = \frac{g_iT_{outSpikePeriod}}{k_i} - T_{outSpike}, \\
            V_{i,0} &= \frac{V_{cc}\left( 1 - e^{-T_{outSpike}A} \right)}{1 - e^{-k_iT_{outSpikeGap}B - T_{outSpike}A}}, \\
            V_{i,thresh} &= V_{i,0}e^{-k_iT_{outSpikeGapInhib}B}, \\
            V_{i,thresh} &= \frac{V_{cc}}{2}, \\
            \frac{V_{cc}}{2e^{-k_iT_{outSpikeGapInhib}B}}
                    &= \frac{V_{cc}\left( 1 - e^{-T_{outSpike}A} \right)}{1 - e^{-k_iT_{outSpikeGap}B - T_{outSpike}A}},


#. TODO BELOW IS CALCULATION FOR RC PRODUCT WITH EXPONENTIAL, ALWAYS DISCHARGING
#. Calculate the inhibition parameters:
    #. Find :math:`T_{fire}` without inhibition to be 0.5 of desired by halving :math:`C_{max}`.
    #. Calculate B (RC product) from :math:`T_{inhib}`:

        .. math::

            V_{i,0} &= \frac{V_{i,max}\left( 1 - e^{-T_{spike}A} \right)}{1 - e^{-T_{spikeGap}B - T_{spike}A}} \\
            V_{i,thresh} &= V_{i,0}e^{-T_{spikeGapInhib}B} \\
                    &= e^{-T_{spikeGapInhib}B}\frac{V_{i,max}\left( 1 - e^{-T_{spike}A} \right)}{1 - e^{-T_{spikeGap}B - T_{spike}A}}, \\
                    &= e^{-T_{spikeGapInhib}B}\frac{V_{cc}\frac{R}{R + R_{cb}}\left( 1 - e^{-T_{spike}A} \right)}{1 - e^{-T_{spikeGap}B - T_{spike}A}}, \\
            \text{where $T_{spikeGap}$ is the time between the end of one spike and the rise of the next,} \span\\
            \text{and $T_{spikeGapInhib}$ is the time from the end of one spike to end of inhibition.} \span

    #. Since :math:`B, A, V_{i,max}` all depend on one of :math:`R, C`, this equation must be solved by numerical methods.

        # find tFire no inhibition
        # calculated tInhib inhibited time
        # fall tFire from vInhib == vThresh, find resistance required to achieve
        # tInhib time.
        # i,max = cc * R / (R + Rcb)
        # i,0 = i,max + i,0 * e^{-T
#. Halve :math:`C_{max}`, set that as the neuron's capacitance.
    #. Note: At this point, if both :math:`R_i` and :math:`C_i` are set to :math:`\infty`, the algorithm will work perfectly.
    #. Lowering :math:`C_i` with :math:`R_i` still at infinity TODO
#. Set :math:`C_i` as low as it will go with desired accuracy in bar problem.
#. Set :math:`R_i` low enough to get desired run-time length.



Members
=======


"""

cimport cython
cimport numpy as np

cdef extern from "math.h":
    double exp(double)
    double log(double)
    double sqrt(double)

from mr.adadelta cimport AdaDelta
from mr.modelBase cimport SklearnModelBase
from mr.util cimport FastRandom

import numpy as np
import scipy.optimize
import sklearn
import sys

cdef class LcaSpikingWoodsAnalyticalInhibition(SklearnModelBase):
    """See the module documentation for the design of this class.  Takes two
    receptive fields, one the "most receptive" and another the "least
    receptive" and generates a network that does sparse approximation
    correctly.

    Example usage:

    .. code-block:: python

        physics = dict(phys_rMax=183e3, phys_rMin=53e3)
        lca = LcaSpikingWoodsAnalyticalInhibition(
                np.r_[ np.ones(90), np.zeros(10) ],
                np.r_[ np.ones(10), np.zeros(90) ],
                **physics)

    """

    UNSUPERVISED = True
    PARAMS = SklearnModelBase.PARAMS + [
            'algTimePerSpike', 'algSpikes', 'algInputKMax', 'algInputWidth',
            'algOutputKMax',
            'rfAvg',
            'rfLeast',
            'phys_rMin',
            'phys_rMax',
            'phys_vcc',
    ]
    PICKLE_VARS = SklearnModelBase.PICKLE_VARS + [
            'ada_',
            '_lcaInternal_crossbar',
            'vFire_',
            'cNeuron_',
            'cInhib_',
            'rInhib_',
    ]


    ## Design parameters
    cdef public object rfAvg, rfLeast
    # Even though rfAvg and rfLeast are arrays of double, sklearn.clone has
    # issues with them in that format, so they are stored as type object.
    cdef public double algTimePerSpike, algSpikes, algInputKMax, algInputWidth
    cdef public double algOutputKMax

    ## Physics parameters
    cdef public double phys_rMin, phys_rMax, phys_vcc

    ## Model values initialized by :meth:`_init`
    cdef public AdaDelta ada_
    cdef public double[:, :] _lcaInternal_crossbar
    property crossbar_:
        """Returns or sets the crossbar used for sparse representation.
        """
        def __get__(self):
            return self._lcaInternal_crossbar
        def __set__(self, double[:, :] crossbar):
            cbShape = (crossbar.shape[0], crossbar.shape[1])
            if crossbar.shape[0] != self.nInputs:
                raise ValueError("Bad shape: {}, not {}".format(cbShape,
                        (self.nInputs, self.nOutputs)))
            if crossbar.shape[1] != self.nOutputs:
                raise ValueError("Bad shape: {}, not {}".format(cbShape,
                        (self.nInputs, self.nOutputs)))
            self._lcaInternal_crossbar = crossbar
    cdef public double vFire_
    """Fire threshold (V)"""
    cdef public double cNeuron_
    """Capacitance of neuron (F)"""
    cdef public double cInhib_
    """Capacitance of inhibition row (F)"""
    cdef public double rInhib_
    """Resistance of RC circuit for inhibition (Ohms)"""

    ## Temporaries that are not saved
    cdef public FastRandom _rand


    def __init__(self, nOutputs=10, rfLeast=None, rfAvg=None, **kwargs):
        defaults = {
                'nOutputs': nOutputs,
                'algTimePerSpike': 1e-9,
                'algSpikes': 10.,
                'algInputKMax': 0.1,
                'algInputWidth': 1e-10,
                'algOutputKMax': 0.1,
                'rfLeast': rfLeast,
                'rfAvg': rfAvg,

                'phys_rMin': 53e3,
                'phys_rMax': 180e3,
                'phys_vcc': 0.8,
        }
        defaults.update(**kwargs)
        super(LcaSpikingWoodsAnalyticalInhibition, self).__init__(**defaults)

        self._rand = FastRandom()


    cpdef _init(self, int nInputs, int nOutputs):
        """Responsible for setting up a new instance of this network.
        """
        cdef int i, j

        self.ada_ = AdaDelta(nInputs, nOutputs)
        self._lcaInternal_crossbar = np.random.uniform(size=(nInputs, nOutputs))

        ## Parameter auto-detection!
        cdef double Q1M = 0., Q2M = 0., Q1L = 0., Q2L = 0., Q, T
        cdef double[:] rfAvg = self.rfAvg
        cdef double[:] rfLeast = self.rfLeast

        if rfAvg is None:
            rfAvg = np.r_[ np.ones(nInputs)*0.5, np.zeros(0) ]
        if rfLeast is None:
            rfLeast = np.r_[ np.ones(1), np.zeros(nInputs - 1) ]

        if rfAvg.shape[0] != nInputs:
            raise ValueError("Bad rfAvg shape: {} != {}".format(
                    rfAvg.shape[0], nInputs))
        if rfLeast.shape[0] != nInputs:
            raise ValueError("Bad rfLeast shape: {} != {}".format(
                    rfLeast.shape[0], nInputs))

        # vFire is determined off the least receptive field's Q ratio
        cdef double G_max = 1. / self.phys_rMin
        cdef double K_max = self.algInputKMax
        cdef double g_min = self.phys_rMin / self.phys_rMax
        for i in range(self.nInputs):
            # TODO - Use max of data average and rfLeast for k in Q2 terms?
            Q1L += max(g_min, rfLeast[i])
            Q2L += max(g_min, rfLeast[i]) * rfLeast[i]
            Q1M += max(g_min, rfAvg[i])
            Q2M += max(g_min, rfAvg[i]) * rfAvg[i]
        Q1L *= G_max
        Q2L *= self.phys_vcc * G_max * K_max
        Q1M *= G_max
        Q2M *= self.phys_vcc * G_max * K_max
        Q = Q2L / Q1L
        self.vFire_ = Q * 0.9
        # TODO 0.9 is magic

        cdef double T_spike = self.algTimePerSpike * self.algOutputKMax
        cdef double T_spikeGap = self.algTimePerSpike * (
                1. - self.algOutputKMax)
        cdef double C = (-T_spikeGap * Q1M / log(1. - self.vFire_ * Q1M / Q2M))
        # Halve that capacitance to make room for inhibition
        C *= 0.5
        self.cNeuron_ = C

        ## Now solve for the inhibition parameters, assuming we want the
        # capacitance of the inhibition to match the neurons
        self.cInhib_ = C = self.cNeuron_

        r = self._init_solveForR(C, k=np.asarray(rfAvg).max(),
                g=np.asarray(rfAvg).max(), Q1=Q1M, Q2=Q2M)
        self.rInhib_ = r


    cpdef _init_getQ(self, double[:] rf):
        """For a given ``rf``, returns (Q1, Q2)."""
        if not self._isInit:
            raise RuntimeError("Must init() first")

        cdef double G_max = 1. / self.phys_rMin
        cdef double K_max = self.algInputKMax
        cdef double g_min = self.phys_rMin / self.phys_rMax
        cdef double Q1 = 0., Q2 = 0.
        cdef int i
        for i in range(self.nInputs):
            Q1 += max(g_min, rf[i])
            Q2 += max(g_min, rf[i]) * rf[i]
        Q1 *= G_max
        Q2 *= self.phys_vcc * G_max * K_max
        return (Q1, Q2)


    cpdef double _init_solveForR(self, double C, double k, double g,
            double Q1, double Q2) except? -1.:
        if not self._isInit:
            raise RuntimeError("Must init() first")

        cdef double Q = Q2 / Q1
        cdef double T_fire = -self.cNeuron_ / Q1 * log((Q - self.vFire_) / Q)
        r = scipy.optimize.fsolve(self._init_solveForR_inner, [1e6],
                args=(C, k, g, T_fire),
                full_output=True)
        if r[2] != 1:
            raise ValueError("Capacitance calculated {} did not produce "
                    "viable inhibition resistance: {}".format(self.cInhib_,
                        r[3]))
        return r[0][0]


    cpdef double _init_solveForR_inner(self, double[:] R0, double C_i,
            double k, double g, double T_fire) except? -1.:
        """Since a viable RC pair is difficult to compute analytically,
        this method's roots are the solutions to (self.cInhib_, R) that
        optimize the inhibition response in this network.

        :param R0: The guess from scipy.optimize.
        :param C_i: The inhibition capacitance to use.
        :param k: The actual input k to use
        :param g: The stored crossbar value to represent k.
        :param T_fire: The time calculation based on Q1 and Q2 for the neuron
                being balanced.
        """
        cdef double R = R0[0]
        if R < 10:
            # Disallow negative / small resistances
            return -0.5

        cdef double vcc = self.phys_vcc
        cdef double G_max = 1. / self.phys_rMin
        cdef double g_min = self.phys_rMin / self.phys_rMax
        cdef double Rcb = 1. / (G_max * max(g, g_min))

        cdef double A = 1. / (Rcb * C_i)
        cdef double B = 1. / (R * C_i)

        # Desired spike period
        cdef double T_outSpikePeriod = self.algTimePerSpike * g / k
        cdef double T_outSpike = self.algTimePerSpike * self.algOutputKMax
        cdef double T_outSpikeGap = T_outSpikePeriod - T_outSpike
        cdef double T_outSpikeGapInhib = T_outSpikeGap - T_fire

        cdef double ke = k * self.algInputKMax
        return (0.5 * exp(ke * T_outSpikeGapInhib * B)
                - (1 - exp(-T_outSpike * A))
                    / (1 - exp(-ke * T_outSpikeGap * B - T_outSpike * A)))


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef _partial_fit(self, double[:] x, double[:] y):
        cdef int i, j
        cdef double[:, :] cb = self._lcaInternal_crossbar
        cdef double[:] bo = self._bufferOut

        self._predict(x, bo)
        self._reconstruct(bo, self._bufferIn)

        cdef np.ndarray[np.double_t, ndim=2] r = np.asmatrix(x) - self._bufferIn
        for i in range(cb.shape[0]):
            for j in range(cb.shape[1]):
                cb[i, j] = max(0.0, min(1.0, cb[i, j]
                        + self.ada_.getDelta(i, j, -2. * bo[j] * r[0, i])))


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef _predict(self, double[:] x, double[:] y):
        ## Local temporaries
        cdef int i, j
        cdef double Q, u, u2
        cdef double inhibMult
        cdef double dt
        cdef bint DEBUG = 0
        cdef int hadSpike
        cdef FastRandom rand = self._rand

        ## Local mirror of relevant physical constants
        cdef double[:, :] crossbar = self._lcaInternal_crossbar
        cdef double[:, :] gCrossbar = np.zeros((self.nInputs, self.nOutputs))
        cdef double vcc = self.phys_vcc
        cdef double vThresh = self.vFire_
        cdef double inhib_vThresh = 0.5 * vcc
        cdef double G_max = 1. / self.phys_rMin
        cdef double g_min = self.phys_rMin / self.phys_rMax
        cdef double K_maxIn = self.algInputKMax, K_maxOut = self.algOutputKMax
        cdef double C = self.cNeuron_
        cdef double C_i = self.cInhib_
        cdef double R_i = self.rInhib_
        cdef double onegBI = -1. * R_i * C_i
        cdef double negBI = 1. / onegBI

        ## Timing calculations
        cdef double T_outSpike = self.algTimePerSpike * K_maxOut
        cdef double T_inSpike = self.algInputWidth
        # Calculate the scalar for the random inbound spikes for each input;
        # note that this is doubled since the mean of a uniform random is 0.5
        cdef double[:] T_inGapRand = np.zeros(self.nInputs)
        for i in range(self.nInputs):
            # Period of one input spike event at full capacity
            u = T_inSpike / (self.algInputKMax * max(1e-4, x[i]))
            T_inGapRand[i] = 2. * (u - T_inSpike)

        ## States
        cdef double[:] Vnode = np.zeros(self.nOutputs)
        cdef double[:] Q1 = np.zeros(self.nOutputs)
        cdef double[:] Q2 = np.zeros(self.nOutputs)
        cdef double[:] Vinhib = np.ones(self.nInputs) * inhib_vThresh
        cdef double[:] inState = np.zeros(self.nInputs)
        cdef double[:] fireCond = np.zeros(self.nInputs)
        cdef double t = 0.
        y[:] = 0.

        ## Calculate the actual crossbar conductances
        for i in range(self.nInputs):
            for j in range(self.nOutputs):
                gCrossbar[i, j] = G_max * max(g_min, crossbar[i, j])

        ## Initialize Q1 values which are constant
        for i in range(self.nInputs):
            for j in range(self.nOutputs):
                Q1[j] += gCrossbar[i, j]

        ## Initialize input spikes (initial phase unknown)
        for i in range(self.nInputs):
            inState[i] = rand.get() * (T_inGapRand[i]*0.5 + T_inSpike)

        if DEBUG:
            dbgK = np.zeros(self.nInputs)
        cdef double algTime = self.algTimePerSpike * self.algSpikes
        while t < algTime:
            dt = algTime - t

            ## Update Q values
            Q2[:] = 0.
            for i in range(self.nInputs):
                if Vinhib[i] > inhib_vThresh or inState[i] > T_inSpike:
                    # Inhibited or not spiking
                    continue

                for j in range(self.nOutputs):
                    Q2[j] += gCrossbar[i, j]
            for i in range(self.nOutputs):
                # K_max and k_i are both 1 here, since we know anything logged
                # is currently spiking.  Note that G_max was already rolled
                # into gCrossbar
                Q2[i] *= vcc

            if DEBUG:
                allTimes = []

            ## Calculate the next phase change
            # For input lines
            for i in range(self.nInputs):
                # Input line
                if inState[i] > T_inSpike:
                    dt = min(dt, inState[i] - T_inSpike)
                else:
                    dt = min(dt, inState[i])

                # Input inhibition
                if Vinhib[i] > inhib_vThresh and inState[i] <= T_inSpike:
                    dt = min(dt, log(inhib_vThresh / Vinhib[i]) * onegBI)

            # For output neurons
            for i in range(self.nOutputs):
                Q = Q2[i] / Q1[i]
                if Q <= vThresh:
                    continue

                u = -C / Q1[i] * log((Q - vThresh) / (Q - Vnode[i]))
                if DEBUG:
                    allTimes.append(u)
                dt = min(dt, u)

            if DEBUG:
                print("At {}: {}, {}.  {}.  {}.  {}.".format(t, dt, allTimes,
                        np.asarray(y), np.asarray(Vnode), np.asarray(Vinhib)))

            ##  Apply time update
            if dt < -1e-300:
                raise ValueError("BAD TIME UPDATE: {}.\n\nIn states: {}\n\nT_inSpike: {}"
                        .format(dt, inState, T_inSpike))

            # Ensure there are no zero-time updates
            dt += min(T_inSpike, T_outSpike) * 1e-8

            # Update inhibition first as regardless of spike, the fall time is
            # the same.
            inhibMult = exp(dt * negBI)
            for i in range(self.nInputs):
                fireCond[i] = 0.
                if inState[i] <= T_inSpike:
                    if DEBUG:
                        dbgK[i] += dt
                    Vinhib[i] *= inhibMult

            # Charge all neurons according to dt, log spikes
            hadSpike = -1
            for i in range(self.nOutputs):
                u = exp(-dt * Q1[i] / C)
                Vnode[i] = Q2[i] / Q1[i] * (1. - u) + Vnode[i] * u
                if Vnode[i] < vThresh:
                    continue

                # Spiking event; tally up inhibition.  We have a flag for
                # spiking because when multiple spikes happen, inhibition
                # should still only be updated once
                hadSpike = i

                Vnode[i] = 0.
                Vnode[:] = 0.
                # NOTE - If inhibition is turned off, the above line must be
                # turned on!  The equations rely on it.  Come to think of it,
                # all of the equations rely on it.
                y[i] += 1.

                for j in range(self.nInputs):
                    # Conductance stacks
                    fireCond[j] += gCrossbar[j, i]

            # Update inhibition of each row based on spiking events and
            # add spike time to dt
            if hadSpike >= 0:
                dt += T_outSpike
                for j in range(self.nInputs):
                    u2 = 1. / fireCond[j]  # R_{cb}
                    u = exp(-T_outSpike / (C_i * u2))  # e^{-T_{spike}A}
                    Vinhib[j] = vcc * (1. - u) + Vinhib[j] * u

            # Finally, update input lines according to dt, including any time
            # added due to spiking
            for i in range(self.nInputs):
                inState[i] -= dt
                # MUST be a while loop - T_outSpike can be comparatively large
                while inState[i] <= 0.:
                    inState[i] += rand.get() * T_inGapRand[i] + T_inSpike

            # And update sim time so far
            t += dt

        if DEBUG:
            print("K: {} / {}".format(dbgK / algTime,
                    np.asarray(x) * self.algInputKMax))

        u = 1. / self.algSpikes
        for i in range(self.nOutputs):
            y[i] *= u


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef _reconstruct(self, double[:] y, double[:] r):
        cdef int i, j
        r[:] = 0.0
        for j in range(self.nOutputs):
            for i in range(self.nInputs):
                r[i] += self._lcaInternal_crossbar[i, j] * y[j]



cdef class LcaSpikingWoodsAnalytical(SklearnModelBase):
    """An analytical (not ODE-simulated) version of LcaSpikingWoods."""
    UNSUPERVISED = True
    PARAMS = SklearnModelBase.PARAMS + [ 'avgInput', 'simTime', 'nHomeo',
            'nSpikes', 'circuit_rMin', 'circuit_rMax', 'untrainedK',
            'spikeDensity', 'tolerance_vFire', 'vEventSet', 'drainAll',
            'learnMomentum', 'adaDeltaRho', 'inhib' ]
    PICKLE_VARS = SklearnModelBase.PICKLE_VARS + [ '_crossbar', '_homeostasis',
            'cNeuron_', 'vFire_',
            '_eG2', '_edX', '_learnMoments' ]

    cdef public double avgInput, simTime, nSpikes, circuit_rMin, circuit_rMax
    cdef public double untrainedK, spikeDensity
    cdef public int nHomeo
    cdef public double tolerance_vFire
    cdef public double vEventSet
    cdef public bint drainAll

    # Inhibition black magick
    cdef public dict inhib

    # Adadelta stuff
    cdef public double learnMomentum
    cdef public double adaDeltaRho
    cdef public double[:, :] _eG2, _edX, _learnMoments

    # Internals
    cdef public double cNeuron_, vFire_

    cdef public double[:, :] _crossbar
    cdef public double[:] _homeostasis

    # Temporaries set on init
    cdef double _condMax
    cdef double _condMin
    cdef double[:] _randBuffer
    cdef int _randIndex, _randLen

    cpdef convergerProps(self):
        return [ self._crossbar, self._eG2, self._edX, self._homeostasis ]


    def __init__(self, nOutputs = 10, **kwargs):
        defaults = {
                'nOutputs': nOutputs,
                'avgInput': 0.1,
                'simTime': 1e-8,
                'nHomeo': 1,
                'nSpikes': 10.0,
                'spikeDensity': 0.1,
                'circuit_rMin': 52279.,
                'circuit_rMax': 206969.,
                'untrainedK': 2.0,
                'vEventSet': 0.7,
                'drainAll': True,

                'adaDeltaRho': 0.9,
                'learnMomentum': -1,

                'inhib': None,

                # Tolerances - +/- for various properties
                'tolerance_vFire': 0.001,
        }
        defaults.update(**kwargs)
        super(LcaSpikingWoodsAnalytical, self).__init__(**defaults)

        self._condMax = 1.0 / self.circuit_rMin
        self._condMin = 1.0 / self.circuit_rMax
        self._randIndex = 0
        self._randLen = 1024
        self._randBuffer = np.random.uniform(size = (self._randLen,))


    cpdef double _getCond(self, double weight):
        cdef double r = self._condMax * weight
        if r < self._condMin:
            return self._condMin
        return r


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
        if log1 < 1e-10:
            log1 = 1e-10
        if log2 < 1e-300:
            log2 = 1e-300
        cdef double r =  tQ1 * log(log1) / (uQ1 * log(log2)) - self.untrainedK
        if r > 1000.0:
            return 9999
            return (1000.0 - self.untrainedK) * (1000.0 - self.untrainedK)
        return r * r


    cdef double _calcVFire(self) except -1:
        """Returns vFire for given network params."""
        # Untrained (independent) and trained conductances
        cdef double cAvg = self._getCond(self.avgInput)
        cdef double cMax = self._getCond(self.avgInput)
        cdef double cHomeo = self.nHomeo * self._condMin
        cdef double ckHomeo = cHomeo * self.spikeDensity

        # Number active inputs, on average
        cdef double T = self._crossbar.shape[0]

        # Untrained vars
        # An untrained network has the same average as a trained, but
        # its RF is as orthogonal as can be.  Since Q2 relies on
        # \sum_n{k_n c_n}, this looks like \sum_n{k_n c(k_n)} for a perfect
        # fit.  Ideally (no lower bound), c(k_n) == k_n c_{max}, this means
        # that Q2 relies on c_{max} \sum_n{k_n^2}.  k_n can be treated as an
        # array of i.i.d. random variables with mean self.avgInput.  Calculating
        # E(k_n^2) will give us the conductance to use for Q2.
        cdef double w, w2, minWeight = self._condMin / self._condMax
        cdef int i
        opt = np.zeros(100000)
        for i in range(len(opt)):
            w = np.random.beta(self.avgInput, 1.0 - self.avgInput)
            opt[i] = max(minWeight, w) * w
        cdef double optimal = opt.mean() * self.spikeDensity
        subop = np.zeros(100000)
        for i in range(len(subop)):
            w = np.random.beta(self.avgInput, 1.0 - self.avgInput)
            w2 = np.random.beta(self.avgInput, 1.0 - self.avgInput)
            subop[i] = max(minWeight, w2) * w
        cdef double suboptimal = subop.mean() * self.spikeDensity
        cdef double uQ1 = cHomeo + T * cAvg
        cdef double uQ2 = self.vEventSet * (ckHomeo + T * suboptimal
                * self._condMax)

        # Trained vars
        cdef double tQ1 = cHomeo + T * cAvg
        cdef double tQ2 = self.vEventSet * (ckHomeo + T * optimal
                * self._condMax)

        # tU / tT
        #O = scipy.optimize.leastsq(self._calcVFire_error, [ 1e-3 ],
        #        args = (uQ1, uQ2, tQ1, tQ2))
        #cdef double vFire = O[0][0]
        qPossible = np.linspace(1e-4, 5e-1, 1001)
        qError = np.asarray([ self._calcVFire_error([ q ], uQ1, uQ2, tQ1, tQ2)
                for  q in qPossible ])
        cdef double vFire = qError.argmin() * (5e-1 - 1e-4) * 1e-3 + 1e-4

        cdef double vFireError = self._calcVFire_error([vFire], uQ1, uQ2, tQ1,
                tQ2)
        if vFireError > 1.:
            raise ValueError("Bad vFire? {}.  Error {}.  All: {}".format(vFire,
                    vFireError ** 0.5 + self.untrainedK, qError))

        #vFire = 0.01
        if True:
            import pandas
            args = np.linspace(1e-4, 5e-1, 1001)
            vFireErrors = [ sqrt(self._calcVFire_error([ q ], uQ1, uQ2, tQ1, tQ2)) + self.untrainedK for q in args ]
            df = pandas.DataFrame(zip(args, vFireErrors), columns = ["vFire","K"])
            #print df.to_string()
            print("FOUND vFIRE {}, K {} / WANTED {}".format(vFire,
                    sqrt(self._calcVFire_error([vFire], uQ1, uQ2, tQ1, tQ2))
                        + self.untrainedK,
                    self.untrainedK))
            print("Others (-/+): {} / {}".format(
                    sqrt(self._calcVFire_error([vFire - 0.001], uQ1, uQ2, tQ1, tQ2))
                        + self.untrainedK,
                    sqrt(self._calcVFire_error([vFire + 0.001], uQ1, uQ2, tQ1, tQ2))
                        + self.untrainedK,))
        return abs(vFire)


    cpdef double _calcVFromC(self, double C) except -1:
        cdef double cAvg = self._getCond(self.avgInput)
        cdef double cHomeo = self.nHomeo * self._condMin
        cdef double ckHomeo = self.spikeDensity * cHomeo

        cdef double T = self._crossbar.shape[0]

        op = np.zeros(100000)
        cdef int i
        cdef double w, wMin = self._condMin / self._condMax
        for i in range(len(op)):
            w = np.random.beta(self.avgInput, 1. - self.avgInput)
            op[i] = max(wMin, w)*w

        cdef double Q1 = cHomeo + T * cAvg
        cdef double Q2 = (
                T * self.vEventSet * self.spikeDensity * self._condMax * op.mean()
                + ckHomeo * self.vEventSet)

        cdef double t = self.simTime / self.nSpikes
        return Q2 / Q1 * (1 - np.exp(-t * Q1 / C))


    cdef double _calcCNeuron(self, double vFire) except -1:
        # Trained vars only
        cdef double cAvg = self._getCond(self.avgInput)
        cdef double cMax = self._getCond(self.avgInput)
        cdef double cHomeo = self.nHomeo * self._condMin
        cdef double ckHomeo = self.spikeDensity * cHomeo

        cdef double T = self._crossbar.shape[0]

        op = np.zeros(100000)
        cdef int i
        cdef double w, wMin = self._condMin / self._condMax
        for i in range(len(op)):
            w = np.random.beta(self.avgInput, 1.0 - self.avgInput)
            op[i] = max(wMin, w) * w
        cdef double optimal = op.mean() * self.spikeDensity
        cdef double Q1 = cHomeo + T * cAvg
        cdef double Q2 = self.vEventSet * (ckHomeo + T * optimal * self._condMax)

        cdef double t = self.simTime / self.nSpikes
        return -t * Q1 / log(1. - vFire * Q1 / Q2)


    cpdef _calcFireTime(self, double voltage, double condSum,
            double condKProduct):
        """Fire time:

        t = -C / Q1 * ln(1 - Q1 / Q2 * Vnode)
        Q1 = sum(cond)
        Q2 = Vevent * sum(k*cond)

        Returns (vMax, fireTime), where fireTime is -1 if never fires"""
        cdef double Q1 = condSum
        cdef double Q2 = self.vEventSet * condKProduct
        if Q2 < 1e-300:
            return (-1.0, -1.0)
        cdef double gap = 1 - Q1 / Q2 * voltage
        if gap <= 0.0:
            return (Q2 / Q1, -1.0)
        return (Q2 / Q1, -self.cNeuron_ / Q1 * log(gap))


    cpdef calcFireTimes(self, states):
        """Populates states with normalized fire times.

        states - [ lambda, ... ] where each lambda returns two values: a [0, 1]
                weight in the learned dictionary element, and a [0, 1] k value
                for the input.

        returns a list of dicts with avgWeight, avgInput, avgCond, avgCondK,
        and normalized fireRate, where 1.0 = on target (simTime / nSpikes), 2.0
        is twice as fast, etc.
        """
        # t = -C / Q1 * ln(1 - Q1 / Q2 * Vnode)
        cdef double avgWeight, avgInput, avgCond, avgCondK
        cdef double condSum, condKProduct, avgFire
        avgFire = self.simTime / self.nSpikes
        results = []
        for d in states:
            jb = np.asarray([ d() for _ in range(10000) ])
            avgWeight = jb[:, 0].mean()
            avgInput = jb[:, 1].mean()
            conds = np.asarray([ self._getCond(w) for w in jb[:, 0] ])
            avgCond = conds.mean()
            condk = conds * jb[:, 1] * self.spikeDensity
            #condk = self._condMax * jb[:, 0] * jb[:, 1] * self.spikeDensity
            avgCondK = condk.mean()

            condSum = self.nInputs * avgCond
            condKProduct = self.nInputs * avgCondK

            ft = self._calcFireTime(self.vFire_, condSum,
                    condKProduct)
            results.append(dict(avgWeight = avgWeight, avgInput = avgInput,
                    avgCond = avgCond, avgCondK = avgCondK,
                    vMax = ft[0],
                    fireRate = avgFire / ft[1]))

        return results


    cpdef _init(self, int nInputs, int nOutputs):
        self._crossbar = np.random.uniform(size = (nInputs, nOutputs))
        self._eG2 = np.zeros((nInputs, nOutputs), dtype = float)
        self._edX = np.zeros((nInputs, nOutputs), dtype = float)
        self._eG2[:, :] = 1e-3
        self._edX[:, :] = 1e-3
        self._homeostasis = np.zeros((nOutputs,), dtype = float)

        self.vFire_ = self._calcVFire()
        self.cNeuron_ = self._calcCNeuron(self.vFire_)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef _partial_fit(self, double[:] x, double[:] y):
        cdef int i, j

        cdef double adRho = self.adaDeltaRho, e = 1e-6
        cdef double g, gScalar, cbDelta

        cdef double[:] bufIn = self._bufferIn, bufOut = self._bufferOut
        cdef double[:,:] _eG2 = self._eG2, _edX = self._edX, _crossbar = self._crossbar

        self._predict(x, bufOut)
        self._reconstruct(bufOut, bufIn)

        '''
        # Bufin is linked...
        ih = bufIn.shape[0] // 2
        for i in range(ih):
            pval = (bufIn[i] - bufIn[i + ih])
            if pval >= 0.0:
                bufIn[i] = pval * 0.5 + 0.5
                bufIn[i + ih] = 0.
            else:
                bufIn[i] = 0.
                bufIn[i + ih] = 0.5 - pval * 0.5'''

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

                if self.learnMomentum > 0:
                    # Traditional momentum
                    cbDelta += self._learnMoments[i, j] * self.learnMomentum
                    self._learnMoments[i, j] = cbDelta

                _crossbar[i, j] = max(0.0, min(1.0,
                        _crossbar[i, j] + cbDelta))

        self._updateHomeostasis(bufOut)


    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef _predict(self, double[:] x, double[:] y):
        cdef int i, j
        cdef int cycleCnt = 0
        cdef double C = self.cNeuron_
        cdef double vThresh, vFire = self.vFire_
        cdef double vThreshWiggle = self.tolerance_vFire * 2.0
        cdef double[:, :] cond = np.zeros((self._crossbar.shape[0],
                self._crossbar.shape[1]), dtype = float)
        cdef double[:] k = np.zeros((self._crossbar.shape[0] + self.nHomeo,),
                dtype = float)

        # Inhibition variables
        cdef double inhibA, inhibB, inhibC
        cdef double[:] inhibs = np.zeros((self._crossbar.shape[0],),
                dtype = float)
        cdef double[:] inhibsAvg = np.zeros((self._crossbar.shape[0],),
                dtype = float)
        cdef double[:] kPreInhib = np.zeros((self._crossbar.shape[0],),
                dtype = float)
        cdef double[:] inhibCond = np.zeros((self._crossbar.shape[0],),
                dtype = float)
        cdef double inhibMin, inhibMax, inhibAvg, inhibEffect
        if self.inhib is not None:
            inhibMin = self.inhib['threshMin']
            inhibMax = self.inhib['threshMax']
            inhibs[:] = inhibMin

        # Homeostasis and Q states
        cdef double[:] homeoCond = np.zeros((self.nOutputs,), dtype = float)
        cdef double[:] Q1 = np.zeros(self.nOutputs), Q2 = np.zeros(self.nOutputs)
        cdef double Q
        cdef double tSpike = self.simTime / self.nSpikes

        # Stash all of our conductances
        for i in range(self._crossbar.shape[0]):
            for j in range(self.nOutputs):
                cond[i, j] = self._getCond(self._crossbar[i, j])
        for j in range(self.nOutputs):
            homeoCond[j] = self._getCond(self._homeostasis[j])
            Q1[j] = 0.0
            for i in range(self._crossbar.shape[0]):
                Q1[j] += cond[i, j]
            for i in range(self.nHomeo):
                Q1[j] += homeoCond[j]

        cdef double spike, best, tmp
        cdef int bestN
        cdef double[:] times = np.zeros((self.nOutputs,), dtype = float)
        cdef double[:] volts = np.zeros((self.nOutputs,), dtype = float)
        cdef double invSpikes = 1.0 / self.nSpikes
        y[:] = 0.0
        cdef double t = 0.0
        while t < self.simTime:
            cycleCnt += 1
            # Each time, wiggle our firing threshold by a bit to show that we
            # need it to be stable
            vThresh = vFire + (self._fastRand() - 0.5) * vThreshWiggle

            best = self.simTime
            bestN = -1
            for i in range(self._crossbar.shape[0]):
                # Re-sample k[i]
                k[i] = self._sampleK(x[i % x.shape[0]], tSpike*10., tSpike)
                k[i] = x[i % x.shape[0]] * self.spikeDensity

                # Apply inhibition.  General process:
                # pre/post means pre- and post- output spike
                # T is time between spike events
                #
                # 1. Inhibition added is proportional to representation by spike:
                #     TODO find constant A in terms of T, R, and C
                #     I_{post} = I_{max} + (I_0 - I_{max})e^-A
                #
                # 2. Each spike un-inhibits the next.
                #     a. Generally, k_i = g_i should result in an asymptotic
                #         average inhibition of A.
                #     b. I drains through an RC circuit -> I_{pre} = I_0 e^{-B},
                #         B = T/RC
                #         I_{avg} = I_0/T \int_0^T e^{-t/RC} dt
                #         = -RCI_0/T (e^{-T/RC} - 1)
                #         = I_0/B (1 - e^{-B})
                #
                #         Rehash: Only k% of the time is that draining
                #         happening.  Thus, if we average everything out, we
                #         actually get kt = \hat{t} where \hat{t} is effective
                #         drain time.
                #
                #         I_{pre} = I_0 e^{-kB}, ...,
                #         I_{avg} = I_0 / (kB) (1 - e^{-kB})
                #     c. To achieve equilibrium, I_0 = I_{post} for pre, I_{pre} for post..
                #         I_{pre} = I_{post} e^{-kB}
                #         I_{post} = I_{max} + (I_{pre} - I_{max})e^{-A}
                #         = I_{max} + (I_{post} e^{-kB} - I_{max})e^{-A}
                #         = I_{max}(1 - e^{-A}) / (1 - e^{-kB}e^{-A})
                #         I_{avg} = I_{post} / (kB) (1 - e^{-kB})
                #         = I_{max}(1 - e^{-A})(1 - e^{-kB}) / [ (kB)(1 - e^{-kB}e^{-A}) ]
                #         = I_{max} (1 - e^{-A} - e^{-kB} + e^{-kB}e^{-A}) / [kB(1 - e^{-kB}e^{-A})]
                #
                #         A and B are both positive, so this works.
                #         A is dependent on the crossbar.  A will be larger for
                #             more conductive rows.
                #         B and k are dependent only on input element... assuming equal distribution,
                #         I_{avg} = C (1 - e^{-A}) / (1 - e^{-kB}{e^{-A}),
                #         C = I_{Max} (1 - e^{-kB}) / (kB)
                #         Double iteration to find the range:
                #         I_{range} = I_{max} + I_{max} e^{-kB}
                #         Where C varies based on B/k statistics but is fixed for the crossbar.
                #         The smaller the value of A, the less hits quantity is affected by row dynamics.
                #         The larger the value of A, the larger the dynamic range of inhibition.
                #
                #     TODO - Calculate A (should be easy), simulate inhibition ranges
                #         for the memristive devices / etc we have.
                #
                #     A k = k_{max} should result in maximum
                #tmp = max(0., inhib[i] - k[i])
                #k[i] *= inhib[i] * 0.5
                #inhib[i] = max(0., inhib[i] - k[i])

                if self.inhib is not None:
                    # Based on sampled K and known constants, integrate inhibition
                    inhibB = tSpike * (1. - self.spikeDensity) / (self.inhib['rDrain'] * self.inhib['capacitance'])
                    tmp = max(0.01, k[i])
                    tmp = 1.
                    inhibAvg = inhibs[i] / (tmp*inhibB) * (1. - exp(-tmp*inhibB))
                    inhibEffect = (inhibAvg - inhibMin) / (inhibMax - inhibMin)
                    inhibEffect = min(1., max(0., inhibEffect))
                    kPreInhib[i] = tmp
                    k[i] *= 1. - inhibEffect
                    # Log pre-spike inhibition
                    inhibsAvg[i] += inhibAvg
            for i in range(self.nHomeo):
                k[self._crossbar.shape[0] + i] = self._sampleK(1.0, tSpike,
                        tSpike)

            for j in range(self.nOutputs):
                # Calculate this event's Q1 and Q2 based on sampled k values
                Q2[j] = 0.0
                for i in range(self._crossbar.shape[0]):
                    Q2[j] += cond[i, j] * k[i]
                for i in range(self.nHomeo):
                    Q2[j] += homeoCond[j] * k[self._crossbar.shape[0] + i]
                Q2[j] *= self.vEventSet

                # Find our time
                Q = Q2[j] / Q1[j]
                tmp = Q - vThresh
                if tmp > 1e-300:
                    spike = -C / Q1[j] * log(tmp / (Q - volts[j]))
                else:
                    spike = self.simTime
                times[j] = spike
                if bestN == -1 or spike < best:
                    best = spike
                    bestN = j

            # Correct for overwiggling
            best = max(self.simTime * 0.001 / self.nSpikes, best)
            if False:
                # Note - we may want to round best here, if the rate at which we
                # do the comparison is not infinite
                best += self.simTime / self.nSpikes * 0.1
            if t + best < self.simTime:
                # bestN fires, then everything resets
                # TODO - Power calculations for drainAll are off!
                if self.drainAll:
                    for j in range(self.nOutputs):
                        if times[j] <= best:
                            y[j] += invSpikes
                        volts[j] = 0.0
                else:
                    for j in range(self.nOutputs):
                        if times[j] <= best:
                            y[j] += invSpikes
                            volts[j] = 0.0
                        else:
                            Q = exp(-best * Q1[j] / C)
                            Q = Q2[j] / Q1[j] * (1 - Q) + Q * volts[j]
                            #print("{} / {}: {} -> {}".format(times[j], best,
                            #        volts[j], Q))
                            volts[j] = Q

                # Update inhibition terms - they drained in the interim and are
                # charged by firing spikes
                if self.inhib is not None:
                    # First calculate inhibition drain
                    inhibC = self.inhib['capacitance']
                    inhibB = best / (self.inhib['rDrain'] * inhibC)
                    for i in range(self._crossbar.shape[0]):
                        inhibs[i] = inhibs[i] * exp(-kPreInhib[i] * inhibB)
                        inhibCond[i] = 0.
                        for j in range(self._crossbar.shape[1]):
                            if times[j] <= best:
                                inhibCond[i] += cond[i, j]

                    # Second (see above), charge the inhibition capacitor
                    for i in range(self._crossbar.shape[0]):
                        inhibA = tSpike * self.spikeDensity / ((self.inhib['rCharge'] + 1. / inhibCond[i]) * inhibC)
                        inhibs[i] = 1. + (inhibs[i] - 1.) * exp(-inhibA)
            else:
                # Clip best for power debugging
                best = self.simTime - t

            # Aggregate power over timeslice (best)
            if self._debug:
                self.debugInfo_.simTime[self.debugInfo_.index] = t + best
                for j in range(self.nOutputs):
                    # tmp is integral of V over (0, best)
                    tmp = Q2[j] / Q1[j] * (best + C / Q1[j] * (exp(
                            -best * Q1[j] / C) - 1.0))
                    for i in range(self._crossbar.shape[0]):
                        self.debugInfo_.energy[self.debugInfo_.index] += (
                                k[i] * cond[i, j] * self.vEventSet ** 2 * (
                                    best - tmp / self.vEventSet))
                    for i in range(self.nHomeo):
                        self.debugInfo_.energy[self.debugInfo_.index] += (
                                k[self._crossbar.shape[0] + i] * homeoCond[j]
                                * self.vEventSet ** 2 * (
                                    best - tmp / self.vEventSet))

                    # Whether or not we fire (end of sim), drain this capacitor
                    tmp = Q2[j] / Q1[j] * (1 - exp(-best * Q1[j] / C))
                    self.debugInfo_.energy[self.debugInfo_.index] += (
                            C * tmp * tmp * 0.5)

            # Set time to after spike
            t += best + tSpike * self.spikeDensity

        if self.inhib is not None and False:
            self._reconstruct(y, self._bufferIn)
            print("{} for input {} reconstructed as {}, {} spikes".format(
                    np.asarray(inhibsAvg)[:5] / cycleCnt,
                    np.asarray(x)[:5],
                    np.asarray(self._bufferIn)[:5],
                    # Last cycle never spikes
                    cycleCnt-1))


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef _reconstruct(self, double[:] y, double[:] r):
        cdef int i, j
        cdef double[:, :] _crossbar = self._crossbar
        r[:] = 0.0
        for j in range(self.nOutputs):
            for i in range(self.nInputs):
                r[i] += _crossbar[i, j] * y[j]


    @cython.cdivision(True)
    cdef inline double _sampleK(self, double k, double tWindow, double tSpike):
        """Returns a sampled input scalar k for given time window and spike
        time (average spike time, e.g. self.simTime / self.nSpikes)."""
        if k < 1e-10:
            k = 1e-10
        cdef double kInv = 1.0 / k
        cdef double tGapInterval = tSpike * kInv
        tSpike *= self.spikeDensity
        tGapInterval -= tSpike

        # U is where the spike starts, including already started positions
        # (-tSpike).
        cdef double R = self._fastRand()
        cdef double U = -tSpike - tGapInterval * 10.0# + R * (tSpike + tGapInterval) / tSpike
        cdef double total = 0.0
        while U < tWindow:
            if U <= -tSpike:
                pass
            elif U + tSpike < tWindow:
                if U < 0:
                    total += tSpike + U
                else:
                    total += tSpike
            else:
                total += tWindow - U
            R = self._fastRand() * 2.0
            U += tSpike + tGapInterval * R

        return total / tWindow


    cpdef double _sampleK_test(self, double k, double tWindow, double tSpike):
        """Returns _sampleK, in python form"""
        return self._sampleK(k, tWindow, tSpike)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _updateHomeostasis(self, double[:] y):
        cdef int i
        cdef double dh, a, b
        cdef double[:] _homeostasis = self._homeostasis
        # Normalize output and drop / recharge homeostasis
        for i in range(self.nOutputs):
            #dh = (1.0 - self._homeostasis[i]) * 0.1 - y[i] * 1.0
            #dh = 0.1 - y[i]
            #dh = -0.2 * y[i] + (1.0 - self._homeostasis[i]) / max(1, self.nOutputs - 1)
            #self._homeostasis[i] = max(0.0, self._homeostasis[i] + dh)
            dh = ((1.0 - _homeostasis[i]) / max(1, self.nOutputs - 1)
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
            a = 1.0 / self.nOutputs
            b = a * (1 - 0.5) / (1.0 / self.nOutputs)
            dh = ((1. - _homeostasis[i]) * a - b * y[i])
            _homeostasis[i] = min(1.0, max(0.0, _homeostasis[i] + dh))
