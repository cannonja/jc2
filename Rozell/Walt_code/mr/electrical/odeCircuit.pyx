
#cython: profile=True

cimport cython
cimport numpy as np
from libc.math cimport exp, fabs, sinh

cdef inline double dmax(double a, double b):
    if a >= b:
        return a
    return b

import math
import numpy as np
import sys

cdef double ANY_DT = 1e6

cdef class OdeComponent:
    def __init__(self, OdeCircuit circuit, list props = []):
        """Installs the component in the circuit, and allocates values for each
        kwarg, then sets the value.

        props - Ordered list of tuples (name, initialValue, maxStableChange).
                maxStableChange is the change in value to trigger instability.
                By default, this value is multiplied by 0.1.  For nodes, for
                instances, maxStableChange == 1.0 for the voltage.
        """
        cdef int i, nKeys

        self.circuit = circuit
        self.circuit._comps.append(self)

        nKeys = len(props)
        self.indices = np.zeros((nKeys,), dtype = np.int64)
        self.indexNames = []
        for i in range(nKeys):
            self.indices[i] = self.circuit.addValue(props[i][0], props[i][1],
                    props[i][2])
            self.indexNames.append(props[i][0])


    cdef int getIndex(self, unicode name) except -1:
        """Returns the integer index for the given attribute."""
        cdef int i
        for i in range(len(self.indexNames)):
            if self.indexNames[i] == name:
                return self.indices[i]
        raise ValueError("No property named {}".format(name))


    cpdef double getValue(self, unicode name) except? -1:
        """Pythonic interface to (slowly) investigate a component's state."""
        return self.circuit.values[self.getIndex(name)]


    cpdef setValue(self, unicode name, double value):
        """Pythonic interface to (slowly) set a component's state."""
        self.circuit.values[self.getIndex(name)] = value


    cdef void updateDerivatives(self, double[:] values, double[:] derivatives):
        """Updates derivatives with values based on this component's
        differential equations and current values."""
        pass



cdef class OdeCircuit:
    """A simulatable network.

    Components are added via OdeComponent.__init__().

    Power is measured only for FixedNodes with voltage flowing out of them.
    """

    def __init__(self):
        self._comps = []
        self.dtMax = ANY_DT
        self.dtMin = 1e-6
        self.stableScale = 0.1
        self.stderrEveryXSeconds = 0.0

        self.nValues = 0
        self.maxValues = 100
        self.valueNames = []
        self.values = np.zeros((self.maxValues,), dtype = float)
        self.derivatives = np.zeros((self.maxValues,), dtype = float)
        self.stableConsts = np.zeros((self.maxValues,), dtype = float)

        self.ground = FixedNode(self, 0.0)

        self.reset()


    cdef int addValue(self, unicode name, double initialValue,
            double stableConst):
        cdef int nmax
        cdef double[:] nvalues, nderivs, nstables
        if self.nValues == self.maxValues:
            nmax = int(self.maxValues * 3 / 2)
            nvalues = np.zeros((nmax,), dtype = float)
            nderivs = np.zeros((nmax,), dtype = float)
            nstables = np.zeros((nmax,), dtype = float)
            nvalues[:self.nValues] = self.values
            nderivs[:self.nValues] = self.derivatives
            nstables[:self.nValues] = self.stableConsts
            self.values = nvalues
            self.derivatives = nderivs
            self.stableConsts = nstables
            self.maxValues = nmax

        nmax = self.nValues
        self.nValues += 1

        self.valueNames.append(name)
        self.values[nmax] = initialValue
        self.stableConsts[nmax] = stableConst

        return nmax


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef double advance(self, double dtMax = -1.0) except -1:
        """Advances one time integral, up to dtMax seconds.  dtMax may be
        different from the class-level dtMax; the min of those two is used.
        """
        cdef int i
        cdef double dt = self.dtMax, dtMin
        cdef double cChange, cChangeFactor
        cdef double[:] values = self.values, derivatives = self.derivatives
        cdef double[:] stableConsts = self.stableConsts

        if dtMax > 0:
            dt = min(dt, dtMax)
        dtMin = min(dt, self.dtMin)

        # Before we simulate the network, run user code and
        # then log our state at this moment
        self.preAdvance(self.lastDt)

        # Fill derivatives based on current state
        self._stepPower = 0.0
        self.updateDerivatives(self.simTime, values)

        # Determine the maximum stable change based on derivatives and
        # acceptable scale factors
        cChangeFactor = 1.0 / self.stableScale
        for i in range(self.nValues):
            #if ccyth.name is not None:
            #    ccyth.addLogs(LogEntry(entry, ccyth.name))

            # Figure out the change at the current dt
            cChange = fabs(derivatives[i]) * cChangeFactor
            if cChange * dt > stableConsts[i]:
                # New dt!
                dt = stableConsts[i] / cChange
                if dt < dtMin:
                    if dt < self.minFlaggedDt * 0.5:
                        sys.stderr.write("Below threshold update {} (from "
                                "{}) / min update {}\n".format(dt,
                                    self.valueNames[i],
                                    dtMin))
                        self.minFlaggedDt = dt
                    dt = dtMin

        # Now that we have dt, apply all derivatives
        for i in range(self.nValues):
            values[i] += derivatives[i] * dt
        #cdef double[:, :] r = scipy.integrate.odeint(
        #        self.updateDerivatives,
        #        np.asarray(values[:self.nValues]),
        #        [ 0.0, dt ])
        #for i in range(self.nValues):
        #    values[i] = r[1, i]

        cdef double oldTime = self.simTime
        self.energy += self._stepPower * dt
        self.simTime += dt
        if self.stderrEveryXSeconds > 0:
            oldI = int(oldTime / self.stderrEveryXSeconds)
            newI = int(self.simTime / self.stderrEveryXSeconds)
            if oldI != newI:
                sys.stderr.write("(simulated {}s)\n".format(self.simTime))
        self.lastDt = dt
        return dt


    cpdef advanceRoundUp(self, roundInterval = 1.0):
        """Rounds time up to next even increment of roundInterval."""
        cdef int simInt

        # Let the system settle and log history
        self.advance(self.dtMin)
        simInt = int(1.0 / roundInterval)
        self.simTime = (math.ceil(simInt * self.simTime + 0.5) * roundInterval
                - self.dtMin)
        self.advance(self.dtMin)


    cpdef clearHistory(self, timeToKeep = 0):
        '''
        if timeToKeep != 0 and len(self.hist) != 0:
            hi = len(self.hist) - 1
            lo = 0
            timeCut = self.hist[hi]['time'] - timeToKeep
            while lo < hi:
                idx = (lo + hi) // 2
                v = self.hist[idx]['time']
                if v < timeCut:
                    lo = idx + 1
                else:
                    hi = idx - 1
            assert self.hist[lo]['time'] >= timeCut
            self.hist = self.hist[lo:]
        else:
            self.hist = []'''
        return


    cdef void preAdvance(self, double lastDt) except *:
        """Called before a step is taken.  Useful for e.g. user code logic.

        lastDt - Time (seconds) simulated between last step and this one.
        """


    cpdef reset(self):
        # Seconds into simulation
        self.simTime = 0.0
        self.lastDt = 0.0
        self.energy = 0.0
        self.minFlaggedDt = ANY_DT
        self.clearHistory()


    cpdef simulate(self, double dt):
        """Simulate exactly dt of simTime, updating history, etc along the
        way."""
        cdef double endTime = self.simTime + dt
        cdef double gap
        while self.simTime < endTime:
            gap = endTime - self.simTime
            self.advance(gap)


    cpdef updateDerivatives(self, double simTime, double[:] values):
        """Populates (and returns, for convenience) our derivatives."""
        cdef int i
        cdef OdeComponent ccyth
        cdef double[:] derivView = self.derivatives[:self.nValues]
        derivView[:] = 0.0
        for ccyth in self._comps:
            ccyth.updateDerivatives(values, derivView)
        return derivView



cdef class Node(OdeComponent):
    """Also a basic wire Node; has currents in / out, changes voltage"""
    cdef double capacitanceInv

    property capacitance:
        def __get__(self):
            return 1.0 / self.capacitanceInv
        def __set__(self, double value):
            self.capacitanceInv = 1.0 / value

    def __init__(self, circuit, cap):
        super(Node, self).__init__(circuit, props = [
                (u'voltage', 0.0, 1.0) ])
        self.capacitance = cap


cdef class FixedNode(Node):
    """For compatibility with other components, exposes a "voltage" parameter
    that is constant."""
    def __init__(self, circuit, voltage):
        super(FixedNode, self).__init__(circuit, 1.0)
        # Infinite capacitance
        self.capacitanceInv = 0.0
        self.providesPower = True
        self.setValue(u"voltage", voltage)


cdef class Resistor(OdeComponent):
    cdef public double conductance
    cdef Node n1, n2
    cdef int n1VoltageIndex, n2VoltageIndex

    property resistance:
        def __get__(self):
            return 1.0 / self.conductance
        def __set__(self, double value):
            self.conductance = 1.0 / value

    def __init__(self, circuit, Node n1, Node n2, double ohms):
        super(Resistor, self).__init__(circuit)
        self.resistance = ohms

        self.n1 = n1
        self.n2 = n2
        self.n1VoltageIndex = n1.getIndex(u"voltage")
        self.n2VoltageIndex = n2.getIndex(u"voltage")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.profile(False)
    cdef void updateDerivatives(self, double[:] vals, double[:] derivs):
        cdef double vdiff = (vals[self.n1VoltageIndex]
                - vals[self.n2VoltageIndex])
        cdef double current = vdiff * self.conductance
        if vdiff > 0:
            if self.n1.providesPower:
                self.circuit._stepPower += vals[self.n1VoltageIndex] * current
        else:
            if self.n2.providesPower:
                self.circuit._stepPower -= vals[self.n2VoltageIndex] * current
        derivs[self.n1VoltageIndex] -= current * self.n1.capacitanceInv
        derivs[self.n2VoltageIndex] += current * self.n2.capacitanceInv


cdef double Memristor_rpConst[8]
Memristor_rpConst[0] = 1.03e-5 # alpha
Memristor_rpConst[1] = 0.515 # beta
Memristor_rpConst[2] = 2.05e-5 # gamma
Memristor_rpConst[3] = 1.03 # delta
Memristor_rpConst[4] = 0.4 # lambda
Memristor_rpConst[5] = 2.06e-7 # eta1
Memristor_rpConst[6] = 18.54 # eta2
Memristor_rpConst[7] = 100 # tau (relaxation time for memristor)
# As per patrick's e-mail, w-update params go up by 3%
#for i in range(4, 7):
#    Memristor_rpConst[i] *= 1.03

cdef class Memristor(OdeComponent):
    cdef Node n1, n2
    cdef int n1VoltageIndex, n2VoltageIndex
    cdef int stateIndex, currentIndex, voltIndex

    property state:
        """Internal state variable, range is specific to model."""
        def __get__(self):
            return self.circuit.values[self.stateIndex]
        def __set__(self, double value):
            self.circuit.values[self.stateIndex] = value


    def __init__(self, circuit, Node n1, Node n2, double initialState):
        super(Memristor, self).__init__(circuit, props = [
                (u'memristorState', initialState, 1.0),
                (u'lastCurrent', 0.0, 1.0),
                (u'lastVoltage', 0.0, 1.0) ])
        self.n1 = n1
        self.n2 = n2
        self.n1VoltageIndex = n1.getIndex(u"voltage")
        self.n2VoltageIndex = n2.getIndex(u"voltage")
        self.stateIndex = self.getIndex(u"memristorState")
        # Debug attributes
        self.currentIndex = self.getIndex(u"lastCurrent")
        self.voltIndex = self.getIndex(u"lastVoltage")


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.profile(False)
    cdef void updateDerivatives(self, double[:] vals, double[:] derivs):
        cdef double v = (vals[self.n1VoltageIndex]
                - vals[self.n2VoltageIndex])
        cdef double dw, w = max(0.0, min(1.0, vals[self.stateIndex]))
        cdef double current

        # Apply clamping to current state before continuing
        vals[self.stateIndex] = w

        current = ((1-w)*Memristor_rpConst[0]*(1-exp(-Memristor_rpConst[1]*v))
                     +w*Memristor_rpConst[2]*sinh(Memristor_rpConst[3] * v))

        c1 = Memristor_rpConst[4] * (Memristor_rpConst[5] * sinh(Memristor_rpConst[6] * v))
        if v > 0:
            window = (1 - exp((w - 1)/2))
        else:
            window = (1 - exp(-w / 6))
        dw = window * c1

        if v > 0:
            if self.n1.providesPower:
                self.circuit._stepPower += vals[self.n1VoltageIndex] * current
        else:
            if self.n2.providesPower:
                self.circuit._stepPower -= vals[self.n2VoltageIndex] * current
        derivs[self.n1VoltageIndex] -= current * self.n1.capacitanceInv
        derivs[self.n2VoltageIndex] += current * self.n2.capacitanceInv
        derivs[self.stateIndex] += dw
        vals[self.currentIndex] = current
        vals[self.voltIndex] = v
