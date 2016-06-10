
cimport numpy as np

cdef class OdeComponent:
    cdef OdeCircuit circuit
    cdef np.int64_t[:] indices
    cdef list indexNames
    # Set to non-zero for nodes whose current contributes to power.
    # TODO - Revamp this system so that node's current is tracked, and then
    # multiplied by conductance.  Or something.  This works for now, though.
    cdef int providesPower

    cdef int getIndex(self, unicode name) except -1
    cpdef double getValue(self, unicode name) except? -1
    cpdef setValue(self, unicode name, double value)
    cdef void updateDerivatives(self, double[:] values, double[:] derivatives)


cdef class OdeCircuit:
    cdef public object ground
    cdef public double stderrEveryXSeconds
    cdef public double dtMax, dtMin, simTime, lastDt, stableScale, energy

    cdef list _comps
    cdef int nValues, maxValues
    cdef list valueNames
    cdef double[:] values, derivatives, stableConsts
    cdef double minFlaggedDt
    cdef double _stepPower

    cdef int addValue(self, unicode name, double initialValue,
            double stableConst)
    cdef double advance(self, double dtMax = ?) except -1
    cpdef advanceRoundUp(self, roundInterval = ?)
    cpdef clearHistory(self, timeToKeep = ?)
    cdef void preAdvance(self, double lastDt) except *
    cpdef reset(self)
    cpdef simulate(self, double dt)
    cpdef updateDerivatives(self, double simTime, double[:] values)
