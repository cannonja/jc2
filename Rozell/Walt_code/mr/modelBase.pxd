
cdef class SklearnModelBaseStats:
    cdef public int index

    # -------- Properties for all learning models --------
    # Number of active outputs per patch / total output activity squared per
    # patch.
    cdef public double[:] activityCount, activitySqr
    # Array of shape (nOutputs,), which tracks how many times each output was
    # active (outputCount) and its sum of squared activity (outputSqr).
    cdef public double[:] outputCount, outputSqr

    # -------- Properties specific to different learning models --------
    # Total energy consumed during computation
    cdef public double[:] energy
    # Time of computation (energy / simTime = avg power)
    cdef public double[:] simTime


cdef class SklearnModelBase:
    cdef public double[:] _bufferIn, _bufferOut
    cdef public int _isInit
    cdef public int nInputs, nOutputs
    # Internally set and used during functions that accept debug as a kwarg.
    cdef public int _debug

    # Variables learned throughout the fitting procedure
    cdef public int fitIters_
    cdef public double t_

    # When a function is called with debug = True as a kwarg, this is populated.
    cdef public SklearnModelBaseStats debugInfo_

    cpdef _init(self, int nInputs, int nOutputs)
    cpdef _partial_fit(self, double[:] x, double[:] y)
    cpdef _predict(self, double[:] x, double[:] y)
    cpdef _reconstruct(self, double[:] y, double[:] r)

    cpdef _resetDebug(self, debug, int lenX)
