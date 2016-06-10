
cdef class AdaDelta:
    cdef public int w, h
    cdef public double rho, initial, epsilon, momentum
    cdef public double[:, :] _edX, _eG2, _last

    cpdef convergerProps(self)
    cpdef double getDelta(self, int i, int j, double gradient) except? -1
