cdef class FastRandom:
    cpdef double get(self) except? -1.

    cdef public double[:] _randBuffer
    """Up-next random values"""
    cdef public int _randIndex
    """The current index into :attr:`_randBuffer`"""
    cdef public int _randLen
    """The length of :attr:`_randBuffer`"""

