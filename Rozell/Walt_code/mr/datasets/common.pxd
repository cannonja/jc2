
from cython cimport view

cimport numpy
ctypedef numpy.float_t float_t
ctypedef numpy.uint8_t uint8_t
ctypedef numpy.uint32_t uint32_t

from libc.stdio cimport FILE

cdef extern from "stdio.h":
    FILE* fopen(const char*, const char*)
    int fclose(FILE*)
    size_t fread(void*, size_t, size_t, FILE*)


cdef class CFile:
    cdef const char* fname
    cdef FILE* ptr
    cdef uint8_t[::view.contiguous] temp

    # __enter__ MUST be cpdef, __exit__ can't be...
    cpdef CFile __enter__(self)

    # close / open not recommended (use __enter__ and __exit__)
    cpdef close(self)
    cpdef open(self, const char* name = ?)
    cpdef tuple read(self, int nbytes = ?)
    cpdef uint8_t readUint8(self) except? 255
    cpdef uint32_t readUint32(self) except? 0x80818080
    cdef uint8_t safeGet(self) except? 255


cdef class ImageSet:
    # --- Client interface ---
    cpdef grayscale(self)
    cpdef split(self, int nTest, noLabels = ?, shuffle = ?)
    cpdef splitPatches(self, int nTest, int patchW, int patchH, int stride,
            noLabels = ?, shuffle = ?)

    # Array of (# images, image data).  Class has public .images getter
    cdef double[:, :] _images

    # Array of (# images, # labels).  Second dimension is 1 if the image matches
    # the label, and 0 otherwise.
    cdef double[:, :] _labels

    cdef int _imWidth, _imHeight, _imChannels
    cdef int _nextIndex

    # Returns the next (_images, _labels) slot.  Both are of type double[:].
    cpdef tuple _getNext(self)

    # Prepare _images and _labels for the given number of images / labels.
    # nLabels may be zero for an unsupervised-only dataset.
    cpdef _reset(self, int nImages, int w, int h, int c, int nLabels)
