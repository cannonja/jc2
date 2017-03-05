
cimport numpy as np

import numpy as np
from skvideo.io import FFmpegWriter

class VideoHelper(object):
    def __init__(self, filename, frameSize, fps=30):
        self._buf = np.zeros((frameSize[1], frameSize[0], 3), dtype=np.uint8)
        self._vid = FFmpegWriter(filename, inputdict={
                '-r': str(fps),
                '-s': '{}x{}'.format(*frameSize),
        })


    def frame(self):
        self._buf[:, :, :] = 0.
        return _VideoFrame(self._vid, self._buf)


    def __enter__(self):
        return self


    def __exit__(self, typ, val, tb):
        self._vid.close()



cdef class _VideoFrame:
    cdef public np.uint8_t[:, :, ::1] _buf
    cdef public object _vid

    @property
    def blit_flat_float(self):
        """Used to copy a 1-d (flattened) array of doubles into our buffer.

        Usage:
            frame.blit_flat_float[x_range, y_range] = buffer matching ranges
        """
        return _VideoFrameBlitFlatFloat(self)


    @property
    def blit_flat_float_mono_as_alpha(self):
        """Used to copy a 1-d (flattened) array of doubles into our buffer,
        using a color to represent the monochromatic data.  Optionally, an
        alpha multiplier may be applied.

        Usage:
            frame.blit_flat_float_mono_as_alpha[x_range, y_range,
                    color, alpha_mult] = src_array

        Both ``color`` and ``alpha_mult`` are optional.
        """
        return _VideoFrameBlitFlatFloatMonoAlpha(self)


    def __init__(self, vid, buf):
        self._vid = vid
        self._buf = buf


    def __enter__(self):
        return self


    def __exit__(self, typ, val, tb):
        self._vid.writeFrame(self._buf)



cdef class _VideoFrameBlitFlatFloat:
    cdef public _VideoFrame _frame
    def __init__(self, videoFrame):
        self._frame = videoFrame
    def __setitem__(self, key, item):
        if len(key) != 2:
            raise ValueError("Requires 2 keys")

        cdef np.uint8_t[:, :, ::1] dst = self._frame._buf
        cdef double[::] src = item

        cdef int ws, we, wt, hs, he, ht, w, h, i
        ws, we, wt = _read_slice(key[0], dst.shape[1])
        hs, he, ht = _read_slice(key[1], dst.shape[0])

        i = 0
        for h in range(hs, he, ht):
            for w in range(ws, we, wt):
                dst[h, w, 0] = max(0, min(255, int(src[i] * 255)))
                dst[h, w, 1] = max(0, min(255, int(src[i+1] * 255)))
                dst[h, w, 2] = max(0, min(255, int(src[i+2] * 255)))
                i += 3



cdef class _VideoFrameBlitFlatFloatMonoAlpha:
    cdef public _VideoFrame _frame
    def __init__(self, frame):
        self._frame = frame
    def __setitem__(self, key, item):
        cdef double amult = 1., u, uu
        cdef np.uint8_t[:] acolor = np.asarray([255, 255, 255], dtype=np.uint8)

        if len(key) < 2 or len(key) > 4:
            raise ValueError("len(key) must be between 2 and 4")
        if len(key) >= 4:
            amult = key[3]
        if len(key) >= 3:
            acolor = np.asarray(key[2], dtype=np.uint8)

        amultu = 1. - amult
        cdef np.uint8_t[:, :, ::1] dst = self._frame._buf
        cdef double[::] src = item

        cdef int ws, we, wt, hs, he, ht, w, h, i
        ws, we, wt = _read_slice(key[0], dst.shape[1])
        hs, he, ht = _read_slice(key[1], dst.shape[0])
        i = 0
        for h in range(hs, he, ht):
            for w in range(ws, we, wt):
                u = max(0., min(1., src[i] * amult))
                uu = 1. - u
                dst[h, w, 0] = max(0, min(255, int(u * acolor[0]
                        + uu * dst[h, w, 0])))
                dst[h, w, 1] = max(0, min(255, int(u * acolor[1]
                        + uu * dst[h, w, 1])))
                dst[h, w, 2] = max(0, min(255, int(u * acolor[2]
                        + uu * dst[h, w, 2])))
                i += 1



cdef tuple _read_slice(key, int arrsize):
    cdef int s, e, t
    if isinstance(key, slice):
        if key.start is None:
            s = 0
        elif key.start < 0:
            s = arrsize + key.start
        else:
            s = key.start
        if key.stop is None:
            e = arrsize
        elif key.stop < 0:
            e = arrsize + key.stop
        else:
            e = key.stop
        t = key.step or 1
    elif isinstance(key, int):
        s = key
        e = key + 1
        t = 1
    else:
        raise ValueError("Not a slice or int! {}".format(key))
    return s, e, t

