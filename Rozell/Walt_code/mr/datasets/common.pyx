
import numpy as np

cdef class CFile:
    def __init__(self, fname):
        self.fname = fname
        self.ptr = <FILE*>0
        self.temp = np.zeros(8, dtype = np.uint8)


    cpdef CFile __enter__(self):
        self.open()
        return self


    def __exit__(self, type, value, traceback):
        self.close()


    cpdef close(self):
        if self.ptr != <FILE*>0:
            fclose(self.ptr)
            self.ptr = <FILE*>0


    cpdef open(self, const char* name = NULL):
        if name == <const char*>0:
            name = self.fname
        self.ptr = fopen(name, "rb")
        if self.ptr == <FILE*>0:
            raise ValueError("Failed to open {}".format(name))


    cpdef tuple read(self, int nbytes = 1):
        """Follows python conventions - reads nbytes, returns them as members
        of a numpy array.  First element is 0 if EOF.

        Returns (nBytesRead, bytes)
        """
        if nbytes > self.temp.shape[0]:
            self.temp = np.zeros((nbytes,), dtype = np.uint8)
        cdef size_t nread = fread(&self.temp[0], 1, nbytes, self.ptr)
        return (nread, self.temp)


    cpdef uint8_t readUint8(self) except? 255:
        return self.safeGet()


    cpdef uint32_t readUint32(self) except? 0x80818080:
        cdef int _
        cdef uint32_t w = 0x0
        for _ in range(4):
            w = (w << 8) + <uint32_t>self.safeGet()
        return w


    cdef uint8_t safeGet(self) except? 255:
        n, d = self.read(1)
        if n == 0:
            raise ValueError("EOF reached")
        return <uint8_t>d[0]


cdef class ImageSet:
    """A set of images"""
    property images:
        def __get__(self):
            return np.asarray(self._images)

    property imChannels:
        def __get__(self):
            return self._imChannels

    property imHeight:
        def __get__(self):
            return self._imHeight

    property imWidth:
        def __get__(self):
            return self._imWidth

    property labels:
        def __get__(self):
            return np.asarray(self._labels)

    property nLabels:
        def __get__(self):
            return self._labels.shape[1] if self._labels is not None else 0

    property nImages:
        def __get__(self):
            return self._images.shape[0]


    cpdef grayscale(self):
        """Returns a single-channel version of this ImageSet (can be self)."""
        cdef ImageSet result
        cdef double[:] image, label
        cdef double r, g, b
        cdef int i, j, k, tmp
        cdef int iw = self.imWidth

        if self.imChannels == 1:
            result = self
        elif self.imChannels == 3:
            result = ImageSet()
            result._reset(self.nImages, self.imWidth, self.imHeight, 1,
                    self.nLabels)
            for k in range(self.nImages):
                image, label = result._getNext()

                for i in range(self.imWidth):
                    for j in range(self.imHeight):
                        tmp = 3 * (iw * j + i)
                        r = self._images[k, tmp]
                        g = self._images[k, tmp+1]
                        b = self._images[k, tmp+2]
                        image[iw * j + i] = 0.2125 * r + 0.7152 * g + 0.0722 * b

                if self._labels is not None:
                    label[:] = self._labels[k, :]
        else:
            raise NotImplementedError(self.imChannels)

        return result


    cpdef split(self, int nTest, noLabels = False, shuffle = False):
        """Splits the loaded images into to categories: (nImages - nTest,
        nTest).

        Returns (trainArgs, testArgs, visualParams).  Usage like
        model.fit(*trainArgs) followed by model.score(*testArgs) and
        model.visualize(visualParams)"""
        return self.splitPatches(nTest, self.imWidth, self.imHeight, 0,
                noLabels = noLabels, shuffle = shuffle)


    cpdef splitPatches(self, int nTest, int patchW, int patchH, int stride,
            noLabels = False, shuffle = False):
        """Lazy splitting of patches.  Optionally, shuffle the patches.

        Returns (trainArgs, testArgs, visualParams), just like split()."""
        train = []
        test = []
        visualParams = (patchW, patchH, self.imChannels)

        if not noLabels:
            if self._labels is None:
                raise ValueError("Cannot split with labels, no labels in "
                        "dataset")

        # Split patches before shuffling, so that the test images are also
        # randomized
        splitter = _PatchSplitter(self, 0, self.nImages, patchW, patchH,
                stride)
        cdef int nSplit = len(splitter) - nTest * splitter.nPer
        patches = splitter.patches
        if shuffle:
            patches = patches.shuffle()
        train.append(patches[:nSplit])
        test.append(patches[nSplit:])

        if not noLabels:
            labels = splitter.labels
            if shuffle:
                # Shuffle uses a constant seed, so shuffling separate from
                # patches is OK.
                labels = labels.shuffle()
            train.append(labels[:nSplit])
            test.append(labels[nSplit:])

        return (tuple(train), tuple(test), visualParams)


    cpdef tuple _getNext(self):
        cdef double[:] r1, r2
        r1 = self._images[self._nextIndex, :]
        if self._labels is not None:
            r2 = self._labels[self._nextIndex, :]
        else:
            r2 = None
        self._nextIndex += 1
        return (r1, r2)


    cpdef _reset(self, int nImages, int w, int h, int c, int nLabels):
        self._imWidth = w
        self._imHeight = h
        self._imChannels = c

        cdef int inputsEach = w*h*c
        self._images = np.zeros((nImages, inputsEach), dtype = float)
        if nLabels > 0:
            self._labels = np.zeros((nImages, nLabels), dtype = float)
        else:
            self._labels = None

        self._nextIndex = 0


class _PatchSplitter(object):
    """Helper class that splits some images (and optionally labels) into a
    patches in a format that SklearnModelBase accepts."""

    @property
    def labels(self):
        return _PatchLabel(self)

    @property
    def nPer(self):
        """Returns the number of patches per source image"""
        return self._nPer

    @property
    def patches(self):
        return _PatchGet(self)

    def __len__(self):
        return (self._idxMax - self._idxMin) * self._nPer

    def __init__(self, ImageSet imageSet, int idxMin, int idxMax, int patchW,
            int patchH, int stride):
        self._set = imageSet
        self._images = self._set.images
        self._idxMin = idxMin
        self._idxMax = idxMax
        self._patchW = patchW
        self._patchH = patchH
        self._stride = stride
        if self._stride == 0:
            self._stride = max(patchW, patchH)

        self._wPer = 1 + (self._set.imWidth - patchW) // self._stride
        self._hPer = 1 + (self._set.imHeight - patchH) // self._stride
        self._nPer = self._wPer * self._hPer


class _ArbitrarySlice(object):
    """Arbitrary slices on an object with __getitem__ and __len__"""

    def __init__(self, gettable, sl):
        if not isinstance(sl, slice):
            raise ValueError("sl must be a slice!  Was {}".format(sl))

        self._s = gettable
        self._inds = sl.indices(len(self._s))
        self._len = (self._inds[1] - self._inds[0]) // self._inds[2]


    def __getitem__(self, i):
        if isinstance(i, slice):
            return _ArbitrarySlice(self, i)
        elif not isinstance(i, int):
            raise TypeError("Bad index: {}".format(i))

        if i >= self._len:
            raise IndexError("{}: {} >= {}".format(self, i, self._len))
        elif i < 0:
            if i < -self._len:
                raise IndexError("{}: {} < -{}".format(self, i, self._len))
            i += self._len

        return self._s[self._inds[0] + i * self._inds[2]]


    def __len__(self):
        return self._len


    def __str__(self):
        return "{}[{}:{}:{}]".format(self._s, *self._inds)


class _RandomSlice(object):
    """Randomizes patches.  Created by calling .shuffle() on a patch thing."""
    def __init__(self, gettable):
        self._s = gettable
        self._len = len(gettable)
        self._inds = np.arange(self._len)
        rs = np.random.RandomState(32)
        rs.shuffle(self._inds)


    def __getitem__(self, i):
        if isinstance(i, slice):
            return _ArbitrarySlice(self, i)
        elif not isinstance(i, int):
            raise TypeError("Bad index: {}".format(i))

        if i >= self._len:
            raise IndexError("{}: {} >= {}".format(self, i, self._len))
        elif i < 0:
            if i < -self._len:
                raise IndexError("{}: {} < -{}".format(self, i, self._len))
            i += self._len

        return self._s[self._inds[i]]


    def __len__(self):
        return self._len


    def __str__(self):
        return "{}[<shuffled>]".format(self._s)


    def mean(self):
        return self._s.mean()


class _PatchGet(object):
    def __init__(self, splitter):
        self.s = splitter
        self.b = np.zeros((self.s._patchW * self.s._patchH
                * self.s._set.imChannels,), dtype = float)


    def __len__(self):
        return len(self.s)


    def __getitem__(self, i):
        if isinstance(i, slice):
            return _ArbitrarySlice(self, i)
        elif not isinstance(i, int):
            raise TypeError("Bad index {}".format(i))

        cdef int ii = i, nPer = self.s._nPer
        if ii >= len(self):
            raise IndexError("Not that many patches")

        cdef int iw = self.s._set.imWidth, ih = self.s._set.imHeight
        cdef int ic = self.s._set.imChannels
        cdef int pw = self.s._patchW, ph = self.s._patchH
        cdef int image = ii // nPer
        cdef double[:] imageP = self.s._images[image, :]
        cdef int patch = ii - image * nPer
        cdef int patchL = (patch % self.s._wPer) * self.s._stride
        cdef int patchT = (patch // self.s._wPer) * self.s._stride

        cdef int j, k, c
        for j in range(pw):
            for k in range(ph):
                for c in range(ic):
                    self.b[ic * (pw * k + j) + c] = imageP[
                            ic * (iw * (patchT + k) + patchL + j) + c]

        return np.asarray(self.b)


    def mean(self):
        """Returns the mean of this dataset."""
        return self.s._images.mean()


    def shuffle(self):
        return _RandomSlice(self)


class _PatchLabel(object):
    def __init__(self, splitter):
        self.s = splitter


    def __len__(self):
        return len(self.s)


    def __getitem__(self, i):
        if isinstance(i, slice):
            return _ArbitrarySlice(self, i)
        elif not isinstance(i, int):
            raise TypeError("Bad index {}".format(i))
        elif i >= len(self.s):
            raise IndexError("No such patch")

        cdef int image = i // self.s._nPer
        return np.asarray(self.s._set.labels[image, :])


    def shuffle(self):
        return _RandomSlice(self)
