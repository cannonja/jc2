
from mr.optimize.maximize import Maximize

import numpy as np
from mr.test.common import MrTestBase

class Problem(object):
    def __init__(self, ndims=4, nwellsPerDim=4):
        self.ndims = ndims
        self._wells = []

        class Well(object):
            def __init__(self):
                self.loc = np.random.uniform(size=(ndims,))
                self.maximum = np.random.uniform()
                self.falloff = 0.05 * np.random.uniform()

        self.maximum = -1.
        for _ in range(nwellsPerDim**ndims):
            w = Well()
            self._wells.append(w)
            if w.maximum > self.maximum:
                self.maximum = w.maximum
                self.maximumLoc = w.loc


    def calculate(self, loc):
        val = 0.
        for w in self._wells:
            dist = ((w.loc - loc) ** 2).sum() ** 0.5
            val = max(val, w.maximum * (0.5 ** (dist / w.falloff)))
        return val



class MaximizeTest(MrTestBase):
    def test_max(self):
        ndims = 4
        dimRanges = [ ('d{}'.format(i), (0., 1.)) for i in range(ndims) ]
        p = Problem(ndims)

        def pip(w):
            @w.job
            def getScore(s):
                parms = []
                for i in range(ndims):
                    parms.append(s['d{}'.format(i)])
                return (s, p.calculate(parms), 0.)

        m = Maximize(4)
        print("Real max: {} at {}".format(p.maximum, p.maximumLoc))
        m.maximizeRbf(dimRanges, pip, solveTime=1.0)


