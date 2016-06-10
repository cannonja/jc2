
from mr.electrical import SpiceNetwork

import numpy as np
import os
from unittest import TestCase

class TestSpiceNetwork(TestCase):
    """Test cases base class.  Will be skipped, deriving classes should
    override _getNetwork."""

    def _getNetwork(self):
        self.skipTest("Base class TestSpiceNetwork being skipped")


    def assertTreeEqual(self, a, b, path = None, problems = None):
        """Test two equivalences, using a tree operation.

        The left side, a, is assumed to be the expected.  It may have meta
        elements, including:

            { '$$almost': 5 } - Used for floating point.  A tolerance of 1% is
                    assumed.
            { '$$almost': (5, 0.1) } - Used for floating point.  The second
                    member of the tuple is the tolerance (flat, not percentage
                    based).
            { '$$almost': [1,2,3,...] } - Used for floating point.  Tests that
                    a list of numbers match the tolerance.  Useful if they all
                    share a tolerance.
        """
        ptr = 'Mismatch: '
        if path is not None:
            ptr = 'Mismatch at {}: '.format(''.join(path))
        else:
            path = ()

        probs = problems
        if probs is None:
            probs = []
        def add(prob):
            probs.append("{} {}".format(ptr, prob))
        def next(na, nb, pathChange):
            self.assertTreeEqual(na, nb, path + pathChange, probs)

        if isinstance(a, dict):
            isNormal = True
            keys = a.keys()[:2]
            if len(keys) == 1 and keys[0].startswith('$$'):
                isNormal = False
                k = keys[0][2:]
                v = a[keys[0]]
                if k == 'almost':
                    if isinstance(v, tuple):
                        val, tol = v
                    else:
                        val = v
                        ptol = 0.01
                        tol = None

                    if isinstance(val, (tuple, list)):
                        for i, v in enumerate(val):
                            vtol = tol
                            if vtol is None:
                                vtol = abs(v * ptol)
                            next({ '$$almost': (v, vtol) }, b[i],
                                    ('[', str(i), ']'))
                    else:
                        vtol = tol
                        if vtol is None:
                            vtol = abs(val * ptol)
                        if val + vtol < b:
                            add("{} out of tolerance ({} +- {})".format(b,
                                    val, tol))
                        elif val - vtol > b:
                            add("{} out of tolerance ({} +- {})".format(b,
                                    val, tol))
                else:
                    isNormal = True

            if isNormal and not isinstance(b, dict):
                add("{} is not a dict".format(b))
            elif isNormal:
                ak = set(a.keys())
                bk = set(b.keys())
                kk = ak.difference(bk)
                if kk:
                    add("Missing keys: {}".format(', '.join(kk)))
                kk = bk.difference(ak)
                if kk:
                    add("Extra keys: {}".format(', '.join(kk)))

                for k in ak.intersection(bk):
                    next(a[k], b[k], ('[', repr(k), ']'))
        elif isinstance(a, (list, tuple)):
            if not isinstance(b, type(a)):
                add("{} is not a list".format(b))
            else:
                for i, na in enumerate(a):
                    if i >= len(b):
                        add("List missing element {}: {}".format(i, b))
                    else:
                        next(na, b[i], ('[', repr(i), ']'))
                if len(a) < len(b):
                    add("List has extra element {}: {}".format(b[len(a)], b))
        elif a != b:
            add("{} != {}".format(a, b))

        if problems is None and probs:
            # Final execution, raise anything that was wrong
            raise ValueError("{} errors: {}".format(len(probs),
                    ''.join([ '\n{}.'.format(p) for p in probs ])))


    def test_assertTreeEqual(self):
        if self.__class__.__name__ != 'TestSpiceNetwork':
            self.skipTest("Base class only")
        self.assertTreeEqual([], [])
        self.assertTreeEqual(5, 5)
        self.assertTreeEqual('8', '8')
        self.assertTreeEqual((), ())
        self.assertTreeEqual({}, {})
        self.assertTreeEqual([1,2,3], [1,2,3])
        self.assertTreeEqual((1,2,3), (1,2,3))
        with self.assertRaises(ValueError):
            self.assertTreeEqual((1,2,3), [1,2,3])
        with self.assertRaises(ValueError):
            self.assertTreeEqual([1,2,3], (1,2,3))
        with self.assertRaises(ValueError):
            self.assertTreeEqual([1,2,3], [1,2])
        with self.assertRaises(ValueError):
            self.assertTreeEqual([1,2], [1,2,3])
        self.assertTreeEqual({ 'a': 1 }, { 'a': 1 })
        self.assertTreeEqual({ '$$a': 2 }, { '$$a': 2 })
        self.assertTreeEqual({ 'a': 8, 'b': 9 }, { 'b': 9, 'a': 8 })
        with self.assertRaises(ValueError):
            self.assertTreeEqual({ 'a': 8, 'b': 9 }, { 'a': 8, 'b': 7 })
        with self.assertRaises(ValueError):
            self.assertTreeEqual({ 'a': 8, 'b': 9 }, { 'a': 8 })
        with self.assertRaises(ValueError):
            self.assertTreeEqual({ 'a': 8 }, { 'a': 8, 'b': 9 })


    def test_avg(self):
        sn = self._getNetwork()
        r = sn.simulate(
                [ SpiceNetwork.Measure("x",
                    method = SpiceNetwork.MeasureMethod.AVERAGE) ],
                1.0, 11,
                """Spice!
Vsrc plus 0 DC 5V
Rcharge plus x 1
Cx x 0 1 IC=0.0
""")
        self.assertTreeEqual([ { '$$almost': (1.80, 0.2) } ], r)


    def test_circuit_basic(self):
        sn = self._getNetwork()
        c = sn.Circuit(1.)
        c.add("Va a 0 PWL(0. 4. 1. 0.)")
        c.add("Rab a b 10")
        c.add("Rb0 b 0 10")
        c.draw("/tmp/circuit.pdf")
        r = c.run([ 'b' ])
        self.assertTreeEqual(
                { 'b': [
                    { '$$almost': 2.0 },
                    { '$$almost': 1.0 },
                    { '$$almost': 0.0 }, ]},
                r)


    def test_power(self):
        sn = self._getNetwork()
        r = sn.simulate(
                [ SpiceNetwork.Measure("Vpwr,vcc", SpiceNetwork.MeasureType.POWER) ],
                1.0, 11,
                """Spice!
Vpwr vcc 0 PWL(0. 0. 1. 1.)
Rdrain vcc 0 4""")
        # Tolerance is so high because Xyce is really inaccurate with this
        # circuit for whatever reason
        self.assertTreeEqual([ { '$$almost': (
                # Pwr = V^2/R
                # 0/4, 0.1^2 / 4, ...
                (-np.linspace(0., 1., 11) ** 2 / 4).tolist(), 0.032) } ], r)


    def test_sample(self):
        sn = self._getNetwork()
        r = sn.simulate(
                [ SpiceNetwork.Measure("x") ],
                1.0, 11,
                """Spice!
Vsrc plus 0 DC 5V
Rcharge plus x 1
Cx x 0 1 IC=0.0
""")
        self.assertEqual(11, len(r[0]))
        self.assertTreeEqual([ { '$$almost': ([0.0, 0.305286, 0.86952, 1.12567,
                1.59131, 1.8027, 2.18697, 2.36142, 2.67854, 2.95754, 3.16118],
                0.2) }], r)



class TestSpiceNetwork_NgSpice(TestSpiceNetwork):
    """Implementation of test cases for ngspice"""
    def _getNetwork(self):
        return SpiceNetwork(os.path.expanduser("/usr/bin/ngspice"),
                SpiceNetwork.SpiceType.NGSPICE)



class TestSpiceNetwork_Xyce(TestSpiceNetwork):
    """Implementation of test cases for Xyce"""
    def _getNetwork(self):
        return SpiceNetwork(os.path.expanduser("~/bin/Xyce"),
                SpiceNetwork.SpiceType.XYCE)
