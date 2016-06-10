
import numpy
import unittest

class MrTestBase(unittest.TestCase):
    """Provides shared utilities (assertDeepEqual), and skips tests whose
    class names end in "Base".
    """

    @classmethod
    def setUpClass(cls):
        if cls.__name__.endswith("Base"):
            raise unittest.SkipTest("Is base class for tests, skipping")
        super(MrTestBase, cls).setUpClass()


    def assertDeepEqual(self, expected, actual):
        compares = [ (['value'], expected, actual) ]
        errors = []

        while compares:
            path, a, b = compares.pop()

            def addError(e):
                """Add an error to be reported.  If the error is terminal, call
                continue afterward."""
                errors.append("Expected differs from actual at {}: {}\n{}\n\n{}"
                        .format(path, e, a, b))

            if type(a) != type(b):
                addError("Type differs: {} / {}".format(type(a), type(b)))
                continue

            if isinstance(a, dict):
                for k, v in a.iteritems():
                    if k not in b:
                        addError("Actual does not contain {}".format(k))
                    else:
                        compares.append((path + [ "[", repr(k), "]" ], v, b[k]))
                for k, v in b.iteritems():
                    if k not in a:
                        addError("Actual contains extra key {}".format(k))
            elif isinstance(a, list):
                if len(a) != len(b):
                    addError("Lists have different length: {} vs {}".format(
                            len(a), len(b)))
                for i in range(min(len(a), len(b))):
                    compares.append((path + [ "[", repr(i), "]" ], a[i], b[i]))
            elif isinstance(a, numpy.ndarray):
                # Dimensionality (number of axes)
                if len(a.shape) != len(b.shape):
                    addError("Different dimensionality: {} vs {}".format(
                            len(a.shape), len(b.shape)))

                # Axis shape comparison
                for axis in range(min(len(a.shape), len(b.shape))):
                    if a.shape[axis] != b.shape[axis]:
                        addError("Axis {} has different sizes ({} vs {})"
                                .format(axis, a.shape[axis], b.shape[axis]))

                # Value compare
                if not numpy.allclose(a, b):
                    addError("Values do not match according to numpy.allclose")
            elif hasattr(a, '__eq__') or isinstance(a, (int, float)):
                if a != b:
                    addError("Expected != actual")
            else:
                addError("Don't know how to compare type {}, and it doesn't "
                        "have an __eq__ method".format(type(a)))

        if errors:
            self.fail('\n\n'.join([ "=== Failure #{} ===\n{}".format(i, e)
                    for i, e in enumerate(errors) ]))
