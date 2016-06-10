
from mr.test.common import MrTestBase

class UnsupervisedTestBase(MrTestBase):
    def getDefaultInstance(self):
        """Returns a default instance of the test for standard tests"""
        raise NotImplementedError()


    def test_repeatability(self):
        # Ensure that predict() gives the same results over and over and over
        inst = self.getDefaultInstance()

        data = [ [ 1, 0, 0 ], [ 0, 1, 0 ], [ 1, 0.5, 1 ] ]
        test = [ [ 1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ] ]
        inst.fit(data)
        first = inst.predict(test)
        for _ in range(10):
            n = inst.predict(test)
            self.assertDeepEqual(first, n)

