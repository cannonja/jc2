
from .common import UnsupervisedTestBase

from mr.unsupervised import Lca

class LcaTest(UnsupervisedTestBase):
    def getDefaultInstance(self):
        return Lca(3)

