"""Classes for dealing with different datasets, both common and trivial examples.

Main classes:

.. autosummary::
    :toctree: _autosummary

    cifar10.Cifar10Dataset
    cifar100.Cifar100Dataset
    imageDataset.ImageDataset
    mnist.MnistDataset
    scanline.ScanlineDataset
"""

from .cifar10 import Cifar10Dataset
from .cifar100 import Cifar100Dataset
from .imageDataset import ImageDataset
from .mnist import MnistDataset
from .scanline import ScanlineDataset

