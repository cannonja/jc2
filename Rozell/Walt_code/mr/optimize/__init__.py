"""Distributed optimization routines using the :mod:`job_stream` library.

Primary Classes
===============

.. autosummary::
    maximize.Maximize
    sweep.Sweep

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    mr.optimize.maximize
    mr.optimize.sweep

Members
=======
"""
from .maximize import Maximize
from .sweep import Sweep
