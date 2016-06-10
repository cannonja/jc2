r"""Provides classes and methods for dealing with simulations of electrical
components and circuits.

Usage
=====

Generally, you'll want to paste together a :class:`circuit.Circuit` object
using a :class:`spiceNetwork.SpiceNetwork`, asking it to measure certain
qualities:

.. code-block:: python

    from mr.electrical import SpiceNetwork, SubCircuit
    import os

    sn = SpiceNetwork(os.path.expanduser("~/bin/Xyce"),
            SpiceNetwork.SpiceType.XYCE)
    cir = sn.Circuit(1e-9, 101)  # Time of simulation, number of samples
    meas = {}

    cir.add('Vpwr vdd 0 1')
    cir.add('R1 vdd 1 500')
    cir.add('R2 1 0 1000')
    meas['1'] = cir.Measure('1')
    meas['pwr'] = cir.Measure('Vpwr,vdd', cir.MeasureType.POWER)
    r = cir.run(meas)
    print(r)  # r has 'time', 'pwr', and '1' indices, which are arrays of
              # values at the sample points.


Using Transistor Models
-----------------------

In the Main :class:`circuit.Circuit`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    trans45nm = SubCircuit.modelsFromFile('/path/to/p045_cmos_models_tt.inc')
    cir.add(trans45nm)
    cir.add('M1 1 i1 2 2 nmos L=45n W=45n')


In a :class:`circuit.SubCircuit`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :ref:`example-not-gate`.


Examples
========

.. _example-not-gate:

OR Gate Using NOT Gate as SubCircuit
------------------------------------

.. code-block:: python

    trans45nm = SubCircuit.modelsFromFile('/path/to/p045_cmos_models_tt.inc')
    scNot = SubCircuit("not_gate", [ 'vdd', 'vee', '1', '2' ],
            depends=[trans45nm])
    scNot.add('''
            Mpos 2 1 vdd vdd pmos L=45n W=45n
            Mneg 2 1 vee vee nmos L=45n W=45n
            ''')

    scOr = SubCircuit("or_gate", [ 'vdd', 'vee', '1', '2', '3' ],
            depends=[trans45nm])
    scOr.add('''
            M1 10 1 vdd vdd pmos L=45n W=45n
            M2 13 2 10 vdd pmos L=45n W=45n
            M3 13 1 vee vee nmos L=45n W=45n
            M4 13 2 vee vee nmos L=45n W=45n''')
    scOr.add(scNot([ 'vdd', 'vee', '13', '3' ]))



Submodules
==========

.. autosummary::
    :toctree: _autosummary

    circuit
    odeCircuit
    spiceLibrary
    spiceNetwork


Members
=======

"""

from circuit import SubCircuit
from spiceLibrary import SpiceLibrary
from spiceNetwork import SpiceNetwork

