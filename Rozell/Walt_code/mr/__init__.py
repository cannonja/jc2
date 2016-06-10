"""See ../README.md for more information.

Facilities provided:

.. autosummary::
    :toctree: _autosummary

    mr.datasets
    mr.electrical
    mr.figureMaker
    mr.optimize
    mr.supervised
    mr.unsupervised

"""

import numpy as np
import os
import socket

# Allow .pyx files to be seamlessly integrated via cython/pyximport with
# default compiler directives.
import functools
import pyximport.pyximport

# Hack pyximport to have default options for profiling and embedding signatures
# in docstrings.
# Anytime pyximport needs to build a file, it ends up calling
# pyximport.pyximport.get_distutils_extension.  This function returns an object
# which has a cython_directives attribute that may be set to a dictionary of
# compiler directives for cython.
_old_get_distutils_extension = pyximport.pyximport.get_distutils_extension
@functools.wraps(_old_get_distutils_extension)
def _get_distutils_extension_new(*args, **kwargs):
    extension_mod, setup_args = _old_get_distutils_extension(*args, **kwargs)

    if not hasattr(extension_mod, 'cython_directives'):
        extension_mod.cython_directives = {}
    extension_mod.cython_directives.setdefault('embedsignature', True)
    extension_mod.cython_directives.setdefault('profile', True)
    return extension_mod, setup_args
pyximport.pyximport.get_distutils_extension = _get_distutils_extension_new

# Finally, install pyximport so that each machine has its own build directory
# (prevents errors with OpenMPI)
pyximport.install(build_dir = os.path.expanduser(
        '~/.pyxbld/{}'.format(socket.gethostname())),
        setup_args={'include_dirs':[np.get_include()]})


from .figureMaker import FigureMaker

