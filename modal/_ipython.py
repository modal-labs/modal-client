# Copyright Modal Labs 2022
import sys
import warnings

ipy_outstream = None
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import ipykernel.iostream

    ipy_outstream = ipykernel.iostream.OutStream
except ImportError:
    pass


def is_notebook(stdout=None):
    if ipy_outstream is None:
        return False
    if stdout is None:
        stdout = sys.stdout
    return isinstance(stdout, ipy_outstream)
