import sys


def is_notebook(stdout=None):
    if stdout is None:
        stdout = sys.stdout
    try:
        import ipykernel.iostream

        return isinstance(stdout, ipykernel.iostream.OutStream)
    except ImportError:
        return False
