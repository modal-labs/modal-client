# Copyright Modal Labs 2022
import sys


def is_notebook(stdout=None):
    ipykernel_iostream = sys.modules.get("ipykernel.iostream")
    if ipykernel_iostream is None:
        return False
    if stdout is None:
        stdout = sys.stdout
    return isinstance(stdout, ipykernel_iostream.OutStream)
