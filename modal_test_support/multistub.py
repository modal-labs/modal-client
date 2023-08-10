# Copyright Modal Labs 2023
import modal

a = modal.Stub()


@a.function()
def a_func(i):
    assert a_func.is_hydrated()
    assert not b_func.is_hydrated()


b = modal.Stub()


@b.function()
def b_func(i):
    assert b_func.is_hydrated()
    assert not a_func.is_hydrated()
