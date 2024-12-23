# Copyright Modal Labs 2023
import modal

a = modal.App()


@a.function()
def a_func(i):
    assert a_func.is_hydrated
    assert not b_func.is_hydrated
    assert modal.App.container_app() == a


b = modal.App()


@b.function()
def b_func(i):
    assert b_func.is_hydrated
    assert not a_func.is_hydrated
    assert modal.App.container_app() == b
