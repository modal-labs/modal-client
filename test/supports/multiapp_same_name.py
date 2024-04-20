# Copyright Modal Labs 2023
import modal

app = modal.App("dummy")


def foo(i):
    return 1


foo_handle = app.function()(foo)


other_app = modal.App("dummy")


@other_app.function()
def bar(i):
    return 2
