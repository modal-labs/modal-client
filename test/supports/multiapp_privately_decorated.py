# Copyright Modal Labs 2023
import modal

app = modal.App()


def foo(i):
    return 1


foo_handle = app.function()(foo)  #  "privately" decorated, by not override the original function


other_app = modal.App()


@other_app.function()
def bar(i):
    return 2
