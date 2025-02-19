# Copyright Modal Labs 2023
import modal

app = modal.App("dummy", include_source=True)  # TODO: remove include_source=True)


def foo(i):
    return 1


foo_handle = app.function()(foo)


other_app = modal.App()


@other_app.function()
def bar(i):
    return 2
