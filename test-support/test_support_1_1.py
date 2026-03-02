import typing

import modal

app = modal.App("test-support-1-1")


@app.function(min_containers=1)
def identity_with_repr(s: typing.Any) -> typing.Any:
    return s, repr(s)
