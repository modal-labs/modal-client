# Copyright Modal Labs 2022
from typing import cast

import modal
from modal._utils.async_utils import synchronizer
from modal.app import _App

app = modal.App()
inner = cast(_App, synchronizer._translate_in(app))

# This is in module scope, so will show what the `description`
# value is at import time, which may be different if some code
# changes the `description` post-import.
print(f"app.description: {inner._description}")


@app.function()
def foo():
    pass
