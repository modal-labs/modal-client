# Copyright Modal Labs 2022
import modal

app = modal.App()

# This is in module scope, so will show what the `description`
# value is at import time, which may be different if some code
# changes the `description` post-import.
print(f"app.description: {app.description}")


@app.function()
def foo():
    pass
