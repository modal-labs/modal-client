# Copyright Modal Labs 2024
import c

import modal

app = modal.App()


@app.function(secrets=[modal.Secret.from_dict({"foo": "bar"})])
def b_func():
    pass


app.include(c.app)
