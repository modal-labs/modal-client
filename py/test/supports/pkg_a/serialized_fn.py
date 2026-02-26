# Copyright Modal Labs 2022
import modal

app = modal.App()


@app.function(serialized=True)
def f():
    pass
