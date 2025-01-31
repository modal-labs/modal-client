# Copyright Modal Labs 2025
import modal

app = modal.App()


@app.function(gpu="NOT_A_GPU")
def f():
    pass
