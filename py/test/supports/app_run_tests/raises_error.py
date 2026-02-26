# Copyright Modal Labs 2024
import modal

app = modal.App()


@app.function(gpu="broken:gpu:string")
def f():
    pass
