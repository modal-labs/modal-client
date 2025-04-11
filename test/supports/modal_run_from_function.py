# Copyright Modal Labs 2025
import modal

app = modal.App("app1")

app2 = modal.App("app2")


@app2.function()
def foo():
    pass


@app.function(image=modal.Image.debian_slim().pip_install("rich"))
def run_other_app():
    with modal.enable_output():
        with app2.run():
            foo.remote()
