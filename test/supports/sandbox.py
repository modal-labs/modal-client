# Copyright Modal Labs 2024
import modal

app = modal.App()


@app.function()
def spawn_sandbox(x):
    modal.Sandbox.create("bash", "-c", "echo bar")
