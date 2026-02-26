# Copyright Modal Labs 2025
import modal

app = modal.App()


@app.local_entrypoint()
def main():
    raise ValueError("This is an error message that should be visible")
