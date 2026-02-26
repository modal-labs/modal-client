# Copyright Modal Labs 2025
import modal

app = modal.App()


@app.local_entrypoint()
def some_main_entrypoint():
    print("main entrypoint")


def main_func():
    print("main func")
