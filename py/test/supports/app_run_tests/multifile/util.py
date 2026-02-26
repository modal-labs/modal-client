# Copyright Modal Labs 2025
from .main import app, main_func


@app.local_entrypoint()
def run_this():
    print("ran util")
    main_func()
