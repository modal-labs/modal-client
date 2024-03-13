# Copyright Modal Labs 2022
from .stub import f, stub

if __name__ == "__main__":
    with stub.run(show_progress=True):
        assert f.remote(2, 4) == 20  # type: ignore
