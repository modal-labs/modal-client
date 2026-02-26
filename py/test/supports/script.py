# Copyright Modal Labs 2022
from .common import app, f

if __name__ == "__main__":
    with app.run():
        assert f.remote(2, 4) == 20  # type: ignore
