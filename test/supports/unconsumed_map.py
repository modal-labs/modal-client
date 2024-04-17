# Copyright Modal Labs 2022
from .stub import app, f

if __name__ == "__main__":
    with app.run():
        f.map([1, 2, 3])  # type: ignore
