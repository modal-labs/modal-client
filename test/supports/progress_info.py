# Copyright Modal Labs 2022
from modal._output import OutputManager

from .common import app, f

if __name__ == "__main__":
    with OutputManager.enable_output():  # TODO(erikbern): turn this into modal.enable_output()
        with app.run():
            assert f.remote(2, 4) == 20  # type: ignore
