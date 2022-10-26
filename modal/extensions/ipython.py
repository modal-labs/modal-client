# Copyright Modal Labs 2022
import atexit
import logging
import sys
from typing import Any

import cloudpickle
from IPython import get_ipython
from IPython.core.magic import register_cell_magic, needs_local_scope

from modal import Stub
from modal.config import config, logger
from modal_utils.async_utils import run_coro_blocking

app_ctx: Any


try:
    get_ipython()
except:
    pass
else:
    @register_cell_magic
    @needs_local_scope
    def modal(line, cell, local_ns):
        "experimental modal magic"
        code = "\n".join(cell.split("\n")[1:])
        import modal
        stub = modal.Stub()

        def pickle_picklable(ns):
            ret = {}
            for k, v in ns.items():
                if k.startswith("_") or k in ("In", "Out"): continue
                try:
                    s = cloudpickle.dumps(v)
                    ret[k] = s
                except Exception:
                    pass
            return ret


        def unpickle_unpicklable(ns):
            ret = {}
            for k, v in ns.items():
                if k.startswith("_") or k in ("In", "Out"): continue
                try:
                    s = cloudpickle.loads(v)
                    ret[k] = s
                except Exception:
                    pass
            return ret

        @stub.function(serialized=True)
        def run_code(code, cell_locals_pickled):
            cell_locals = unpickle_unpicklable(cell_locals_pickled)
            exec(code, globals(), cell_locals)
            return pickle_picklable(cell_locals)

        with stub.run(show_progress=True):
            pickled_results = run_code(code, pickle_picklable(local_ns))

        res = unpickle_unpicklable(pickled_results)
        local_ns.update(res)


    def load_ipython_extension(ipython):
        global app_ctx

        # Set logger for notebook sys.stdout
        logger.addHandler(logging.StreamHandler(stream=sys.stdout))
        logger.setLevel(config["loglevel"])

        # Create a app and provide it in the IPython app
        stub = Stub(blocking_late_creation_ok=True)
        ipython.push({"stub": stub})

        app_ctx = stub.run()

        # Notebooks have an event loop present, but we want this function
        # to be blocking. This is fairly hacky.
        run_coro_blocking(app_ctx.__aenter__())

        def exit_app():
            print("Exiting modal app")
            run_coro_blocking(app_ctx.__aexit__(None, None, None))

        atexit.register(exit_app)


def unload_ipython_extension(ipython):
    global app_ctx

    run_coro_blocking(app_ctx.__aexit__(None, None, None))
