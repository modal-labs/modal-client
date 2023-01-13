# Copyright Modal Labs 2023
import asyncio
import os
import sys
from asyncio.subprocess import Process
from pathlib import Path

from ._output import OutputManager

MODAL_AUTORELOAD_ENV = "MODAL_AUTORELOAD_SERVE"


async def restart_serve(existing_app_id: str, prev_proc: Process, output_mgr: OutputManager) -> Process:
    if prev_proc:
        try:
            prev_proc.terminate()
        except ProcessLookupError:
            output_mgr.print_if_visible("[yellow]⚡️ Previous serving app process crashed.[/yellow]")
    env = {**os.environ, **{MODAL_AUTORELOAD_ENV: existing_app_id}}
    return await asyncio.create_subprocess_exec(
        sys.executable,
        *_get_child_arguments(),
        env=env,
    )


def _get_child_arguments():
    """
    Return the executable. This contains a workaround for Windows if the
    executable is reported to not have the .exe extension which can cause bugs
    on reloading.

    ---

    Copyright (c) Django Software Foundation and individual contributors.
    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:

        1. Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.

        2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

        3. Neither the name of Django nor the names of its contributors may be used
        to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
    ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    import __main__

    py_script = Path(sys.argv[0])

    args = ["-W%s" % o for o in sys.warnoptions]
    if sys.implementation.name == "cpython":
        args.extend(f"-X{key}" if value is True else f"-X{key}={value}" for key, value in sys._xoptions.items())
    # __spec__ is set when the server was started with the `-m` option,
    # see https://docs.python.org/3/reference/import.html#main-spec
    # __spec__ may not exist, e.g. when running in a Conda env.
    if getattr(__main__, "__spec__", None) is not None:
        spec = __main__.__spec__
        if (spec.name == "__main__" or spec.name.endswith(".__main__")) and spec.parent:
            name = spec.parent
        else:
            name = spec.name
        args += ["-m", name]
        args += sys.argv[1:]
    elif not py_script.exists():
        # sys.argv[0] may not exist for several reasons on Windows.
        # It may exist with a .exe extension or have a -script.py suffix.
        exe_entrypoint = py_script.with_suffix(".exe")
        if exe_entrypoint.exists():
            # Should be executed directly, ignoring sys.executable.
            return [exe_entrypoint, *sys.argv[1:]]
        script_entrypoint = py_script.with_name("%s-script.py" % py_script.name)
        if script_entrypoint.exists():
            # Should be executed as usual.
            return [*args, script_entrypoint, *sys.argv[1:]]
        raise RuntimeError("Script %s does not exist." % py_script)
    else:
        args += sys.argv
    return args
