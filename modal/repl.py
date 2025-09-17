import ast
import asyncio
import os
import signal
import sys
import uuid
from typing import List, Literal, Optional, Tuple

import httpx

from modal.app import App
from modal.image import Image
from modal.output import enable_output
from modal.sandbox import Sandbox
from modal.snapshot import SandboxSnapshot


class Repl:
    port = 8000

    def __init__(self, sandbox: Sandbox, sb_url: str, id: Optional[str] = None):
        self.sb = sandbox
        self.sb_url = sb_url
        self.id = id or str(uuid.uuid4())

    @staticmethod
    def parse_command(code: str) -> List[Tuple[str, Literal["exec", "eval"]]]:
        try:
            tree = ast.parse(code, mode="exec")
            if tree.body and len(tree.body) > 0 and isinstance(tree.body[-1], ast.Expr):  # ast.Expr should be eval()'d
                last_expr = tree.body[-1]
                lines = code.splitlines(keepends=True)
                start_line = getattr(last_expr, "lineno", None)
                start_col = getattr(last_expr, "col_offset", None)
                end_line = getattr(last_expr, "end_lineno", None)
                end_col = getattr(last_expr, "end_col_offset", None)
                # print(start_line, start_col, end_line, end_col)
                if end_line is None or end_col is None or start_line is None or start_col is None:
                    return [(code, "exec")]
                start_line -= 1
                end_line -= 1  # ast parser returns 1-indexed lines.our list of strings is 0-indexed
                prefix_parts = []
                if start_line > 0:
                    prefix_parts.append("".join(lines[:start_line]))
                prefix_parts.append(lines[start_line][:start_col])
                prefix_code = "".join(prefix_parts)
                # puts everything before last expression into one str. this is all exec()'d
                last_expr_parts = []
                if start_line == end_line:
                    last_expr_parts.append(lines[start_line][start_col:end_col])
                else:
                    last_expr_parts.append(lines[start_line][start_col:])
                    if end_line - start_line > 1:
                        last_expr_parts.append("\n".join(lines[start_line + 1 : end_line]))
                    last_expr_parts.append(lines[end_line][:end_col])
                last_expr_code = "".join(last_expr_parts)

                commands = []
                if prefix_code.strip():
                    commands.append((prefix_code, "exec"))
                commands.append((last_expr_code, "eval"))
            else:
                commands = [(code, "exec")]  # whole thing exec()'d
            returnCommands = []
            for cmd in commands:
                if cmd[0].strip():
                    returnCommands.append(cmd)
            return returnCommands
        except Exception as e:
            print(str(e))
            return []

    @staticmethod
    def create(python_version: str = "3.13", port: int = port, packages: List[str] = []) -> "Repl":
        packages.append("fastapi")
        packages.append("uvicorn")
        packages.append("pydantic")
        image = Image.debian_slim(python_version=python_version)
        image = image.pip_install(*packages)
        repl_server_path = os.path.join(os.path.dirname(__file__), "_utils", "repl_server.py")
        image = image.add_local_file(local_path=repl_server_path, remote_path="/root/repl_server.py")
        app = App.lookup(name="repl", create_if_missing=True)
        with enable_output():
            start_cmd = ["bash", "-c", "cd root && python -m repl_server.py"]
            sb = Sandbox.create(
                *start_cmd, app=app, image=image, encrypted_ports=[port], _experimental_enable_snapshot=True
            )
            sb_url = sb.tunnels()[port].url
        return Repl(sb, sb_url)

    @staticmethod
    def from_snapshot(snapshot_id: str, id: Optional[str] = None) -> "Repl":
        snapshot = SandboxSnapshot.from_id(snapshot_id)
        sb = Sandbox._experimental_from_snapshot(snapshot)
        sb_url = sb.tunnels()[8000].url
        return Repl(sb, sb_url, id)

    @staticmethod
    def create_prompt() -> "Repl":
        print("enter the python packages you want to install in a comma separated list")
        packages = input()
        return Repl.create(packages=packages.split(","))

    @staticmethod
    async def start_repl():
        try:
            createNew = None
            while createNew not in ["y", "n"]:
                print("Would you like to use a previously snapshotted repl? (y/n)")
                createNew = input()
                createNew = createNew.lower()
                if createNew not in ["y", "n"]:
                    print("invalid choice, please enter y or n")
                    createNew = None
            if createNew == "y":
                repl = Repl.from_snapshot(input("Enter the id of the snapshot you want to restore from: "))
            else:
                repl = Repl.create_prompt()
            signal.signal(signal.SIGINT, lambda signum, frame: repl.signal_handler(signum, frame))
            with enable_output():
                print("Welcome to Modal REPL")
                while True:
                    print(">> ", end="")
                    command = input()
                    if command == "exit()":
                        asyncio.run(repl.signal_handler(signal.SIGINT, None))
                    commands = Repl.parse_command(command)
                    res = await repl.run(commands)
                    output = res.json()["result"]
                    if output:
                        print(output)
        except Exception as e:
            raise Exception(f"Error running commands: {e}")

    def run(self, commands: List[Tuple[str, Literal["exec", "eval"]]]):
        try:
            response = httpx.post(self.sb_url, json={"code": commands})
            return response
        except Exception as e:
            raise Exception(f"Error running commands: {e}")

    def kill(self):
        if self.sb:
            snapshot = self.sb._experimental_snapshot()
            self.sb.terminate()
            return snapshot.object_id
        raise ValueError("repl not found")

    def signal_handler(self, signum, frame):
        self.kill()
        sys.exit(0)
