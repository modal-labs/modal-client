import asyncio
from typing import Dict, List, Literal, Optional

import fastapi
from pydantic import BaseModel

from modal.repl import Repl


class ReplMCPCreateRequest(BaseModel):
    python_version: Optional[str]
    packages: Optional[List[str]]
    port: Optional[int]
    timeout: int = 5  # num of minutes to keep repl alive


class ReplMCPCreateResponse(BaseModel):
    repl_id: str
    status_code: Literal[200, 400, 500]
    error_message: Optional[str]


class ReplMCPExecRequest(BaseModel):
    repl_id: str
    command: str


class ReplMCPExecResponse(BaseModel):
    status_code: Literal[200, 400, 500]
    output: str
    error_message: Optional[str]


app = fastapi.FastAPI()
aliveRepls: Dict[str, Repl] = {}  # repl id -> Repl object
curSnapshotRefs: Dict[str, str] = {}  # repl id -> current snapshot id
replTimers: Dict[str, asyncio.TimerHandle] = {}
replTimeouts: Dict[str, int] = {}  # repl id -> timeout


@app.post("/create_repl")
async def create_repl(request: ReplMCPCreateRequest) -> ReplMCPCreateResponse:
    try:
        required_packages = ["fastapi", "pydantic", "uvicorn"]
        if request.packages:
            request.packages.extend(required_packages)
        else:
            request.packages = required_packages
        repl = Repl.create(request.python_version or "3.13", request.port or 8000, request.packages)
        aliveRepls[repl.uuid] = repl
        loop = asyncio.get_event_loop()
        replTimeouts[repl.uuid] = request.timeout
        replTimers[repl.uuid] = loop.call_later(request.timeout * 60, terminate_repl, repl.uuid)
        return ReplMCPCreateResponse(repl_id=repl.uuid, status_code=200)
    except Exception as e:
        return ReplMCPCreateResponse(repl_id="", status_code=500, error_message=str(e))


@app.post("/exec")
async def exec_cmd(request: ReplMCPExecRequest) -> ReplMCPExecResponse:
    try:
        repl = get_repl(request.repl_id)
        replTimers[repl.uuid].cancel()
        loop = asyncio.get_event_loop()
        replTimers[repl.uuid] = loop.call_later(replTimeouts[repl.uuid] * 60, terminate_repl, repl.uuid)
        commands = Repl.parse_command(request.command)
    except ValueError:
        return ReplMCPExecResponse(output="", status_code=400, error_message="Repl not found")
    try:
        output = repl.run(commands)
        return ReplMCPExecResponse(output=output.json()["result"], status_code=200)
    except Exception as e:
        return ReplMCPExecResponse(output="", status_code=500, error_message=str(e))


def get_repl(repl_id: str) -> Repl:
    if repl_id in aliveRepls:
        return aliveRepls[repl_id]
    if repl_id in curSnapshotRefs:
        return Repl.from_snapshot(curSnapshotRefs[repl_id])
    raise ValueError(f"Repl {repl_id} not found")


def terminate_repl(repl_id: str):
    repl = get_repl(repl_id)
    repl.kill()
    replTimers[repl_id].cancel()
    del aliveRepls[repl_id]
    del curSnapshotRefs[repl_id]
    del replTimers[repl_id]


asyncio.run(app.run("0.0.0.0", 8000))
