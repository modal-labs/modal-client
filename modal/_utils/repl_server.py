# Copyright Modal Labs 2025

import asyncio
from typing import Any, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="REPL Server")

_exec_context: Dict[str, Any] = {"__builtins__": __builtins__}


class ReplCommand(BaseModel):
    code: List[Tuple[str, Literal["exec", "eval"]]]


class ReplCommandResponse(BaseModel):
    status_code: Literal[400, 200]
    result: str


@app.post("/")
def run_exec(body: ReplCommand) -> ReplCommandResponse:
    commands = body.code
    try:
        for command in commands:
            if command[1] == "exec":
                exec(command[0], _exec_context, _exec_context)
            else:
                res = eval(command[0], _exec_context, _exec_context)
                return ReplCommandResponse(status_code=200, result=str(res))
        return ReplCommandResponse(status_code=200, result="")  # just send back blank str if all commands are exec'd
    except Exception as exc:
        return ReplCommandResponse(status_code=400, result=str(exc))


async def _main() -> None:
    print("Starting REPL server")
    config = uvicorn.Config(app=app, host="0.0.0.0", port=8000, reload=False)
    server = uvicorn.Server(config)
    await server.serve()


asyncio.run(_main())
