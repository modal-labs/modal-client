# Copyright Modal Labs 2025
# type: ignore

"""Test deployment of @app.server() decorator."""

import modal
import modal.experimental

app = modal.App("test-server-deploy")


@app.server(port=8000, min_containers=1, proxy_regions=["us-east"])
class SimpleServer:
    @modal.enter()
    def start(self):
        import subprocess

        self.proc = subprocess.Popen(["python3", "-m", "http.server", "8000"])

    @modal.exit()
    def stop(self):
        self.proc.terminate()


@app.server(port=8000, proxy_regions=["us-east"], image=modal.Image.debian_slim().pip_install("fastapi", "uvicorn"))
@modal.experimental.asgi_app_on_flash()
def create_app():
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/")
    def root():
        return {"message": "Hello World"}

    return app
