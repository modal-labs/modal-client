import fastapi

import modal

stub = modal.Stub()


@stub.cls(container_idle_timeout=180)
class MyCls:
    def __init__(self, param):
        print("Running __init__")
        self.param = param

    @modal.enter()
    async def setup(self):
        print("Running enter step", self.param)
        self.other = "hello"

    @modal.method()
    async def run_inference(self):
        print("inference", self.param, self.other)


@stub.local_entrypoint()
def main():
    MyCls("hello").run_inference.remote()


@stub.function()
@modal.asgi_app()
def app():
    app = fastapi.FastAPI()

    @app.get("/")
    async def inference_request():
        MyCls("some_param").run_inference.remote()
