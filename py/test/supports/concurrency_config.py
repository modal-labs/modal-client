# Copyright Modal Labs 2025
import modal

app = modal.App("concurrency-config", include_source=False)

CONFIG_VALS = {"MAX": 100, "TARGET": 50}


@app.function()
@modal.concurrent(max_inputs=CONFIG_VALS["MAX"], target_inputs=CONFIG_VALS["TARGET"])
def has_concurrent_config():
    pass


@app.function()
@modal.concurrent(max_inputs=CONFIG_VALS["MAX"], target_inputs=CONFIG_VALS["TARGET"])
@modal.fastapi_endpoint()
def has_concurrent_config_and_fastapi_endpoint():
    pass


@app.function()
@modal.fastapi_endpoint()
@modal.concurrent(max_inputs=CONFIG_VALS["MAX"], target_inputs=CONFIG_VALS["TARGET"])
def has_fastapi_endpoint_and_concurrent_config():
    pass


@app.function()
def has_no_config():
    pass


@app.cls()
@modal.concurrent(max_inputs=CONFIG_VALS["MAX"], target_inputs=CONFIG_VALS["TARGET"])
class HasConcurrentConfig:
    @modal.method()
    def method(self): ...


@app.cls()
@modal.concurrent(max_inputs=CONFIG_VALS["MAX"], target_inputs=CONFIG_VALS["TARGET"])
class HasConcurrentConfigAndFastapiEndpoint:
    @modal.fastapi_endpoint()
    def method(self): ...


@app.cls()
class HasNoConfig:
    @modal.method()
    def method(self): ...
