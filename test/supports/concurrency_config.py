# Copyright Modal Labs 2025
import modal

app = modal.App("concurrency-config")

CONFIG_VALS = {"OLD_MAX": 100, "NEW_MAX": 1000, "TARGET": 500}


@app.function(allow_concurrent_inputs=CONFIG_VALS["OLD_MAX"])
def has_old_config():
    pass


@app.function()
@modal.concurrent(max_inputs=CONFIG_VALS["NEW_MAX"], target_inputs=CONFIG_VALS["TARGET"])
def has_new_config():
    pass


@app.function()
@modal.concurrent(max_inputs=CONFIG_VALS["NEW_MAX"], target_inputs=CONFIG_VALS["TARGET"])
@modal.fastapi_endpoint()
def has_new_config_and_fastapi_endpoint():
    pass


@app.function()
@modal.fastapi_endpoint()
@modal.concurrent(max_inputs=CONFIG_VALS["NEW_MAX"], target_inputs=CONFIG_VALS["TARGET"])
def has_fastapi_endpoint_and_new_config():
    pass


@app.function()
def has_no_config():
    pass


@app.cls(allow_concurrent_inputs=CONFIG_VALS["OLD_MAX"])
class HasOldConfig:
    @modal.method()
    def method(self): ...


@app.cls()
@modal.concurrent(max_inputs=CONFIG_VALS["NEW_MAX"], target_inputs=CONFIG_VALS["TARGET"])
class HasNewConfig:
    @modal.method()
    def method(self): ...


@app.cls()
class HasNoConfig:
    @modal.method()
    def method(self): ...


@app.cls()
@modal.concurrent(max_inputs=CONFIG_VALS["NEW_MAX"], target_inputs=CONFIG_VALS["TARGET"])
class HasNewConfigAndFastapiEndpoint:
    @modal.fastapi_endpoint()
    def method(self): ...
