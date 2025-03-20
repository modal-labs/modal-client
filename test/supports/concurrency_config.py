# Copyright Modal Labs 2025
import modal

app = modal.App("concurrency-config")

CONFIG_VALS = {"OLD": 100, "NEW": 1000}


@app.function(allow_concurrent_inputs=CONFIG_VALS["OLD"])
def has_old_concurrency_config():
    pass


@app.function()
@modal.concurrent(max_inputs=CONFIG_VALS["NEW"])
def has_new_concurrency_config():
    pass


@app.function()
def has_no_concurrency_config():
    pass
