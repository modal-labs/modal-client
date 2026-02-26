# Copyright Modal Labs 2025
import modal

app = modal.App("batching-config", include_source=False)

CONFIG_VALS = {"MAX_SIZE": 100, "WAIT_MS": 1000}


@app.function()
@modal.batched(max_batch_size=CONFIG_VALS["MAX_SIZE"], wait_ms=CONFIG_VALS["WAIT_MS"])
def has_batch_config():
    pass


@app.cls()
class HasBatchConfig:
    @modal.batched(max_batch_size=CONFIG_VALS["MAX_SIZE"], wait_ms=CONFIG_VALS["WAIT_MS"])
    def is_batched(self):
        pass
