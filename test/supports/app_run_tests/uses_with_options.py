# Copyright Modal Labs 2025

import modal

app = modal.App("uses-with-options")


@app.cls()
class C:
    @modal.method()
    def f(self):
        print("Done!")


C_with_gpu = C.with_options(gpu="H100")  # type: ignore  # Type masking is a problem for with_options
