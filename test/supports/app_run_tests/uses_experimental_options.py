# Copyright Modal Labs 2025

import modal

app = modal.App()


@app.function(experimental_options={"warn_me": 1})
def gets_warning():
    print("Done!")
