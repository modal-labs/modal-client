# Copyright Modal Labs 2024
import subprocess

import modal
from modal import App

app = App(include_source=False)


class S:
    @modal.enter()
    def start(self):
        self.process = subprocess.Popen(["python3", "-m", "http.server", "8000"])

    @modal.exit()
    def stop(self):
        self.process.terminate()


UndecoratedS = S  # keep a reference to original class before overwriting

DecoratedS = app._experimental_server(port=8000, proxy_regions=["us-east"])(S)  # type: ignore # "decorator" of S
