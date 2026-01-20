# Copyright Modal Labs 2024
import subprocess

import modal
from modal import App

app = App(include_source=False)


class S:
    @modal.enter(snap=True)
    def start(self):
        self.process = subprocess.Popen(["python3", "-m", "http.server", "8000"])

    @modal.enter(snap=False)
    def enter_post_snapshot(self):
        return "Done starting server."

    @modal.exit()
    def stop(self):
        self.process.terminate()


UndecoratedS = S  # keep a reference to original class before overwriting

DecoratedS = app.server(port=8000, proxy_regions=["us-east"])(S)  # "decorator" of S
