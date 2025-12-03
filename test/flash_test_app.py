import subprocess

import modal
import modal.experimental
from modal import App

app = App("flash-app-default")


@app.cls(enable_memory_snapshot=True)
@modal.concurrent(target_inputs=10)
@modal.experimental.http_server(8080, proxy_regions=["us-east", "us-west"], exit_grace_period=10)
class FlashClassDefault:
    @modal.enter(snap=True)
    def serve(self):
        self.value = 2
        self.process = subprocess.Popen(["python3", "-m", "http.server", "8080"])
        print("pre snapshot", self.value)

    @modal.enter(snap=False)
    def enter_post_snapshot(self):
        self.value = 3
        print("post snapshot", self.value)

    @modal.exit()
    def exit(self):
        print("exit", self.value)
