# Copyright Modal Labs 2025
"""Test deployment of @app.server() decorator."""

import modal

app = modal.App("test-server-deploy")


@app.server(port=8000, min_containers=1)
class SimpleServer:
    @modal.enter()
    def start(self):
        import subprocess
        self.proc = subprocess.Popen(["python3", "-m", "http.server", "8000"])

    @modal.exit()
    def stop(self):
        self.proc.terminate()

