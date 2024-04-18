# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + tags=["parameters"]
server_addr = None
# -

from modal.client import Client
from modal_proto import api_pb2

client = Client(server_addr, api_pb2.CLIENT_TYPE_CLIENT, ("foo-id", "foo-secret"))

# +
import modal

app = modal.App()


@app.function()
def hello():
    print("running")


# + tags=["main"]
with client:
    with app.run(client=client, show_progress=True):
        hello.remote()
# -
