# Developing `modal-client`

Check out the repo:

```bash
git clone https://github.com/modal-labs/modal-client.git
cd modal-client
```

Create a virtual environment:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install -r requirements.dev.txt
```

Build protobufs:

```bash
inv protoc type-stubs
```

`inv` refers to [Invoke](https://www.pyinvoke.org/), a Python CLI task runner.

Run modal:

```bash
source .venv/bin/activate
modal
```
