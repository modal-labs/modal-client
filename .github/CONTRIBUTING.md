# Contributing

## Set Up Development Environment

1. Create and activate a Python virtual environment with your preferred tool. The Python version
   used for development is specified in `.github/workflows/ci-cd.yml` under
   `.github/actions/setup-cached-python`.
    1. If you use `uv`, run `uv venv .venv --python 3.11 --seed && source .venv/bin/activate`.
    1. If you use `pyenv`, run `pyenv virtualenv -p python3.11 3.11.12 modal-client && pyenv
       activate modal-client`.
1. Install development dependencies: `pip install -r requirements.dev.txt`
1. Compile protobuf files: `inv protoc`
1. Install the repo in editable mode: `pip install -e .`
1. Build type Python stubs and check types: `inv type-check`
1. Install pre-commit: `pre-commit install`
