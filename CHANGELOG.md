# Changelog

This changelog documents user-facing updates (features, enhancements, fixes, and deprecations) to the `modal` client library. Patch releases are made on every change.

The client library is still in pre-1.0 development, and sometimes breaking changes are necessary. We try to minimize them and publish deprecation warnings / migration guides in advance, typically providing a transition window of several months.

We appreciate your patience while we speedily work towards a stable release of the client.

## Latest

<!-- NEW CONTENT GENERATED BELOW. PLEASE PRESERVE THIS COMMENT. -->

### 0.56.4715 (2024-01-24)

* `Mount.from_local_python_packages` now places mounted packages at `/root` in the Modal runtime by default (used to be `/pkg`). To override this behavior, the function now takes a `remote_dir: Union[str, PurePosixPath]` argument.



### 0.56.4707 (2024-01-23)

The Modal client library is now compatible with Python 3.12, although there are a few limitations:

- Images that use Python 3.12 without explicitly specifing it through `python_version` or `add_python` will not build
  properly unless the modal client is also running on Python 3.12.
- The `conda` and `microconda` base images currently do not support Python 3.12 because an upstream dependency is not yet compatible.



### 0.56.4700 (2024-01-22)

- `gpu.A100` class now supports specifying GiB memory configuration using a `size: str` parameter. The `memory: int` parameter is deprecated.



### 0.56.4693 (2024-01-22)

You can now execute commands in running containers with `modal container exec [container-id] [command]`.



### 0.56.4691 (2024-01-22)

* The `modal` cli now works more like the `python` cli in regard to script/module loading:
    - Running `modal my_dir/my_script.py` now puts `my_dir` on the PYTHONPATH.
    - `modal my_package.my_module` will now mount to /root/my_package/my_module.py in your Modal container, regardless if using automounting or not (and any intermediary `__init__.py` files will also be mounted)



### 0.56.4687 (2024-01-20)

- Modal now uses the current profile if `MODAL_PROFILE` is set to the empty string.



### 0.56.4649 (2024-01-17)

- Dropped support for building Python 3.7 based `modal.Image`s. Python 3.7 is end-of-life since late June 2023.



### 0.56.4620 (2024-01-16)

* modal.Stub.function now takes a `block_network` argument.



### 0.56.4616 (2024-01-16)

* modal.Stub now takes a `volumes` argument for setting the default volumes of all the stub's functions, similarly to the `mounts` and `secrets` argument.



### 0.56.4590 (2024-01-13)

`modal serve`: Setting MODAL_LOGLEVEL=DEBUG now displays which files cause an app reload during serve



### 0.56.4570 (2024-01-12)

- `modal run` cli command now properly propagates `--env` values to object lookups in global scope of user code


