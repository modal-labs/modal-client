# Changelog

This changelog documents user-facing updates (features, enhancements, fixes, and deprecations) to the `modal` client library. Patch releases are made on every change.

The client library is still in pre-1.0 development, and sometimes breaking changes are necessary. We try to minimize them and publish deprecation warnings / migration guides in advance, typically providing a transition window of several months.

We appreciate your patience while we speedily work towards a stable release of the client.

## Latest

<!-- NEW CONTENT GENERATED BELOW. PLEASE PRESERVE THIS COMMENT. -->

### 0.57.13 (2024-02-07)

Adds support for mounting an S3 bucket as a volume.



### 0.57.9 (2024-02-07)

- Support for an implicit 'default' profile is now deprecated. If you have more than one profile in your Modal config file, one must be explicitly set to `active` (use `modal profile activate` or edit your `.modal.toml` file to resolve).
- An error is now raised when more than one profile is set to `active`.



### 0.57.2 (2024-02-06)

- Improve error message when generator functions are called with `.map(...)`.



### 0.57.0 (2024-02-06)

- Greatly improved streaming performance of generators and WebSocket web endpoints.
- **Breaking change:** You cannot use `.map()` to call a generator function. (In previous versions, this merged the results onto a single stream, but the behavior was undocumented and not widely used.)
- **Incompatibility:** Generator outputs are now on a different internal system. Modal code on client versions before 0.57 cannot trigger [deployed functions](https://modal.com/docs/guide/trigger-deployed-functions) with `.remote_gen()` that are on client version 0.57, and vice versa.



## 0.56

Note that in version 0.56 and prior, Modal used a different numbering system for patch releases.


### 0.56.4964 (2024-02-05)

- When using `modal token new` or `model token set`, the profile containing the new token will now be activated by default. Use the `--no-activate` switch to update the `modal.toml` file without activating the corresponding profile.



### 0.56.4953 (2024-02-05)

- The `modal profile list` output now indicates when the workspace is determined by a token stored in environment variables.



### 0.56.4952 (2024-02-05)

- Variadic parameters (e.g. *args and **kwargs) can now be used in scheduled functions as long as the function doesn't have any other parameters without a default value



### 0.56.4903 (2024-02-01)

- `modal container exec`'s `--no-tty` flag has been renamed to `--no-pty`.



### 0.56.4902 (2024-02-01)

- The singular form of the `secret` parameter in `Stub.function`, `Stub.cls`, and `Image.run_function` has been deprecated. Please update your code to use the plural form instead:`secrets=[Secret(...)]`.



### 0.56.4885 (2024-02-01)

- In `modal profile list`, the user's GitHub username is now shown as the name for the "Personal" workspace.



### 0.56.4874 (2024-01-31)

- The `modal token new` and `modal token set` commands now create profiles that are more closely associated with workspaces, and they have more explicit profile activation behavior:
  - By default, these commands will create/update a profile named after the workspace that the token points to, rather than a profile named "default"
  - Both commands now have an `--activate` flag that will activate the profile associated with the new token
  - If no other profiles exist at the time of creation, the new profile will have its `active` metadata set to True
- With these changes, we are moving away from the concept of a "default" profile. Implicit usage of the "default" profile will be deprecated in a future update.



### 0.56.4849 (2024-01-29)

- Adds tty support to `modal container exec` for fully-interactive commands. Example: `modal container exec [container-id] /bin/bash`



### 0.56.4792 (2024-01-26)

- The `modal profile list` command now shows the workspace associated with each profile.



### 0.56.4715 (2024-01-24)

- `Mount.from_local_python_packages` now places mounted packages at `/root` in the Modal runtime by default (used to be `/pkg`). To override this behavior, the function now takes a `remote_dir: Union[str, PurePosixPath]` argument.



### 0.56.4707 (2024-01-23)

- The Modal client library is now compatible with Python 3.12, although there are a few limitations:

  - Images that use Python 3.12 without explicitly specifing it through `python_version` or `add_python` will not build
    properly unless the modal client is also running on Python 3.12.
  - The `conda` and `microconda` base images currently do not support Python 3.12 because an upstream dependency is not yet compatible.



### 0.56.4700 (2024-01-22)

- `gpu.A100` class now supports specifying GiB memory configuration using a `size: str` parameter. The `memory: int` parameter is deprecated.



### 0.56.4693 (2024-01-22)

- You can now execute commands in running containers with `modal container exec [container-id] [command]`.



### 0.56.4691 (2024-01-22)

- The `modal` cli now works more like the `python` cli in regard to script/module loading:
  - Running `modal my_dir/my_script.py` now puts `my_dir` on the PYTHONPATH.
  - `modal my_package.my_module` will now mount to /root/my_package/my_module.py in your Modal container, regardless if using automounting or not (and any intermediary `__init__.py` files will also be mounted)



### 0.56.4687 (2024-01-20)

- Modal now uses the current profile if `MODAL_PROFILE` is set to the empty string.



### 0.56.4649 (2024-01-17)

- Dropped support for building Python 3.7 based `modal.Image`s. Python 3.7 is end-of-life since late June 2023.



### 0.56.4620 (2024-01-16)

- modal.Stub.function now takes a `block_network` argument.



### 0.56.4616 (2024-01-16)

- modal.Stub now takes a `volumes` argument for setting the default volumes of all the stub's functions, similarly to the `mounts` and `secrets` argument.



### 0.56.4590 (2024-01-13)

- `modal serve`: Setting MODAL_LOGLEVEL=DEBUG now displays which files cause an app reload during serve



### 0.56.4570 (2024-01-12)

- `modal run` cli command now properly propagates `--env` values to object lookups in global scope of user code


