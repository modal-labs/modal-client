# Changelog

This changelog documents user-facing updates (features, enhancements, fixes, and deprecations) to the `modal` client library. Patch releases are made on every change.

The client library is still in pre-1.0 development, and sometimes breaking changes are necessary. We try to minimize them and publish deprecation warnings / migration guides in advance, typically providing a transition window of several months.

We appreciate your patience while we speedily work towards a stable release of the client.

## Latest

<!-- NEW CONTENT GENERATED BELOW. PLEASE PRESERVE THIS COMMENT. -->

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


