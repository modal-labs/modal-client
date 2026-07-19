---
description: Complete API reference for the Modal Python SDK. Documentation for App, Function, Image, Sandbox, Volume, and other Modal primitives.
---

# Python SDK Reference

This is the API reference for the [`modal`](https://pypi.org/project/modal/)
Python SDK, which allows you to programmatically interact with Modal.

## Application construction

|  |  |
| --- | --- |
| [`App`](/docs/sdk/py/latest/App) | The main unit of deployment for code on Modal |
| [`App.function`](/docs/sdk/py/latest/App#function) | Decorator for registering a function with an App |
| [`App.cls`](/docs/sdk/py/latest/App#cls) | Decorator for registering a class with an App |
| [`App.server`](/docs/sdk/py/latest/App#server) | Decorator for registering a server with an App |

## Serverless execution

|  |  |
| --- | --- |
| [`Function`](/docs/sdk/py/latest/Function) | A serverless function backed by an autoscaling container pool |
| [`Cls`](/docs/sdk/py/latest/Cls) | A serverless class supporting parametrization and lifecycle hooks |
| [`Server`](/docs/sdk/py/latest/Server) | A serverless HTTP application with low-latency request routing |

## Extended Function configuration

### Class parametrization

|  |  |
| --- | --- |
| [`parameter`](/docs/sdk/py/latest/parameter) | Used to define class parameters, akin to a Dataclass field |

### Lifecycle hooks

|  |  |
| --- | --- |
| [`enter`](/docs/sdk/py/latest/enter) | Decorator for a method that will be executed during container startup |
| [`exit`](/docs/sdk/py/latest/exit) | Decorator for a method that will be executed during container shutdown |
| [`method`](/docs/sdk/py/latest/method) | Decorator for exposing a method as an invokable function |

### Web integrations

|  |  |
| --- | --- |
| [`fastapi_endpoint`](/docs/sdk/py/latest/fastapi_endpoint) | Decorator for exposing a simple FastAPI-based endpoint |
| [`asgi_app`](/docs/sdk/py/latest/asgi_app) | Decorator for functions that construct an ASGI web application |
| [`wsgi_app`](/docs/sdk/py/latest/wsgi_app) | Decorator for functions that construct a WSGI web application |
| [`web_server`](/docs/sdk/py/latest/web_server) | Decorator for functions that construct an HTTP web server |

### Function semantics

|  |  |
| --- | --- |
| [`batched`](/docs/sdk/py/latest/batched) | Decorator that enables [dynamic input batching](/docs/guide/dynamic-batching) |
| [`concurrent`](/docs/sdk/py/latest/concurrent) | Decorator that enables [input concurrency](/docs/guide/concurrent-inputs) |

### Scheduling

|  |  |
| --- | --- |
| [`Cron`](/docs/sdk/py/latest/Cron) | A schedule that runs based on cron syntax |
| [`Period`](/docs/sdk/py/latest/Period) | A schedule that runs at a fixed interval |

### Exception handling

|  |  |
| --- | --- |
| [`Retries`](/docs/sdk/py/latest/Retries) | Function retry policy for input failures |

## Sandboxed execution

|  |  |
| --- | --- |
| [`Sandbox`](/docs/sdk/py/latest/Sandbox) | An interface for restricted code execution |
| [`ContainerProcess`](/docs/sdk/py/latest/container_process#containerprocess) | An object representing a sandboxed process |
| [`FileIO`](/docs/sdk/py/latest/file_io#fileio) | A handle for a file in the Sandbox filesystem |

## Container configuration

|  |  |
| --- | --- |
| [`Image`](/docs/sdk/py/latest/Image) | An API for specifying container images |
| [`Secret`](/docs/sdk/py/latest/Secret) | A pointer to secrets that will be exposed as environment variables |

## Data primitives

### Persistent storage

|  |  |
| --- | --- |
| [`Volume`](/docs/sdk/py/latest/Volume) | Distributed storage supporting highly performant parallel reads |
| [`CloudBucketMount`](/docs/sdk/py/latest/CloudBucketMount) | Storage backed by a third-party cloud bucket (S3, etc.) |

### In-memory storage

|  |  |
| --- | --- |
| [`Dict`](/docs/sdk/py/latest/Dict) | A distributed key-value store |
| [`Queue`](/docs/sdk/py/latest/Queue) | A distributed FIFO queue |

## Account configuration

|  |  |
| --- | --- |
| [`Workspace`](/docs/sdk/py/latest/Workspace) | Workspace-level configuration and observability |
| [`Environment`](/docs/sdk/py/latest/Environment) | Manage workspace subdivisions |

## Networking

|  |  |
| --- | --- |
| [`Proxy`](/docs/sdk/py/latest/Proxy) | An object that provides a static outbound IP address for containers |
| [`forward`](/docs/sdk/py/latest/forward) | A context manager for publicly exposing a port from a container |
