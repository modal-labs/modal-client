# Copyright Modal Labs 2023
import contextlib
import inspect
from collections import deque, defaultdict
from typing import Any, Optional, List, Tuple

from grpclib import GRPCError, Status


def patch_mock_servicer(cls):
    """Adds an `.intercept()` context manager method

    This allows for context-local tracking and assertions of all calls
    performed on the servicer during a context, e.g.:

    ```python notest
    with servicer.intercept() as ctx:
        await some_complex_method()
    assert ctx.calls == [("SomeMethod", MyMessage(foo="bar"))]
    ```
    Also allows to set a predefined queue of responses, temporarily replacing a mock servicer's default responses for a method:

    ```python notest
    with servicer.intercept() as ctx:
        ctx.add_response("SomeMethod", [
            MyResponse(bar="baz")
        ])
        ctx.add_response("SomeMethod", [
            MyResponse(bar="baz2")
        ])
        await service_stub.SomeMethod(Empty())  # receives MyResponse(bar="baz")
        await service_stub.SomeMethod(Empty())  # receives MyResponse(bar="baz2")
    ```

    Also patches all unimplemented abstract methods in a mock servicer with default error implementations.
    """

    async def fallback(self, stream) -> None:
        raise GRPCError(Status.UNIMPLEMENTED, "Not implemented in mock servicer " + repr(cls))

    @contextlib.contextmanager
    def intercept(servicer):
        ctx = InterceptionContext()
        servicer.interception_context = ctx
        yield ctx
        servicer.interception_context = None

    cls.intercept = intercept
    cls.interception_context = None

    def make_interceptable(method_name, original_method):
        async def intercepted_method(servicer_self, stream):
            ctx = servicer_self.interception_context
            if ctx:
                intercepted_stream = InterceptedStream(ctx, method_name, stream)
                custom_responder = ctx.next_custom_responder(method_name)
                if custom_responder:
                    return await custom_responder(servicer_self, intercepted_stream)
                else:
                    # use default servicer, but intercept messages for assertions
                    return await original_method(servicer_self, intercepted_stream)
            else:
                return await original_method(servicer_self, stream)

        return intercepted_method

    # Fill in the remaining methods on the class
    for name in dir(cls):
        method = getattr(cls, name)
        if getattr(method, "__isabstractmethod__", False):
            setattr(cls, name, make_interceptable(name, fallback))
        elif name[0].isupper() and inspect.isfunction(method):
            setattr(cls, name, make_interceptable(name, method))

    cls.__abstractmethods__ = frozenset()
    return cls


class InterceptionContext:
    def __init__(self):
        self.calls: List[Tuple[str, Any]] = []  # List[Tuple[method_name, message]]
        self.custom_responses: dict[str, deque[List[Any]]] = defaultdict(deque)

    def add_recv(self, method_name: str, msg):
        self.calls.append((method_name, msg))

    def add_response(self, method_name: str, custom_response: Optional[List[Any]] = None):
        # adds one response to a queue of responses for method method_name
        if custom_response is not None:
            assert isinstance(
                custom_response, list
            ), "custom_response should be a list of messages sent back by the servicer"

        if custom_response:
            self.custom_responses[method_name].append(custom_response)

    def next_custom_responder(self, method_name):
        method_responses = self.custom_responses[method_name]
        if not method_responses:
            return None
        next_response_messages = method_responses.popleft()

        async def responder(servicer_self, stream):
            await stream.recv_message()  # get the input message so we can track that
            for msg in next_response_messages:
                await stream.send_message(msg)

        return responder


class InterceptedStream:
    def __init__(self, interception_context, method_name, stream):
        self.interception_context = interception_context
        self.method_name = method_name
        self.stream = stream

    async def recv_message(self):
        msg = await self.stream.recv_message()
        self.interception_context.add_recv(self.method_name, msg)
        return msg

    async def send_message(self, msg):
        await self.stream.send_message(msg)

    def __getattr__(self, attr):
        return getattr(self.stream, attr)
