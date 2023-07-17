# Copyright Modal Labs 2023
import contextlib
import inspect
import logging
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Tuple

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
        ctx.assert_responses_consumed()
        servicer.interception_context = None

    cls.intercept = intercept
    cls.interception_context = None

    def make_interceptable(method_name, original_method):
        async def intercepted_method(servicer_self, stream):
            ctx = servicer_self.interception_context
            if ctx:
                intercepted_stream = await InterceptedStream(ctx, method_name, stream).initialize()
                custom_responder = ctx.next_custom_responder(method_name, intercepted_stream.request_message)
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


class ResponseNotConsumed(Exception):
    def __init__(self, unconsumed_requests: List[str]):
        self.unconsumed_requests = unconsumed_requests
        request_count = Counter(unconsumed_requests)
        super().__init__(f"Expected but did not receive the following requests: {request_count}")


class InterceptionContext:
    def __init__(self):
        self.calls: List[Tuple[str, Any]] = []  # List[Tuple[method_name, message]]
        self.custom_responses: Dict[str, List[Tuple[Callable[[Any], bool], List[Any]]]] = defaultdict(list)

    def add_recv(self, method_name: str, msg):
        self.calls.append((method_name, msg))

    def add_response(
        self, method_name: str, first_payload, *, request_filter: Callable[[Any], bool] = lambda req: True
    ):
        # adds one response to a queue of responses for requests of the specified type
        self.custom_responses[method_name].append((request_filter, [first_payload]))

    def next_custom_responder(self, method_name, request):
        method_responses = self.custom_responses[method_name]

        for i, (request_filter, response_messages) in enumerate(method_responses):
            try:
                request_matches = request_filter(request)
            except Exception:
                logging.exception("Error when filtering requests")
                raise

            if request_matches:
                next_response_messages = response_messages
                self.custom_responses[method_name] = method_responses[:i] + method_responses[i + 1 :]
                break
        else:
            return None

        async def responder(servicer_self, stream):
            try:
                await stream.recv_message()  # get the input message so we can track that
                for msg in next_response_messages:
                    await stream.send_message(msg)
            except Exception:
                logging.exception("Error when sending response")
                raise

        return responder

    def assert_responses_consumed(self):
        unconsumed = []
        for method_name, queued_responses in self.custom_responses.items():
            unconsumed += [method_name] * len(queued_responses)

        if unconsumed:
            raise ResponseNotConsumed(unconsumed)

    def pop_request(self, method_name):
        # fast forward to the next request of type method_name
        # dropping any preceding requests if there is a match
        # returns the payload of the request
        for i, (_method_name, msg) in enumerate(self.calls):
            if _method_name == method_name:
                self.calls = self.calls[i + 1 :]
                return msg

        raise Exception(f"No message of that type in call list: {self.calls}")


class InterceptedStream:
    def __init__(self, interception_context, method_name, stream):
        self.interception_context = interception_context
        self.method_name = method_name
        self.stream = stream
        self.request_message = None

    async def initialize(self):
        self.request_message = await self.recv_message()
        return self

    async def recv_message(self):
        if self.request_message:
            ret = self.request_message
            self.request_message = None
            return ret

        msg = await self.stream.recv_message()
        self.interception_context.add_recv(self.method_name, msg)
        return msg

    async def send_message(self, msg):
        await self.stream.send_message(msg)

    def __getattr__(self, attr):
        return getattr(self.stream, attr)
