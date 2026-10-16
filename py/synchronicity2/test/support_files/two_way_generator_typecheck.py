# Copyright Modal Labs 2026
import typing

import two_way_generator

# sync tests
gen = two_way_generator.echo_generator()

gen.send(None)  # type: ignore  # First send must be None, even though signature specifies str
gen.send("hello")  # str send values


async def async_check():
    agen = two_way_generator.echo_generator.aio()

    await agen.asend(None)  # type: ignore  # First send must be None, even though signature specifies str
    await agen.asend("hello")  # str send values

    payload_agen: typing.AsyncGenerator[two_way_generator.Payload, two_way_generator.Payload] = (
        two_way_generator.async_payload_generator.aio()
    )
    payload = await anext(payload_agen)
    await payload_agen.asend(payload)


payload_gen: typing.Generator[two_way_generator.Payload, two_way_generator.Payload, two_way_generator.Payload] = (
    two_way_generator.sync_payload_generator()
)
payload = next(payload_gen)
payload_gen.send(payload)

payload_iter: typing.Iterator[two_way_generator.Payload] = two_way_generator.iter_payloads()
next(payload_iter)
