# Copyright Modal Labs 2026
import include_underscored_methods

tool = include_underscored_methods.Tool("typecheck")

private_result: str = tool._private("value")
made_tool: include_underscored_methods.Tool = include_underscored_methods.Tool._make("made")


async def check_async() -> None:
    async_private_result: str = await tool._private.aio("value")
    async_made_tool: include_underscored_methods.Tool = await include_underscored_methods.Tool._make.aio("made")
    stream_values: list[str] = [value async for value in tool._stream.aio(2)]

    assert async_private_result
    assert async_made_tool
    assert stream_values
