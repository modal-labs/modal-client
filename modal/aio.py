# Copyright Modal Labs 2022
import datetime

from modal.exception import deprecation_error

deprecation_error(
    datetime.date(2023, 5, 12),
    "The `modal.aio` module and `Aio*` classes have been deprecated.\n"
    "For calling functions asynchronously, use `await some_function.aio(...)`\n"
    "Instead of separate classes for async usage, the interface now only changes how to call the methods."
    "Where you would have previously used `await AioDict.lookup(...)` you now use "
    "`await Dict.lookup.aio(...)` instead.\nObjects that are themselves generators or context managers "
    "now conform to both the blocking and async interfaces, and returned objects of all functions/methods",
)
