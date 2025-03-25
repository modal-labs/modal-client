import time
from collections import defaultdict
from contextlib import contextmanager

import modal
from modal._runtime.user_code_imports import DATA_FORMAT
from modal_proto import api_pb2

app = modal.App(
    secrets=[modal.Secret.from_local_environ(["MODAL_DATA_FORMAT"])],
    image=modal.Image.debian_slim().pip_install("cbor2"),
)


stat = {"data_format": api_pb2.DataFormat.Name(DATA_FORMAT)}
stat_queue = modal.Queue.from_name("stat-queue", create_if_missing=True)


@contextmanager
def timer(name):
    t = time.monotonic_ns()
    yield
    dt = time.monotonic_ns() - t
    stat[name] = dt
    # print(f"{name}: {dt:_}")


num_inputs = 50


@app.function(min_containers=num_inputs, region="us-east", max_inputs=1)
def serialize_stuff():
    stat["function"] = "serialize_stuff"
    return ["a"] * 100_000


@app.function(region="us-east", min_containers=num_inputs)
def wrapper():
    # records e2e times
    stat["function"] = "wrapper"
    with timer("e2e"):
        serialize_stuff.remote()


@app.local_entrypoint()
def d():
    time.sleep(10)  # wait for containers to come online
    print("starting")
    stat_queue.clear()
    t = []
    t0 = time.monotonic()
    for _ in range(num_inputs):
        t.append(wrapper.spawn())
        modal.FunctionCall.gather(*t)
    dt = time.monotonic() - t0

    print("Got data")
    function_data = defaultdict(list)
    for data in stat_queue.iterate():
        function_data[data["function"]].append(data)
    # haxx:
    output_rows = function_data["serialize_stuff"]
    for output_row, metadata in zip(output_rows, function_data["wrapper"]):
        output_row["e2e"] = metadata["e2e"]
        output_row["deserialize"] = metadata["deserialize"]

    import pandas

    df = pandas.DataFrame(output_rows)
    print(df)
    print(df.describe().loc[["mean", "std", "50%", "max"]])
    print(f"Total duration: {dt:_}")
