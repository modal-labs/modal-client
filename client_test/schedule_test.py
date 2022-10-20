# Copyright Modal Labs 2022
from modal import Period, Stub
from modal_proto import api_pb2

stub = Stub()


@stub.function(schedule=Period(seconds=5))
def f():
    pass


def test_schedule(servicer, client):
    with stub.run(client=client):
        assert servicer.function2schedule == {"fu-1": api_pb2.Schedule(period=api_pb2.Schedule.Period(seconds=5.0))}
