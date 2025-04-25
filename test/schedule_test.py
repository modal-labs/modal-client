# Copyright Modal Labs 2022
from modal import App, Period
from modal_proto import api_pb2

app = App(include_source=True)  # TODO: remove include_source=True when automount is disabled by default


@app.function(schedule=Period(seconds=5))
def f():
    pass


def test_schedule(servicer, client):
    with app.run(client=client):
        assert servicer.function2schedule == {"fu-1": api_pb2.Schedule(period=api_pb2.Schedule.Period(seconds=5.0))}
