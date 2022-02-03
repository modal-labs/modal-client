from modal import Schedule, Session, function

session = Session()


@Schedule.factory
def my_schedule_1():
    return Schedule.create(period="5s", session=session)


@function(session=session, schedule=my_schedule_1)
def f():
    pass


def test_schedule(servicer, client):
    with session.run(client=client):
        assert servicer.function2schedule == {"fu-1": "sc-1"}
