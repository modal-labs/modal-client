from modal import Schedule, Session, function

session = Session()


@function(session=session, schedule=Schedule.period("5s"))
def f():
    pass


def test_schedule(servicer, client):
    with session.run(client=client):
        assert servicer.function2schedule == {"fu-1": "sc-1"}
