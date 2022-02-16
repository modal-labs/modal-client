from modal import App, Schedule, function

app = App()


@function(app=app, schedule=Schedule.period("5s"))
def f():
    pass


def test_schedule(servicer, client):
    with app.run(client=client):
        assert servicer.function2schedule == {"fu-1": ("", 5.0)}
