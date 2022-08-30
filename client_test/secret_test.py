from modal import Secret, Stub


def test_secret(servicer, client):
    stub = Stub()
    stub.secret = Secret({"FOO": "BAR"})
    with stub.run(client=client) as running_app:
        assert running_app.secret.object_id == "st-123"
