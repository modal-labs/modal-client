from modal import Secret, Stub


def test_secret(servicer, client):
    stub = Stub()
    with stub.run(client=client) as running_app:
        secret = Secret({"FOO": "BAR"}).create(running_app)
        assert secret.object_id == "st-123"
