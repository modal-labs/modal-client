from modal import App, Secret


def test_secret(servicer, client):
    app = App()
    with app.run(client=client) as running_app:
        secret = Secret({"FOO": "BAR"}).create(running_app)
        assert secret.object_id == "st-123"
