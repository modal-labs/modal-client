from modal import App, Secret


def test_secret(servicer, client):
    app = App()
    with app.run(client=client):
        secret = Secret({"FOO": "BAR"}).create(app)
        assert secret.object_id == "st-123"
