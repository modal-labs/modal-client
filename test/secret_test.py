from modal import App, Secret


def test_secret(servicer, client):
    app = App()
    with app.run(client=client):
        secret = Secret.create(env_dict={"FOO": "BAR"})
        assert secret.object_id == "st-123"
