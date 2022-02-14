from modal import Secret, Session


def test_secret(servicer, client):
    session = Session()
    with session.run(client=client):
        secret = Secret.create(env_dict={"FOO": "BAR"})
        assert secret.object_id == "st-123"
