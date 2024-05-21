# Copyright Modal Labs 2024
import modal


def dummy():
    pass


def test_volume_mount(client, servicer):
    app = modal.App()
    secret = modal.Secret.from_dict({"AWS_ACCESS_KEY_ID": "1", "AWS_SECRET_ACCESS_KEY": "2"})
    cld_bckt_mnt = modal.CloudBucketMount(
        bucket_name="foo",
        key_prefix="dir/",
        bucket_endpoint_url="https://1234.r2.cloudflarestorage.com",
        secret=secret,
        read_only=False,
    )

    _ = app.function(volumes={"/root/foo": cld_bckt_mnt})(dummy)

    with app.run(client=client):
        pass
