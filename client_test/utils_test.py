from modal.app import print_logs
from modal.proto import api_pb2


def test_print_logs(capsys):
    print_logs("foo", api_pb2.FILE_DESCRIPTOR_STDOUT)
    print_logs("bar", api_pb2.FILE_DESCRIPTOR_STDERR)
    print_logs("baz", api_pb2.FILE_DESCRIPTOR_INFO)
    captured = capsys.readouterr()
    assert "foo" in captured.out
    assert "bar" in captured.err
    assert "baz" in captured.err
