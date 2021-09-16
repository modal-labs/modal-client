from polyester.utils import print_logs


def test_print_logs(capsys):
    print_logs(b"foo", "stdout")
    print_logs(b"bar", "stderr")
    print_logs(b"baz", "server")
    captured = capsys.readouterr()
    assert "foo" in captured.out
    assert "bar" in captured.err
    assert "baz" in captured.err
