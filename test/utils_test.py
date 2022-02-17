from modal.app import print_logs


def test_print_logs(capsys):
    print_logs("foo", "stdout")
    print_logs("bar", "stderr")
    print_logs("baz", "server")
    captured = capsys.readouterr()
    assert "foo" in captured.out
    assert "bar" in captured.err
    assert "baz" in captured.err
