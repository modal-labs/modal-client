from modal_proto import api_pb2

from modal._logging import LogPrinter


def test_print_logs(capsys):
    log_printer = LogPrinter()
    log_printer.feed(api_pb2.TaskLogs(data="foo", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT))
    log_printer.feed(api_pb2.TaskLogs(data="bar", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDERR))
    log_printer.feed(api_pb2.TaskLogs(data="baz", file_descriptor=api_pb2.FILE_DESCRIPTOR_INFO))
    captured = capsys.readouterr()
    assert "foo" in captured.out
    assert "bar" in captured.err
    assert "baz" in captured.err
