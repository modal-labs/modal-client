from modal._logging import LogPrinter
from modal_proto import api_pb2
from modal_utils.app_utils import is_valid_deployment_name


def test_print_logs(capsys):
    log_printer = LogPrinter()
    log_printer.feed(api_pb2.TaskLogs(data="foo", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDOUT))
    log_printer.feed(api_pb2.TaskLogs(data="bar", file_descriptor=api_pb2.FILE_DESCRIPTOR_STDERR))
    log_printer.feed(api_pb2.TaskLogs(data="baz", file_descriptor=api_pb2.FILE_DESCRIPTOR_INFO))
    captured = capsys.readouterr()
    assert "foo" in captured.out
    assert "bar" in captured.err
    assert "baz" in captured.err


def test_deployment_name():
    assert is_valid_deployment_name("banana")
    assert is_valid_deployment_name("BaNaNa")
    assert not is_valid_deployment_name(" ")
    assert not is_valid_deployment_name("ban/ana")
