# Copyright Modal Labs 2024
import threading
import traceback

success = threading.Event()


def main():
    try:
        import modal  # noqa
    except BaseException:
        traceback.print_exc()
    success.set()


if __name__ == "__main__":
    t = threading.Thread(target=main, daemon=True)
    t.start()
    was_success = success.wait(timeout=1)
    assert was_success
