# Copyright Modal Labs 2025
import threading

success = threading.Event()


def main():
    import modal  # noqa

    success.set()


if __name__ == "__main__":
    t = threading.Thread(target=main, daemon=True)
    t.start()
    was_success = success.wait(timeout=5)
    assert was_success
