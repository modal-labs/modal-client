# Copyright Modal Labs 2026
"""Fork while _loop_creation_lock is held and verify the child does not deadlock.

threading.Lock is a pthread mutex: if the parent holds it at fork time, the child inherits
it permanently locked, which would hang _start_loop() in the child without the at-fork hook.
"""

import multiprocessing
import os
import threading

from synchronicity2.synchronizer import Synchronizer


def main():
    multiprocessing.set_start_method("fork")

    synchronizer = Synchronizer()

    async def hello():
        return f"hello from pid={os.getpid()}"

    synchronizer._run_function_sync(hello())  # start the loop in the parent

    hangs = 0
    total = 10

    for _ in range(total):
        synchronizer._close_loop()

        started = threading.Event()

        def bg():
            started.set()
            synchronizer._start_loop()

        t = threading.Thread(target=bg, daemon=True)
        t.start()
        started.wait()

        rq = multiprocessing.Queue()

        def child(rq):
            try:
                rq.put(synchronizer._run_function_sync(hello()))
            except Exception as e:
                rq.put(f"error: {e}")

        p = multiprocessing.Process(target=child, args=(rq,))
        p.start()
        p.join(timeout=5)

        if p.is_alive():
            hangs += 1
            p.kill()
            p.join()

        t.join(timeout=2)

    if hangs > 0:
        print(f"FAIL: {hangs}/{total} children deadlocked", flush=True)
        raise SystemExit(1)
    print(f"PASS: 0/{total} deadlocks", flush=True)


if __name__ == "__main__":
    main()
