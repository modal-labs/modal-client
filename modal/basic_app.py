import time

import modal

app = modal.App("basic-app2", image=modal.Image.debian_slim())


@app.cls(region="us-west-2", min_containers=1, max_containers=1)
class MyClass:
    @modal.enter()
    def enter(self):
        self.remote_fn = modal.Cls.from_name("basic-app2-2", "Run")()

    @modal.method()
    def run(self, x: str):
        st = time.time()
        remote_fn = self.remote_fn
        handle = remote_fn.run.remote.aio(1)
        elapsed = time.time() - st
        return elapsed, handle.object_id


if __name__ == "__main__":
    clazz = modal.Cls.from_name("basic-app2", "MyClass")()
    times = []
    for k in range(1000):
        elapsed, object_id = clazz.run.remote("hello")
        print(object_id, elapsed)
        times.append(elapsed)
    p50 = sorted(times)[len(times) // 2]
    print("p50:", p50)
