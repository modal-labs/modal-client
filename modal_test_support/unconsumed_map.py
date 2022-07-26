from .stub import f, stub

if __name__ == "__main__":
    with stub.run():
        f.map([1, 2, 3])
