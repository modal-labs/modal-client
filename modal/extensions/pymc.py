import ctypes
import sys
import traceback
from collections import namedtuple
from typing import Any, List

from aiostream import stream
from synchronicity.interface import Interface

import modal
from modal_utils.async_utils import synchronize_apis, synchronizer

pymc_stub = modal.Stub(
    image=modal.Image.conda()
    .conda_install(["theano-pymc==1.1.2", "pymc3==3.11.2", "scikit-learn", "mkl-service"])
    .apt_install(["zlib1g"])
)

# HACK: we need the aio version of the pymc app, so we can merge the sample processes
# as async generators.
aio_pymc_stub = synchronizer._translate_out(synchronizer._translate_in(pymc_stub), Interface.ASYNC)

if aio_pymc_stub.is_inside():
    import numpy as np
    from fastprogress.fastprogress import progress_bar
    from pymc3 import theanof
    from pymc3.exceptions import SamplingError


class ParallelSamplingError(Exception):
    def __init__(self, message, chain, warnings=None):
        super().__init__(message)
        if warnings is None:
            warnings = []
        self._chain = chain
        self._warnings = warnings


# Taken from https://hg.python.org/cpython/rev/c4f92b597074
class RemoteTraceback(Exception):
    def __init__(self, tb):
        self.tb = tb

    def __str__(self):
        return self.tb


class ExceptionWithTraceback:
    def __init__(self, exc, tb):
        tb = traceback.format_exception(type(exc), exc, tb)
        tb = "".join(tb)
        self.exc = exc
        self.tb = '\n"""\n%s"""' % tb

    def __reduce__(self):
        return rebuild_exc, (self.exc, self.tb)


def rebuild_exc(exc, tb):
    exc.__cause__ = RemoteTraceback(tb)
    return exc


@aio_pymc_stub.generator
async def sample_process(
    draws: int,
    tune: int,
    step_method,
    chain: int,
    seed,
    start,  # Dict[str, np.ndarray],
):
    tt_seed = seed + 1

    np.random.seed(seed)
    theanof.set_tt_rng(tt_seed)

    point = {}
    stats = None

    for name, (shape, dtype) in step_method.vars_shape_dtype.items():
        size = 1
        for dim in shape:
            size *= int(dim)
        size *= dtype.itemsize
        if size != ctypes.c_size_t(size).value:
            raise ValueError("Variable %s is too large" % name)

        array = bytearray(size)
        point[name] = (array, shape, dtype)

        array_np = np.frombuffer(array, dtype).reshape(shape)
        array_np[...] = start[name]
        point[name] = array_np  # type: ignore

    def compute_point():
        nonlocal point, stats
        if step_method.generates_stats:
            point, stats = step_method.step(point)
        else:
            point = step_method.step(point)
            stats = None

    def collect_warnings():
        if hasattr(step_method, "warnings"):
            return step_method.warnings()
        else:
            return []

    draw = 0
    tuning = True

    while True:
        if draw == tune:
            step_method.stop_tuning()
            tuning = False

        if draw < draws + tune:
            try:
                compute_point()
            except SamplingError as e:
                warns = collect_warnings()
                e = ExceptionWithTraceback(e, e.__traceback__)
                raise e  # TODO: deal w/ warns
        else:
            return

        is_last = draw + 1 == draws + tune
        if is_last:
            warns = collect_warnings()
        else:
            warns = None
        yield point, is_last, draw, tuning, stats, warns, chain
        draw += 1


Draw = namedtuple("Draw", ["chain", "is_last", "draw_idx", "tuning", "stats", "point", "warnings"])


class _ModalSampler:
    def __init__(
        self,
        draws: int,
        tune: int,
        chains: int,
        cores: int,
        seeds: list,
        start_points,  # Sequence[Dict[str, np.ndarray]],
        step_method,
        start_chain_num: int = 0,
        progressbar: bool = True,
        mp_ctx=None,
        pickle_backend: str = "pickle",
    ):

        if any(len(arg) != chains for arg in [seeds, start_points]):
            raise ValueError("Number of seeds and start_points must be %s." % chains)

        self._finished: List[Any] = []
        self._active: List[Any] = []
        self._max_active = cores

        self._in_context = False
        self._start_chain_num = start_chain_num

        self._progress = None
        self._divergences = 0
        self._total_draws = 0
        self._desc = "Sampling {0._chains:d} chains, {0._divergences:,d} divergences"
        self._chains = chains
        if progressbar:
            self._progress = progress_bar(range(chains * (draws + tune)), display=progressbar)
            self._progress.comment = self._desc.format(self)

            # HACK: fastprogress checks sys.stdout.isatty(), and there's no way to override.
            sys.stdout.isatty = lambda: True  # type: ignore

        self._draws = draws
        self._tune = tune
        self._step_method = step_method
        self._seeds = seeds
        self._start_points = start_points

    async def __aiter__(self):
        samplers = [
            sample_process(
                self._draws,
                self._tune,
                self._step_method,
                chain + self._start_chain_num,
                seed,
                start,
            )
            for chain, seed, start in zip(range(self._chains), self._seeds, self._start_points)
        ]
        print(f"{len(samplers)} samplers")

        if not self._in_context:
            raise ValueError("Use ParallelSampler as context manager.")

        merged_samplers = stream.merge(*samplers)
        if self._progress:
            self._progress.update(self._total_draws)

        async with merged_samplers.stream() as streamer:
            async for result in streamer:
                point, is_last, draw, tuning, stats, warns, chain = result
                self._total_draws += 1
                if not tuning and stats and stats[0].get("diverging"):
                    self._divergences += 1
                    if self._progress:
                        self._progress.comment = self._desc.format(self)
                if self._progress:
                    self._progress.update(self._total_draws)

                yield Draw(chain, is_last, draw, tuning, stats, point, warns)

    def __enter__(self):
        self._in_context = True
        return self

    def __exit__(self, *args):
        pass


ModalSampler, _ = synchronize_apis(_ModalSampler)
