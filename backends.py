import os

import numpy


def convert_to_numpy(arr, backend, device="cpu"):
    """Converts an array or collection of arrays to np.ndarray"""
    if isinstance(arr, (list, tuple)):
        return [convert_to_numpy(subarr, backend, device) for subarr in arr]

    if type(arr) is numpy.ndarray:
        # this is stricter than isinstance,
        # we don't want subclasses to get passed through
        return arr

    if backend == "cupy":
        return arr.get()

    if backend == "jax":
        return numpy.asarray(arr)

    if backend == "pytorch":
        if device == "gpu":
            return numpy.asarray(arr.cpu())
        else:
            return numpy.asarray(arr)

    if backend == "tensorflow":
        return numpy.asarray(arr)

    if backend == "aesara":
        return numpy.asarray(arr)

    if backend == "taichi":
        return arr.to_numpy()

    raise RuntimeError(
        f"Got unexpected array / backend combination: {type(arr)} / {backend}"
    )


class BackendNotSupported(Exception):
    pass


class BackendConflict(Exception):
    pass


def check_backend_conflicts(backends, device):
    if device == "gpu":
        gpu_backends = set(backends) - {"numba", "numpy", "aesara"}
        if len(gpu_backends) > 1:
            raise BackendConflict(
                f"Can only use one GPU backend at the same time (got: {gpu_backends})"
            )


class SetupContext:
    def __init__(self, f):
        self._f = f
        self._f_args = (tuple(), dict())

    def __call__(self, *args, **kwargs):
        self._f_args = (args, kwargs)
        return self

    def __enter__(self):
        self._env = os.environ.copy()
        args, kwargs = self._f_args
        self._f_iter = iter(self._f(*args, **kwargs))

        try:
            module = next(self._f_iter)
        except Exception as e:
            raise BackendNotSupported(str(e)) from None

        return module

    def __exit__(self, *args, **kwargs):
        try:
            next(self._f_iter)
        except StopIteration:
            pass
        os.environ = self._env


setup_function = SetupContext


# setup function definitions


@setup_function
def setup_cupy(device="cpu"):
    if device != "gpu":
        raise RuntimeError("cupy requires GPU mode")
    import cupy

    yield cupy


@setup_function
def setup_jax(device="cpu"):
    os.environ.update(
        XLA_FLAGS=(
            "--xla_cpu_multi_thread_eigen=false "
            "intra_op_parallelism_threads=1 "
            "inter_op_parallelism_threads=1 "
        ),
    )

    if device in ("cpu", "gpu"):
        os.environ.update(JAX_PLATFORM_NAME=device)

    import jax

    if device == "tpu":
        jax.config.update("jax_xla_backend", "tpu_driver")
        jax.config.update("jax_backend_target", os.environ.get("JAX_BACKEND_TARGET"))

    if device != "tpu":
        # use 64 bit floats (not supported on TPU)
        jax.config.update("jax_enable_x64", True)

    if device == "gpu":
        assert len(jax.devices()) > 0

    yield jax


@setup_function
def setup_pytorch(device="cpu"):
    os.environ.update(
        OMP_NUM_THREADS="1",
    )
    import torch

    if device == "gpu":
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() > 0

    yield torch




TAICHI_SETUP_DONE = False

@setup_function
def setup_taichi(device="cpu"):
    global TAICHI_SETUP_DONE
    import taichi

    if not TAICHI_SETUP_DONE:
        taichi.init(
            arch=taichi.cpu if device == "cpu" else taichi.gpu, 
            cpu_max_num_threads=1,
            default_fp=taichi.f64,
        )
        TAICHI_SETUP_DONE = True

    yield taichi

__backends__ = {
    "cupy": setup_cupy,
    "jax": setup_jax,
    "pytorch": setup_pytorch,
    "taichi": setup_taichi,
}
