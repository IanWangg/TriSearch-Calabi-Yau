"""Opt-in CPU startup for CUDA-enabled PyTorch/PyG installations.

Prepend this directory to PYTHONPATH with TRISEARCH_CPU_ONLY=1. Spawned Python
workers inherit it. No installed packages are changed. PyG otherwise calls
torch.__config__.show() during import, which can probe an unhealthy GPU driver,
and prefers its CUDA extensions even when CUDA_VISIBLE_DEVICES is empty.
"""

import os

if os.environ.get("TRISEARCH_CPU_ONLY") == "1":
    import importlib.machinery

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _original_find_spec = importlib.machinery.PathFinder.find_spec
    _pyg_extensions = ("torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv")

    @classmethod
    def _cpu_find_spec(cls, fullname, path=None, target=None):
        if fullname.endswith("_cuda") and path and any(
            package in str(directory) for directory in path for package in _pyg_extensions
        ):
            return None
        return _original_find_spec(fullname, path, target)

    importlib.machinery.PathFinder.find_spec = _cpu_find_spec

    import torch

    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0

    def _reject_cuda(*args, **kwargs):
        raise RuntimeError("CUDA is disabled by TRISEARCH_CPU_ONLY.")

    torch.cuda._lazy_init = _reject_cuda
    # PyG and Dynamo inspect USE_MKL through this function. The native build
    # report probes the driver, so expose only the runtime capability they need,
    # explicitly identifying this as an abbreviated CPU-runtime report.
    torch.__config__.show = lambda: "TriSearch CPU runtime (abbreviated); USE_MKL=" + (
        "ON" if torch.backends.mkl.is_available() else "OFF")
