"""Physical device selection and extensible, execution-only hardware profiles."""

import copy

from pvz_rl.config import _deep_update, _read_config


def select_gpu(selection="0"):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")
    devices = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    text = str(selection)
    if text.isdecimal():
        index = int(text)
        if index >= len(devices):
            raise ValueError(f"Visible GPU index {index} is unavailable")
    else:
        matches = [i for i, p in enumerate(devices)
                   if text.removeprefix("GPU-") == str(p.uuid).removeprefix("GPU-")]
        if len(matches) != 1:
            raise ValueError("--gpu must be a visible index or a complete visible CUDA UUID")
        index = matches[0]
    torch.cuda.set_device(index)
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("The selected GPU must support native BF16")
    from pvz_game.cuda.backend import cupy_runtime

    cp = cupy_runtime()
    cp.cuda.Device(index).use()
    p = devices[index]
    return dict(index=index, uuid=str(p.uuid), name=p.name, memory_mib=p.total_memory / 2**20,
                capability=list(torch.cuda.get_device_capability(index)))


def apply_gpu_profile(cfg, name="auto", *, device=None, path=None):
    """Profiles never change weights, learning settings or collector topology."""
    catalogue = _read_config(path).get("gpu_profiles", {})
    # A user configuration may intentionally contain only research settings.
    # Keep the generic native-BF16 interface available without requiring every
    # custom TOML to copy the bundled catalogue.
    catalogue = copy.deepcopy(catalogue)
    catalogue.setdefault("generic-bf16", _read_config(None)["gpu_profiles"]["generic-bf16"])
    if name == "auto":
        name = next((key for key, value in catalogue.items()
                     if device and device["name"] in value.get("device_names", [])), "generic-bf16")
    if name not in catalogue:
        raise ValueError(f"Unknown GPU profile {name!r}; available: {', '.join(catalogue)}")
    profile = catalogue[name]
    allowed = {"device_names", "status", "policy", "performance", "torch_threads"}
    if set(profile) - allowed:
        raise ValueError("GPU profiles may contain only execution settings and match metadata")
    from pvz_rl.learning.performance import ENCODER_SETTINGS, PERFORMANCE_SETTINGS

    if set(profile.get("policy", {})) - set(ENCODER_SETTINGS):
        raise ValueError("GPU profile cannot change the model architecture")
    if set(profile.get("performance", {})) - set(PERFORMANCE_SETTINGS):
        raise ValueError("GPU profile contains unsupported execution settings")
    result = copy.deepcopy(cfg)
    _deep_update(result["policy"], profile.get("policy", {}))
    _deep_update(result["training"]["performance"], profile.get("performance", {}))
    result["training"]["torch_threads"] = profile.get("torch_threads", cfg["training"]["torch_threads"])
    return result, name


def apply_learning_profile(cfg, name="current", *, path=None):
    result = copy.deepcopy(cfg)
    if name == "current":
        return result
    catalogue = _read_config(path).get("learning_profiles", {})
    if name not in catalogue:
        raise ValueError(f"Unknown learning profile {name!r}")
    # Keep hardware calibration separate from experimental protocol choices.
    _deep_update(result["training"], catalogue[name])
    return result
