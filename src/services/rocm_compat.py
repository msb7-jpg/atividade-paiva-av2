import os

import torch
import torch.nn.functional as F


def apply_rocm_windows_workarounds() -> None:
    """Contorna o BatchNorm do MIOpen nos wheels ROCm para Windows.

    O MIOpen compila o kernel de BatchNorm em tempo de execução (HIPRTC), e no Windows
    essa compilação falha com "'type_traits' file not found" -> miopenStatusUnknownError
    (ROCm/TheRock#9206). Roteia apenas o BatchNorm para o kernel HIP nativo do PyTorch;
    as convoluções continuam no MIOpen. Sem efeito em CUDA ou Linux.
    """
    if os.name != "nt" or torch.version.hip is None:
        return
    if getattr(F.batch_norm, "_rocm_compat", False):
        return

    original_batch_norm = F.batch_norm

    def batch_norm_without_miopen(*args, **kwargs):
        with torch.backends.cudnn.flags(enabled=False):
            return original_batch_norm(*args, **kwargs)

    batch_norm_without_miopen._rocm_compat = True
    F.batch_norm = batch_norm_without_miopen
