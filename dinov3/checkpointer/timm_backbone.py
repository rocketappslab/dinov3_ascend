# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Load a timm DINOv3 backbone (safetensors) into an FSDP student."""

import json
import logging
import struct
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

logger = logging.getLogger("dinov3")

_CHECKPOINT_PREFIX = "_checkpoint_wrapped_module."
_SAFETENSORS_DTYPE = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def resolve_timm_weights(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path)
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"timm weights not found: {checkpoint_path}")
    return path


def remap_timm_key(key: str) -> str | None:
    """Map a timm DINOv3 key onto this repo's student backbone."""
    if key == "reg_token":
        return "backbone.storage_tokens"
    if key in ("cls_token",) or key.startswith(("patch_embed.", "norm.")):
        return f"backbone.{key}"
    if not key.startswith("blocks."):
        return None
    key = key.replace(".gamma_1", ".ls1.gamma")
    key = key.replace(".gamma_2", ".ls2.gamma")
    key = key.replace(".mlp.fc1_g.", ".mlp.w1.")
    key = key.replace(".mlp.fc1_x.", ".mlp.w2.")
    key = key.replace(".mlp.fc2.", ".mlp.w3.")
    return f"backbone.{key}"


def canonical_parameter_names(model: nn.Module) -> set[str]:
    return {name.replace(_CHECKPOINT_PREFIX, "") for name, _ in model.named_parameters()}


def _read_safetensors(path: Path, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    tensors = {}
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        data_start = 8 + header_len
        for key, info in header.items():
            if key == "__metadata__":
                continue
            mapped = remap_timm_key(key)
            if mapped is None:
                logger.warning(f"Skipping unrecognized timm key {key}")
                continue
            start, end = info["data_offsets"]
            handle.seek(data_start + start)
            raw = handle.read(end - start)
            source_dtype = _SAFETENSORS_DTYPE[info["dtype"]]
            tensor = torch.frombuffer(bytearray(raw), dtype=source_dtype).reshape(info["shape"]).clone()
            if tensor.dtype != dtype:
                tensor = tensor.to(dtype=dtype)
            tensors[mapped] = tensor
    return tensors


def load_timm_dinov3_backbone(model: nn.Module, checkpoint_path: str | Path) -> None:
    """Load timm DINOv3 backbone weights into an FSDP ``student`` module.

    DINO and iBOT heads are absent from the timm checkpoint and stay as initialized.
    ``local_cls_norm`` is also absent; it is copied from the pretrained final norm.
    Rank 0 reads the file and the tensors are broadcast into the FSDP shards.
    """
    path = resolve_timm_weights(checkpoint_path)
    names = canonical_parameter_names(model)
    param_dtype = next(model.parameters()).dtype
    rank = dist.get_rank()
    logger.info(f"Loading timm DINOv3 backbone from {path}")

    state_dict: dict[str, torch.Tensor] = {}
    if rank == 0:
        state_dict = _read_safetensors(path, param_dtype)
        if "backbone.local_cls_norm.weight" in names and "backbone.norm.weight" in state_dict:
            state_dict["backbone.local_cls_norm.weight"] = state_dict["backbone.norm.weight"].clone()
            state_dict["backbone.local_cls_norm.bias"] = state_dict["backbone.norm.bias"].clone()
        unexpected = sorted(key for key in state_dict if key not in names)
        for key in unexpected:
            state_dict.pop(key)
        missing_backbone = sorted(
            key for key in names if key.startswith("backbone.") and key not in state_dict and "mask_token" not in key
        )
        logger.info(f"timm backbone tensors matched: {len(state_dict)}")
        if unexpected:
            logger.warning(f"Dropped {len(unexpected)} timm keys with no parameter, e.g. {unexpected[:8]}")
        if missing_backbone:
            logger.warning(f"Backbone parameters left at init: {missing_backbone}")
        if len(state_dict) < 100:
            raise RuntimeError(f"Only matched {len(state_dict)} tensors from {path}")

    incompatible = set_model_state_dict(
        model,
        state_dict,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            strict=False,
        ),
    )
    if rank == 0:
        logger.info(f"Loaded timm backbone with msg: {incompatible}")
