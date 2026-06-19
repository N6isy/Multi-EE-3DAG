"""Adapter for using PointNeXt/OpenPoints as a per-point backbone.

The adapter keeps the project training contract simple:

    input:  points [B, N, 3] or [B, N, 6]
    output: point features [B, N, C]

OpenPoints remains an external dependency under `MultiEEAffordance/external`
or `/home/lzq/data/MultiEEAffordance/external` and is not vendored into the
training package.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


class AttrDict(dict):
    """Small dict with attribute access for OpenPoints config fragments."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _attrdict(value: dict[str, Any] | None) -> AttrDict:
    result = AttrDict()
    for key, item in (value or {}).items():
        result[key] = _attrdict(item) if isinstance(item, dict) else item
    return result


def _default_pointnext_root() -> Path:
    return Path(__file__).resolve().parents[2] / "external" / "backbones" / "PointNeXt-master"


def _resolve_pointnext_root(pointnext_root: str | Path | None) -> Path:
    root = Path(pointnext_root).expanduser() if pointnext_root else _default_pointnext_root()
    return root.resolve()


def _ensure_openpoints_importable(pointnext_root: Path) -> None:
    if not pointnext_root.exists():
        raise FileNotFoundError(
            f"PointNeXt root does not exist: {pointnext_root}. "
            "Set config field `pointnext_root` to the external PointNeXt/OpenPoints directory."
        )
    if not (pointnext_root / "openpoints").exists():
        raise FileNotFoundError(f"Missing openpoints package under PointNeXt root: {pointnext_root}")
    root_text = str(pointnext_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


class PointNeXtBackbone(nn.Module):
    """Thin wrapper around OpenPoints PointNextEncoder + PointNextDecoder."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        pointnext_root: str | Path | None = None,
        width: int = 32,
        blocks: list[int] | None = None,
        strides: list[int] | None = None,
        nsample: int | list[int] = 32,
        radius: float | list[float] = 0.1,
        decoder_layers: int = 2,
        decoder_stages: int = 4,
        sa_layers: int = 1,
        sa_use_res: bool = False,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.pointnext_root = _resolve_pointnext_root(pointnext_root)
        _ensure_openpoints_importable(self.pointnext_root)
        try:
            from openpoints.models.backbone.pointnext import PointNextDecoder, PointNextEncoder
        except Exception as exc:  # pragma: no cover - depends on external compiled deps.
            raise RuntimeError(
                "Failed to import OpenPoints PointNeXt. Install the PointNeXt/OpenPoints "
                "dependencies in the active environment, then rerun the backbone smoke test."
            ) from exc

        blocks = [1, 1, 1, 1, 1] if blocks is None else [int(item) for item in blocks]
        strides = [1, 2, 2, 2, 2] if strides is None else [int(item) for item in strides]
        if len(blocks) != len(strides):
            raise ValueError(f"PointNeXt blocks/strides length mismatch: {blocks} vs {strides}")
        group_args = _attrdict({"NAME": "ballquery"})
        aggr_args = _attrdict({"feature_type": "dp_fj", "reduction": "max"})
        norm_args = _attrdict({"norm": "bn"})
        act_args = _attrdict({"act": "relu"})
        conv_args = _attrdict({"order": "conv-norm-act"})
        self.encoder = PointNextEncoder(
            in_channels=self.input_channels,
            width=int(width),
            blocks=blocks,
            strides=strides,
            nsample=nsample,
            radius=radius,
            group_args=group_args,
            aggr_args=aggr_args,
            norm_args=norm_args,
            act_args=act_args,
            conv_args=conv_args,
            sa_layers=int(sa_layers),
            sa_use_res=bool(sa_use_res),
        )
        self.decoder = PointNextDecoder(
            encoder_channel_list=self.encoder.channel_list,
            decoder_layers=int(decoder_layers),
            decoder_stages=int(decoder_stages),
            in_channels=self.input_channels,
        )
        self.out_channels = int(self.decoder.out_channels)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        coords = points[:, :, :3].contiguous()
        features = points[:, :, : self.input_channels].transpose(1, 2).contiguous()
        p, f = self.encoder.forward_seg_feat(coords, features)
        out = self.decoder(p, f)
        if out.ndim == 4 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        if out.ndim != 3:
            raise RuntimeError(f"Unexpected PointNeXt output shape: {tuple(out.shape)}")
        return out.transpose(1, 2).contiguous()
