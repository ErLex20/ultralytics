# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Hardware-first building blocks for the YOLO-DLA model family."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .conv import Conv, RepConv
from .head import DetectDLA

__all__ = "DLALocalContext", "DLARepCSP", "YOLODLADetect"


class DLARepCSP(nn.Module):
    """CSP block whose training-time RepConv branches fuse to plain 3x3 convolutions.

    Unlike split-based CSP implementations, both branches use independent 1x1
    projections. This keeps the exported graph four-dimensional and avoids ONNX
    ``Slice``/shape-helper nodes that can fragment a TensorRT DLA loadable.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5) -> None:
        """Initialize a DLA-safe re-parameterizable CSP block.

        Args:
            c1: Number of input channels.
            c2: Number of output channels.
            n: Number of RepConv units in the transformed branch.
            e: Hidden-channel expansion ratio.
        """
        super().__init__()
        hidden = max(int(c2 * e), 16)
        self.short = Conv(c1, hidden, 1, 1)
        self.main = Conv(c1, hidden, 1, 1)
        self.blocks = nn.Sequential(
            *(RepConv(hidden, hidden, 3, 1, bn=True, act=Conv.default_act) for _ in range(n))
        )
        self.fuse = Conv(2 * hidden, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process two projected branches and fuse them along the channel axis."""
        return self.fuse(torch.cat((self.short(x), self.blocks(self.main(x))), dim=1))


class DLALocalContext(nn.Module):
    """Data-dependent local context mixer using only DLA-native tensor operations.

    The block replaces global self-attention with a depthwise spatial gate. It
    uses Conv, Sigmoid, ElementWise Product, Add, and a final projection; all
    tensors remain static-rank NCHW. There is no MatMul, global reduction,
    channel split, or Softmax.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 1.0) -> None:
        """Initialize one or more residual local-context mixers.

        Args:
            c1: Number of input channels.
            c2: Number of output channels.
            n: Number of residual mixer stages.
            e: Hidden-channel expansion ratio.
        """
        super().__init__()
        hidden = max(int(c2 * e), 16)
        self.input = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.mixers = nn.ModuleList(_DLALocalMixer(c2, hidden) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual local-context mixing."""
        x = self.input(x)
        for mixer in self.mixers:
            x = x + mixer(x)
        return x


class _DLALocalMixer(nn.Module):
    """Single local-context stage used internally by :class:`DLALocalContext`."""

    def __init__(self, channels: int, hidden: int) -> None:
        """Initialize value, gate, and output projections."""
        super().__init__()
        self.value = Conv(channels, hidden, 1, 1)
        self.gate = nn.Sequential(
            Conv(channels, channels, 5, 1, g=channels, act=False),
            nn.Sigmoid(),
        )
        self.output = Conv(hidden, channels, 1, 1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gate locally mixed values with an input-dependent spatial mask."""
        return self.output(self.value(x) * self.gate(x))


class YOLODLADetect(DetectDLA):
    """NMS-free YOLO-DLA detection head with re-parameterizable Conv towers.

    Training uses the standard Ultralytics one-to-many and one-to-one branches.
    Model fusion removes the one-to-many branch, while export inherits the
    ``DetectDLA`` contract and emits one packed four-dimensional tensor per
    pyramid level. The model YAML selects ``reg_max=1`` so no DFL operation is
    required by the deployment decoder.
    """

    tower_reps = 2  # RepConv units per tower (ablation knob)
    cls_width = None  # classification tower width; None uses max(64, min(nc, 128))

    def __init__(self, nc: int = 80, reg_max: int = 1, end2end: bool = True, ch: tuple = ()) -> None:
        """Initialize hardware-aligned box and classification towers."""
        super().__init__(nc, reg_max, end2end, ch)
        box_channels = max(32, ch[0] // 2, 4 * reg_max)
        cls_channels = self.cls_width or max(64, min(nc, 128))
        self.cv2 = nn.ModuleList(self._make_tower(x, box_channels, 4 * reg_max) for x in ch)
        self.cv3 = nn.ModuleList(self._make_tower(x, cls_channels, nc) for x in ch)
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @classmethod
    def _make_tower(cls, c1: int, hidden: int, c2: int) -> nn.Sequential:
        """Build a train-time RepConv tower that becomes plain convolutions after fusion."""
        return nn.Sequential(
            Conv(c1, hidden, 1, 1),
            *(RepConv(hidden, hidden, 3, 1, bn=True, act=Conv.default_act) for _ in range(cls.tower_reps)),
            nn.Conv2d(hidden, c2, 1),
        )
