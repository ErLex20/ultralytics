# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CPU post-processing helpers for raw four-dimensional DLA head outputs."""

from __future__ import annotations

from typing import Any

import torch


def decode_dla_outputs(
    outputs: list[torch.Tensor] | tuple[torch.Tensor, ...],
    strides: list[float] | tuple[float, ...] | torch.Tensor,
    reg_max: int,
    nc: int,
    *,
    nm: int = 0,
    end2end: bool,
    max_det: int = 300,
    agnostic: bool = False,
    device: str | torch.device | None = "cpu",
) -> torch.Tensor:
    """Decode packed DLA P3/P4/P5 maps into the standard Ultralytics prediction format.

    Args:
        outputs: Packed maps with shape ``(B, 4 * reg_max + nc, H, W)``.
        strides: Feature-pyramid strides in output order.
        reg_max: Number of regression bins; one selects direct DFL-free distances.
        nc: Number of classes.
        nm: Number of instance-mask coefficients packed after class logits.
        end2end: Whether outputs come from the one-to-one branch.
        max_det: Maximum number of one-to-one predictions returned.
        agnostic: Select only the best class per anchor before ranking.
        device: Post-processing device. The default moves DLA outputs to CPU.

    Returns:
        For one-to-one heads, a ``(B, K, 6 + nm)`` xyxy tensor ready for the
        normal end-to-end predictor path. Otherwise, a ``(B, 4 + nc + nm, A)``
        xywh tensor ready for the standard NMS path.
    """
    from ultralytics.utils.tal import dist2bbox, make_anchors

    if len(outputs) != len(strides):
        raise ValueError(f"Expected {len(strides)} DLA prediction maps, received {len(outputs)}")
    if reg_max < 1:
        raise ValueError(f"reg_max must be positive, received {reg_max}")

    maps = [x.detach().to(device=device, dtype=torch.float32) for x in outputs]
    box_channels = 4 * reg_max
    expected_channels = box_channels + nc + nm
    for i, prediction in enumerate(maps):
        if prediction.ndim != 4 or prediction.shape[1] < expected_channels:
            raise ValueError(
                f"Invalid DLA output {i}: expected (B, >= {expected_channels}, H, W), got {tuple(prediction.shape)}"
            )

    boxes_4d = [x[:, :box_channels] for x in maps]
    scores_4d = [x[:, box_channels : box_channels + nc] for x in maps]
    masks_4d = [x[:, box_channels + nc : expected_channels] for x in maps] if nm else []
    stride_tensor = torch.as_tensor(strides, dtype=torch.float32, device=maps[0].device)
    anchors, stride_per_anchor = (x.transpose(0, 1) for x in make_anchors(boxes_4d, stride_tensor, 0.5))

    batch = maps[0].shape[0]
    boxes = torch.cat([x.reshape(batch, box_channels, -1) for x in boxes_4d], dim=-1)
    scores = torch.cat([x.reshape(batch, nc, -1) for x in scores_4d], dim=-1).sigmoid()
    masks = torch.cat([x.reshape(batch, nm, -1) for x in masks_4d], dim=-1) if nm else None
    if reg_max > 1:
        anchors_count = boxes.shape[-1]
        distribution = boxes.reshape(batch, 4, reg_max, anchors_count).softmax(dim=2)
        projection = torch.arange(reg_max, dtype=distribution.dtype, device=distribution.device).view(1, 1, -1, 1)
        boxes = (distribution * projection).sum(dim=2)

    decoded = dist2bbox(boxes, anchors.unsqueeze(0), xywh=not end2end, dim=1) * stride_per_anchor
    predictions = (
        torch.cat((decoded, scores, masks), dim=1) if masks is not None else torch.cat((decoded, scores), dim=1)
    )
    return _rank_one2one(predictions, nc, max_det, agnostic) if end2end else predictions


def decode_dla_backend_outputs(
    outputs: Any,
    model: Any,
    *,
    max_det: int,
    agnostic: bool = False,
) -> Any:
    """Decode backend outputs only when export metadata identifies a raw DLA head."""
    if not getattr(model, "dla_raw", False):
        return outputs
    if isinstance(outputs, torch.Tensor):
        return outputs  # Already decoded by a task-specific predictor before calling its detection base class.
    if not isinstance(outputs, (list, tuple)):
        raise TypeError(f"Raw DLA metadata requires a list of output maps, got {type(outputs).__name__}")
    count = int(getattr(model, "dla_outputs", 3))
    nm = int(getattr(model, "nm", 0))
    predictions = decode_dla_outputs(
        outputs[:count],
        getattr(model, "dla_strides", (8.0, 16.0, 32.0)),
        int(getattr(model, "reg_max", 1)),
        len(model.names),
        nm=nm,
        end2end=bool(getattr(model, "end2end", False)),
        max_det=max_det,
        agnostic=agnostic,
    )
    if nm:
        if len(outputs) <= count:
            raise ValueError("Raw DLA segmentation output is missing its prototype tensor")
        return predictions, outputs[count].detach().to(device="cpu", dtype=torch.float32)
    return predictions


def _rank_one2one(predictions: torch.Tensor, nc: int, max_det: int, agnostic: bool) -> torch.Tensor:
    """Apply the Ultralytics one-to-one ranking contract outside the DLA engine."""
    predictions = predictions.transpose(1, 2)
    boxes, scores, extra = predictions.split((4, nc, predictions.shape[-1] - 4 - nc), dim=-1)
    batch, anchors, classes = scores.shape
    k = min(max_det, anchors)

    if agnostic:
        confidence, labels = scores.max(dim=-1, keepdim=True)
        confidence, indices = confidence.topk(k, dim=1)
        labels = labels.gather(dim=1, index=indices)
        boxes = boxes.gather(dim=1, index=indices.expand(-1, -1, 4))
        extra = extra.gather(dim=1, index=indices.expand(-1, -1, extra.shape[-1]))
        return torch.cat((boxes, confidence, labels.float(), extra), dim=-1)

    anchor_indices = scores.amax(dim=-1).topk(k, dim=1).indices.unsqueeze(-1)
    selected_scores = scores.gather(dim=1, index=anchor_indices.expand(-1, -1, classes))
    confidence, class_indices = selected_scores.flatten(1).topk(k, dim=1)
    box_indices = anchor_indices[torch.arange(batch, device=scores.device)[:, None], class_indices // classes]
    boxes = boxes.gather(dim=1, index=box_indices.expand(-1, -1, 4))
    extra = extra.gather(dim=1, index=box_indices.expand(-1, -1, extra.shape[-1]))
    return torch.cat(
        (boxes, confidence.unsqueeze(-1), (class_indices % classes).float().unsqueeze(-1), extra), dim=-1
    )
