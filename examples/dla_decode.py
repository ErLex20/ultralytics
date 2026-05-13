"""External decode for the headless DLA-exported YOLO11 engine.

The DLA-compatible head (`DetectDLA` / `SegmentDLA` in export mode) emits raw
4-D feature maps so every op in the engine stays DLA-compatible. The decode
that was previously baked into the engine (DFL → dist2bbox → sigmoid → flatten)
must therefore run externally on GPU after the engine returns.

Engine output order for `SegmentDLA` (10 tensors):

    0: box_P3         (B, 4*reg_max, H_P3, W_P3)   # e.g. (1, 64, 64, 64)
    1: score_P3       (B, nc,         H_P3, W_P3)   # e.g. (1, 80, 64, 64)
    2: box_P4         (B, 4*reg_max, H_P4, W_P4)
    3: score_P4       (B, nc,         H_P4, W_P4)
    4: box_P5         (B, 4*reg_max, H_P5, W_P5)
    5: score_P5       (B, nc,         H_P5, W_P5)
    6: proto          (B, nm,        2*H_P3, 2*W_P3)
    7: mask_P3        (B, nm,         H_P3, W_P3)
    8: mask_P4        (B, nm,         H_P4, W_P4)
    9: mask_P5        (B, nm,         H_P5, W_P5)

For `DetectDLA` the engine returns only outputs 0-5.

Run `python dla_decode.py --verify` to check numerical parity against the
standard 3-D `Detect._inference` path on a synthetic input.
"""

from __future__ import annotations

import argparse

import torch


def decode_heads(
    boxes_per_scale: list[torch.Tensor],
    scores_per_scale: list[torch.Tensor],
    strides: torch.Tensor,
    reg_max: int = 16,
    nc: int = 80,
) -> torch.Tensor:
    """Decode the 6 raw 4-D heads from a headless engine into `(B, 4+nc, num_anchors)`.

    Args:
        boxes_per_scale: list of 3 box maps, each `(B, 4*reg_max, H_i, W_i)`.
        scores_per_scale: list of 3 score maps, each `(B, nc, H_i, W_i)`.
        strides: tensor of strides per scale, e.g. `torch.tensor([8., 16., 32.])`.
        reg_max: DFL bin count (16 for YOLO11-n).
        nc: number of classes.

    Returns:
        `(B, 4+nc, total_anchors)` — xywh box centres followed by class probabilities.
    """
    from ultralytics.utils.tal import dist2bbox, make_anchors

    bs = boxes_per_scale[0].shape[0]
    device, dtype = boxes_per_scale[0].device, boxes_per_scale[0].dtype

    anchors, strides_t = (a.transpose(0, 1) for a in make_anchors(boxes_per_scale, strides, 0.5))

    boxes_flat = torch.cat([b.view(bs, 4 * reg_max, -1) for b in boxes_per_scale], dim=-1)
    scores_flat = torch.cat([s.view(bs, nc, -1) for s in scores_per_scale], dim=-1)

    b, _, a = boxes_flat.shape
    dist = boxes_flat.view(b, 4, reg_max, a).softmax(2)
    dfl_w = torch.arange(reg_max, dtype=dtype, device=device).view(1, 1, reg_max, 1)
    dist = (dist * dfl_w).sum(2)

    dbox = dist2bbox(dist, anchors.unsqueeze(0), xywh=True, dim=1) * strides_t
    return torch.cat([dbox, scores_flat.sigmoid()], dim=1)


def decode_masks(
    masks_per_scale: list[torch.Tensor],
    nm: int = 32,
) -> torch.Tensor:
    """Flatten + concat the 3 mask-coefficient feature maps into `(B, nm, num_anchors)`."""
    bs = masks_per_scale[0].shape[0]
    return torch.cat([m.view(bs, nm, -1) for m in masks_per_scale], dim=-1)


def _verify() -> None:
    """Numerical-parity test: external decode vs standard 3-D `Detect._inference`."""
    from ultralytics import YOLO

    torch.manual_seed(0)
    model = YOLO("ultralytics/cfg/models/11/yolo11-dla-seg.yaml").model
    model.eval()
    head = model.model[-1]
    print(f"Head: {head.__class__.__name__}  nc={head.nc} reg_max={head.reg_max} nm={head.nm}")

    x = torch.randn(1, 3, 512, 512)

    head.export = False
    with torch.no_grad():
        (y_ref, _proto_ref), _preds = model(x)
    head.export = True
    with torch.no_grad():
        outs = model(x)
    head.export = False

    boxes_per_scale = [outs[0], outs[2], outs[4]]
    scores_per_scale = [outs[1], outs[3], outs[5]]
    proto_dla = outs[6]
    masks_per_scale = [outs[7], outs[8], outs[9]]

    y_dla = decode_heads(boxes_per_scale, scores_per_scale, head.stride, reg_max=head.reg_max, nc=head.nc)
    mask_coeff_dla = decode_masks(masks_per_scale, nm=head.nm)
    y_dla_full = torch.cat([y_dla, mask_coeff_dla], dim=1)

    diff = (y_ref - y_dla_full).abs()
    print(f"Reference shape:  {tuple(y_ref.shape)}")
    print(f"DLA-decoded shape: {tuple(y_dla_full.shape)}")
    print(f"Max abs diff: {diff.max().item():.3e}   Mean abs diff: {diff.mean().item():.3e}")
    proto_diff = (_proto_ref - proto_dla).abs().max().item()
    print(f"Proto max abs diff: {proto_diff:.3e}")
    if diff.max().item() < 1e-4 and proto_diff < 1e-4:
        print("PASS — external decode matches in-engine decode")
    else:
        print("FAIL — numerical mismatch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Numerical-parity test vs Detect._inference")
    args = parser.parse_args()
    if args.verify:
        _verify()
    else:
        parser.print_help()
