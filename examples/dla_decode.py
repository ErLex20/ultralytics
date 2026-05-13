"""External decode for the DLA-exported YOLO11 engine (packed-channel layout).

The DLA-compatible head (`DetectDLA` / `SegmentDLA` in export mode) packs all
per-anchor predictions for each pyramid scale into one 4-D tensor so every op
in the engine stays DLA-compatible. The decode that was previously baked into
the engine (DFL → dist2bbox → sigmoid → flatten) must therefore run externally
on GPU after the engine returns.

Engine output order:

    DetectDLA  (3 outputs):
        pred_p3  (B, 4*reg_max + nc, H_P3, W_P3)         # e.g. (1, 144, 64, 64)
        pred_p4  (B, 4*reg_max + nc, H_P4, W_P4)         # e.g. (1, 144, 32, 32)
        pred_p5  (B, 4*reg_max + nc, H_P5, W_P5)         # e.g. (1, 144, 16, 16)

    SegmentDLA (4 outputs):
        pred_p3  (B, 4*reg_max + nc + nm, H_P3, W_P3)    # e.g. (1, 176, 64, 64)
        pred_p4  (B, 4*reg_max + nc + nm, H_P4, W_P4)    # e.g. (1, 176, 32, 32)
        pred_p5  (B, 4*reg_max + nc + nm, H_P5, W_P5)    # e.g. (1, 176, 16, 16)
        proto    (B, nm, 2*H_P3, 2*W_P3)                  # e.g. (1, 32, 128, 128)

Per scale, channels are packed in the order  `[box | score | mask]`:

    box   = pred[:,                       :  4*reg_max         , :, :]
    score = pred[:,  4*reg_max            :  4*reg_max + nc     , :, :]
    mask  = pred[:,  4*reg_max + nc       :  4*reg_max + nc + nm, :, :]

Run `python dla_decode.py --verify` to check numerical parity against the
standard 3-D `Detect._inference` path on a synthetic input.
"""

from __future__ import annotations

import argparse

import torch


def unpack_scale(
    pred: torch.Tensor,
    reg_max: int,
    nc: int,
    nm: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Split a packed per-scale tensor into (box, score, mask) on the channel axis."""
    box = pred[:, : 4 * reg_max]
    score = pred[:, 4 * reg_max : 4 * reg_max + nc]
    mask = pred[:, 4 * reg_max + nc : 4 * reg_max + nc + nm] if nm else None
    return box, score, mask


def decode_heads(
    preds_per_scale: list[torch.Tensor],
    strides: torch.Tensor,
    reg_max: int = 16,
    nc: int = 80,
    nm: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Decode the 3 packed 4-D heads into `(B, 4+nc, total_anchors)` (+ mask coeffs).

    Args:
        preds_per_scale: list of 3 packed maps, each `(B, 4*reg_max + nc + nm, H_i, W_i)`.
        strides: tensor of strides per scale, e.g. `torch.tensor([8., 16., 32.])`.
        reg_max: DFL bin count.
        nc: number of classes.
        nm: number of mask prototypes (0 for detection, 32 for segmentation).

    Returns:
        `(dets, mask_coeff)` where
            dets       = `(B, 4 + nc, total_anchors)` — xywh box centres + class probs.
            mask_coeff = `(B, nm, total_anchors)` or `None` if `nm == 0`.
    """
    from ultralytics.utils.tal import dist2bbox, make_anchors

    bs = preds_per_scale[0].shape[0]
    device, dtype = preds_per_scale[0].device, preds_per_scale[0].dtype

    boxes_4d, scores_4d, masks_4d = [], [], []
    for p in preds_per_scale:
        box, score, mask = unpack_scale(p, reg_max, nc, nm)
        boxes_4d.append(box)
        scores_4d.append(score)
        if mask is not None:
            masks_4d.append(mask)

    anchors, strides_t = (a.transpose(0, 1) for a in make_anchors(boxes_4d, strides, 0.5))

    boxes_flat = torch.cat([b.view(bs, 4 * reg_max, -1) for b in boxes_4d], dim=-1)
    scores_flat = torch.cat([s.view(bs, nc, -1) for s in scores_4d], dim=-1)

    b, _, a = boxes_flat.shape
    dist = boxes_flat.view(b, 4, reg_max, a).softmax(2)
    dfl_w = torch.arange(reg_max, dtype=dtype, device=device).view(1, 1, reg_max, 1)
    dist = (dist * dfl_w).sum(2)

    dbox = dist2bbox(dist, anchors.unsqueeze(0), xywh=True, dim=1) * strides_t
    dets = torch.cat([dbox, scores_flat.sigmoid()], dim=1)

    if masks_4d:
        mask_coeff = torch.cat([m.view(bs, nm, -1) for m in masks_4d], dim=-1)
        return dets, mask_coeff
    return dets, None


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
        (y_ref, proto_ref), _preds = model(x)
    head.export = True
    with torch.no_grad():
        outs = model(x)
    head.export = False

    preds_per_scale = list(outs[:3])
    proto_dla = outs[3]
    dets, mask_coeff = decode_heads(preds_per_scale, head.stride, reg_max=head.reg_max, nc=head.nc, nm=head.nm)
    y_dla_full = torch.cat([dets, mask_coeff], dim=1)

    diff = (y_ref - y_dla_full).abs()
    print(f"Reference shape:  {tuple(y_ref.shape)}")
    print(f"DLA-decoded shape: {tuple(y_dla_full.shape)}")
    print(f"Max abs diff: {diff.max().item():.3e}   Mean abs diff: {diff.mean().item():.3e}")
    proto_diff = (proto_ref - proto_dla).abs().max().item()
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
