"""Train a DLA-YOLO11 variant with warm-start + hot-start + two-phase schedule.

Pipeline:

    1. Build the DLA architecture from YAML (`yolo11-dla.yaml` / `yolo11-dla-seg.yaml`).
    2. Warm-start: partial-load all matching weights from the standard pretrained
       checkpoint (`yolo11n-seg.pt` / `yolo11n.pt`) via `YOLO.load()`. This
       transfers layers 0-9 and 11-22 verbatim — `C2DLA` at layer 10 keeps
       random init.
    3. Hot-start: copy the directly transferable submodules of the original
       `Attention` into `DLAAttention`:
         - `pe`  ← `attn.pe`        (3×3 DWConv, identical)
         - `proj_out` ← `attn.proj` (1×1 Conv, identical)
         - `proj_v`  ← V slice of `attn.qkv` (channels gathered per head)
         - `proj_k`  ← K slice of `attn.qkv` *only* when `k_dim == nh_kd`
                       (else leave random — typically true for default `k_dim`)
       `proj_attn` always stays random (no analog in the original).
    4. Phase 1 — freeze layers 0-9 (`freeze=10`) for `--warmup-epochs` epochs so
       `proj_attn` finds a sensible point while the backbone stays put.
    5. Phase 2 — unfreeze and fine-tune for `--finetune-epochs` epochs from the
       phase-1 `last.pt`.

Example:
    python examples/train_dla.py \\
        --model ultralytics/cfg/models/11/yolo11-dla-seg.yaml \\
        --pretrained yolo11n-seg.pt \\
        --data coco-seg.yaml \\
        --imgsz 512 --batch 64 \\
        --warmup-epochs 5 --finetune-epochs 95 \\
        --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ultralytics import YOLO


def _copy_conv_bn(dst, src) -> None:
    """Copy an ultralytics `Conv` (Conv2d + BN) module: weight + BN params + running stats."""
    dst.conv.weight.copy_(src.conv.weight)
    if getattr(dst.conv, "bias", None) is not None and getattr(src.conv, "bias", None) is not None:
        dst.conv.bias.copy_(src.conv.bias)
    if hasattr(src, "bn") and hasattr(dst, "bn"):
        dst.bn.weight.copy_(src.bn.weight)
        dst.bn.bias.copy_(src.bn.bias)
        dst.bn.running_mean.copy_(src.bn.running_mean)
        dst.bn.running_var.copy_(src.bn.running_var)
        if hasattr(src.bn, "num_batches_tracked") and hasattr(dst.bn, "num_batches_tracked"):
            dst.bn.num_batches_tracked.copy_(src.bn.num_batches_tracked)


def hot_start_c2dla(dst_root: torch.nn.Module, src_pt: str, layer_idx: int = 10) -> dict[str, int]:
    """Copy transferable submodules from `C2PSA` (source `.pt`) to `C2DLA` (dst model).

    `Attention.qkv` is `Conv(dim, dim + 2·nh_kd, 1)` with channels laid out **per head**:
    each head contributes a contiguous block `[q_i (key_dim) | k_i (key_dim) | v_i (head_dim)]`.
    The K and V "slices" are therefore non-contiguous in the global channel axis — we
    gather them by index. The total K block has `num_heads·key_dim = nh_kd` channels;
    the V block has `num_heads·head_dim = dim` channels.

    Returns a counter describing how many submodules transferred across all DLABlocks.
    """
    ckpt = torch.load(src_pt, map_location="cpu", weights_only=False)
    src_model = ckpt["model"]
    src_c2psa = src_model.model[layer_idx]   # C2PSA
    dst_c2dla = dst_root.model[layer_idx]    # C2DLA

    n = min(len(src_c2psa.m), len(dst_c2dla.m))
    counts = {"pe": 0, "proj_out": 0, "proj_v": 0, "proj_k": 0, "skipped_k_shape_mismatch": 0}

    for i in range(n):
        src_attn = src_c2psa.m[i].attn   # Attention
        dst_attn = dst_c2dla.m[i].attn   # DLAAttention

        with torch.no_grad():
            # pe (3×3 DWConv on V) — direct copy
            if dst_attn.pe.conv.weight.shape == src_attn.pe.conv.weight.shape:
                _copy_conv_bn(dst_attn.pe, src_attn.pe)
                counts["pe"] += 1

            # proj_out  ←  src.proj
            if dst_attn.proj_out.conv.weight.shape == src_attn.proj.conv.weight.shape:
                _copy_conv_bn(dst_attn.proj_out, src_attn.proj)
                counts["proj_out"] += 1

            # Gather per-head K and V indices out of qkv
            num_heads = src_attn.num_heads
            key_dim = src_attn.key_dim
            head_dim = src_attn.head_dim
            per_head = 2 * key_dim + head_dim

            k_idx: list[int] = []
            v_idx: list[int] = []
            for h in range(num_heads):
                base = h * per_head
                k_idx += list(range(base + key_dim, base + 2 * key_dim))
                v_idx += list(range(base + 2 * key_dim, base + per_head))
            k_idx_t = torch.tensor(k_idx, dtype=torch.long)
            v_idx_t = torch.tensor(v_idx, dtype=torch.long)

            qkv_w = src_attn.qkv.conv.weight
            qkv_bn = src_attn.qkv.bn

            # V transfer — usually shapes match (dim = num_heads * head_dim)
            v_w = qkv_w.index_select(0, v_idx_t)
            if v_w.shape == dst_attn.proj_v.conv.weight.shape:
                dst_attn.proj_v.conv.weight.copy_(v_w)
                dst_attn.proj_v.bn.weight.copy_(qkv_bn.weight.index_select(0, v_idx_t))
                dst_attn.proj_v.bn.bias.copy_(qkv_bn.bias.index_select(0, v_idx_t))
                dst_attn.proj_v.bn.running_mean.copy_(qkv_bn.running_mean.index_select(0, v_idx_t))
                dst_attn.proj_v.bn.running_var.copy_(qkv_bn.running_var.index_select(0, v_idx_t))
                counts["proj_v"] += 1

            # K transfer — only when dst.k_dim == nh_kd
            k_w = qkv_w.index_select(0, k_idx_t)
            if k_w.shape == dst_attn.proj_k.conv.weight.shape:
                dst_attn.proj_k.conv.weight.copy_(k_w)
                dst_attn.proj_k.bn.weight.copy_(qkv_bn.weight.index_select(0, k_idx_t))
                dst_attn.proj_k.bn.bias.copy_(qkv_bn.bias.index_select(0, k_idx_t))
                dst_attn.proj_k.bn.running_mean.copy_(qkv_bn.running_mean.index_select(0, k_idx_t))
                dst_attn.proj_k.bn.running_var.copy_(qkv_bn.running_var.index_select(0, k_idx_t))
                counts["proj_k"] += 1
            else:
                counts["skipped_k_shape_mismatch"] += 1

    return counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="ultralytics/cfg/models/11/yolo11-dla-seg.yaml",
                   help="DLA architecture YAML")
    p.add_argument("--pretrained", default="yolo11n-seg.pt",
                   help="Path to standard pretrained .pt (yolo11n-seg.pt for seg, yolo11n.pt for detect)")
    p.add_argument("--data", default="coco-seg.yaml", help="Dataset YAML")
    p.add_argument("--imgsz", type=int, default=512, help="Train/export image size")
    p.add_argument("--batch", type=int, default=64, help="Batch size")
    p.add_argument("--warmup-epochs", type=int, default=5, help="Phase 1 frozen-backbone epochs")
    p.add_argument("--finetune-epochs", type=int, default=95, help="Phase 2 full fine-tune epochs")
    p.add_argument("--warmup-lr0", type=float, default=0.001, help="Phase 1 initial LR")
    p.add_argument("--finetune-lr0", type=float, default=0.0005, help="Phase 2 initial LR")
    p.add_argument("--project", default="runs/dla", help="Ultralytics project dir")
    p.add_argument("--name", default="exp", help="Run name prefix")
    p.add_argument("--device", default="0", help="cuda device id (e.g. '0' or '0,1') or 'cpu'")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable AMP mixed precision")
    p.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False, help="Generate result plots")
    p.add_argument("--phase1-only", action="store_true", help="Run only phase 1 (useful for plumbing tests)")
    p.add_argument("--no-train", action="store_true", help="Build + warm-start + hot-start only; skip training")
    args = p.parse_args()

    # 1. Build + warm-start
    print(f"[1/4] Building {args.model}")
    model = YOLO(args.model)
    print(f"[1/4] Warm-starting from {args.pretrained}  (partial weight load; unmatched layers stay random)")
    model.load(args.pretrained)

    # 2. Hot-start C2DLA
    print("[2/4] Hot-starting C2DLA.attn submodules from pretrained C2PSA.attn")
    counts = hot_start_c2dla(model.model, args.pretrained)
    for k, v in counts.items():
        print(f"        {k:30s} {v}")

    if args.no_train:
        print("--no-train set — exiting before training")
        return

    common = dict(
        data=args.data, imgsz=args.imgsz, batch=args.batch,
        device=args.device, amp=args.amp, plots=args.plots,
        project=args.project,
    )

    # 3. Phase 1 — frozen backbone warmup
    print(f"[3/4] Phase 1 — freeze=10, epochs={args.warmup_epochs}, lr0={args.warmup_lr0}")
    model.train(**common, epochs=args.warmup_epochs, freeze=10, lr0=args.warmup_lr0,
                name=f"{args.name}_warmup")
    save_dir = Path(model.trainer.save_dir)
    last_pt = save_dir / "weights" / "last.pt"
    print(f"        phase-1 last checkpoint: {last_pt}")

    if args.phase1_only:
        print("--phase1-only set — exiting after phase 1")
        return

    # 4. Phase 2 — full fine-tune
    print(f"[4/4] Phase 2 — freeze=0, epochs={args.finetune_epochs}, lr0={args.finetune_lr0}")
    model2 = YOLO(str(last_pt))
    model2.train(**common, epochs=args.finetune_epochs, freeze=0, lr0=args.finetune_lr0,
                 name=f"{args.name}_finetune")
    print(f"        phase-2 last checkpoint: {Path(model2.trainer.save_dir) / 'weights' / 'last.pt'}")
    print("        best checkpoint:        ", Path(model2.trainer.save_dir) / "weights" / "best.pt")


if __name__ == "__main__":
    main()
