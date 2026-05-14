"""Single-image inference for the DLA-compiled YOLO11-seg engine.

Run on the Jetson where the engine was compiled (TRT engines are not portable):

    trtexec --onnx=best.onnx --saveEngine=best.engine --fp16 \\
            --useDLACore=0 --allowGPUFallback

    python examples/test_dla_engine.py \\
        --engine best.engine --image sample.jpg --output out.jpg

Engine emits 4 packed 4-D outputs (SegmentDLA):
    pred_p3  (1, 4*reg_max + nc + nm, 64,  64)
    pred_p4  (1, 4*reg_max + nc + nm, 32,  32)
    pred_p5  (1, 4*reg_max + nc + nm, 16,  16)
    proto    (1, nm,                  128, 128)

Decode (DFL → dist2bbox → sigmoid) and NMS run on host GPU after the engine returns.
"""

from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
import tensorrt as trt
import torch

try:
    from ultralytics.utils.nms import non_max_suppression  # fork layout
except ImportError:
    from ultralytics.utils.ops import non_max_suppression  # stock pip layout
from ultralytics.utils.ops import process_mask, scale_boxes
from ultralytics.utils.tal import dist2bbox, make_anchors


def decode_heads(preds_per_scale, strides, reg_max=16, nc=80, nm=0):
    """Decode 3 packed 4-D heads into (B, 4+nc, A) xywh+probs and (B, nm, A) mask coeffs."""
    bs = preds_per_scale[0].shape[0]
    device, dtype = preds_per_scale[0].device, preds_per_scale[0].dtype

    boxes_4d, scores_4d, masks_4d = [], [], []
    for p in preds_per_scale:
        boxes_4d.append(p[:, : 4 * reg_max])
        scores_4d.append(p[:, 4 * reg_max : 4 * reg_max + nc])
        if nm:
            masks_4d.append(p[:, 4 * reg_max + nc : 4 * reg_max + nc + nm])

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
        return dets, torch.cat([m.view(bs, nm, -1) for m in masks_4d], dim=-1)
    return dets, None

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
}


def letterbox(im: np.ndarray, new_shape: int = 512, color=(114, 114, 114)):
    """Resize keeping aspect ratio, pad to square. Returns (image, scale, (pad_x, pad_y))."""
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = round(h * r), round(w * r)
    dh, dw = new_shape - nh, new_shape - nw
    top, left = dh // 2, dw // 2
    bottom, right = dh - top, dw - left
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)


def _meta(engine: trt.ICudaEngine, name: str) -> dict:
    """Collect dtype/format/components/padded-shape for a TRT tensor."""
    shape = tuple(engine.get_tensor_shape(name))
    fmt = engine.get_tensor_format(name)
    dtype = _TRT_TO_TORCH[engine.get_tensor_dtype(name)]
    comp = engine.get_tensor_components_per_element(name)
    vec_dim = engine.get_tensor_vectorized_dim(name)
    padded = list(shape)
    if vec_dim >= 0:
        padded[vec_dim] = math.ceil(padded[vec_dim] / comp) * comp
    n_elem = 1
    for s in padded:
        n_elem *= s
    return dict(shape=shape, fmt=fmt, dtype=dtype, comp=comp, vec_dim=vec_dim,
                padded=tuple(padded), n_elem=n_elem)


class TRTEngine:
    """TRT 10.x wrapper that handles `dla_hwc4` input and `chw{16,32}` outputs."""

    def __init__(self, engine_path: str, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        self.in_meta = {n: _meta(self.engine, n) for n in self.input_names}
        self.out_meta = {n: _meta(self.engine, n) for n in self.output_names}

        print("[engine] I/O bindings:")
        for n, m in {**self.in_meta, **self.out_meta}.items():
            print(f"  {n:20s} logical={m['shape']} fmt={m['fmt']} dtype={m['dtype']} comp={m['comp']} vec_dim={m['vec_dim']}")

        self.in_buf: dict[str, torch.Tensor] = {}
        for n, m in self.in_meta.items():
            self.in_buf[n] = torch.zeros(m["n_elem"], dtype=m["dtype"], device=self.device)
            self.context.set_tensor_address(n, self.in_buf[n].data_ptr())

        self.out_buf: dict[str, torch.Tensor] = {}
        for n, m in self.out_meta.items():
            self.out_buf[n] = torch.empty(m["n_elem"], dtype=m["dtype"], device=self.device)
            self.context.set_tensor_address(n, self.out_buf[n].data_ptr())

    _CHW_TILED = {trt.TensorFormat.CHW2, trt.TensorFormat.CHW4,
                  trt.TensorFormat.CHW16, trt.TensorFormat.CHW32}

    def _pack_input(self, name: str, x_nchw: torch.Tensor) -> None:
        m = self.in_meta[name]
        x = x_nchw.to(self.device)
        B, C, H, W = x.shape
        if m["fmt"] in self._CHW_TILED:
            comp = m["comp"]
            tiles = math.ceil(C / comp)
            xp = torch.zeros((B, tiles * comp, H, W), dtype=m["dtype"], device=self.device)
            xp[:, :C] = x.to(m["dtype"])
            # (B, tiles, comp, H, W) -> (B, tiles, H, W, comp)
            xp = xp.view(B, tiles, comp, H, W).permute(0, 1, 3, 4, 2).contiguous()
            self.in_buf[name].copy_(xp.reshape(-1))
        elif m["fmt"] == trt.TensorFormat.DLA_HWC4:
            packed = torch.zeros((B, H, W, m["comp"]), dtype=m["dtype"], device=self.device)
            packed[..., :C] = x.permute(0, 2, 3, 1).to(m["dtype"])
            self.in_buf[name].copy_(packed.reshape(-1))
        elif m["fmt"] == trt.TensorFormat.LINEAR:
            self.in_buf[name].copy_(x.to(m["dtype"]).reshape(-1))
        else:
            raise NotImplementedError(f"Input format {m['fmt']} not supported")
        self.context.set_input_shape(name, (B, C, H, W))

    def _unpack_output(self, name: str) -> torch.Tensor:
        m = self.out_meta[name]
        B, C, H, W = m["shape"]
        buf = self.out_buf[name]
        if m["fmt"] in self._CHW_TILED:
            tiles = math.ceil(C / m["comp"])
            t = buf.view(B, tiles, H, W, m["comp"]).permute(0, 1, 4, 2, 3).contiguous()
            return t.view(B, tiles * m["comp"], H, W)[:, :C].float()
        if m["fmt"] == trt.TensorFormat.LINEAR:
            return buf.view(*m["shape"]).float()
        raise NotImplementedError(f"Output format {m['fmt']} not supported")

    def infer(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        for name in self.input_names:
            self._pack_input(name, x)
        stream = torch.cuda.current_stream(self.device)
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return {n: self._unpack_output(n) for n in self.output_names}


def _palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(n, 3), dtype=np.uint8)


def overlay(im: np.ndarray, boxes: torch.Tensor, classes: torch.Tensor, confs: torch.Tensor,
            masks: torch.Tensor | None, alpha: float = 0.5) -> np.ndarray:
    colors = _palette(int(classes.max().item()) + 1 if classes.numel() else 1)
    out = im.copy()
    if masks is not None:
        H, W = im.shape[:2]
        for m, c in zip(masks, classes):
            m_full = cv2.resize(m.cpu().numpy().astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
            color = colors[int(c.item())]
            out[m_full] = (alpha * color + (1 - alpha) * out[m_full]).astype(np.uint8)
    for box, cls, conf in zip(boxes, classes, confs):
        x1, y1, x2, y2 = box.int().tolist()
        color = tuple(int(v) for v in colors[int(cls.item())])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{int(cls.item())} {conf.item():.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, help="Path to .engine compiled with trtexec --useDLACore=0")
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--output", default="out.jpg", help="Output overlay image path")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--nc", type=int, default=80)
    ap.add_argument("--reg-max", type=int, default=16)
    ap.add_argument("--nm", type=int, default=32)
    args = ap.parse_args()

    im0 = cv2.imread(args.image)
    if im0 is None:
        raise FileNotFoundError(args.image)
    im_pad, _, _ = letterbox(im0, args.imgsz)
    arr = im_pad[:, :, ::-1].transpose(2, 0, 1)
    arr = np.ascontiguousarray(arr, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr)[None]

    eng = TRTEngine(args.engine)
    raw = eng.infer(x)

    head_outs = [raw[n].float() for n in eng.output_names]
    print("\n=== Engine outputs ===")
    for name, t in zip(eng.output_names, head_outs):
        print(f"  {name:20s} shape={tuple(t.shape)} dtype={t.dtype}  min={t.min().item():.3f}  max={t.max().item():.3f}  mean={t.mean().item():.3f}")

    proto = next(t for t in head_outs if t.shape[1] == args.nm)
    pred_maps = [t for t in head_outs if t is not proto]
    pred_maps.sort(key=lambda t: -t.shape[-1])  # P3, P4, P5
    print(f"\n  proto: shape={tuple(proto.shape)}")
    print(f"  pred_maps (P3,P4,P5): {[tuple(p.shape) for p in pred_maps]}")

    strides = torch.tensor([args.imgsz / p.shape[-1] for p in pred_maps],
                           device=pred_maps[0].device, dtype=pred_maps[0].dtype)
    print(f"  strides={strides.tolist()}")
    dets, mask_coeff = decode_heads(pred_maps, strides, args.reg_max, args.nc, args.nm)
    preds = torch.cat([dets, mask_coeff], dim=1)

    cls_scores = preds[:, 4:4 + args.nc]
    max_per_anchor = cls_scores.amax(1)[0]
    topk = max_per_anchor.topk(min(10, max_per_anchor.numel()))
    print(f"\n=== Score stats (after sigmoid) ===")
    print(f"  max={max_per_anchor.max().item():.4f}  mean={max_per_anchor.mean().item():.4f}")
    print(f"  top-10 anchor scores: {[f'{v:.3f}' for v in topk.values.tolist()]}")
    print(f"  #anchors > 0.001 : {(max_per_anchor > 0.001).sum().item()}")
    print(f"  #anchors > 0.05  : {(max_per_anchor > 0.05).sum().item()}")
    print(f"  #anchors > {args.conf:.2f} : {(max_per_anchor > args.conf).sum().item()}")

    nms_out = non_max_suppression(preds, conf_thres=args.conf, iou_thres=args.iou, nc=args.nc)[0]

    if nms_out is None or nms_out.numel() == 0:
        print("No detections.")
        cv2.imwrite(args.output, im0)
        return

    masks = process_mask(proto[0], nms_out[:, 6:], nms_out[:, :4],
                         (args.imgsz, args.imgsz), upsample=True)
    boxes_orig = scale_boxes((args.imgsz, args.imgsz), nms_out[:, :4].clone(), im0.shape[:2])

    confs = nms_out[:, 4]
    classes = nms_out[:, 5].int()
    print(f"Detections: {len(nms_out)}")
    for box, cls, conf in zip(boxes_orig, classes, confs):
        x1, y1, x2, y2 = box.int().tolist()
        print(f"  cls={int(cls.item()):3d}  conf={conf.item():.3f}  box=({x1},{y1},{x2},{y2})")

    out_img = overlay(im0, boxes_orig, classes, confs, masks)
    cv2.imwrite(args.output, out_img)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
