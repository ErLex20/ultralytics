# YOLO-DLA

YOLO-DLA is a hardware-first, single-stage detector for NVIDIA DLA and similar
fixed-function convolutional accelerators. It is a separate model family; the
existing YOLO11 and YOLO11-DLA implementations are unchanged and remain the two
primary experimental baselines.

## Research comparison

| Model | Purpose |
|---|---|
| YOLO11 | Unconstrained reference architecture |
| YOLO11-DLA | Minimal conversion of YOLO11 to the DLA operator contract |
| YOLO-DLA | New YOLO architecture designed around that contract |

YOLO-DLA retains the defining YOLO structure: one forward pass, a fully
convolutional backbone and PAN neck, dense P3/P4/P5 predictions, and direct box
and class regression. It does not use proposals, learned queries, or a
Transformer decoder.

## YOLO-DLA-N v0

The initial model is defined in
`ultralytics/cfg/models/dla/yolo-dla.yaml` and introduces three components:

- `DLARepCSP`: independent CSP projections avoid channel Slice operations;
  RepConv training branches fuse into one 3x3 convolution for deployment.
- `DLALocalContext`: a local, input-dependent depthwise gate replaces global
  self-attention using only Conv, Sigmoid, Product, and Add.
- `YOLODLADetect`: a decoupled RepConv head trained with one-to-many and
  one-to-one assignment. Deployment retains only the one-to-one branch.

The v0 head uses `reg_max=1`, so deployment is DFL-free. Export emits three raw
NCHW tensors, ordered as P3/P4/P5 and packed as `[box | class_logits]`. For COCO
each tensor has 84 logical channels. Physical CHW16 padding is handled by the
TensorRT binding format.

The unfused training model has 3.019M parameters because both assignment heads
are present. After fusion removes the auxiliary head and collapses RepConv, the
deployment graph has 2.393M parameters.

## Operator contract

The deployable neural graph is restricted to static four-dimensional tensors
and the following operation families:

- convolution and depthwise convolution;
- ReLU and Sigmoid;
- elementwise Add and Product;
- channel-axis Concat;
- MaxPool through SPPF;
- fixed nearest-neighbor resize.

There is no MatMul, dynamic reshape, transpose, channel Slice, global pooling,
DFL Softmax, TopK, Gather, or NMS inside the exported engine.

## Build and train

```bash
yolo detect train \
  model=ultralytics/cfg/models/dla/yolo-dla.yaml \
  data=coco.yaml imgsz=640 epochs=300 batch=128
```

The generic end-to-end criterion automatically trains the auxiliary one-to-many
and deployment one-to-one heads. Distillation and alternative training recipes
should be introduced as controlled ablations after establishing this baseline.

## Export

```bash
yolo export \
  model=path/to/yolo-dla-n.pt format=onnx imgsz=640 \
  dynamic=False simplify=True
```

The ONNX outputs are named `pred_p3`, `pred_p4`, and `pred_p5`. Export metadata
records `dla_raw`, `reg_max`, and pyramid strides so the standard Ultralytics
predictor and validator can decode them externally. The decoder intentionally
runs on CPU by default. For the one-to-one head it performs box decoding,
Sigmoid, confidence ranking, and thresholding; it does not run NMS.

A strict build on the target Jetson is the authoritative compliance check:

```bash
trtexec \
  --onnx=yolo-dla.onnx \
  --saveEngine=yolo-dla-fp16.engine \
  --useDLACore=0 --fp16 \
  --inputIOFormats=fp16:dla_hwc4 \
  --outputIOFormats=fp16:chw16
```

Do not pass `--allowGPUFallback`. A valid result is a successful strict build
with no GPU layers. Record neural latency, CPU postprocess latency, end-to-end
latency, energy, and concurrent-GPU workload latency separately.

### Verified Jetson result

The randomly initialized YOLO-DLA-N v0 ONNX was compiled and benchmarked on an
NVIDIA Orin with TensorRT 10.3.0 at 640x640, using DLA core 0, FP16 DLA-native
I/O, and GPU fallback disabled. The strict build completed without unsupported
layer or fallback warnings.

| Metric | Result |
|---|---:|
| Engine size | 5.446 MiB |
| Throughput | 46.164 qps |
| Mean host latency | 22.131 ms |
| Median host latency | 22.115 ms |
| p95 host latency | 22.219 ms |
| Mean device compute time | 21.504 ms |
| Mean H2D / D2H | 0.409 / 0.218 ms |

TensorRT labels device timing as `GPU Compute Time` even when the selected
device is DLA. In this run `DLACore: 0`, `Allow GPU fallback for DLA: Disabled`,
and the successful build without fallback messages establish DLA execution.

## Current scope

This commit establishes the detection architecture and deployment interface.
It does not yet claim trained COCO accuracy. FP16 DLA compliance and raw-network
latency are confirmed on Orin; CPU postprocess, end-to-end application latency,
energy, and GPU-concurrency measurements remain to be collected. Instance
segmentation, distillation, INT8 QAT, and context/head ablations belong to later
experimental stages after training the detection baseline.
