# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tests for the hardware-first YOLO-DLA model family."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import C2DLA, DLARepCSP, YOLODLADetect
from ultralytics.utils.dla import decode_dla_outputs

ROOT = Path(__file__).resolve().parents[1]
YOLO_DLA_CFG = ROOT / "ultralytics/cfg/models/dla/yolo-dla.yaml"
YOLO11_DLA_CFG = ROOT / "ultralytics/cfg/models/11/yolo11-dla.yaml"
YOLO11_DLA_SEG_CFG = ROOT / "ultralytics/cfg/models/11/yolo11-dla-seg.yaml"


def test_yolo_dla_train_and_export_contract():
    """YOLO-DLA should use dual assignment and export three packed NCHW maps."""
    model = YOLO(YOLO_DLA_CFG).model
    head = model.model[-1]
    assert isinstance(head, YOLODLADetect)
    assert head.end2end and head.reg_max == 1
    assert any(isinstance(module, DLARepCSP) for module in model.modules())

    model.train()
    with torch.no_grad():
        predictions = model(torch.randn(1, 3, 128, 128))
    assert predictions.keys() == {"one2many", "one2one"}
    assert predictions["one2many"]["boxes"].shape[1] == 4
    assert predictions["one2one"]["scores"].shape[1] == 80

    model.eval()
    head.export = True
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 256, 256))
    assert [tuple(output.shape) for output in outputs] == [
        (1, 84, 32, 32),
        (1, 84, 16, 16),
        (1, 84, 8, 8),
    ]


def test_yolo_dla_external_decode_matches_python_inference():
    """The CPU raw-map decoder should preserve the normal end-to-end prediction contract."""
    torch.manual_seed(0)
    model = YOLO(YOLO_DLA_CFG).model.eval()
    head = model.model[-1]
    image = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        reference, _ = model(image)
        head.export = True
        raw = model(image)
        head.export = False
    decoded = decode_dla_outputs(
        raw,
        head.stride,
        head.reg_max,
        head.nc,
        end2end=True,
        max_det=head.max_det,
    )
    torch.testing.assert_close(decoded, reference)


def test_yolo_dla_dual_assignment_loss_backpropagates():
    """Both training-only and deployment heads should receive gradients from the end-to-end loss."""
    model = YOLO(YOLO_DLA_CFG).model.train()
    model.args = get_cfg()
    batch = {
        "img": torch.rand(2, 3, 128, 128),
        "batch_idx": torch.tensor([0, 1]),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.1, 0.15]]),
    }
    loss, _ = model(batch)
    loss.sum().backward()
    head = model.model[-1]
    assert head.cv2[0][-1].weight.grad.abs().sum() > 0
    assert head.one2one_cv2[0][-1].weight.grad.abs().sum() > 0


def test_yolo_dla_repconv_fuses_without_changing_raw_outputs():
    """All RepConv training branches should collapse before deployment export."""
    torch.manual_seed(1)
    model = YOLO(YOLO_DLA_CFG).model.eval()
    image = torch.randn(1, 3, 256, 256)
    model.model[-1].export = True
    with torch.no_grad():
        before = model(image)

    fused = deepcopy(model).fuse(verbose=False).eval()
    fused.model[-1].export = True
    with torch.no_grad():
        after = fused(image)
    assert all(hasattr(module, "conv") for module in fused.modules() if module.__class__.__name__ == "RepConv")
    for expected, actual in zip(before, after):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_yolo11_dla_configuration_remains_independent():
    """The existing YOLO11-DLA YAML must keep its original C2DLA and DetectDLA design."""
    model = YOLO(YOLO11_DLA_CFG).model
    assert isinstance(model.model[10], C2DLA)
    assert model.model[-1].__class__.__name__ == "DetectDLA"
    assert not isinstance(model.model[-1], YOLODLADetect)


def test_existing_yolo11_dla_segmentation_maps_still_decode():
    """The shared CPU decoder should also preserve the existing YOLO11-DLA segmentation contract."""
    torch.manual_seed(2)
    model = YOLO(YOLO11_DLA_SEG_CFG).model.eval()
    head = model.model[-1]
    image = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        (reference, proto_reference), _ = model(image)
        head.export = True
        raw = model(image)
        head.export = False
    decoded = decode_dla_outputs(
        raw[:3],
        head.stride,
        head.reg_max,
        head.nc,
        nm=head.nm,
        end2end=False,
    )
    torch.testing.assert_close(decoded, reference)
    torch.testing.assert_close(raw[3], proto_reference)


def test_yolo_dla_onnx_operator_contract(tmp_path):
    """The fused ONNX graph must not reintroduce rank-changing or selection operators."""
    onnx = pytest.importorskip("onnx")
    model = deepcopy(YOLO(YOLO_DLA_CFG).model).fuse(verbose=False).eval()
    model.model[-1].export = True
    output = tmp_path / "yolo-dla.onnx"
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 128, 128),
        output,
        opset_version=17,
        input_names=["images"],
        output_names=["pred_p3", "pred_p4", "pred_p5"],
        do_constant_folding=True,
        dynamo=False,
    )
    graph = onnx.load(output)
    operators = {node.op_type for node in graph.graph.node}
    forbidden = {
        "Div",
        "Gather",
        "Gemm",
        "MatMul",
        "ReduceMean",
        "Reshape",
        "Slice",
        "Softmax",
        "TopK",
        "Transpose",
    }
    assert not operators.intersection(forbidden)
    assert [tensor.name for tensor in graph.graph.output] == ["pred_p3", "pred_p4", "pred_p5"]
