#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import shape_inference
from test_support import (
    assert_all_nodes_run_on_ggml,
    cached_model_path,
    cpu_session,
    ggml_session,
)

_MNIST_MODEL_URL = (
    "https://media.githubusercontent.com/media/onnx/models/"
    "c32b9776d06d2ebc7888d705e3a558f62b20e7a8/"
    "validated/vision/classification/mnist/model/mnist-8.onnx"
)
_MOSAIC_MODEL_URL = (
    "https://github.com/onnx/models/raw/"
    "c32b9776d06d2ebc7888d705e3a558f62b20e7a8/"
    "validated/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx"
)
_TINY_YOLOV3_MODEL_URL = (
    "https://media.githubusercontent.com/media/onnx/models/"
    "c32b9776d06d2ebc7888d705e3a558f62b20e7a8/"
    "validated/vision/object_detection_segmentation/tiny-yolov3/model/"
    "tiny-yolov3-11.onnx"
)
_OPENWAKEWORD_ALEXA_URL = (
    "https://github.com/dscripka/openWakeWord/releases/download/"
    "v0.5.1/alexa_v0.1.onnx"
)
_OPENWAKEWORD_EMBEDDING_URL = (
    "https://github.com/dscripka/openWakeWord/releases/download/"
    "v0.5.1/embedding_model.onnx"
)
_ARCFACE_MODEL_URL = (
    "https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8/resolve/main/"
    "arcfaceresnet100-8.onnx?download=true"
)
_RESNET18_MODEL_URL = (
    "https://github.com/onnx/models/raw/"
    "c32b9776d06d2ebc7888d705e3a558f62b20e7a8/"
    "validated/vision/classification/resnet/model/resnet18-v2-7.onnx"
)
_MOBILENETV2_MODEL_URL = (
    "https://github.com/onnx/models/raw/"
    "c32b9776d06d2ebc7888d705e3a558f62b20e7a8/"
    "validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
)


def _concretize_input_dims(model_path: Path, overrides: dict[str, list[int]]) -> Path:
    # tiny-yolov3's inputs have dynamic batch and spatial dims (N, 3, ?, ?).
    # GGONNX relies on complete shape information propagated across partition
    # boundaries, so we pin the inputs and re-run shape inference before the
    # model reaches ORT.
    static_path = model_path.with_name(model_path.stem + "_static.onnx")
    if static_path.exists():
        return static_path
    model = onnx.load(str(model_path))
    for graph_input in model.graph.input:
        dims = overrides.get(graph_input.name)
        if dims is None:
            continue
        shape = graph_input.type.tensor_type.shape
        assert len(shape.dim) == len(dims), (
            f"override for {graph_input.name} expects {len(shape.dim)} dims"
        )
        for slot, value in zip(shape.dim, dims):
            slot.ClearField("dim_param")
            slot.dim_value = int(value)
    onnx.save(shape_inference.infer_shapes(model), str(static_path))
    return static_path


def _mnist_one_input() -> np.ndarray:
    image = np.zeros((1, 1, 28, 28), dtype=np.float32)
    image[0, 0, 4:24, 14:16] = 1.0
    return image


@pytest.mark.integration
def test_mnist_model_matches_cpu(ep_library: Path) -> None:
    model_path = cached_model_path("mnist-8.onnx", _MNIST_MODEL_URL)
    inputs = {"Input3": _mnist_one_input()}

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    cpu_out = cpu.run(["Plus214_Output_0"], inputs)[0]
    ggml_out = ggml.run(["Plus214_Output_0"], inputs)[0]

    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-5, atol=1e-5)
    assert int(np.argmax(ggml_out, axis=1)[0]) == int(np.argmax(cpu_out, axis=1)[0])


@pytest.mark.integration
def test_mosaic_model_matches_cpu(ep_library: Path) -> None:
    model_path = cached_model_path("mosaic-9.onnx", _MOSAIC_MODEL_URL)
    rng = np.random.default_rng(0)
    inputs = {"input1": rng.standard_normal((1, 3, 224, 224)).astype(np.float32)}

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    cpu_out = cpu.run(["output1"], inputs)[0]
    ggml_out = ggml.run(["output1"], inputs)[0]

    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-3, atol=1e-3)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.integration
def test_tiny_yolov3_model_matches_cpu(ep_library: Path) -> None:
    raw_path = cached_model_path("tiny-yolov3-11.onnx", _TINY_YOLOV3_MODEL_URL)
    model_path = _concretize_input_dims(
        raw_path,
        {"input_1": [1, 3, 416, 416], "image_shape": [1, 2]},
    )
    rng = np.random.default_rng(0)
    inputs = {
        "input_1": rng.standard_normal((1, 3, 416, 416)).astype(np.float32),
        "image_shape": np.array([[416, 416]], dtype=np.float32),
    }
    output_names = ["yolonms_layer_1", "yolonms_layer_1:1", "yolonms_layer_1:2"]

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    cpu_out = cpu.run(output_names, inputs)
    ggml_out = ggml.run(output_names, inputs)

    # Box coords + class scores must match the CPU reference; the int32 indices
    # tensor is an exact match (NMS output).
    np.testing.assert_allclose(ggml_out[0], cpu_out[0], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(ggml_out[1], cpu_out[1], rtol=1e-3, atol=1e-3)
    np.testing.assert_array_equal(ggml_out[2], cpu_out[2])


@pytest.mark.integration
def test_openwakeword_alexa_matches_cpu(ep_library: Path) -> None:
    model_path = cached_model_path("alexa_v0.1.onnx", _OPENWAKEWORD_ALEXA_URL)

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    rng = np.random.default_rng(0)
    inputs = {}
    for graph_input in cpu.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in graph_input.shape]
        inputs[graph_input.name] = rng.standard_normal(shape).astype(np.float32)

    output_names = [out.name for out in cpu.get_outputs()]
    cpu_out = cpu.run(output_names, inputs)
    ggml_out = ggml.run(output_names, inputs)

    for got, expected in zip(ggml_out, cpu_out):
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.integration
def test_arcface_resnet100_matches_cpu(ep_library: Path) -> None:
    model_path = cached_model_path("arcfaceresnet100-8.onnx", _ARCFACE_MODEL_URL)

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    input_info = cpu.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_info.shape]
    rng = np.random.default_rng(0)
    inputs = {input_info.name: rng.standard_normal(shape).astype(np.float32)}

    output_names = [out.name for out in cpu.get_outputs()]
    cpu_out = cpu.run(output_names, inputs)
    ggml_out = ggml.run(output_names, inputs)

    for got, expected in zip(ggml_out, cpu_out):
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.integration
def test_resnet18_v2_matches_cpu(ep_library: Path) -> None:
    raw_path = cached_model_path("resnet18-v2-7.onnx", _RESNET18_MODEL_URL)
    model_path = _concretize_input_dims(raw_path, {"data": [1, 3, 224, 224]})

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    rng = np.random.default_rng(0)
    inputs = {"data": rng.standard_normal((1, 3, 224, 224)).astype(np.float32)}

    output_names = [out.name for out in cpu.get_outputs()]
    cpu_out = cpu.run(output_names, inputs)
    ggml_out = ggml.run(output_names, inputs)

    for got, expected in zip(ggml_out, cpu_out):
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.integration
def test_mobilenetv2_matches_cpu(ep_library: Path) -> None:
    raw_path = cached_model_path("mobilenetv2-12.onnx", _MOBILENETV2_MODEL_URL)
    model_path = _concretize_input_dims(raw_path, {"input": [1, 3, 224, 224]})

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    rng = np.random.default_rng(0)
    inputs = {"input": rng.standard_normal((1, 3, 224, 224)).astype(np.float32)}

    output_names = [out.name for out in cpu.get_outputs()]
    cpu_out = cpu.run(output_names, inputs)
    ggml_out = ggml.run(output_names, inputs)

    for got, expected in zip(ggml_out, cpu_out):
        np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.integration
def test_openwakeword_embedding_matches_cpu(ep_library: Path) -> None:
    raw_path = cached_model_path("embedding_model.onnx", _OPENWAKEWORD_EMBEDDING_URL)
    model_path = _concretize_input_dims(raw_path, {"input_1": [1, 76, 32, 1]})

    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    rng = np.random.default_rng(0)
    inputs = {"input_1": rng.standard_normal((1, 76, 32, 1)).astype(np.float32)}

    output_names = [out.name for out in cpu.get_outputs()]
    cpu_out = cpu.run(output_names, inputs)
    ggml_out = ggml.run(output_names, inputs)

    for got, expected in zip(ggml_out, cpu_out):
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
    assert_all_nodes_run_on_ggml(ggml)
