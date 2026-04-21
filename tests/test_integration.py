#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pytest
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
