#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper

from test_support import cpu_session, ggml_session, save_model


def build_single_add_model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3])
    node = helper.make_node("Add", ["x", "y"], ["z"], name="add_0")
    graph = helper.make_graph([node], "single_add", [x, y], [z])
    save_model(path, graph)


def build_dynamic_add_model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", "cols"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", "cols"])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, ["batch", "cols"])
    node = helper.make_node("Add", ["x", "y"], ["z"], name="add_dynamic")
    graph = helper.make_graph([node], "dynamic_add", [x, y], [z])
    save_model(path, graph)


def build_broadcast_add_model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", "cols"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["cols"])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, ["batch", "cols"])
    node = helper.make_node("Add", ["x", "y"], ["z"], name="add_broadcast")
    graph = helper.make_graph([node], "broadcast_add", [x, y], [z])
    save_model(path, graph)


def build_single_binary_model(path: Path, op_type: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3])
    node = helper.make_node(op_type, ["x", "y"], ["z"], name=f"{op_type.lower()}_0")
    graph = helper.make_graph([node], f"single_{op_type.lower()}", [x, y], [z])
    save_model(path, graph)


def build_dynamic_mixed_binary_model(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", "cols"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", "cols"])
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, ["batch", "cols"])
    m = helper.make_tensor_value_info("m", TensorProto.FLOAT, ["batch", "cols"])
    s = helper.make_tensor_value_info("s", TensorProto.FLOAT, ["batch", "cols"])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, ["batch", "cols"])
    nodes = [
        helper.make_node("Add", ["x", "y"], ["a"], name="add_0"),
        helper.make_node("Mul", ["a", "y"], ["m"], name="mul_0"),
        helper.make_node("Sub", ["m", "x"], ["s"], name="sub_0"),
        helper.make_node("Div", ["s", "y"], ["z"], name="div_0"),
    ]
    graph = helper.make_graph(nodes, "dynamic_mixed_binary", [x, y], [z], value_info=[a, m, s])
    save_model(path, graph)


def build_single_gru_model(path: Path) -> None:
    seq_length = 3
    batch_size = 2
    input_size = 4
    hidden_size = 3

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [seq_length, batch_size, input_size])
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, [1, 3 * hidden_size, input_size])
    r = helper.make_tensor_value_info("r", TensorProto.FLOAT, [1, 3 * hidden_size, hidden_size])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 6 * hidden_size])
    initial_h = helper.make_tensor_value_info("initial_h", TensorProto.FLOAT, [1, batch_size, hidden_size])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [seq_length, 1, batch_size, hidden_size])
    y_h = helper.make_tensor_value_info("y_h", TensorProto.FLOAT, [1, batch_size, hidden_size])
    node = helper.make_node(
        "GRU",
        ["x", "w", "r", "b", "", "initial_h"],
        ["y", "y_h"],
        name="gru_0",
        hidden_size=hidden_size,
    )
    graph = helper.make_graph([node], "single_gru", [x, w, r, b, initial_h], [y, y_h])
    save_model(path, graph)


def standard_inputs() -> dict[str, np.ndarray]:
    return {
        "x": np.array([[1.0, 2.0, 3.0], [4.5, -2.0, 7.0]], dtype=np.float32),
        "y": np.array([[0.5, -1.0, 8.0], [1.5, 3.0, -4.0]], dtype=np.float32),
    }


def assert_model_matches_cpu(model_path: Path, ep_library: Path, output_name: str, inputs: dict[str, np.ndarray]) -> None:
    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)
    cpu_out = cpu.run([output_name], inputs)[0]
    ggml_out = ggml.run([output_name], inputs)[0]
    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-6, atol=1e-6)


def gru_inputs() -> dict[str, np.ndarray]:
    return {
        "x": np.array(
            [
                [[0.1, -0.2, 0.3, 0.5], [0.4, 0.0, -0.1, 0.2]],
                [[-0.3, 0.6, 0.2, -0.4], [0.7, -0.5, 0.1, 0.3]],
                [[0.2, 0.1, -0.6, 0.8], [-0.4, 0.9, 0.5, -0.2]],
            ],
            dtype=np.float32,
        ),
        "w": np.array(
            [
                [
                    [0.15, -0.20, 0.10, 0.05],
                    [-0.30, 0.25, 0.40, -0.10],
                    [0.20, 0.35, -0.15, 0.30],
                    [0.05, -0.10, 0.25, 0.40],
                    [-0.20, 0.30, -0.35, 0.15],
                    [0.45, -0.25, 0.05, -0.30],
                    [0.10, 0.15, -0.20, 0.35],
                    [-0.40, 0.05, 0.30, -0.15],
                    [0.25, -0.35, 0.20, 0.10],
                ]
            ],
            dtype=np.float32,
        ),
        "r": np.array(
            [
                [
                    [0.12, -0.18, 0.07],
                    [-0.22, 0.14, 0.31],
                    [0.28, -0.09, 0.16],
                    [0.05, 0.20, -0.11],
                    [-0.17, 0.26, 0.08],
                    [0.34, -0.21, 0.13],
                    [0.09, -0.04, 0.27],
                    [-0.29, 0.18, -0.07],
                    [0.16, 0.11, -0.24],
                ]
            ],
            dtype=np.float32,
        ),
        "b": np.array(
            [
                [
                    0.02,
                    -0.03,
                    0.01,
                    -0.04,
                    0.05,
                    -0.02,
                    0.03,
                    -0.01,
                    0.04,
                    -0.02,
                    0.01,
                    0.03,
                    0.05,
                    -0.04,
                    0.02,
                    -0.03,
                    0.02,
                    -0.01,
                ]
            ],
            dtype=np.float32,
        ),
        "initial_h": np.array(
            [[[0.05, -0.10, 0.15], [-0.20, 0.25, -0.05]]],
            dtype=np.float32,
        ),
    }


def test_single_add(suite_tmpdir, ep_library: Path) -> None:
    model_path = suite_tmpdir / "single_add.onnx"
    build_single_add_model(model_path)
    assert_model_matches_cpu(model_path, ep_library, "z", standard_inputs())


def test_dynamic_add_cache_reuse(suite_tmpdir, ep_library: Path, debug_api) -> None:
    dynamic_model_path = suite_tmpdir / "dynamic_add.onnx"
    build_dynamic_add_model(dynamic_model_path)
    cpu = cpu_session(dynamic_model_path)
    ggml = ggml_session(dynamic_model_path, ep_library)

    debug_api.reset()
    first_inputs = standard_inputs()
    cpu_out = cpu.run(["z"], first_inputs)[0]
    ggml_out = ggml.run(["z"], first_inputs)[0]
    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-6, atol=1e-6)
    assert debug_api.graph_build_count() == 1

    same_shape_inputs = {
        "x": np.array([[9.0, 1.0, -2.0], [5.0, 6.0, 7.0]], dtype=np.float32),
        "y": np.array([[1.0, 1.5, 2.0], [0.0, -3.0, 10.0]], dtype=np.float32),
    }
    cpu_out = cpu.run(["z"], same_shape_inputs)[0]
    ggml_out = ggml.run(["z"], same_shape_inputs)[0]
    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-6, atol=1e-6)
    assert debug_api.graph_build_count() == 1

    second_shape_inputs = {
        "x": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        "y": np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32),
    }
    cpu_out = cpu.run(["z"], second_shape_inputs)[0]
    ggml_out = ggml.run(["z"], second_shape_inputs)[0]
    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-6, atol=1e-6)
    assert debug_api.graph_build_count() == 2


def test_broadcast_add_runs_on_ggml(suite_tmpdir, ep_library: Path, debug_api) -> None:
    model_path = suite_tmpdir / "broadcast_add.onnx"
    build_broadcast_add_model(model_path)
    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    debug_api.reset()
    broadcast_inputs = {
        "x": np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32),
        "y": np.array([0.5, -1.0, 8.0], dtype=np.float32),
    }
    cpu_out = cpu.run(["z"], broadcast_inputs)[0]
    ggml_out = ggml.run(["z"], broadcast_inputs)[0]
    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-6, atol=1e-6)
    assert debug_api.graph_build_count() == 1


@pytest.mark.parametrize("op_type", ["Add", "Sub", "Mul", "Div"])
def test_single_binary_ops(suite_tmpdir, ep_library: Path, op_type: str) -> None:
    model_path = suite_tmpdir / f"single_{op_type.lower()}.onnx"
    build_single_binary_model(model_path, op_type)
    assert_model_matches_cpu(model_path, ep_library, "z", standard_inputs())


def test_dynamic_mixed_binary_cache_reuse(suite_tmpdir, ep_library: Path, debug_api) -> None:
    model_path = suite_tmpdir / "dynamic_mixed_binary.onnx"
    build_dynamic_mixed_binary_model(model_path)
    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    debug_api.reset()
    first_inputs = {
        "x": np.array([[1.0, 2.0, 4.0], [4.5, -2.0, 7.0]], dtype=np.float32),
        "y": np.array([[0.5, -1.0, 8.0], [1.5, 3.0, -4.0]], dtype=np.float32),
    }
    cpu_out = cpu.run(["z"], first_inputs)[0]
    ggml_out = ggml.run(["z"], first_inputs)[0]
    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-6, atol=1e-6)
    assert debug_api.graph_build_count() == 1

    same_shape_inputs = {
        "x": np.array([[2.0, 5.0, 6.0], [8.0, 3.0, -1.0]], dtype=np.float32),
        "y": np.array([[4.0, 2.0, 3.0], [2.0, -4.0, 5.0]], dtype=np.float32),
    }
    cpu_out = cpu.run(["z"], same_shape_inputs)[0]
    ggml_out = ggml.run(["z"], same_shape_inputs)[0]
    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-6, atol=1e-6)
    assert debug_api.graph_build_count() == 1

    second_shape_inputs = {
        "x": np.array([[4.0, 6.0], [8.0, 10.0], [12.0, 14.0]], dtype=np.float32),
        "y": np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], dtype=np.float32),
    }
    cpu_out = cpu.run(["z"], second_shape_inputs)[0]
    ggml_out = ggml.run(["z"], second_shape_inputs)[0]
    np.testing.assert_allclose(ggml_out, cpu_out, rtol=1e-6, atol=1e-6)
    assert debug_api.graph_build_count() == 2


def test_single_gru_matches_cpu(suite_tmpdir, ep_library: Path) -> None:
    model_path = suite_tmpdir / "single_gru.onnx"
    build_single_gru_model(model_path)
    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    inputs = gru_inputs()
    cpu_y, cpu_y_h = cpu.run(["y", "y_h"], inputs)
    ggml_y, ggml_y_h = ggml.run(["y", "y_h"], inputs)
    np.testing.assert_allclose(ggml_y, cpu_y, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ggml_y_h, cpu_y_h, rtol=1e-6, atol=1e-6)
