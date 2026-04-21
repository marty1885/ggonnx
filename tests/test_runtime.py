#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pytest
from onnx import TensorProto, helper

from test_support import (
    assert_all_nodes_run_on_ggml,
    cpu_session,
    ggml_session,
    ensure_model,
    ensure_model_with_opset,
)


def build_single_add_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3])
    node = helper.make_node("Add", ["x", "y"], ["z"], name="add_0")
    graph = helper.make_graph([node], "single_add", [x, y], [z])
    return ensure_model(tmpdir, graph)


def build_dynamic_add_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", "cols"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", "cols"])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, ["batch", "cols"])
    node = helper.make_node("Add", ["x", "y"], ["z"], name="add_dynamic")
    graph = helper.make_graph([node], "dynamic_add", [x, y], [z])
    return ensure_model(tmpdir, graph)


def build_broadcast_add_model(tmpdir: Path) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", "cols"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["cols"])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, ["batch", "cols"])
    node = helper.make_node("Add", ["x", "y"], ["z"], name="add_broadcast")
    graph = helper.make_graph([node], "broadcast_add", [x, y], [z])
    return ensure_model(tmpdir, graph)


def build_single_binary_model(tmpdir: Path, op_type: str) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3])
    node = helper.make_node(op_type, ["x", "y"], ["z"], name=f"{op_type.lower()}_0")
    graph = helper.make_graph([node], f"single_{op_type.lower()}", [x, y], [z])
    return ensure_model(tmpdir, graph)


def build_dynamic_mixed_binary_model(tmpdir: Path) -> Path:
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
    return ensure_model(tmpdir, graph)


def build_single_gru_model(tmpdir: Path) -> Path:
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
    return ensure_model(tmpdir, graph)


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
    assert_all_nodes_run_on_ggml(ggml)


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
    model_path = build_single_add_model(suite_tmpdir)
    assert_model_matches_cpu(model_path, ep_library, "z", standard_inputs())


def test_dynamic_add_cache_reuse(suite_tmpdir, ep_library: Path, debug_api) -> None:
    dynamic_model_path = build_dynamic_add_model(suite_tmpdir)
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
    assert_all_nodes_run_on_ggml(ggml)


def test_broadcast_add_runs_on_ggml(suite_tmpdir, ep_library: Path, debug_api) -> None:
    model_path = build_broadcast_add_model(suite_tmpdir)
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
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.parametrize("op_type", ["Add", "Sub", "Mul", "Div"])
def test_single_binary_ops(suite_tmpdir, ep_library: Path, op_type: str) -> None:
    model_path = build_single_binary_model(suite_tmpdir, op_type)
    assert_model_matches_cpu(model_path, ep_library, "z", standard_inputs())


def test_dynamic_mixed_binary_cache_reuse(suite_tmpdir, ep_library: Path, debug_api) -> None:
    model_path = build_dynamic_mixed_binary_model(suite_tmpdir)
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
    assert_all_nodes_run_on_ggml(ggml)


def build_single_unary_model(tmpdir: Path, op_type: str, shape=(2, 3), **attrs) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(shape))
    node = helper.make_node(op_type, ["x"], ["y"], name=f"{op_type.lower()}_0", **attrs)
    graph = helper.make_graph([node], f"single_{op_type.lower()}", [x], [y])
    return ensure_model(tmpdir, graph)


def build_matmul_model(tmpdir: Path, a_shape, b_shape, out_shape) -> Path:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape))
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape))
    c = helper.make_tensor_value_info("c", TensorProto.FLOAT, list(out_shape))
    node = helper.make_node("MatMul", ["a", "b"], ["c"], name="matmul_0")
    graph = helper.make_graph([node], "single_matmul", [a, b], [c])
    return ensure_model(tmpdir, graph)


_UNARY_INPUT_OVERRIDES = {
    # Inputs suited to each op's valid domain.
    "Sqrt": np.array([[0.25, 1.0, 4.0], [9.0, 0.5, 2.0]], dtype=np.float32),
    "Log":  np.array([[0.5, 1.0, 2.0], [3.0, 4.0, 0.1]], dtype=np.float32),
}


@pytest.mark.parametrize(
    "op_type",
    ["Relu", "Sigmoid", "Tanh", "Neg", "Abs", "Sqrt", "Exp", "Log", "Softplus", "Elu"],
)
def test_single_unary_ops(suite_tmpdir, ep_library: Path, op_type: str) -> None:
    model_path = build_single_unary_model(suite_tmpdir, op_type)
    default_x = np.array([[1.0, -2.0, 0.5], [-0.25, 3.0, -1.5]], dtype=np.float32)
    inputs = {"x": _UNARY_INPUT_OVERRIDES.get(op_type, default_x)}
    assert_model_matches_cpu(model_path, ep_library, "y", inputs)


@pytest.mark.parametrize("alpha", [0.01, 0.2])
def test_single_leaky_relu(suite_tmpdir, ep_library: Path, alpha: float) -> None:
    model_path = build_single_unary_model(suite_tmpdir, "LeakyRelu", alpha=alpha)
    inputs = {"x": np.array([[1.0, -2.0, 0.5], [-0.25, 3.0, -1.5]], dtype=np.float32)}
    assert_model_matches_cpu(model_path, ep_library, "y", inputs)


@pytest.mark.parametrize("shape,axis", [((2, 4), -1), ((2, 4), 1), ((2, 3, 4), -1)])
def test_single_softmax(suite_tmpdir, ep_library: Path, shape, axis: int) -> None:
    model_path = build_single_unary_model(suite_tmpdir, "Softmax", shape=shape, axis=axis)
    rng = np.random.default_rng(0)
    inputs = {"x": rng.standard_normal(shape).astype(np.float32)}
    assert_model_matches_cpu(model_path, ep_library, "y", inputs)


@pytest.mark.parametrize(
    "a_shape,b_shape,out_shape",
    [
        ((3, 4), (4, 5), (3, 5)),
        ((2, 3, 4), (2, 4, 5), (2, 3, 5)),
        ((2, 2, 3, 4), (2, 2, 4, 5), (2, 2, 3, 5)),
    ],
)
def test_single_matmul(suite_tmpdir, ep_library: Path, a_shape, b_shape, out_shape) -> None:
    model_path = build_matmul_model(suite_tmpdir, a_shape, b_shape, out_shape)
    rng = np.random.default_rng(1)
    inputs = {
        "a": rng.standard_normal(a_shape).astype(np.float32),
        "b": rng.standard_normal(b_shape).astype(np.float32),
    }
    model = cpu_session(model_path)
    expected = model.run(["c"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["c"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert_all_nodes_run_on_ggml(ggml)


def build_conv_model(
    tmpdir: Path,
    x_shape,
    w_shape,
    y_shape,
    *,
    with_bias: bool = False,
    **attrs,
) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    w = helper.make_tensor_value_info("w", TensorProto.FLOAT, list(w_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))
    graph_inputs = [x, w]
    node_inputs = ["x", "w"]
    if with_bias:
        b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [w_shape[0]])
        graph_inputs.append(b)
        node_inputs.append("b")
    node = helper.make_node("Conv", node_inputs, ["y"], name="conv_0", **attrs)
    graph = helper.make_graph([node], "single_conv", graph_inputs, [y])
    return ensure_model(tmpdir, graph)


def _conv_inputs(x_shape, w_shape, with_bias: bool):
    rng = np.random.default_rng(2)
    inputs = {
        "x": rng.standard_normal(x_shape).astype(np.float32),
        "w": rng.standard_normal(w_shape).astype(np.float32),
    }
    if with_bias:
        inputs["b"] = rng.standard_normal((w_shape[0],)).astype(np.float32)
    return inputs


# (x_shape, w_shape, y_shape, attrs, with_bias)
_CONV_CASES = [
    # plain 3x3, no pad, stride 1 — [N=1, C=1, 5, 5] * [4,1,3,3] -> [1,4,3,3]
    ((1, 1, 5, 5), (4, 1, 3, 3), (1, 4, 3, 3),
     dict(kernel_shape=[3, 3]), False),
    # with bias
    ((1, 1, 5, 5), (4, 1, 3, 3), (1, 4, 3, 3),
     dict(kernel_shape=[3, 3]), True),
    # stride 2
    ((1, 3, 8, 8), (6, 3, 3, 3), (1, 6, 3, 3),
     dict(kernel_shape=[3, 3], strides=[2, 2]), True),
    # padding 1 (preserves spatial with 3x3 s=1)
    ((2, 3, 7, 7), (5, 3, 3, 3), (2, 5, 7, 7),
     dict(kernel_shape=[3, 3], pads=[1, 1, 1, 1]), True),
    # LeNet-style: 1x28x28 with 5x5 kernel, 6 out channels, no pad, s=1 -> 1x6x24x24
    ((1, 1, 28, 28), (6, 1, 5, 5), (1, 6, 24, 24),
     dict(kernel_shape=[5, 5]), True),
    # LeNet conv2: 6->16, 5x5, no pad -> from 12x12 -> 8x8
    ((1, 6, 12, 12), (16, 6, 5, 5), (1, 16, 8, 8),
     dict(kernel_shape=[5, 5]), True),
    # dilation 2
    ((1, 2, 9, 9), (4, 2, 3, 3), (1, 4, 5, 5),
     dict(kernel_shape=[3, 3], dilations=[2, 2]), False),
    # asymmetric kernel
    ((1, 2, 6, 8), (3, 2, 1, 3), (1, 3, 6, 6),
     dict(kernel_shape=[1, 3]), True),
]


@pytest.mark.parametrize("x_shape,w_shape,y_shape,attrs,with_bias", _CONV_CASES)
def test_single_conv(suite_tmpdir, ep_library: Path, x_shape, w_shape, y_shape, attrs, with_bias) -> None:
    model_path = build_conv_model(
        suite_tmpdir, x_shape, w_shape, y_shape, with_bias=with_bias, **attrs
    )
    inputs = _conv_inputs(x_shape, w_shape, with_bias)
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)
    assert_all_nodes_run_on_ggml(ggml)


def build_pool_model(
    tmpdir: Path,
    op_type: str,
    x_shape,
    y_shape,
    **attrs,
) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))
    node = helper.make_node(op_type, ["x"], ["y"], name=f"{op_type.lower()}_0", **attrs)
    graph = helper.make_graph([node], f"single_{op_type.lower()}", [x], [y])
    return ensure_model(tmpdir, graph)


def build_pad_model(tmpdir: Path, x_shape, y_shape, *, pads, mode="reflect", variant="legacy_attr") -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))
    if variant == "legacy_attr":
        node = helper.make_node("Pad", ["x"], ["y"], name="pad_0", pads=pads, mode=mode)
        graph = helper.make_graph([node], "single_pad_legacy", [x], [y])
        return ensure_model_with_opset(tmpdir, graph, 2)
    if variant == "input_pads":
        pads_init = helper.make_tensor("pads", TensorProto.INT64, [len(pads)], list(pads))
        node = helper.make_node("Pad", ["x", "pads"], ["y"], name="pad_0", mode=mode)
        graph = helper.make_graph([node], "single_pad_input_pads", [x], [y], initializer=[pads_init])
        return ensure_model_with_opset(tmpdir, graph, 13)
    raise ValueError(f"unsupported Pad variant: {variant}")


def build_instance_norm_model(tmpdir: Path, x_shape) -> Path:
    c = x_shape[1]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [c])
    bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [c])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x_shape))
    node = helper.make_node(
        "InstanceNormalization",
        ["x", "scale", "bias"],
        ["y"],
        name="instance_norm_0",
        epsilon=1e-5,
    )
    graph = helper.make_graph([node], "single_instance_norm", [x, scale, bias], [y])
    return ensure_model(tmpdir, graph)


def build_upsample_model(
    tmpdir: Path, x_shape, y_shape, scales, *, variant="input_scales"
) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))
    if variant == "legacy_attr":
        node = helper.make_node("Upsample", ["x"], ["y"], name="upsample_0", mode="nearest", scales=scales)
        graph = helper.make_graph([node], "single_upsample_legacy", [x], [y])
        return ensure_model_with_opset(tmpdir, graph, 7)
    if variant == "input_scales":
        scales_init = helper.make_tensor("scales", TensorProto.FLOAT, [4], list(scales))
        node = helper.make_node("Upsample", ["x", "scales"], ["y"], name="upsample_0", mode="nearest")
        graph = helper.make_graph([node], "single_upsample_input_scales", [x], [y], initializer=[scales_init])
        return ensure_model_with_opset(tmpdir, graph, 9)
    if variant == "resize":
        scales_init = helper.make_tensor("scales", TensorProto.FLOAT, [4], list(scales))
        node = helper.make_node(
            "Resize",
            ["x", "", "scales"],
            ["y"],
            name="resize_0",
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        )
        graph = helper.make_graph([node], "single_resize", [x], [y], initializer=[scales_init])
        return ensure_model_with_opset(tmpdir, graph, 13)
    raise ValueError(f"unsupported upsample variant: {variant}")


# (op_type, x_shape, y_shape, attrs)
_POOL_CASES = [
    # MaxPool, no pad, k=2 s=2
    ("MaxPool", (1, 3, 8, 8), (1, 3, 4, 4), dict(kernel_shape=[2, 2], strides=[2, 2])),
    # AveragePool, no pad, k=2 s=2
    ("AveragePool", (1, 3, 8, 8), (1, 3, 4, 4), dict(kernel_shape=[2, 2], strides=[2, 2])),
    # MaxPool with symmetric padding (padding matters less here since max ignores pad)
    ("MaxPool", (2, 4, 7, 7), (2, 4, 7, 7),
     dict(kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])),
    # AveragePool with count_include_pad=1 to match GGML's semantics
    ("AveragePool", (1, 2, 6, 6), (1, 2, 6, 6),
     dict(kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1], count_include_pad=1)),
    # asymmetric kernel, stride 1
    ("MaxPool", (1, 2, 6, 8), (1, 2, 6, 6), dict(kernel_shape=[1, 3])),
    # auto_pad VALID
    ("AveragePool", (1, 3, 10, 10), (1, 3, 5, 5),
     dict(kernel_shape=[2, 2], strides=[2, 2], auto_pad="VALID")),
]


@pytest.mark.parametrize("op_type,x_shape,y_shape,attrs", _POOL_CASES)
def test_single_pool(suite_tmpdir, ep_library: Path, op_type, x_shape, y_shape, attrs) -> None:
    model_path = build_pool_model(suite_tmpdir, op_type, x_shape, y_shape, **attrs)
    rng = np.random.default_rng(9)
    inputs = {"x": rng.standard_normal(x_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert_all_nodes_run_on_ggml(ggml)


_GLOBAL_POOL_CASES = [
    ("GlobalMaxPool", (1, 4, 7, 7), (1, 4, 1, 1)),
    ("GlobalAveragePool", (2, 3, 5, 8), (2, 3, 1, 1)),
]


@pytest.mark.parametrize("op_type,x_shape,y_shape", _GLOBAL_POOL_CASES)
def test_single_global_pool(suite_tmpdir, ep_library: Path, op_type, x_shape, y_shape) -> None:
    model_path = build_pool_model(suite_tmpdir, op_type, x_shape, y_shape)
    rng = np.random.default_rng(10)
    inputs = {"x": rng.standard_normal(x_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.parametrize("variant", ["legacy_attr", "input_pads"])
@pytest.mark.parametrize(
    "x_shape,y_shape,pads",
    [
        ((1, 1, 4, 5), (1, 1, 6, 7), [0, 0, 1, 1, 0, 0, 1, 1]),
        ((1, 2, 6, 6), (1, 2, 14, 14), [0, 0, 4, 4, 0, 0, 4, 4]),
    ],
)
def test_single_pad_reflect(suite_tmpdir, ep_library: Path, variant, x_shape, y_shape, pads) -> None:
    model_path = build_pad_model(suite_tmpdir, x_shape, y_shape, pads=pads, variant=variant)
    rng = np.random.default_rng(11)
    inputs = {"x": rng.standard_normal(x_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.parametrize("x_shape", [(1, 3, 8, 8), (2, 4, 5, 7)])
def test_single_instance_normalization(suite_tmpdir, ep_library: Path, x_shape) -> None:
    model_path = build_instance_norm_model(suite_tmpdir, x_shape)
    c = x_shape[1]
    rng = np.random.default_rng(12)
    inputs = {
        "x": rng.standard_normal(x_shape).astype(np.float32),
        "scale": rng.standard_normal((c,)).astype(np.float32),
        "bias": rng.standard_normal((c,)).astype(np.float32),
    }
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert_all_nodes_run_on_ggml(ggml)


@pytest.mark.parametrize("variant", ["legacy_attr", "input_scales", "resize"])
@pytest.mark.parametrize(
    "x_shape,y_shape,scales",
    [
        ((1, 3, 8, 8), (1, 3, 16, 16), [1.0, 1.0, 2.0, 2.0]),
        ((1, 2, 5, 7), (1, 2, 10, 14), [1.0, 1.0, 2.0, 2.0]),
    ],
)
def test_single_upsample_nearest(
    suite_tmpdir, ep_library: Path, variant, x_shape, y_shape, scales
) -> None:
    model_path = build_upsample_model(suite_tmpdir, x_shape, y_shape, scales, variant=variant)
    rng = np.random.default_rng(13)
    inputs = {"x": rng.standard_normal(x_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


def build_matmul_const_b_model(tmpdir: Path, a_shape, b_shape, out_shape, b_values) -> Path:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape))
    c = helper.make_tensor_value_info("c", TensorProto.FLOAT, list(out_shape))
    b_init = helper.make_tensor("b", TensorProto.FLOAT, list(b_shape), b_values.flatten().tolist())
    node = helper.make_node("MatMul", ["a", "b"], ["c"], name="matmul_const")
    graph = helper.make_graph([node], "matmul_const_b", [a], [c], initializer=[b_init])
    return ensure_model(tmpdir, graph)


@pytest.mark.parametrize(
    "a_shape,b_shape,out_shape",
    [
        ((3, 4), (4, 5), (3, 5)),
        ((2, 3, 4), (2, 4, 5), (2, 3, 5)),
    ],
)
def test_matmul_constant_b(suite_tmpdir, ep_library: Path, a_shape, b_shape, out_shape) -> None:
    rng = np.random.default_rng(7)
    b_values = rng.standard_normal(b_shape).astype(np.float32)
    model_path = build_matmul_const_b_model(suite_tmpdir, a_shape, b_shape, out_shape, b_values)
    inputs = {"a": rng.standard_normal(a_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["c"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["c"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert_all_nodes_run_on_ggml(ggml)


def build_gemm_const_b_model(
    tmpdir: Path, a_shape, b_shape, y_shape, b_values, *, trans_b: int = 0
) -> Path:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))
    b_init = helper.make_tensor("b", TensorProto.FLOAT, list(b_shape), b_values.flatten().tolist())
    node = helper.make_node("Gemm", ["a", "b"], ["y"], name="gemm_const", transB=trans_b)
    graph = helper.make_graph([node], "gemm_const_b", [a], [y], initializer=[b_init])
    return ensure_model(tmpdir, graph)


@pytest.mark.parametrize(
    "a_shape,b_shape,y_shape,trans_b",
    [
        ((4, 5), (5, 3), (4, 3), 0),
        ((4, 5), (3, 5), (4, 3), 1),
    ],
)
def test_gemm_constant_b(suite_tmpdir, ep_library: Path, a_shape, b_shape, y_shape, trans_b) -> None:
    rng = np.random.default_rng(8)
    b_values = rng.standard_normal(b_shape).astype(np.float32)
    model_path = build_gemm_const_b_model(
        suite_tmpdir, a_shape, b_shape, y_shape, b_values, trans_b=trans_b
    )
    inputs = {"a": rng.standard_normal(a_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert_all_nodes_run_on_ggml(ggml)


def build_gemm_model(
    tmpdir: Path,
    a_shape,
    b_shape,
    y_shape,
    *,
    with_c: bool = False,
    c_shape=None,
    **attrs,
) -> Path:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a_shape))
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))
    graph_inputs = [a, b]
    node_inputs = ["a", "b"]
    if with_c:
        c = helper.make_tensor_value_info("c", TensorProto.FLOAT, list(c_shape))
        graph_inputs.append(c)
        node_inputs.append("c")
    node = helper.make_node("Gemm", node_inputs, ["y"], name="gemm_0", **attrs)
    graph = helper.make_graph([node], "single_gemm", graph_inputs, [y])
    return ensure_model(tmpdir, graph)


# (a_shape, b_shape, y_shape, attrs, c_shape or None)
_GEMM_CASES = [
    # plain: A[M,K] @ B[K,N]
    ((4, 5), (5, 3), (4, 3), {}, None),
    # with bias vector [N]
    ((4, 5), (5, 3), (4, 3), {}, (3,)),
    # with bias [M,N]
    ((4, 5), (5, 3), (4, 3), {}, (4, 3)),
    # transA
    ((5, 4), (5, 3), (4, 3), dict(transA=1), (3,)),
    # transB
    ((4, 5), (3, 5), (4, 3), dict(transB=1), (3,)),
    # transA + transB
    ((5, 4), (3, 5), (4, 3), dict(transA=1, transB=1), (3,)),
    # alpha + beta
    ((4, 5), (5, 3), (4, 3), dict(alpha=0.5, beta=2.0), (3,)),
]


@pytest.mark.parametrize("a_shape,b_shape,y_shape,attrs,c_shape", _GEMM_CASES)
def test_single_gemm(suite_tmpdir, ep_library: Path, a_shape, b_shape, y_shape, attrs, c_shape) -> None:
    with_c = c_shape is not None
    model_path = build_gemm_model(
        suite_tmpdir, a_shape, b_shape, y_shape, with_c=with_c, c_shape=c_shape, **attrs
    )
    rng = np.random.default_rng(3)
    inputs = {
        "a": rng.standard_normal(a_shape).astype(np.float32),
        "b": rng.standard_normal(b_shape).astype(np.float32),
    }
    if with_c:
        inputs["c"] = rng.standard_normal(c_shape).astype(np.float32)
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert_all_nodes_run_on_ggml(ggml)


def build_reshape_model(tmpdir: Path, in_shape, out_shape) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(in_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(out_shape))
    shape_init = helper.make_tensor(
        "shape", TensorProto.INT64, [len(out_shape)], list(out_shape)
    )
    node = helper.make_node("Reshape", ["x", "shape"], ["y"], name="reshape_0")
    graph = helper.make_graph(
        [node], "single_reshape", [x], [y], initializer=[shape_init]
    )
    return ensure_model(tmpdir, graph)


@pytest.mark.parametrize(
    "in_shape,out_shape",
    [
        ((2, 3, 4), (6, 4)),
        ((1, 6, 12, 12), (1, 864)),
        ((24,), (2, 3, 4)),
        ((2, 3, 4, 5), (2, 60)),
    ],
)
def test_single_reshape(suite_tmpdir, ep_library: Path, in_shape, out_shape) -> None:
    model_path = build_reshape_model(suite_tmpdir, in_shape, out_shape)
    rng = np.random.default_rng(4)
    inputs = {"x": rng.standard_normal(in_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


def build_slice_model(tmpdir: Path, x_shape, y_shape, starts, ends, axes=None, steps=None) -> Path:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(y_shape))
    inits = [
        helper.make_tensor("starts", TensorProto.INT64, [len(starts)], list(starts)),
        helper.make_tensor("ends", TensorProto.INT64, [len(ends)], list(ends)),
    ]
    node_inputs = ["x", "starts", "ends"]
    if axes is not None:
        inits.append(helper.make_tensor("axes", TensorProto.INT64, [len(axes)], list(axes)))
        node_inputs.append("axes")
    if steps is not None:
        if axes is None:
            inits.append(helper.make_tensor("axes", TensorProto.INT64, [len(steps)],
                                            list(range(len(steps)))))
            node_inputs.append("axes")
        inits.append(helper.make_tensor("steps", TensorProto.INT64, [len(steps)], list(steps)))
        node_inputs.append("steps")
    node = helper.make_node("Slice", node_inputs, ["y"], name="slice_0")
    graph = helper.make_graph([node], "single_slice", [x], [y], initializer=inits)
    return ensure_model(tmpdir, graph)


@pytest.mark.parametrize(
    "x_shape,y_shape,starts,ends,axes",
    [
        # Basic rank-4 slice on spatial dim
        ((1, 3, 8, 8), (1, 3, 4, 8), [0], [4], [2]),
        # Multi-axis slice
        ((1, 3, 8, 8), (1, 2, 4, 4), [1, 0, 2], [3, 4, 6], [1, 2, 3]),
        # Negative indexing
        ((1, 3, 8, 8), (1, 3, 8, 4), [-4], [8], [3]),
        # Rank 2
        ((6, 10), (3, 5), [1, 2], [4, 7], [0, 1]),
        # Rank 1
        ((12,), (4,), [2], [6], [0]),
        # Clamping: end beyond dim
        ((4, 5), (4, 3), [2], [99], [1]),
        # Default axes (none passed) — slice first len(starts) dims
        ((6, 8), (3, 8), [0], [3], None),
    ],
)
def test_single_slice(suite_tmpdir, ep_library: Path, x_shape, y_shape, starts, ends, axes) -> None:
    model_path = build_slice_model(suite_tmpdir, x_shape, y_shape, starts, ends, axes=axes)
    rng = np.random.default_rng(21)
    inputs = {"x": rng.standard_normal(x_shape).astype(np.float32)}
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)


def build_batchnorm_model(tmpdir: Path, x_shape, *, epsilon=1e-5) -> Path:
    c = x_shape[1]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))
    scale = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [c])
    bias = helper.make_tensor_value_info("bias", TensorProto.FLOAT, [c])
    mean = helper.make_tensor_value_info("mean", TensorProto.FLOAT, [c])
    var = helper.make_tensor_value_info("var", TensorProto.FLOAT, [c])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x_shape))
    node = helper.make_node(
        "BatchNormalization",
        ["x", "scale", "bias", "mean", "var"],
        ["y"],
        name="bn_0",
        epsilon=epsilon,
    )
    graph = helper.make_graph([node], "single_bn", [x, scale, bias, mean, var], [y])
    return ensure_model(tmpdir, graph)


@pytest.mark.parametrize("x_shape", [(1, 4, 6, 6), (2, 8, 3, 3), (1, 3, 16, 16)])
def test_single_batchnorm(suite_tmpdir, ep_library: Path, x_shape) -> None:
    model_path = build_batchnorm_model(suite_tmpdir, x_shape)
    rng = np.random.default_rng(22)
    c = x_shape[1]
    inputs = {
        "x": rng.standard_normal(x_shape).astype(np.float32),
        "scale": rng.standard_normal((c,)).astype(np.float32),
        "bias": rng.standard_normal((c,)).astype(np.float32),
        "mean": rng.standard_normal((c,)).astype(np.float32),
        # var must be positive for sqrt to be meaningful
        "var": np.abs(rng.standard_normal((c,)).astype(np.float32)) + 0.1,
    }
    cpu = cpu_session(model_path)
    expected = cpu.run(["y"], inputs)[0]
    ggml = ggml_session(model_path, ep_library)
    got = ggml.run(["y"], inputs)[0]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    assert_all_nodes_run_on_ggml(ggml)


def test_single_gru_matches_cpu(suite_tmpdir, ep_library: Path) -> None:
    model_path = build_single_gru_model(suite_tmpdir)
    cpu = cpu_session(model_path)
    ggml = ggml_session(model_path, ep_library)

    inputs = gru_inputs()
    cpu_y, cpu_y_h = cpu.run(["y", "y_h"], inputs)
    ggml_y, ggml_y_h = ggml.run(["y", "y_h"], inputs)
    np.testing.assert_allclose(ggml_y, cpu_y, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ggml_y_h, cpu_y_h, rtol=1e-6, atol=1e-6)
    assert_all_nodes_run_on_ggml(ggml)
