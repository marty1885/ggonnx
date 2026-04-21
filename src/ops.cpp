#include "ops.hpp"
#include "inner/helpers.hpp"
#include <unordered_map>

TensorMetadata getTensorMetadata(Ort::ConstValueInfo value_info) {
  const auto tensor_info = value_info.TypeInfo().GetTensorTypeAndShapeInfo();

  TensorMetadata result;
  result.element_type = tensor_info.GetElementType();
  result.dims = tensor_info.GetShape();
  return result;
}

TensorMetadata getTensorMetadata(Ort::ConstValue value) {
  const auto tensor_info = value.GetTensorTypeAndShapeInfo();

  TensorMetadata result;
  result.element_type = tensor_info.GetElementType();
  result.dims = tensor_info.GetShape();
  return result;
}

template <typename T>
std::optional<T> readNodeAttribute(Ort::ConstNode node, const char* attribute_name) {
  Ort::ConstOpAttr attr{nullptr};
  const Ort::Status status = node.GetAttributeByName(attribute_name, attr);
  if (!status.IsOK() || attr == nullptr) {
    return std::nullopt;
  }

  T value{};
  if constexpr (std::is_same_v<T, std::vector<std::string>> ||
                std::is_same_v<T, std::vector<int64_t>> ||
                std::is_same_v<T, std::vector<float>>) {
    T values;
    Ort::ThrowOnError(attr.GetValueArray(values));
    return values;
  } else {
    Ort::ThrowOnError(attr.GetValue(value));
    return value;
  }
}

bool IsSupportedElementwiseBinaryNode(Ort::ConstNode node, std::string_view op_type) {
  if (!(op_type == "Add" || op_type == "Sub" || op_type == "Mul" || op_type == "Div")) {
    return false;
  }

  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  const auto implicit_inputs = node.GetImplicitInputs();
  const size_t num_inputs = inputs.size();
  const size_t num_outputs = outputs.size();
  const size_t num_implicit_inputs = implicit_inputs.size();
  if (num_inputs != 2 || num_outputs != 1 || num_implicit_inputs != 0) {
    return false;
  }

  GGONNX_ASSERT(inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                "ORT returned null binary op input/output metadata");

  const TensorMetadata lhs = getTensorMetadata(inputs[0]);
  const TensorMetadata rhs = getTensorMetadata(inputs[1]);
  const TensorMetadata out = getTensorMetadata(outputs[0]);

  if (lhs.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      rhs.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      out.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (!rankSupportedByGGML(lhs) || !rankSupportedByGGML(rhs) || !rankSupportedByGGML(out)) {
    return false;
  }
  if (!broadcastSupportedByGGML(lhs.dims, rhs.dims)) {
    return false;
  }
  if (!broadcastSupportedByGGML(lhs.dims, out.dims)) {
    return false;
  }

  return true;
}

bool IsSupportedElementwiseBinaryOpNode(Ort::ConstNode node) {
  return IsSupportedElementwiseBinaryNode(node, node.GetOperatorType());
}

bool IsSupportedGRUNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  const auto implicit_inputs = node.GetImplicitInputs();
  const size_t num_inputs = inputs.size();
  const size_t num_outputs = outputs.size();
  const size_t num_implicit_inputs = implicit_inputs.size();
  if (num_inputs < 3 || num_outputs == 0 || num_implicit_inputs != 0) {
    return false;
  }

  std::string direction = readNodeAttribute<std::string>(node, "direction").value_or("forward");
  if (direction != "forward") {
    return false;
  }

  const int64_t layout = readNodeAttribute<int64_t>(node, "layout").value_or(0);
  if (layout != 0) {
    return false;
  }

  const int64_t linear_before_reset =
      readNodeAttribute<int64_t>(node, "linear_before_reset").value_or(0);
  if (linear_before_reset != 0) {
    return false;
  }

  if (readNodeAttribute<float>(node, "clip").has_value()) {
    return false;
  }

  if (const auto activations = readNodeAttribute<std::vector<std::string>>(node, "activations")) {
    if (activations->size() != 2 || (*activations)[0] != "Sigmoid" || (*activations)[1] != "Tanh") {
      return false;
    }
  }

  if (inputs.size() < 3 || inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr) {
    return false;
  }
  if (inputs.size() > 4 && inputs[4] != nullptr) {
    return false;
  }

  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata w = getTensorMetadata(inputs[1]);
  const TensorMetadata r = getTensorMetadata(inputs[2]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      w.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      r.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (x.dims.size() != 3 || w.dims.size() != 3 || r.dims.size() != 3) {
    return false;
  }
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(w) || !rankSupportedByGGML(r)) {
    return false;
  }

  const auto hidden_size = readNodeAttribute<int64_t>(node, "hidden_size");
  if (!hidden_size.has_value()) {
    return false;
  }
  if (*hidden_size <= 0 || w.dims[0] != 1 || r.dims[0] != 1 || w.dims[1] != *hidden_size * 3 ||
      r.dims[1] != *hidden_size * 3 || w.dims[2] <= 0 || r.dims[2] != *hidden_size) {
    return false;
  }
  if (x.dims[2] >= 0 && x.dims[2] != w.dims[2]) {
    return false;
  }

  if (inputs.size() > 3 && inputs[3] != nullptr) {
    const TensorMetadata b = getTensorMetadata(inputs[3]);
    if (b.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || b.dims.size() != 2 || b.dims[0] != 1 ||
        b.dims[1] != *hidden_size * 6) {
      return false;
    }
  }

  if (inputs.size() > 5 && inputs[5] != nullptr) {
    const TensorMetadata initial_h = getTensorMetadata(inputs[5]);
    if (initial_h.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || initial_h.dims.size() != 3 ||
        initial_h.dims[0] != 1 || initial_h.dims[2] != *hidden_size) {
      return false;
    }
    if (x.dims[1] >= 0 && initial_h.dims[1] >= 0 && x.dims[1] != initial_h.dims[1]) {
      return false;
    }
  }

  for (Ort::ConstValueInfo output : outputs) {
    if (output == nullptr) {
      continue;
    }
    const TensorMetadata meta = getTensorMetadata(output);
    if (meta.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || !rankSupportedByGGML(meta)) {
      return false;
    }
  }

  return true;
}

void CompileGRUAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const std::string direction = readNodeAttribute<std::string>(node, "direction").value_or("forward");
  GGONNX_ASSERT(direction == "forward", "only forward GRU direction is supported");
  const auto hidden_size = readNodeAttribute<int64_t>(node, "hidden_size");
  GGONNX_ASSERT(hidden_size.has_value() && *hidden_size > 0,
                "GRU hidden_size attribute must be present and positive");
  compiled_node->attrs = NodeDesc::GRUAttrs{.hidden_size = *hidden_size};
}

EmitResult EmitElementwiseBinaryNode(ggml_context* ctx,
                                     const NodeDesc& node,
                                     const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  if (node.inputs.size() != 2 || node.outputs.size() != 1) {
    throw std::runtime_error("compiled binary op node has invalid arity");
  }
  GGONNX_ASSERT(node.inputs[0] < values.size() && node.inputs[1] < values.size(),
                "compiled binary op node input index out of range");
  GGONNX_ASSERT(node.outputs[0] < values.size(), "compiled binary op node output index out of range");

  ggml_tensor* lhs = values[node.inputs[0]];
  ggml_tensor* rhs = values[node.inputs[1]];
  if (lhs == nullptr || rhs == nullptr) {
    throw std::runtime_error("compiled binary op node missing GGML inputs");
  }

  const std::string_view op_type(node.op_type);
  if (op_type == "Add") {
    return EmitOutputs{ggml_add(ctx, lhs, rhs)};
  } else if (op_type == "Sub") {
    return EmitOutputs{ggml_sub(ctx, lhs, rhs)};
  } else if (op_type == "Mul") {
    return EmitOutputs{ggml_mul(ctx, lhs, rhs)};
  } else if (op_type == "Div") {
    return EmitOutputs{ggml_div(ctx, lhs, rhs)};
  }

  return std::nullopt;
}

EmitResult EmitGRUNode(ggml_context* ctx,
                       const NodeDesc& node,
                       const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() >= 3, "compiled GRU node has invalid input arity");
  const auto* attrs = std::get_if<NodeDesc::GRUAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled GRU node missing GRU attributes");
  GGONNX_ASSERT(attrs->hidden_size > 0, "compiled GRU node must have positive hidden_size");

  ggml_tensor* x = values[node.inputs[0]];
  ggml_tensor* w = values[node.inputs[1]];
  ggml_tensor* r = values[node.inputs[2]];
  ggml_tensor* b = (node.inputs.size() > 3 && node.inputs[3] != kOptionalValueAbsent) ? values[node.inputs[3]] : nullptr;
  ggml_tensor* initial_h =
      (node.inputs.size() > 5 && node.inputs[5] != kOptionalValueAbsent) ? values[node.inputs[5]] : nullptr;
  if (x == nullptr || w == nullptr || r == nullptr) {
    throw std::runtime_error("compiled GRU node missing required GGML inputs");
  }

  const int64_t input_size = x->ne[0];
  const int64_t batch_size = x->ne[1];
  const int64_t seq_length = x->ne[2];
  const int64_t hidden_size = attrs->hidden_size;
  GGONNX_ASSERT(w->ne[0] == input_size && w->ne[1] == hidden_size * 3 && w->ne[2] == 1,
                "compiled GRU W tensor shape mismatch");
  GGONNX_ASSERT(r->ne[0] == hidden_size && r->ne[1] == hidden_size * 3 && r->ne[2] == 1,
                "compiled GRU R tensor shape mismatch");

  auto matrix_slice_rows = [&](ggml_tensor* matrix, int64_t row_offset, int64_t row_count) -> ggml_tensor* {
    return ggml_view_2d(ctx,
                        matrix,
                        matrix->ne[0],
                        row_count,
                        matrix->nb[1],
                        static_cast<size_t>(row_offset) * matrix->nb[1]);
  };
  auto vector_slice = [&](ggml_tensor* vector, int64_t offset, int64_t length) -> ggml_tensor* {
    return ggml_view_1d(
        ctx, vector, length, static_cast<size_t>(offset) * ggml_element_size(vector));
  };
  auto timestep_slice = [&](ggml_tensor* sequence, int64_t step) -> ggml_tensor* {
    return ggml_view_2d(ctx,
                        sequence,
                        sequence->ne[0],
                        sequence->ne[1],
                        sequence->nb[1],
                        static_cast<size_t>(step) * sequence->nb[2]);
  };
  auto broadcast_add = [&](ggml_tensor* matrix, ggml_tensor* bias) -> ggml_tensor* {
    return ggml_add(ctx, matrix, ggml_repeat(ctx, bias, matrix));
  };

  ggml_tensor* wb = nullptr;
  ggml_tensor* rb = nullptr;
  if (b != nullptr) {
    GGONNX_ASSERT(b->ne[0] == hidden_size * 6 && b->ne[1] == 1, "compiled GRU bias tensor shape mismatch");
    wb = vector_slice(b, 0, hidden_size * 3);
    rb = vector_slice(b, hidden_size * 3, hidden_size * 3);
  }

  ggml_tensor* h_t = nullptr;
  if (initial_h != nullptr) {
    GGONNX_ASSERT(initial_h->ne[0] == hidden_size && initial_h->ne[1] == batch_size && initial_h->ne[2] == 1,
                  "compiled GRU initial_h tensor shape mismatch");
    h_t = ggml_view_2d(ctx, initial_h, initial_h->ne[0], initial_h->ne[1], initial_h->nb[1], 0);
  } else {
    h_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, batch_size);
    std::memset(h_t->data, 0, ggml_nbytes(h_t));
  }

  ggml_tensor* y = nullptr;
  if (!node.outputs.empty() && node.outputs[0] != kOptionalValueAbsent) {
    y = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden_size, batch_size, 1, seq_length);
  }

  ggml_tensor* w_z = matrix_slice_rows(w, 0, hidden_size);
  ggml_tensor* w_r = matrix_slice_rows(w, hidden_size, hidden_size);
  ggml_tensor* w_h = matrix_slice_rows(w, hidden_size * 2, hidden_size);
  ggml_tensor* r_z = matrix_slice_rows(r, 0, hidden_size);
  ggml_tensor* r_r = matrix_slice_rows(r, hidden_size, hidden_size);
  ggml_tensor* r_h = matrix_slice_rows(r, hidden_size * 2, hidden_size);
  ggml_tensor* wb_z = wb != nullptr ? vector_slice(wb, 0, hidden_size) : nullptr;
  ggml_tensor* wb_r = wb != nullptr ? vector_slice(wb, hidden_size, hidden_size) : nullptr;
  ggml_tensor* wb_h = wb != nullptr ? vector_slice(wb, hidden_size * 2, hidden_size) : nullptr;
  ggml_tensor* rb_z = rb != nullptr ? vector_slice(rb, 0, hidden_size) : nullptr;
  ggml_tensor* rb_r = rb != nullptr ? vector_slice(rb, hidden_size, hidden_size) : nullptr;
  ggml_tensor* rb_h = rb != nullptr ? vector_slice(rb, hidden_size * 2, hidden_size) : nullptr;

  for (int64_t step = 0; step < seq_length; ++step) {
    ggml_tensor* x_t = timestep_slice(x, step);

    ggml_tensor* z = ggml_add(ctx, ggml_mul_mat(ctx, w_z, x_t), ggml_mul_mat(ctx, r_z, h_t));
    if (wb_z != nullptr) {
      z = broadcast_add(z, wb_z);
    }
    if (rb_z != nullptr) {
      z = broadcast_add(z, rb_z);
    }
    z = ggml_sigmoid(ctx, z);

    ggml_tensor* r_gate = ggml_add(ctx, ggml_mul_mat(ctx, w_r, x_t), ggml_mul_mat(ctx, r_r, h_t));
    if (wb_r != nullptr) {
      r_gate = broadcast_add(r_gate, wb_r);
    }
    if (rb_r != nullptr) {
      r_gate = broadcast_add(r_gate, rb_r);
    }
    r_gate = ggml_sigmoid(ctx, r_gate);

    ggml_tensor* h_candidate = ggml_mul_mat(ctx, w_h, x_t);
    ggml_tensor* recurrent_term = ggml_mul_mat(ctx, r_h, ggml_mul(ctx, r_gate, h_t));
    h_candidate = ggml_add(ctx, h_candidate, recurrent_term);
    if (wb_h != nullptr) {
      h_candidate = broadcast_add(h_candidate, wb_h);
    }
    if (rb_h != nullptr) {
      h_candidate = broadcast_add(h_candidate, rb_h);
    }
    h_candidate = ggml_tanh(ctx, h_candidate);

    h_t = ggml_add(ctx, h_candidate, ggml_mul(ctx, z, ggml_sub(ctx, h_t, h_candidate)));

    if (y != nullptr) {
      y = ggml_set(ctx, y, h_t, y->nb[1], y->nb[2], y->nb[3], static_cast<size_t>(step) * y->nb[3]);
    }
  }

  EmitOutputs outputs;
  if (!node.outputs.empty() && node.outputs[0] != kOptionalValueAbsent) {
    outputs.push_back(ggml_cont(ctx, y));
  }
  if (node.outputs.size() > 1 && node.outputs[1] != kOptionalValueAbsent) {
    outputs.push_back(ggml_reshape_3d(ctx, ggml_cont(ctx, h_t), hidden_size, batch_size, 1));
  }
  return outputs;
}

// Shape-preserving unary float op with no attributes. Covers Relu, Sigmoid, Tanh, Neg,
// Abs, Sqrt, Exp, Log, Softplus, Elu (ONNX default alpha=1.0 only).
bool IsSupportedUnaryFloatNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || outputs[0] == nullptr) {
    return false;
  }
  const TensorMetadata in = getTensorMetadata(inputs[0]);
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  if (in.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      out.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (!rankSupportedByGGML(in) || !rankSupportedByGGML(out)) {
    return false;
  }

  // ONNX Elu supports an alpha attribute; GGML's ggml_elu is fixed at alpha=1.0.
  if (node.GetOperatorType() == "Elu") {
    const float alpha = readNodeAttribute<float>(node, "alpha").value_or(1.0f);
    if (alpha != 1.0f) {
      return false;
    }
  }
  return true;
}

EmitResult EmitUnaryFloatNode(ggml_context* ctx,
                              const NodeDesc& node,
                              const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() == 1 && node.outputs.size() == 1,
                "compiled unary op node has invalid arity");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled unary op node missing GGML input");

  const std::string_view op(node.op_type);
  if (op == "Relu")     return EmitOutputs{ggml_relu(ctx, x)};
  if (op == "Sigmoid")  return EmitOutputs{ggml_sigmoid(ctx, x)};
  if (op == "Tanh")     return EmitOutputs{ggml_tanh(ctx, x)};
  if (op == "Neg")      return EmitOutputs{ggml_neg(ctx, x)};
  if (op == "Abs")      return EmitOutputs{ggml_abs(ctx, x)};
  if (op == "Sqrt")     return EmitOutputs{ggml_sqrt(ctx, x)};
  if (op == "Exp")      return EmitOutputs{ggml_exp(ctx, x)};
  if (op == "Log")      return EmitOutputs{ggml_log(ctx, x)};
  if (op == "Softplus") return EmitOutputs{ggml_softplus(ctx, x)};
  if (op == "Elu")      return EmitOutputs{ggml_elu(ctx, x)};
  return std::nullopt;
}

bool IsSupportedLeakyReluNode(Ort::ConstNode node) {
  return IsSupportedUnaryFloatNode(node);
}

void CompileLeakyReluAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const float alpha = readNodeAttribute<float>(node, "alpha").value_or(0.01f);
  compiled_node->attrs = NodeDesc::AlphaAttrs{.alpha = alpha};
}

EmitResult EmitLeakyReluNode(ggml_context* ctx,
                             const NodeDesc& node,
                             const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::AlphaAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled LeakyRelu node missing alpha attribute");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled LeakyRelu node missing GGML input");
  return EmitOutputs{ggml_leaky_relu(ctx, x, attrs->alpha, /*inplace=*/false)};
}

// ONNX Softmax (opset >= 13): softmax along `axis` (default -1). GGML's ggml_soft_max
// operates along ne[0], which is the last ONNX dim.. so we only accept axis=-1 or
// axis=rank-1. Pre-opset-13 Softmax with its "coerce to 2D" semantics is rejected.
bool IsSupportedSoftmaxNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || outputs[0] == nullptr) {
    return false;
  }
  const TensorMetadata in = getTensorMetadata(inputs[0]);
  if (in.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || !rankSupportedByGGML(in)) {
    return false;
  }
  if (in.dims.empty()) {
    return false;
  }
  const int64_t rank = static_cast<int64_t>(in.dims.size());
  const int64_t axis = readNodeAttribute<int64_t>(node, "axis").value_or(-1);
  const int64_t normalized = axis < 0 ? axis + rank : axis;
  if (normalized != rank - 1) {
    return false;
  }
  return true;
}

void CompileSoftmaxAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  compiled_node->attrs =
      NodeDesc::AxisAttrs{.axis = readNodeAttribute<int64_t>(node, "axis").value_or(-1)};
}

EmitResult EmitSoftmaxNode(ggml_context* ctx,
                           const NodeDesc& node,
                           const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled Softmax node missing GGML input");
  return EmitOutputs{ggml_soft_max(ctx, x)};
}

// ONNX MatMul: A[..., M, K] @ B[..., K, N] -> [..., M, N], with numpy-style batch
// broadcasting. In GGML dim ordering (reversed), A is [K, M, batch...] (ne[0]=K)
// and B is [N, K, batch...] (ne[0]=N). ggml_mul_mat(w, x) requires w->ne[0]==x->ne[0]
// (the shared K), so we transpose+materialize B to get [K, N, batch...] and pass
// that as the "weight" operand. Result shape is [N, M, batch...] which is the
// reversed ONNX [..., M, N]. We require batch dims to match exactly for now.
bool IsSupportedMatMulNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || inputs[1] == nullptr || outputs[0] == nullptr) {
    return false;
  }
  const TensorMetadata a = getTensorMetadata(inputs[0]);
  const TensorMetadata b = getTensorMetadata(inputs[1]);
  const TensorMetadata c = getTensorMetadata(outputs[0]);
  if (a.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      b.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      c.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (a.dims.size() < 2 || b.dims.size() < 2) {
    return false;
  }
  if (a.dims.size() != b.dims.size()) {
    return false;  // skip rank-broadcasting matmul for now
  }
  if (!rankSupportedByGGML(a) || !rankSupportedByGGML(b) || !rankSupportedByGGML(c)) {
    return false;
  }
  // Inner dims must agree when both are concrete.
  const int64_t a_k = a.dims[a.dims.size() - 1];
  const int64_t b_k = b.dims[b.dims.size() - 2];
  if (a_k >= 0 && b_k >= 0 && a_k != b_k) {
    return false;
  }
  // Batch dims must match exactly (concrete or symbolic-identical not checked — conservative).
  for (size_t i = 0; i + 2 < a.dims.size(); ++i) {
    if (a.dims[i] >= 0 && b.dims[i] >= 0 && a.dims[i] != b.dims[i]) {
      return false;
    }
  }
  return true;
}

ConstantLayout MatMulConstantLayout(const NodeDesc& /*node*/, size_t input_idx) {
  // ONNX MatMul: A[...,M,K] @ B[...,K,N]. Only B (input 1) benefits from being
  // stored pre-transposed so that ggml_mul_mat(B, A) can skip the runtime
  // transpose + ggml_cont.
  return input_idx == 1 ? ConstantLayout::MATMUL_WEIGHT_TRANSPOSED
                        : ConstantLayout::AS_IS;
}

EmitResult EmitMatMulNode(ggml_context* ctx,
                          const NodeDesc& node,
                          const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() == 2 && node.outputs.size() == 1,
                "compiled MatMul node has invalid arity");
  ggml_tensor* a = values[node.inputs[0]];
  ggml_tensor* b = values[node.inputs[1]];
  GGONNX_NOT_NULL(a, "compiled MatMul node missing GGML A input");
  GGONNX_NOT_NULL(b, "compiled MatMul node missing GGML B input");
  // Fast path: B is a pre-transposed constant (compile-time materialized with
  // ne[0]==K). Detected by shape match against A's contraction dim. Otherwise
  // we pay the runtime transpose+cont. ggml_mul_mat requires !is_transposed on
  // its first operand and a unit innermost stride on the second, so a plain
  // ggml_transpose view is not accepted by the CPU kernel.
  ggml_tensor* b_eff = (b->ne[0] == a->ne[0])
                           ? b
                           : ggml_cont(ctx, ggml_transpose(ctx, b));
  return EmitOutputs{ggml_mul_mat(ctx, b_eff, a)};
}

// ONNX Conv (2D only for now): X[N,C,H,W] @ W[OC,IC/group,KH,KW] (+ optional B[OC]).
// GGML's reverse-dim convention lines up with ggml_conv_2d_direct's expected layouts:
// X -> ne=[W,H,C,N] and W -> ne=[KW,KH,IC,OC]. No transposes needed.
// Limitations: group=1, symmetric pads only, auto_pad in {NOTSET, VALID}, rank-4 only.
bool IsSupportedConvNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if ((inputs.size() != 2 && inputs.size() != 3) || outputs.size() != 1 ||
      node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || inputs[1] == nullptr || outputs[0] == nullptr) {
    return false;
  }

  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata w = getTensorMetadata(inputs[1]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      w.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      y.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (x.dims.size() != 4 || w.dims.size() != 4 || y.dims.size() != 4) {
    return false;
  }
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(w) || !rankSupportedByGGML(y)) {
    return false;
  }

  if (readNodeAttribute<int64_t>(node, "group").value_or(1) != 1) {
    return false;
  }

  const std::string auto_pad = readNodeAttribute<std::string>(node, "auto_pad").value_or("NOTSET");
  if (auto_pad != "NOTSET" && auto_pad != "VALID") {
    return false;
  }

  if (const auto pads = readNodeAttribute<std::vector<int64_t>>(node, "pads")) {
    if (pads->size() != 4) return false;
    if ((*pads)[0] != (*pads)[2] || (*pads)[1] != (*pads)[3]) return false;
    if (auto_pad == "VALID" && ((*pads)[0] != 0 || (*pads)[1] != 0)) return false;
  }
  if (const auto strides = readNodeAttribute<std::vector<int64_t>>(node, "strides")) {
    if (strides->size() != 2) return false;
  }
  if (const auto dilations = readNodeAttribute<std::vector<int64_t>>(node, "dilations")) {
    if (dilations->size() != 2) return false;
  }
  if (const auto kernel_shape = readNodeAttribute<std::vector<int64_t>>(node, "kernel_shape")) {
    if (kernel_shape->size() != 2) return false;
    if (w.dims[2] >= 0 && (*kernel_shape)[0] != w.dims[2]) return false;
    if (w.dims[3] >= 0 && (*kernel_shape)[1] != w.dims[3]) return false;
  }

  if (x.dims[1] >= 0 && w.dims[1] >= 0 && x.dims[1] != w.dims[1]) {
    return false;  // IC mismatch (group=1 path).
  }

  if (inputs.size() == 3 && inputs[2] != nullptr) {
    const TensorMetadata b = getTensorMetadata(inputs[2]);
    if (b.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || b.dims.size() != 1) return false;
    if (b.dims[0] >= 0 && w.dims[0] >= 0 && b.dims[0] != w.dims[0]) return false;
  }

  return true;
}

void CompileConvAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  NodeDesc::Conv2DAttrs attrs;

  if (const auto strides = readNodeAttribute<std::vector<int64_t>>(node, "strides")) {
    // ONNX [stride_h, stride_w] -> ggml (s0=width, s1=height).
    attrs.s0 = static_cast<int>((*strides)[1]);
    attrs.s1 = static_cast<int>((*strides)[0]);
  }
  if (const auto dilations = readNodeAttribute<std::vector<int64_t>>(node, "dilations")) {
    attrs.d0 = static_cast<int>((*dilations)[1]);
    attrs.d1 = static_cast<int>((*dilations)[0]);
  }

  const std::string auto_pad = readNodeAttribute<std::string>(node, "auto_pad").value_or("NOTSET");
  if (auto_pad == "NOTSET") {
    if (const auto pads = readNodeAttribute<std::vector<int64_t>>(node, "pads")) {
      // pads=[h_begin, w_begin, h_end, w_end], already validated symmetric.
      attrs.p0 = static_cast<int>((*pads)[1]);
      attrs.p1 = static_cast<int>((*pads)[0]);
    }
  }
  // VALID => pads remain 0.

  compiled_node->attrs = attrs;
}

EmitResult EmitConvNode(ggml_context* ctx,
                        const NodeDesc& node,
                        const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() == 2 || node.inputs.size() == 3,
                "compiled Conv node has invalid input arity");
  const auto* attrs = std::get_if<NodeDesc::Conv2DAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Conv node missing Conv2D attributes");

  ggml_tensor* x = values[node.inputs[0]];
  ggml_tensor* w = values[node.inputs[1]];
  GGONNX_NOT_NULL(x, "compiled Conv node missing GGML X input");
  GGONNX_NOT_NULL(w, "compiled Conv node missing GGML W input");

  ggml_tensor* out = ggml_conv_2d_direct(ctx, w, x,
                                         attrs->s0, attrs->s1,
                                         attrs->p0, attrs->p1,
                                         attrs->d0, attrs->d1);

  if (node.inputs.size() == 3 && node.inputs[2] != kOptionalValueAbsent) {
    ggml_tensor* bias = values[node.inputs[2]];
    GGONNX_NOT_NULL(bias, "compiled Conv node missing GGML B input");
    // bias ne=[OC] -> [1,1,OC,1] to broadcast across W,H,N of conv output ne=[OW,OH,OC,N].
    ggml_tensor* bias_4d = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
    out = ggml_add(ctx, out, bias_4d);
  }

  return EmitOutputs{out};
}

// ONNX Gemm: Y = alpha * A' @ B' + beta * C, where A' = transA ? A^T : A (shape [M,K])
// and B' = transB ? B^T : B (shape [K,N]). Maps to ggml_mul_mat after making both
// operands have the shared K dim at ne[0].
bool IsSupportedGemmNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if ((inputs.size() != 2 && inputs.size() != 3) || outputs.size() != 1 ||
      node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || inputs[1] == nullptr || outputs[0] == nullptr) {
    return false;
  }

  const TensorMetadata a = getTensorMetadata(inputs[0]);
  const TensorMetadata b = getTensorMetadata(inputs[1]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (a.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      b.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      y.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (a.dims.size() != 2 || b.dims.size() != 2 || y.dims.size() != 2) {
    return false;
  }

  const bool trans_a = readNodeAttribute<int64_t>(node, "transA").value_or(0) != 0;
  const bool trans_b = readNodeAttribute<int64_t>(node, "transB").value_or(0) != 0;
  const int64_t a_k = trans_a ? a.dims[0] : a.dims[1];
  const int64_t b_k = trans_b ? b.dims[1] : b.dims[0];
  if (a_k >= 0 && b_k >= 0 && a_k != b_k) {
    return false;
  }

  if (inputs.size() == 3 && inputs[2] != nullptr) {
    const TensorMetadata c = getTensorMetadata(inputs[2]);
    if (c.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return false;
    if (!rankSupportedByGGML(c)) return false;
    // Must broadcast to [M, N] == y.dims.
    if (!broadcastSupportedByGGML(c.dims, y.dims)) return false;
  }

  return true;
}

void CompileGemmAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  NodeDesc::GemmAttrs attrs;
  attrs.alpha = readNodeAttribute<float>(node, "alpha").value_or(1.0f);
  attrs.beta = readNodeAttribute<float>(node, "beta").value_or(1.0f);
  attrs.trans_a = readNodeAttribute<int64_t>(node, "transA").value_or(0) != 0;
  attrs.trans_b = readNodeAttribute<int64_t>(node, "transB").value_or(0) != 0;
  compiled_node->attrs = attrs;
}

ConstantLayout GemmConstantLayout(const NodeDesc& node, size_t input_idx) {
  // Only B benefits from pre-transpose, and only when transB is false — with
  // transB the ONNX B is already [N, K] which lands as ggml ne=[K, N] with no
  // runtime transpose needed.
  if (input_idx != 1) return ConstantLayout::AS_IS;
  const auto* attrs = std::get_if<NodeDesc::GemmAttrs>(&node.attrs);
  if (attrs == nullptr || attrs->trans_b) return ConstantLayout::AS_IS;
  return ConstantLayout::MATMUL_WEIGHT_TRANSPOSED;
}

EmitResult EmitGemmNode(ggml_context* ctx,
                        const NodeDesc& node,
                        const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::GemmAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Gemm node missing attributes");

  ggml_tensor* a = values[node.inputs[0]];
  ggml_tensor* b = values[node.inputs[1]];
  GGONNX_NOT_NULL(a, "compiled Gemm node missing GGML A input");
  GGONNX_NOT_NULL(b, "compiled Gemm node missing GGML B input");

  // Build a_eff with K at ne[0]; then require the same of b_eff. A plain view
  // from ggml_transpose violates ggml_mul_mat's invariants, so we materialize
  // when needed. The b_eff branch collapses to a no-op either when transB=1 or
  // when B was pre-transposed as a constant.
  ggml_tensor* a_eff = attrs->trans_a ? ggml_cont(ctx, ggml_transpose(ctx, a)) : a;
  const int64_t K = a_eff->ne[0];
  ggml_tensor* b_eff = (b->ne[0] == K) ? b : ggml_cont(ctx, ggml_transpose(ctx, b));

  ggml_tensor* out = ggml_mul_mat(ctx, b_eff, a_eff);
  if (attrs->alpha != 1.0f) {
    out = ggml_scale(ctx, out, attrs->alpha);
  }

  if (node.inputs.size() == 3 && node.inputs[2] != kOptionalValueAbsent) {
    ggml_tensor* c = values[node.inputs[2]];
    GGONNX_NOT_NULL(c, "compiled Gemm node missing GGML C input");
    if (attrs->beta != 1.0f) {
      c = ggml_scale(ctx, c, attrs->beta);
    }
    out = ggml_add(ctx, out, c);
  }

  return EmitOutputs{out};
}

// ONNX Reshape: data + shape input. We require the output shape to be fully static
// (resolved by shape inference) and snapshot it at compile time — the runtime shape
// tensor is ignored.
bool IsSupportedReshapeNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || outputs[0] == nullptr) {
    return false;
  }
  const TensorMetadata in = getTensorMetadata(inputs[0]);
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  if (in.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      out.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (!rankSupportedByGGML(in) || !rankSupportedByGGML(out)) {
    return false;
  }
  if (!shapeIsFullyStatic(out.dims)) {
    return false;
  }
  return true;
}

void CompileReshapeAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto outputs = node.GetOutputs();
  GGONNX_ASSERT(outputs.size() == 1 && outputs[0] != nullptr,
                "Reshape must have a single output");
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  compiled_node->attrs = NodeDesc::ReshapeAttrs{.onnx_dims = out.dims};
}

EmitResult EmitReshapeNode(ggml_context* ctx,
                           const NodeDesc& node,
                           const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::ReshapeAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Reshape node missing attributes");

  ggml_tensor* data = values[node.inputs[0]];
  GGONNX_NOT_NULL(data, "compiled Reshape node missing GGML data input");

  const std::array<int64_t, GGML_MAX_DIMS> target = ToPaddedGGMLDims(attrs->onnx_dims);
  ggml_tensor* src = ggml_is_contiguous(data) ? data : ggml_cont(ctx, data);
  ggml_tensor* out = ggml_reshape_4d(ctx, src, target[0], target[1], target[2], target[3]);
  return EmitOutputs{out};
}

// ONNX MaxPool / AveragePool (2D only). ONNX layout X[N,C,H,W] -> ggml ne=[W,H,C,N],
// so ggml_pool_2d's k0/s0/p0 apply to W and k1/s1/p1 apply to H — ONNX kernel_shape/
// strides/pads are [h, w] so we swap. GGML's AveragePool divides by the full kernel
// area regardless of how many in-bounds samples were summed (i.e. count_include_pad=1),
// so we reject AveragePool with non-zero padding unless count_include_pad=1.
bool IsSupportedPool2DNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.empty() || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (outputs.size() > 1) {
    return false;  // MaxPool Indices output not supported.
  }
  if (inputs[0] == nullptr || outputs[0] == nullptr) {
    return false;
  }

  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      y.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (x.dims.size() != 4 || y.dims.size() != 4) {
    return false;
  }
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(y)) {
    return false;
  }

  if (readNodeAttribute<int64_t>(node, "ceil_mode").value_or(0) != 0) {
    return false;
  }
  if (readNodeAttribute<int64_t>(node, "storage_order").value_or(0) != 0) {
    return false;
  }
  if (const auto dilations = readNodeAttribute<std::vector<int64_t>>(node, "dilations")) {
    for (int64_t d : *dilations) {
      if (d != 1) return false;
    }
  }

  const auto kernel_shape = readNodeAttribute<std::vector<int64_t>>(node, "kernel_shape");
  if (!kernel_shape.has_value() || kernel_shape->size() != 2) {
    return false;
  }
  if (const auto strides = readNodeAttribute<std::vector<int64_t>>(node, "strides")) {
    if (strides->size() != 2) return false;
  }

  const std::string auto_pad = readNodeAttribute<std::string>(node, "auto_pad").value_or("NOTSET");
  if (auto_pad != "NOTSET" && auto_pad != "VALID") {
    return false;
  }

  std::array<int64_t, 4> pads{0, 0, 0, 0};
  if (const auto pads_attr = readNodeAttribute<std::vector<int64_t>>(node, "pads")) {
    if (pads_attr->size() != 4) return false;
    if ((*pads_attr)[0] != (*pads_attr)[2] || (*pads_attr)[1] != (*pads_attr)[3]) return false;
    if (auto_pad == "VALID" && ((*pads_attr)[0] != 0 || (*pads_attr)[1] != 0)) return false;
    for (size_t i = 0; i < 4; ++i) pads[i] = (*pads_attr)[i];
  }

  if (node.GetOperatorType() == "AveragePool") {
    const bool has_padding = pads[0] != 0 || pads[1] != 0;
    const int64_t count_include_pad =
        readNodeAttribute<int64_t>(node, "count_include_pad").value_or(0);
    if (has_padding && count_include_pad != 1) {
      return false;
    }
  }

  return true;
}

void CompilePool2DAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  NodeDesc::Pool2DAttrs attrs;
  attrs.op = node.GetOperatorType() == "MaxPool" ? GGML_OP_POOL_MAX : GGML_OP_POOL_AVG;

  const auto kernel_shape = readNodeAttribute<std::vector<int64_t>>(node, "kernel_shape");
  GGONNX_ASSERT(kernel_shape.has_value() && kernel_shape->size() == 2,
                "Pool2D kernel_shape must be present and 2D");
  attrs.k0 = static_cast<int>((*kernel_shape)[1]);
  attrs.k1 = static_cast<int>((*kernel_shape)[0]);

  if (const auto strides = readNodeAttribute<std::vector<int64_t>>(node, "strides")) {
    attrs.s0 = static_cast<int>((*strides)[1]);
    attrs.s1 = static_cast<int>((*strides)[0]);
  } else {
    attrs.s0 = 1;
    attrs.s1 = 1;
  }

  const std::string auto_pad = readNodeAttribute<std::string>(node, "auto_pad").value_or("NOTSET");
  if (auto_pad == "NOTSET") {
    if (const auto pads = readNodeAttribute<std::vector<int64_t>>(node, "pads")) {
      attrs.p0 = static_cast<int>((*pads)[1]);
      attrs.p1 = static_cast<int>((*pads)[0]);
    }
  }
  compiled_node->attrs = attrs;
}

bool IsSupportedGlobalPoolNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 1 || outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || outputs[0] == nullptr) {
    return false;
  }
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      y.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (x.dims.size() != 4 || y.dims.size() != 4) {
    return false;
  }
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(y)) {
    return false;
  }
  // Spatial dims must be concrete so the kernel is known at emit time.
  if (x.dims[2] < 0 || x.dims[3] < 0) {
    return false;
  }
  return true;
}

void CompileGlobalPoolAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  NodeDesc::Pool2DAttrs attrs;
  attrs.op = node.GetOperatorType() == "GlobalMaxPool" ? GGML_OP_POOL_MAX : GGML_OP_POOL_AVG;
  attrs.is_global = true;
  compiled_node->attrs = attrs;
}

EmitResult EmitPool2DNode(ggml_context* ctx,
                          const NodeDesc& node,
                          const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() == 1 && node.outputs.size() == 1,
                "compiled Pool2D node has invalid arity");
  const auto* attrs = std::get_if<NodeDesc::Pool2DAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Pool2D node missing attributes");

  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled Pool2D node missing GGML input");

  int k0 = attrs->k0, k1 = attrs->k1;
  int s0 = attrs->s0, s1 = attrs->s1;
  int p0 = attrs->p0, p1 = attrs->p1;
  if (attrs->is_global) {
    k0 = static_cast<int>(x->ne[0]);
    k1 = static_cast<int>(x->ne[1]);
    s0 = k0;
    s1 = k1;
    p0 = 0;
    p1 = 0;
  }

  ggml_tensor* out = ggml_pool_2d(ctx, x, attrs->op, k0, k1, s0, s1,
                                  static_cast<float>(p0), static_cast<float>(p1));
  return EmitOutputs{out};
}

const OpDefinition* FindOpDefinition(std::string_view domain, std::string_view op_type) {

  struct PairHash {
    size_t operator()(const std::pair<std::string_view, std::string_view>& p) const {
      return std::hash<std::string_view>{}(p.first) ^ std::hash<std::string_view>{}(p.second);
    }
  };

  static const std::unordered_map<std::pair<std::string_view, std::string_view>, const OpDefinition, PairHash> ops_table = {
      {{"", "Add"}, {IsSupportedElementwiseBinaryOpNode, nullptr, EmitElementwiseBinaryNode}},
      {{"", "Sub"}, {IsSupportedElementwiseBinaryOpNode, nullptr, EmitElementwiseBinaryNode}},
      {{"", "Mul"}, {IsSupportedElementwiseBinaryOpNode, nullptr, EmitElementwiseBinaryNode}},
      {{"", "Div"}, {IsSupportedElementwiseBinaryOpNode, nullptr, EmitElementwiseBinaryNode}},
      {{"", "GRU"}, {IsSupportedGRUNode, CompileGRUAttributes, EmitGRUNode}},
      {{"", "Relu"},     {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Sigmoid"},  {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Tanh"},     {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Neg"},      {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Abs"},      {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Sqrt"},     {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Exp"},      {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Log"},      {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Softplus"}, {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Elu"},      {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "LeakyRelu"}, {IsSupportedLeakyReluNode, CompileLeakyReluAttributes, EmitLeakyReluNode}},
      {{"", "Softmax"},  {IsSupportedSoftmaxNode, CompileSoftmaxAttributes, EmitSoftmaxNode}},
      {{"", "MatMul"},   {IsSupportedMatMulNode, nullptr, EmitMatMulNode, MatMulConstantLayout}},
      {{"", "Conv"},     {IsSupportedConvNode, CompileConvAttributes, EmitConvNode, nullptr}},
      {{"", "Gemm"},     {IsSupportedGemmNode, CompileGemmAttributes, EmitGemmNode, GemmConstantLayout}},
      {{"", "Reshape"},  {IsSupportedReshapeNode, CompileReshapeAttributes, EmitReshapeNode}},
      {{"", "MaxPool"},       {IsSupportedPool2DNode, CompilePool2DAttributes, EmitPool2DNode}},
      {{"", "AveragePool"},   {IsSupportedPool2DNode, CompilePool2DAttributes, EmitPool2DNode}},
      {{"", "GlobalMaxPool"}, {IsSupportedGlobalPoolNode, CompileGlobalPoolAttributes, EmitPool2DNode}},
      {{"", "GlobalAveragePool"}, {IsSupportedGlobalPoolNode, CompileGlobalPoolAttributes, EmitPool2DNode}},
  };

  auto it = ops_table.find({domain, op_type});
  if (it != ops_table.end()) {
    return &it->second;
  }
  return nullptr;
}
