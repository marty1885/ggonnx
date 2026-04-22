#include "ops.hpp"
#include "inner/helpers.hpp"
#include "inner/ort_api_helpers.hpp"
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace {

thread_local const ConstantValueMap* g_active_compile_time_constants = nullptr;
thread_local bool g_force_matmul_f32_precision = false;

void ComputeErf(ggml_tensor* dst, const ggml_tensor* src, int ith, int nth, void* /*userdata*/) {
  GGONNX_NOT_NULL(dst, "Erf custom op destination must not be null");
  GGONNX_NOT_NULL(src, "Erf custom op source must not be null");

  const int64_t n = ggml_nelements(src);
  const int64_t start = (n * ith) / nth;
  const int64_t end = (n * (ith + 1)) / nth;
  float* dst_data = ggml_get_data_f32(dst);
  float* src_data = ggml_get_data_f32(src);
  GGONNX_NOT_NULL(dst_data, "Erf custom op destination data must not be null");
  GGONNX_NOT_NULL(src_data, "Erf custom op source data must not be null");
  for (int64_t i = start; i < end; ++i) {
    dst_data[i] = std::erf(src_data[i]);
  }
}

std::optional<ConstantTensor> LookupCompileTimeConstant(Ort::ConstValueInfo value_info) {
  if (value_info == nullptr) return std::nullopt;
  if (value_info.IsConstantInitializer()) {
    Ort::ConstValue value{nullptr};
    Ort::ThrowOnError(value_info.GetInitializer(value));
    if (value != nullptr) {
      const TensorMetadata meta = getTensorMetadata(value);
      const auto tensor_info = value.GetTensorTypeAndShapeInfo();
      const size_t bytes = tensor_info.GetElementCount() *
                           (meta.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? sizeof(float)
                            : meta.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ? sizeof(int32_t)
                            : meta.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ? sizeof(int64_t)
                            : 0);
      if (bytes == 0 && tensor_info.GetElementCount() != 0) {
        return std::nullopt;
      }
      const void* raw_data = nullptr;
      ggonnx::ort_internal::THROW_ON_ERROR(
          ggonnx::ort_internal::GetOrtApi().GetTensorData(value, &raw_data));
      GGONNX_NOT_NULL(raw_data, "ORT returned null tensor data for constant");
      ConstantTensor tensor;
      tensor.element_type = meta.element_type;
      tensor.dims = meta.dims;
      tensor.data.resize(bytes);
      if (bytes > 0) {
        std::memcpy(tensor.data.data(), raw_data, bytes);
      }
      return tensor;
    }
  }
  if (g_active_compile_time_constants == nullptr) return std::nullopt;
  auto it = g_active_compile_time_constants->find(value_info.GetName());
  if (it == g_active_compile_time_constants->end()) return std::nullopt;
  return it->second;
}

}  // namespace

TensorMetadata getTensorMetadata(Ort::ConstValueInfo value_info) {
  const auto tensor_info = value_info.TypeInfo().GetTensorTypeAndShapeInfo();

  TensorMetadata result;
  result.element_type = tensor_info.GetElementType();
  result.dims = tensor_info.GetShape();
  return result;
}

void SetActiveCompileTimeConstants(const ConstantValueMap* constants) {
  g_active_compile_time_constants = constants;
}

const ConstantValueMap* GetActiveCompileTimeConstants() {
  return g_active_compile_time_constants;
}

void SetForceMatMulF32Precision(bool force) {
  g_force_matmul_f32_precision = force;
}

bool GetForceMatMulF32Precision() {
  return g_force_matmul_f32_precision;
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

size_t elementCount(const std::vector<int64_t>& dims) {
  size_t count = 1;
  for (const int64_t dim : dims) {
    GGONNX_ASSERT(dim >= 0, "tensor element count requested for dynamic or invalid shape");
    count *= static_cast<size_t>(dim);
  }
  return count;
}

template <typename T>
std::optional<std::vector<T>> readConstantInputArray(Ort::ConstNode node,
                                                     size_t input_idx,
                                                     ONNXTensorElementDataType element_type) {
  const auto inputs = node.GetInputs();
  if (input_idx >= inputs.size() || inputs[input_idx] == nullptr) {
    return std::nullopt;
  }

  const Ort::ConstValueInfo input = inputs[input_idx];
  const auto constant = LookupCompileTimeConstant(input);
  if (!constant.has_value()) return std::nullopt;
  const TensorMetadata meta{.element_type = constant->element_type, .dims = constant->dims};
  if (meta.element_type != element_type) {
    return std::nullopt;
  }

  const size_t count = elementCount(meta.dims);
  if (count == 0) {
    return std::vector<T>{};
  }

  const T* typed_data = reinterpret_cast<const T*>(constant->data.data());
  GGONNX_NOT_NULL(typed_data, "ORT returned null tensor data for constant");
  return std::vector<T>(typed_data, typed_data + count);
}

std::optional<std::vector<int64_t>> readPadVector(Ort::ConstNode node) {
  if (const auto pads = readNodeAttribute<std::vector<int64_t>>(node, "pads")) {
    return pads;
  }
  if (const auto paddings = readNodeAttribute<std::vector<int64_t>>(node, "paddings")) {
    return paddings;
  }
  return readConstantInputArray<int64_t>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
}

bool inferIntegerSpatialScale(const TensorMetadata& x,
                              const TensorMetadata& y,
                              int* scale_h,
                              int* scale_w) {
  GGONNX_NOT_NULL(scale_h, "scale_h output must not be null");
  GGONNX_NOT_NULL(scale_w, "scale_w output must not be null");
  if (x.dims.size() != 4 || y.dims.size() != 4) {
    return false;
  }
  if (!shapeIsFullyStatic(x) || !shapeIsFullyStatic(y)) {
    return false;
  }
  if (y.dims[0] != x.dims[0] || y.dims[1] != x.dims[1]) {
    return false;
  }
  if (x.dims[2] <= 0 || x.dims[3] <= 0 || y.dims[2] <= 0 || y.dims[3] <= 0) {
    return false;
  }
  if (y.dims[2] % x.dims[2] != 0 || y.dims[3] % x.dims[3] != 0) {
    return false;
  }

  const int64_t inferred_h = y.dims[2] / x.dims[2];
  const int64_t inferred_w = y.dims[3] / x.dims[3];
  if (inferred_h <= 0 || inferred_w <= 0) {
    return false;
  }

  *scale_h = static_cast<int>(inferred_h);
  *scale_w = static_cast<int>(inferred_w);
  return true;
}

bool IsSupportedElementwiseBinaryNode(Ort::ConstNode node, std::string_view op_type) {
  if (!(op_type == "Add" || op_type == "Sub" || op_type == "Mul" || op_type == "Div" ||
        op_type == "Max" || op_type == "Min")) {
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
  // Either operand may broadcast to the output shape — check both sides against
  // the output rather than against each other. The older lhs→rhs check only
  // accepted broadcasts where rhs was the larger side.
  if (!broadcastSupportedByGGML(lhs.dims, out.dims)) {
    return false;
  }
  if (!broadcastSupportedByGGML(rhs.dims, out.dims)) {
    return false;
  }

  // GGML's binary ops require the second operand to broadcast into the first
  // (the first operand's shape determines the output). For commutative ops we
  // can swap operands at emit time, but for Sub/Div the order is meaningful, so
  // only accept the node if lhs already matches the output shape.
  const bool commutative =
      (op_type == "Add" || op_type == "Mul" || op_type == "Max" || op_type == "Min");
  if (!commutative && ToPaddedGGMLDims(lhs.dims) != ToPaddedGGMLDims(out.dims)) {
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
  if (w.dims.size() != 3 || r.dims.size() != 3) {
    return false;
  }
  if (!x.dims.empty() && x.dims.size() != 3) {
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
  if (!x.dims.empty() && x.dims[2] >= 0 && x.dims[2] != w.dims[2]) {
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
  // GGML's elementwise binary ops require the second operand to broadcast into
  // the first. For commutative ops we can swap when the caller passed the
  // smaller tensor first (e.g. `bias [N] + matmul [B, S, N]`).
  const bool commutative =
      (op_type == "Add" || op_type == "Mul" || op_type == "Max" || op_type == "Min");
  if (commutative && !ggml_can_repeat(rhs, lhs)) {
    std::swap(lhs, rhs);
  }
  if (op_type == "Add") {
    return EmitOutputs{ggml_add(ctx, lhs, rhs)};
  } else if (op_type == "Sub") {
    return EmitOutputs{ggml_sub(ctx, lhs, rhs)};
  } else if (op_type == "Mul") {
    return EmitOutputs{ggml_mul(ctx, lhs, rhs)};
  } else if (op_type == "Div") {
    return EmitOutputs{ggml_div(ctx, lhs, rhs)};
  } else if (op_type == "Max" || op_type == "Min") {
    // ggml has no elementwise max/min, so synthesize:
    //   max(a, b) = (a + b + |a - b|) / 2
    //   min(a, b) = (a + b - |a - b|) / 2
    // Broadcast is handled by the underlying add/sub — matches the support check.
    ggml_tensor* sum = ggml_add(ctx, lhs, rhs);
    ggml_tensor* diff = ggml_sub(ctx, lhs, rhs);
    ggml_tensor* abs_diff = ggml_abs(ctx, diff);
    ggml_tensor* combined = (op_type == "Max") ? ggml_add(ctx, sum, abs_diff)
                                               : ggml_sub(ctx, sum, abs_diff);
    return EmitOutputs{ggml_scale(ctx, combined, 0.5f)};
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
      z = ggml_add(ctx, z, wb_z);
    }
    if (rb_z != nullptr) {
      z = ggml_add(ctx, z, rb_z);
    }
    z = ggml_sigmoid(ctx, z);

    ggml_tensor* r_gate = ggml_add(ctx, ggml_mul_mat(ctx, w_r, x_t), ggml_mul_mat(ctx, r_r, h_t));
    if (wb_r != nullptr) {
      r_gate = ggml_add(ctx, r_gate, wb_r);
    }
    if (rb_r != nullptr) {
      r_gate = ggml_add(ctx, r_gate, rb_r);
    }
    r_gate = ggml_sigmoid(ctx, r_gate);

    ggml_tensor* h_candidate = ggml_mul_mat(ctx, w_h, x_t);
    ggml_tensor* recurrent_term = ggml_mul_mat(ctx, r_h, ggml_mul(ctx, r_gate, h_t));
    h_candidate = ggml_add(ctx, h_candidate, recurrent_term);
    if (wb_h != nullptr) {
      h_candidate = ggml_add(ctx, h_candidate, wb_h);
    }
    if (rb_h != nullptr) {
      h_candidate = ggml_add(ctx, h_candidate, rb_h);
    }
    h_candidate = ggml_tanh(ctx, h_candidate);

    h_t = ggml_add(ctx, h_candidate, ggml_mul(ctx, z, ggml_sub(ctx, h_t, h_candidate)));

    if (y != nullptr) {
      y = ggml_set(ctx, y, h_t, y->nb[1], y->nb[2], y->nb[3], static_cast<size_t>(step) * y->nb[3]);
    }
  }

  EmitOutputs outputs;
  if (!node.outputs.empty() && node.outputs[0] != kOptionalValueAbsent) {
    outputs.push_back(y);
  }
  if (node.outputs.size() > 1 && node.outputs[1] != kOptionalValueAbsent) {
    outputs.push_back(ggml_reshape_3d(ctx, h_t, hidden_size, batch_size, 1));
  }
  return outputs;
}

bool IsSupportedLSTMNode(Ort::ConstNode node) {
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

  const int64_t input_forget = readNodeAttribute<int64_t>(node, "input_forget").value_or(0);
  if (input_forget != 0) {
    return false;
  }

  if (readNodeAttribute<float>(node, "clip").has_value()) {
    return false;
  }

  if (const auto activations = readNodeAttribute<std::vector<std::string>>(node, "activations")) {
    if (activations->size() != 3 || (*activations)[0] != "Sigmoid" ||
        (*activations)[1] != "Tanh" || (*activations)[2] != "Tanh") {
      return false;
    }
  }

  if (inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr) {
    return false;
  }
  // sequence_lens (input 4) must be absent
  if (inputs.size() > 4 && inputs[4] != nullptr) {
    return false;
  }
  // P (peepholes, input 7) must be absent
  if (inputs.size() > 7 && inputs[7] != nullptr) {
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
  if (*hidden_size <= 0 || w.dims[0] != 1 || r.dims[0] != 1 || w.dims[1] != *hidden_size * 4 ||
      r.dims[1] != *hidden_size * 4 || w.dims[2] <= 0 || r.dims[2] != *hidden_size) {
    return false;
  }
  if (x.dims[2] >= 0 && x.dims[2] != w.dims[2]) {
    return false;
  }

  if (inputs.size() > 3 && inputs[3] != nullptr) {
    const TensorMetadata b = getTensorMetadata(inputs[3]);
    if (b.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || b.dims.size() != 2 ||
        b.dims[0] != 1 || b.dims[1] != *hidden_size * 8) {
      return false;
    }
  }

  auto check_initial_state = [&](size_t idx) -> bool {
    if (inputs.size() <= idx || inputs[idx] == nullptr) {
      return true;
    }
    const TensorMetadata s = getTensorMetadata(inputs[idx]);
    if (s.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      return false;
    }
    if (!s.dims.empty() && (s.dims.size() != 3 || s.dims[0] != 1 || s.dims[2] != *hidden_size)) {
      return false;
    }
    if (!x.dims.empty() && !s.dims.empty() && x.dims[1] >= 0 && s.dims[1] >= 0 && x.dims[1] != s.dims[1]) {
      return false;
    }
    return true;
  };
  if (!check_initial_state(5) || !check_initial_state(6)) {
    return false;
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

void CompileLSTMAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const std::string direction = readNodeAttribute<std::string>(node, "direction").value_or("forward");
  GGONNX_ASSERT(direction == "forward", "only forward LSTM direction is supported");
  const auto hidden_size = readNodeAttribute<int64_t>(node, "hidden_size");
  GGONNX_ASSERT(hidden_size.has_value() && *hidden_size > 0,
                "LSTM hidden_size attribute must be present and positive");
  compiled_node->attrs = NodeDesc::LSTMAttrs{.hidden_size = *hidden_size};
}

EmitResult EmitLSTMNode(ggml_context* ctx,
                        const NodeDesc& node,
                        const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() >= 3, "compiled LSTM node has invalid input arity");
  const auto* attrs = std::get_if<NodeDesc::LSTMAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled LSTM node missing LSTM attributes");
  GGONNX_ASSERT(attrs->hidden_size > 0, "compiled LSTM node must have positive hidden_size");

  auto input_or_null = [&](size_t idx) -> ggml_tensor* {
    if (node.inputs.size() <= idx || node.inputs[idx] == kOptionalValueAbsent) {
      return nullptr;
    }
    return values[node.inputs[idx]];
  };

  ggml_tensor* x = values[node.inputs[0]];
  ggml_tensor* w = values[node.inputs[1]];
  ggml_tensor* r = values[node.inputs[2]];
  ggml_tensor* b = input_or_null(3);
  ggml_tensor* initial_h = input_or_null(5);
  ggml_tensor* initial_c = input_or_null(6);
  if (x == nullptr || w == nullptr || r == nullptr) {
    throw std::runtime_error("compiled LSTM node missing required GGML inputs");
  }

  const int64_t input_size = x->ne[0];
  const int64_t batch_size = x->ne[1];
  const int64_t seq_length = x->ne[2];
  const int64_t hidden_size = attrs->hidden_size;
  GGONNX_ASSERT(w->ne[0] == input_size && w->ne[1] == hidden_size * 4 && w->ne[2] == 1,
                "compiled LSTM W tensor shape mismatch");
  GGONNX_ASSERT(r->ne[0] == hidden_size && r->ne[1] == hidden_size * 4 && r->ne[2] == 1,
                "compiled LSTM R tensor shape mismatch");

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

  // ONNX LSTM gate order in W/R/B: i, o, f, c.
  ggml_tensor* wb = nullptr;
  ggml_tensor* rb = nullptr;
  if (b != nullptr) {
    GGONNX_ASSERT(b->ne[0] == hidden_size * 8 && b->ne[1] == 1,
                  "compiled LSTM bias tensor shape mismatch");
    wb = vector_slice(b, 0, hidden_size * 4);
    rb = vector_slice(b, hidden_size * 4, hidden_size * 4);
  }

  ggml_tensor* h_t = nullptr;
  if (initial_h != nullptr) {
    GGONNX_ASSERT(initial_h->ne[0] == hidden_size && initial_h->ne[1] == batch_size &&
                      initial_h->ne[2] == 1,
                  "compiled LSTM initial_h tensor shape mismatch");
    h_t = ggml_view_2d(ctx, initial_h, initial_h->ne[0], initial_h->ne[1], initial_h->nb[1], 0);
  } else {
    h_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, batch_size);
    std::memset(h_t->data, 0, ggml_nbytes(h_t));
  }

  ggml_tensor* c_t = nullptr;
  if (initial_c != nullptr) {
    GGONNX_ASSERT(initial_c->ne[0] == hidden_size && initial_c->ne[1] == batch_size &&
                      initial_c->ne[2] == 1,
                  "compiled LSTM initial_c tensor shape mismatch");
    c_t = ggml_view_2d(ctx, initial_c, initial_c->ne[0], initial_c->ne[1], initial_c->nb[1], 0);
  } else {
    c_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, batch_size);
    std::memset(c_t->data, 0, ggml_nbytes(c_t));
  }

  ggml_tensor* y = nullptr;
  if (!node.outputs.empty() && node.outputs[0] != kOptionalValueAbsent) {
    y = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hidden_size, batch_size, 1, seq_length);
  }

  ggml_tensor* w_i = matrix_slice_rows(w, 0, hidden_size);
  ggml_tensor* w_o = matrix_slice_rows(w, hidden_size, hidden_size);
  ggml_tensor* w_f = matrix_slice_rows(w, hidden_size * 2, hidden_size);
  ggml_tensor* w_c = matrix_slice_rows(w, hidden_size * 3, hidden_size);
  ggml_tensor* r_i = matrix_slice_rows(r, 0, hidden_size);
  ggml_tensor* r_o = matrix_slice_rows(r, hidden_size, hidden_size);
  ggml_tensor* r_f = matrix_slice_rows(r, hidden_size * 2, hidden_size);
  ggml_tensor* r_c = matrix_slice_rows(r, hidden_size * 3, hidden_size);
  ggml_tensor* wb_i = wb != nullptr ? vector_slice(wb, 0, hidden_size) : nullptr;
  ggml_tensor* wb_o = wb != nullptr ? vector_slice(wb, hidden_size, hidden_size) : nullptr;
  ggml_tensor* wb_f = wb != nullptr ? vector_slice(wb, hidden_size * 2, hidden_size) : nullptr;
  ggml_tensor* wb_c = wb != nullptr ? vector_slice(wb, hidden_size * 3, hidden_size) : nullptr;
  ggml_tensor* rb_i = rb != nullptr ? vector_slice(rb, 0, hidden_size) : nullptr;
  ggml_tensor* rb_o = rb != nullptr ? vector_slice(rb, hidden_size, hidden_size) : nullptr;
  ggml_tensor* rb_f = rb != nullptr ? vector_slice(rb, hidden_size * 2, hidden_size) : nullptr;
  ggml_tensor* rb_c = rb != nullptr ? vector_slice(rb, hidden_size * 3, hidden_size) : nullptr;

  auto gate_preact = [&](ggml_tensor* wg,
                         ggml_tensor* rg,
                         ggml_tensor* x_t,
                         ggml_tensor* wbg,
                         ggml_tensor* rbg) -> ggml_tensor* {
    ggml_tensor* acc = ggml_add(ctx, ggml_mul_mat(ctx, wg, x_t), ggml_mul_mat(ctx, rg, h_t));
    if (wbg != nullptr) {
      acc = ggml_add(ctx, acc, wbg);
    }
    if (rbg != nullptr) {
      acc = ggml_add(ctx, acc, rbg);
    }
    return acc;
  };

  for (int64_t step = 0; step < seq_length; ++step) {
    ggml_tensor* x_t = timestep_slice(x, step);

    ggml_tensor* i_gate = ggml_sigmoid(ctx, gate_preact(w_i, r_i, x_t, wb_i, rb_i));
    ggml_tensor* f_gate = ggml_sigmoid(ctx, gate_preact(w_f, r_f, x_t, wb_f, rb_f));
    ggml_tensor* c_tilde = ggml_tanh(ctx, gate_preact(w_c, r_c, x_t, wb_c, rb_c));

    c_t = ggml_add(ctx, ggml_mul(ctx, f_gate, c_t), ggml_mul(ctx, i_gate, c_tilde));

    ggml_tensor* o_gate = ggml_sigmoid(ctx, gate_preact(w_o, r_o, x_t, wb_o, rb_o));
    h_t = ggml_mul(ctx, o_gate, ggml_tanh(ctx, c_t));

    if (y != nullptr) {
      y = ggml_set(ctx, y, h_t, y->nb[1], y->nb[2], y->nb[3], static_cast<size_t>(step) * y->nb[3]);
    }
  }

  EmitOutputs outputs;
  if (!node.outputs.empty() && node.outputs[0] != kOptionalValueAbsent) {
    outputs.push_back(y);
  }
  if (node.outputs.size() > 1 && node.outputs[1] != kOptionalValueAbsent) {
    outputs.push_back(ggml_reshape_3d(ctx, h_t, hidden_size, batch_size, 1));
  }
  if (node.outputs.size() > 2 && node.outputs[2] != kOptionalValueAbsent) {
    outputs.push_back(ggml_reshape_3d(ctx, c_t, hidden_size, batch_size, 1));
  }
  return outputs;
}

// Shape-preserving unary float op with no attributes. Covers Relu, Sigmoid, Tanh, Neg,
// Abs, Sqrt, Exp, Log, Erf, Softplus, Elu (ONNX default alpha=1.0 only).
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
  if (op == "Erf")      return EmitOutputs{ggml_map_custom1(ctx, ggml_cont(ctx, x), ComputeErf,
                                                            GGML_N_TASKS_MAX, nullptr)};
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

// ONNX PRelu: Y[i] = X[i] if X[i] >= 0 else slope[i] * X[i]. Slope is
// unidirectionally broadcastable to X. ggml has no native PRelu, so we emit
// Y = relu(X) - slope * relu(-X), which evaluates to relu(X) when X >= 0 and
// to slope * X when X < 0 (since -relu(-X) == X on that branch).
bool IsSupportedPReluNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || inputs[1] == nullptr || outputs[0] == nullptr) {
    return false;
  }
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata slope = getTensorMetadata(inputs[1]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      slope.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      y.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(slope) || !rankSupportedByGGML(y)) {
    return false;
  }
  // ggml's broadcast is per-dim "dst[i] % src[i] == 0" in reversed layout,
  // which subsumes ONNX unidirectional broadcasting of slope -> X.
  if (!broadcastSupportedByGGML(slope.dims, x.dims)) {
    return false;
  }
  return true;
}

EmitResult EmitPReluNode(ggml_context* ctx,
                         const NodeDesc& node,
                         const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() == 2 && node.outputs.size() == 1,
                "compiled PRelu node has invalid arity");
  ggml_tensor* x = values[node.inputs[0]];
  ggml_tensor* slope = values[node.inputs[1]];
  GGONNX_NOT_NULL(x, "compiled PRelu node missing GGML X input");
  GGONNX_NOT_NULL(slope, "compiled PRelu node missing GGML slope input");
  ggml_tensor* relu_x = ggml_relu(ctx, x);
  ggml_tensor* relu_neg_x = ggml_relu(ctx, ggml_neg(ctx, x));
  ggml_tensor* scaled = ggml_mul(ctx, relu_neg_x, slope);
  return EmitOutputs{ggml_sub(ctx, relu_x, scaled)};
}

// ONNX Clip: Y = min(max(X, min_val), max_val). Pre-opset-11 reads min/max from
// node attributes; opset 11+ reads them from inputs[1] and inputs[2] (both
// optional, defaulting to -inf / +inf). ggml_clamp takes the two bounds as
// plain floats, so we require the input forms to be compile-time scalars.
bool IsSupportedClipNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.empty() || inputs.size() > 3 || outputs.size() != 1 ||
      node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || outputs[0] == nullptr) return false;
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      y.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(y)) return false;

  // Opset 11+: min/max come in as scalar tensors. We need them as compile-time
  // constants so ggml_clamp can take float literals.
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i] == nullptr) continue;  // optional, absent
    const auto v = readConstantInputArray<float>(node, i, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    if (!v.has_value() || v->size() != 1) return false;
  }
  return true;
}

void CompileClipAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  NodeDesc::ClipAttrs attrs;
  // Opset <11: attributes. Opset >=11: inputs.
  if (const auto min_attr = readNodeAttribute<float>(node, "min")) attrs.min = *min_attr;
  if (const auto max_attr = readNodeAttribute<float>(node, "max")) attrs.max = *max_attr;
  const auto inputs = node.GetInputs();
  if (inputs.size() >= 2 && inputs[1] != nullptr) {
    const auto v = readConstantInputArray<float>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    if (v && v->size() == 1) attrs.min = (*v)[0];
  }
  if (inputs.size() >= 3 && inputs[2] != nullptr) {
    const auto v = readConstantInputArray<float>(node, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    if (v && v->size() == 1) attrs.max = (*v)[0];
  }
  compiled_node->attrs = attrs;
}

EmitResult EmitClipNode(ggml_context* ctx,
                        const NodeDesc& node,
                        const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::ClipAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Clip node missing attributes");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled Clip node missing GGML input");
  // ggml_clamp returns an in-place view that mutates its input; duplicate first
  // so other consumers of x keep their original values.
  ggml_tensor* src = ggml_dup(ctx, x);
  return EmitOutputs{ggml_clamp(ctx, src, attrs->min, attrs->max)};
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
  ggml_tensor* out = ggml_mul_mat(ctx, b_eff, a);
  if (GetForceMatMulF32Precision()) {
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
  }
  return EmitOutputs{out};
}

// ONNX Conv (2D only for now): X[N,C,H,W] @ W[OC,IC/group,KH,KW] (+ optional B[OC]).
// GGML's reverse-dim convention lines up with ggml_conv_2d_direct's expected layouts:
// X -> ne=[W,H,C,N] and W -> ne=[KW,KH,IC,OC]. No transposes needed.
// Limitations: group=1 for regular conv; auto_pad is lowered via direct padding
// when symmetric, or symmetric-overpad + output crop for one-sided SAME_*.
namespace {

struct AutoPadAxis {
  int begin{0};
  int end{0};
  int symmetric{0};
  int crop_begin{0};
  int output{0};
};

AutoPadAxis ResolveAutoPadAxis(int64_t input,
                               int64_t kernel,
                               int64_t stride,
                               int64_t dilation,
                               int64_t output,
                               bool same_upper) {
  GGONNX_ASSERT(input > 0 && kernel > 0 && stride > 0 && dilation > 0 && output > 0,
                "auto_pad requires positive concrete dims");
  const int64_t effective_kernel = dilation * (kernel - 1) + 1;
  const int64_t total_pad =
      std::max<int64_t>(0, (output - 1) * stride + effective_kernel - input);
  const int64_t begin = same_upper ? (total_pad / 2) : ((total_pad + 1) / 2);
  const int64_t end = total_pad - begin;
  const int64_t symmetric = std::max(begin, end);
  const int64_t crop_begin = symmetric - begin;
  return AutoPadAxis{
      .begin = static_cast<int>(begin),
      .end = static_cast<int>(end),
      .symmetric = static_cast<int>(symmetric),
      .crop_begin = static_cast<int>(crop_begin),
      .output = static_cast<int>(output),
  };
}

ggml_tensor* CropSpatialOutputIfNeeded(ggml_context* ctx,
                                       ggml_tensor* x,
                                       int crop0_begin,
                                       int crop1_begin,
                                       int out_w,
                                       int out_h) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_NOT_NULL(x, "crop source tensor must not be null");
  GGONNX_ASSERT(out_w > 0 && out_h > 0, "crop target shape must be positive");
  if (crop0_begin == 0 && crop1_begin == 0 &&
      x->ne[0] == out_w && x->ne[1] == out_h) {
    return x;
  }
  GGONNX_ASSERT(crop0_begin >= 0 && crop1_begin >= 0, "crop offset must be non-negative");
  GGONNX_ASSERT(crop0_begin + out_w <= x->ne[0] && crop1_begin + out_h <= x->ne[1],
                "crop exceeds spatial output bounds");
  ggml_tensor* cropped = ggml_view_4d(
      ctx, x, out_w, out_h, x->ne[2], x->ne[3], x->nb[1], x->nb[2], x->nb[3],
      crop1_begin * x->nb[1] + crop0_begin * x->nb[0]);
  return ggml_cont(ctx, cropped);
}

}  // namespace

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
  if (w.dims.size() != 3 && w.dims.size() != 4) {
    return false;
  }
  const size_t spatial_rank = w.dims.size() - 2;
  if ((x.dims.empty() ? false : x.dims.size() != spatial_rank + 2) ||
      (y.dims.empty() ? false : y.dims.size() != spatial_rank + 2)) {
    return false;
  }
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(w) || !rankSupportedByGGML(y)) {
    return false;
  }

  // Accept group == 1 (regular conv) and depthwise (group == C_in == C_out with
  // weight shape [C, 1, kH, kW]). Depthwise maps to ggml_conv_2d_dw_direct.
  const int64_t group = readNodeAttribute<int64_t>(node, "group").value_or(1);
  if (group != 1) {
    if (!x.dims.empty() && x.dims[1] < 0) return false;
    if (w.dims[0] < 0 || w.dims[1] < 0) return false;
    if (!y.dims.empty() && y.dims[1] < 0) return false;
    if (!x.dims.empty() && group != x.dims[1]) return false;  // must equal C_in
    if (w.dims[0] != group) return false;           // C_out == group (multiplier 1)
    if (w.dims[1] != 1) return false;               // weight IC/group == 1
    if (!y.dims.empty() && y.dims[1] != group) return false;
  }

  const std::string auto_pad = readNodeAttribute<std::string>(node, "auto_pad").value_or("NOTSET");
  if (auto_pad != "NOTSET" && auto_pad != "VALID" &&
      auto_pad != "SAME_UPPER" && auto_pad != "SAME_LOWER") {
    return false;
  }
  const bool same_auto_pad = auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER";
  if (same_auto_pad) {
    if (spatial_rank == 1) return false;
    if (x.dims[2] < 0 || x.dims[3] < 0 || y.dims[2] < 0 || y.dims[3] < 0 ||
        w.dims[2] < 0 || w.dims[3] < 0) {
      return false;
    }
  }

  if (const auto pads = readNodeAttribute<std::vector<int64_t>>(node, "pads")) {
    if (spatial_rank == 1) {
      if (pads->size() != 2) return false;
      if ((*pads)[0] != (*pads)[1]) return false;
      if (auto_pad == "VALID" && (*pads)[0] != 0) return false;
    } else {
      if (pads->size() != 4) return false;
      if ((*pads)[0] != (*pads)[2] || (*pads)[1] != (*pads)[3]) return false;
      if (auto_pad == "VALID" && ((*pads)[0] != 0 || (*pads)[1] != 0)) return false;
    }
  }
  if (const auto strides = readNodeAttribute<std::vector<int64_t>>(node, "strides")) {
    if (strides->size() != spatial_rank) return false;
  }
  if (const auto dilations = readNodeAttribute<std::vector<int64_t>>(node, "dilations")) {
    if (dilations->size() != spatial_rank) return false;
  }
  if (const auto kernel_shape = readNodeAttribute<std::vector<int64_t>>(node, "kernel_shape")) {
    if (kernel_shape->size() != spatial_rank) return false;
    if (spatial_rank == 1) {
      if (w.dims[2] >= 0 && (*kernel_shape)[0] != w.dims[2]) return false;
    } else {
      if (w.dims[2] >= 0 && (*kernel_shape)[0] != w.dims[2]) return false;
      if (w.dims[3] >= 0 && (*kernel_shape)[1] != w.dims[3]) return false;
    }
  }

  if (group == 1 && !x.dims.empty() && x.dims[1] >= 0 && w.dims[1] >= 0 && x.dims[1] != w.dims[1]) {
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
  const auto inputs = node.GetInputs();
  GGONNX_ASSERT(inputs.size() >= 2 && inputs[1] != nullptr,
                "Conv must have a weight input");
  const TensorMetadata w = getTensorMetadata(inputs[1]);
  GGONNX_ASSERT(w.dims.size() == 3 || w.dims.size() == 4,
                "Conv compile requires 1D or 2D weight rank");
  attrs.spatial_rank = static_cast<int>(w.dims.size() - 2);

  if (const auto strides = readNodeAttribute<std::vector<int64_t>>(node, "strides")) {
    if (attrs.spatial_rank == 1) {
      attrs.s0 = static_cast<int>((*strides)[0]);
    } else {
      // ONNX [stride_h, stride_w] -> ggml (s0=width, s1=height).
      attrs.s0 = static_cast<int>((*strides)[1]);
      attrs.s1 = static_cast<int>((*strides)[0]);
    }
  }
  if (const auto dilations = readNodeAttribute<std::vector<int64_t>>(node, "dilations")) {
    if (attrs.spatial_rank == 1) {
      attrs.d0 = static_cast<int>((*dilations)[0]);
    } else {
      attrs.d0 = static_cast<int>((*dilations)[1]);
      attrs.d1 = static_cast<int>((*dilations)[0]);
    }
  }

  const std::string auto_pad = readNodeAttribute<std::string>(node, "auto_pad").value_or("NOTSET");
  if (auto_pad == "NOTSET") {
    if (const auto pads = readNodeAttribute<std::vector<int64_t>>(node, "pads")) {
      if (attrs.spatial_rank == 1) {
        attrs.p0 = static_cast<int>((*pads)[0]);
      } else {
        // pads=[h_begin, w_begin, h_end, w_end], already validated symmetric.
        attrs.p0 = static_cast<int>((*pads)[1]);
        attrs.p1 = static_cast<int>((*pads)[0]);
      }
    }
  } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
    const TensorMetadata x = getTensorMetadata(inputs[0]);
    const TensorMetadata y = getTensorMetadata(node.GetOutputs()[0]);
    const bool same_upper = auto_pad == "SAME_UPPER";
    const int kw = static_cast<int>(w.dims[3]);
    const int kh = static_cast<int>(w.dims[2]);
    const AutoPadAxis w_axis = ResolveAutoPadAxis(
        x.dims[3], kw, attrs.s0, attrs.d0, y.dims[3], same_upper);
    const AutoPadAxis h_axis = ResolveAutoPadAxis(
        x.dims[2], kh, attrs.s1, attrs.d1, y.dims[2], same_upper);
    attrs.p0 = w_axis.symmetric;
    attrs.p1 = h_axis.symmetric;
    attrs.crop0_begin = w_axis.crop_begin;
    attrs.crop1_begin = h_axis.crop_begin;
    attrs.out_w = w_axis.output;
    attrs.out_h = h_axis.output;
  }
  // VALID => pads remain 0.

  attrs.is_depthwise = readNodeAttribute<int64_t>(node, "group").value_or(1) != 1;
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

  ggml_tensor* out = nullptr;
  if (attrs->spatial_rank == 1) {
    // Lower ONNX Conv1D as a degenerate Conv2D so we stay on the same direct
    // kernel family as Conv2D. Layout:
    //   X [N,C,W]      -> [N,C,1,W]
    //   W [OC,IC,K]    -> [OC,IC,1,K]
    //   Y [N,OC,OW]    -> [N,OC,1,OW] -> squeeze H
    ggml_tensor* x_4d = ggml_reshape_4d(ctx, x, x->ne[0], 1, x->ne[1], x->ne[2]);
    ggml_tensor* w_4d = ggml_reshape_4d(ctx, w, w->ne[0], 1, w->ne[1], w->ne[2]);
    ggml_tensor* out_4d = attrs->is_depthwise
        ? ggml_conv_2d_dw_direct(ctx, w_4d, x_4d,
                                 attrs->s0, 1,
                                 attrs->p0, 0,
                                 attrs->d0, 1)
        : ggml_conv_2d_direct(ctx, w_4d, x_4d,
                              attrs->s0, 1,
                              attrs->p0, 0,
                              attrs->d0, 1);
    out = ggml_reshape_3d(ctx, out_4d, out_4d->ne[0], out_4d->ne[2], out_4d->ne[3]);
  } else {
    out = attrs->is_depthwise
        ? ggml_conv_2d_dw_direct(ctx, w, x,
                                 attrs->s0, attrs->s1,
                                 attrs->p0, attrs->p1,
                                 attrs->d0, attrs->d1)
        : ggml_conv_2d_direct(ctx, w, x,
                              attrs->s0, attrs->s1,
                              attrs->p0, attrs->p1,
                              attrs->d0, attrs->d1);
  }

  if (node.inputs.size() == 3 && node.inputs[2] != kOptionalValueAbsent) {
    ggml_tensor* bias = values[node.inputs[2]];
    GGONNX_NOT_NULL(bias, "compiled Conv node missing GGML B input");
    if (attrs->spatial_rank == 1) {
      // bias ne=[OC] -> [1,OC,1] to broadcast across W,N of conv output ne=[OW,OC,N].
      ggml_tensor* bias_3d = ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1);
      out = ggml_add(ctx, out, bias_3d);
    } else {
      // bias ne=[OC] -> [1,1,OC,1] to broadcast across W,H,N of conv output ne=[OW,OH,OC,N].
      ggml_tensor* bias_4d = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
      out = ggml_add(ctx, out, bias_4d);
    }
  }

  if (attrs->spatial_rank == 2 && attrs->out_w > 0 && attrs->out_h > 0) {
    out = CropSpatialOutputIfNeeded(
        ctx, out, attrs->crop0_begin, attrs->crop1_begin, attrs->out_w, attrs->out_h);
  }

  return EmitOutputs{out};
}

// ONNX ConvTranspose: 2D only, square stride, symmetric pads. Maps to
// ggml_conv_transpose_2d_p0 (no built-in padding) followed by a center crop
// when ONNX pads are non-zero, since output = (in-1)*s - 2p + kernel.
bool IsSupportedConvTransposeNode(Ort::ConstNode node) {
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
  if (x.dims.size() != 4 || w.dims.size() != 4 || y.dims.size() != 4) return false;
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(w) || !rankSupportedByGGML(y)) return false;

  if (readNodeAttribute<int64_t>(node, "group").value_or(1) != 1) return false;
  const std::string auto_pad = readNodeAttribute<std::string>(node, "auto_pad").value_or("NOTSET");
  if (auto_pad != "NOTSET") return false;
  if (readNodeAttribute<std::vector<int64_t>>(node, "output_padding").has_value()) {
    const auto op = *readNodeAttribute<std::vector<int64_t>>(node, "output_padding");
    for (int64_t v : op) if (v != 0) return false;
  }
  if (readNodeAttribute<std::vector<int64_t>>(node, "output_shape").has_value()) return false;

  if (const auto dilations = readNodeAttribute<std::vector<int64_t>>(node, "dilations")) {
    if (dilations->size() != 2 || (*dilations)[0] != 1 || (*dilations)[1] != 1) return false;
  }

  const auto strides = readNodeAttribute<std::vector<int64_t>>(node, "strides");
  if (!strides || strides->size() != 2) return false;
  if ((*strides)[0] != (*strides)[1]) return false;

  if (const auto pads = readNodeAttribute<std::vector<int64_t>>(node, "pads")) {
    if (pads->size() != 4) return false;
    if ((*pads)[0] != (*pads)[2] || (*pads)[1] != (*pads)[3]) return false;
    if ((*pads)[0] < 0 || (*pads)[1] < 0) return false;
  }

  if (inputs.size() == 3 && inputs[2] != nullptr) {
    const TensorMetadata b = getTensorMetadata(inputs[2]);
    if (b.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || b.dims.size() != 1) return false;
  }
  return true;
}

void CompileConvTransposeAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  NodeDesc::ConvTransposeAttrs attrs;
  const auto strides = readNodeAttribute<std::vector<int64_t>>(node, "strides");
  GGONNX_ASSERT(strides && strides->size() == 2, "ConvTranspose needs 2D strides");
  attrs.stride = static_cast<int>((*strides)[0]);
  if (const auto pads = readNodeAttribute<std::vector<int64_t>>(node, "pads")) {
    attrs.pad_h = static_cast<int>((*pads)[0]);
    attrs.pad_w = static_cast<int>((*pads)[1]);
  }
  compiled_node->attrs = attrs;
}

EmitResult EmitConvTransposeNode(ggml_context* ctx,
                                 const NodeDesc& node,
                                 const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::ConvTransposeAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled ConvTranspose node missing attributes");
  ggml_tensor* x = values[node.inputs[0]];
  ggml_tensor* w = values[node.inputs[1]];
  GGONNX_NOT_NULL(x, "ConvTranspose missing X input");
  GGONNX_NOT_NULL(w, "ConvTranspose missing W input");

  ggml_tensor* out = ggml_conv_transpose_2d_p0(ctx, w, x, attrs->stride);
  if (attrs->pad_w != 0 || attrs->pad_h != 0) {
    // Emulate ONNX symmetric padding by cropping p pixels from each spatial edge.
    const int64_t new_w = out->ne[0] - 2 * attrs->pad_w;
    const int64_t new_h = out->ne[1] - 2 * attrs->pad_h;
    GGONNX_ASSERT(new_w > 0 && new_h > 0, "ConvTranspose crop produced non-positive size");
    ggml_tensor* cropped = ggml_view_4d(
        ctx, out, new_w, new_h, out->ne[2], out->ne[3],
        out->nb[1], out->nb[2], out->nb[3],
        static_cast<size_t>(attrs->pad_w) * out->nb[0] +
            static_cast<size_t>(attrs->pad_h) * out->nb[1]);
    out = ggml_cont(ctx, cropped);
  }
  if (node.inputs.size() == 3 && node.inputs[2] != kOptionalValueAbsent) {
    ggml_tensor* bias = values[node.inputs[2]];
    GGONNX_NOT_NULL(bias, "ConvTranspose missing B input");
    ggml_tensor* bias_4d = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
    out = ggml_add(ctx, out, bias_4d);
  }
  return EmitOutputs{out};
}

// ONNX Expand: broadcast input to a given shape. We require the shape input to
// be a compile-time constant so the target dims land in the compiled attrs —
// ORT's shape inference typically leaves the output shape symbolic when the
// shape tensor comes from a Where/Equal chain.
bool IsSupportedExpandNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 2 || outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || inputs[1] == nullptr || outputs[0] == nullptr) return false;
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      y.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(y)) return false;
  if (!shapeIsFullyStatic(x)) return false;

  const auto target = readConstantInputArray<int64_t>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  if (!target) return false;
  std::vector<int64_t> target_dims(target->begin(), target->end());
  if (!rankSupportedByGGML({ONNXTensorElementDataType{}, target_dims})) return false;

  // Align to the longer rank (ONNX Expand broadcasts leading dims).
  const size_t out_rank = std::max(x.dims.size(), target_dims.size());
  std::vector<int64_t> padded_x(out_rank, 1);
  std::vector<int64_t> padded_t(out_rank, 1);
  std::copy_backward(x.dims.begin(), x.dims.end(), padded_x.end());
  std::copy_backward(target_dims.begin(), target_dims.end(), padded_t.end());
  for (size_t i = 0; i < out_rank; ++i) {
    if (padded_x[i] != 1 && padded_t[i] != 1 && padded_x[i] != padded_t[i]) return false;
    if (padded_t[i] < 0) return false;
  }
  return true;
}

void CompileExpandAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto inputs = node.GetInputs();
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const auto target = readConstantInputArray<int64_t>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  GGONNX_ASSERT(target.has_value(), "Expand compile: shape must be constant");
  std::vector<int64_t> target_dims = *target;
  const size_t out_rank = std::max(x.dims.size(), target_dims.size());
  std::vector<int64_t> padded_x(out_rank, 1);
  std::vector<int64_t> padded_t(out_rank, 1);
  std::copy_backward(x.dims.begin(), x.dims.end(), padded_x.end());
  std::copy_backward(target_dims.begin(), target_dims.end(), padded_t.end());
  std::vector<int64_t> out_dims(out_rank);
  for (size_t i = 0; i < out_rank; ++i) {
    out_dims[i] = std::max<int64_t>(padded_x[i], padded_t[i]);
  }
  compiled_node->attrs = NodeDesc::ExpandAttrs{.onnx_dims = std::move(out_dims)};
}

EmitResult EmitExpandNode(ggml_context* ctx,
                          const NodeDesc& node,
                          const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::ExpandAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Expand node missing attributes");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "Expand missing input");

  const std::array<int64_t, GGML_MAX_DIMS> ggml_ne = ToPaddedGGMLDims(attrs->onnx_dims);
  ggml_tensor* src = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
  // ggml_repeat_4d handles integer-multiple broadcasting; source dims of 1
  // become N = target / 1 = target, which is what ONNX Expand means for those
  // axes.
  ggml_tensor* out = ggml_repeat_4d(ctx, src, ggml_ne[0], ggml_ne[1], ggml_ne[2], ggml_ne[3]);
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

// TODO(channel-shuffle): fuse the Reshape→Transpose→Reshape triple that ShuffleNet
// (and similar) use for channel shuffle. The 5D intermediate ([N,g,k,H,W] with
// perm=[0,2,1,3,4]) cannot be held in ggml (GGML_MAX_DIMS=4), but the whole
// triple collapses to a pure 4D view+permute+cont+reshape because H,W are
// pass-through:
//   flat    = reshape_4d(in, H*W, k, g, N)
//   swapped = permute(flat, 0, 2, 1, 3)
//   packed  = cont(swapped)
//   out     = reshape_4d(packed, W, H, C, N)
// Needs: pattern detector in EpGetCapability (verify single-consumer on both
// 5D intermediates), skip ensure_value for the rank-5 values (the rank check
// at src/ggml_execution_provider.cc:309 trips otherwise), and a synthetic
// fused NodeDesc (e.g. "__ChannelShuffle" with attrs {g,k}) emitting the above.
// Without this, those Reshape/Transpose nodes fall back to CPU — numerically
// correct but breaks assert_all_nodes_run_on_ggml on shufflenet.

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

// ONNX Flatten: collapses input dims around `axis` into a 2D tensor. We reuse
// the Reshape machinery — the output shape is fully determined by shape
// inference, so we snapshot it and emit a ggml reshape.
bool IsSupportedFlattenNode(Ort::ConstNode node) {
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
  if (!shapeIsFullyStatic(out.dims)) {
    return false;
  }
  return true;
}

// ONNX Squeeze: removes axes of length 1. Like Reshape/Flatten, once shape
// inference has run the runtime axes input is redundant, so we only require the
// output shape to be fully static and lower to a ggml reshape.
bool IsSupportedSqueezeNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if ((inputs.size() != 1 && inputs.size() != 2) || outputs.size() != 1 ||
      node.GetImplicitInputs().size() != 0) {
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

void CompileSqueezeAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto outputs = node.GetOutputs();
  GGONNX_ASSERT(outputs.size() == 1 && outputs[0] != nullptr,
                "Squeeze must have a single output");
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  compiled_node->attrs = NodeDesc::ReshapeAttrs{.onnx_dims = out.dims};
}

void CompileFlattenAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto outputs = node.GetOutputs();
  GGONNX_ASSERT(outputs.size() == 1 && outputs[0] != nullptr,
                "Flatten must have a single output");
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  compiled_node->attrs = NodeDesc::ReshapeAttrs{.onnx_dims = out.dims};
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
  if (auto_pad != "NOTSET" && auto_pad != "VALID" &&
      auto_pad != "SAME_UPPER" && auto_pad != "SAME_LOWER") {
    return false;
  }
  if ((auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") &&
      (x.dims[2] < 0 || x.dims[3] < 0 || y.dims[2] < 0 || y.dims[3] < 0)) {
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
  } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
    const TensorMetadata x = getTensorMetadata(node.GetInputs()[0]);
    const TensorMetadata y = getTensorMetadata(node.GetOutputs()[0]);
    const bool same_upper = auto_pad == "SAME_UPPER";
    const AutoPadAxis w_axis = ResolveAutoPadAxis(
        x.dims[3], (*kernel_shape)[1], attrs.s0, 1, y.dims[3], same_upper);
    const AutoPadAxis h_axis = ResolveAutoPadAxis(
        x.dims[2], (*kernel_shape)[0], attrs.s1, 1, y.dims[2], same_upper);
    attrs.p0 = w_axis.symmetric;
    attrs.p1 = h_axis.symmetric;
    attrs.crop0_begin = w_axis.crop_begin;
    attrs.crop1_begin = h_axis.crop_begin;
    attrs.out_w = w_axis.output;
    attrs.out_h = h_axis.output;
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
  if (!attrs->is_global && attrs->out_w > 0 && attrs->out_h > 0) {
    out = CropSpatialOutputIfNeeded(
        ctx, out, attrs->crop0_begin, attrs->crop1_begin, attrs->out_w, attrs->out_h);
  }
  return EmitOutputs{out};
}

bool IsSupportedPadNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.empty() || inputs.size() > 4 || outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
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

  const std::string mode = readNodeAttribute<std::string>(node, "mode").value_or("constant");
  if (mode != "reflect" && mode != "constant") {
    return false;
  }

  const auto pads = readPadVector(node);
  if (!pads.has_value() || pads->size() != 8) {
    return false;
  }

  // Per-ONNX-axis validation. Pad can be nonzero on any subset of axes; reflect
  // mode additionally requires non-negative pads strictly smaller than the src
  // length along each active axis.
  const int rank = static_cast<int>(x.dims.size());
  int active_axes = 0;
  for (int a = 0; a < rank; ++a) {
    const int64_t begin = (*pads)[a];
    const int64_t end = (*pads)[a + rank];
    if (begin == 0 && end == 0) continue;
    ++active_axes;
    if (mode == "reflect" && (begin < 0 || end < 0)) return false;
    if (x.dims[a] < 0) continue;  // unresolved dim: trust the export
    if (begin + end + x.dims[a] <= 0) return false;
    if (mode == "reflect" && (begin >= x.dims[a] || end >= x.dims[a])) return false;
  }
  // Reflect support is 1D in ggml, so we pad one axis at a time; ≤ 2 active
  // axes keeps the emitted sequence short. Constant mode uses ggml_pad_ext in
  // one shot and already tolerates any axis set — keep the same cap for
  // simplicity.
  if (active_axes > 2) return false;

  if (mode == "constant") {
    // Only zero fill is supported. Accept an absent constant input or one
    // that folds to a single zero float.
    const auto node_inputs = node.GetInputs();
    if (node_inputs.size() >= 3 && node_inputs[2] != nullptr) {
      const auto v = readConstantInputArray<float>(node, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      if (!v) return false;
      if (v->size() > 1) return false;
      if (!v->empty() && (*v)[0] != 0.0f) return false;
    }
  }

  return true;
}

void CompilePadAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto pads = readPadVector(node);
  GGONNX_ASSERT(pads.has_value() && pads->size() == 8,
                "Pad node must provide 8-element pads");
  const auto inputs = node.GetInputs();
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const int rank = static_cast<int>(x.dims.size());
  const std::string mode = readNodeAttribute<std::string>(node, "mode").value_or("constant");

  NodeDesc::PadAttrs attrs;
  attrs.mode = mode == "reflect" ? NodeDesc::PadAttrs::Mode::Reflect
                                 : NodeDesc::PadAttrs::Mode::Constant;
  // ONNX axis a maps to ggml axis (rank-1-a).
  for (int a = 0; a < rank; ++a) {
    const int ggml_axis = rank - 1 - a;
    attrs.pad_begin[ggml_axis] = static_cast<int>((*pads)[a]);
    attrs.pad_end[ggml_axis] = static_cast<int>((*pads)[a + rank]);
  }
  compiled_node->attrs = attrs;
}

EmitResult EmitPadNode(ggml_context* ctx,
                       const NodeDesc& node,
                       const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(!node.inputs.empty() && node.outputs.size() == 1,
                "compiled Pad node has invalid arity");
  const auto* attrs = std::get_if<NodeDesc::PadAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Pad node missing attributes");

  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled Pad node missing GGML input");

  ggml_tensor* out = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
  if (attrs->mode == NodeDesc::PadAttrs::Mode::Constant) {
    // Crop phase (negative pads) via a 4D view + cont.
    std::array<int, GGML_MAX_DIMS> crop_begin{};
    std::array<int, GGML_MAX_DIMS> crop_end{};
    bool any_crop = false;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
      crop_begin[i] = std::max(0, -attrs->pad_begin[i]);
      crop_end[i] = std::max(0, -attrs->pad_end[i]);
      any_crop = any_crop || crop_begin[i] || crop_end[i];
    }
    if (any_crop) {
      std::array<int64_t, GGML_MAX_DIMS> new_ne{};
      for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        new_ne[i] = out->ne[i] - crop_begin[i] - crop_end[i];
        GGONNX_ASSERT(new_ne[i] > 0, "Pad crop produced non-positive size");
      }
      size_t offset = 0;
      for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        offset += static_cast<size_t>(crop_begin[i]) * out->nb[i];
      }
      ggml_tensor* cropped = ggml_view_4d(
          ctx, out, new_ne[0], new_ne[1], new_ne[2], new_ne[3],
          out->nb[1], out->nb[2], out->nb[3], offset);
      out = ggml_cont(ctx, cropped);
    }
    // Positive-pad phase: ggml_pad_ext fills with zeros on any axis set.
    std::array<int, GGML_MAX_DIMS> pos_begin{};
    std::array<int, GGML_MAX_DIMS> pos_end{};
    bool any_pad = false;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
      pos_begin[i] = std::max(0, attrs->pad_begin[i]);
      pos_end[i] = std::max(0, attrs->pad_end[i]);
      any_pad = any_pad || pos_begin[i] || pos_end[i];
    }
    if (any_pad) {
      out = ggml_pad_ext(ctx, out,
                         pos_begin[0], pos_end[0],
                         pos_begin[1], pos_end[1],
                         pos_begin[2], pos_end[2],
                         pos_begin[3], pos_end[3]);
    }
    return EmitOutputs{out};
  }
  // Reflect mode: ggml_pad_reflect_1d works on ggml axis 0 only. For any other
  // active axis, rotate it into axis 0 with ggml_permute + ggml_cont, pad,
  // and rotate back.
  for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
    const int begin = attrs->pad_begin[axis];
    const int end = attrs->pad_end[axis];
    if (begin == 0 && end == 0) continue;
    if (axis == 0) {
      out = ggml_pad_reflect_1d(ctx, out, begin, end);
      continue;
    }
    std::array<int, GGML_MAX_DIMS> perm{0, 1, 2, 3};
    std::swap(perm[0], perm[axis]);
    ggml_tensor* rotated =
        ggml_cont(ctx, ggml_permute(ctx, out, perm[0], perm[1], perm[2], perm[3]));
    rotated = ggml_pad_reflect_1d(ctx, rotated, begin, end);
    out = ggml_cont(ctx, ggml_permute(ctx, rotated, perm[0], perm[1], perm[2], perm[3]));
  }

  return EmitOutputs{out};
}

bool IsSupportedInstanceNormalizationNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 3 || outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr || outputs[0] == nullptr) {
    return false;
  }

  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata scale = getTensorMetadata(inputs[1]);
  const TensorMetadata bias = getTensorMetadata(inputs[2]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      scale.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      bias.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      y.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (x.dims.size() != 4 || y.dims.size() != 4 || scale.dims.size() != 1 || bias.dims.size() != 1) {
    return false;
  }
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(y) ||
      !rankSupportedByGGML(scale) || !rankSupportedByGGML(bias)) {
    return false;
  }
  if (x.dims[1] >= 0 && scale.dims[0] >= 0 && x.dims[1] != scale.dims[0]) {
    return false;
  }
  if (scale.dims[0] >= 0 && bias.dims[0] >= 0 && scale.dims[0] != bias.dims[0]) {
    return false;
  }
  return true;
}

void CompileInstanceNormalizationAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  compiled_node->attrs = NodeDesc::InstanceNormAttrs{
      .epsilon = readNodeAttribute<float>(node, "epsilon").value_or(1e-5f),
  };
}

EmitResult EmitInstanceNormalizationNode(ggml_context* ctx,
                                         const NodeDesc& node,
                                         const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() == 3 && node.outputs.size() == 1,
                "compiled InstanceNormalization node has invalid arity");
  const auto* attrs = std::get_if<NodeDesc::InstanceNormAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled InstanceNormalization node missing attributes");

  ggml_tensor* x = values[node.inputs[0]];
  ggml_tensor* scale = values[node.inputs[1]];
  ggml_tensor* bias = values[node.inputs[2]];
  GGONNX_NOT_NULL(x, "compiled InstanceNormalization node missing GGML input");
  GGONNX_NOT_NULL(scale, "compiled InstanceNormalization node missing scale input");
  GGONNX_NOT_NULL(bias, "compiled InstanceNormalization node missing bias input");

  GGONNX_ASSERT(x->ne[2] > 0, "InstanceNormalization channel dimension must be positive");
  ggml_tensor* norm = ggml_group_norm(ctx, x, static_cast<int>(x->ne[2]), attrs->epsilon);
  ggml_tensor* scale_4d = ggml_reshape_4d(ctx, scale, 1, 1, scale->ne[0], 1);
  ggml_tensor* bias_4d = ggml_reshape_4d(ctx, bias, 1, 1, bias->ne[0], 1);
  return EmitOutputs{ggml_add(ctx, ggml_mul(ctx, norm, scale_4d), bias_4d)};
}

bool IsSupportedUpsampleNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  const std::string op_type = node.GetOperatorType();
  const bool is_resize = op_type == "Resize";
  if ((!is_resize && (inputs.empty() || inputs.size() > 2)) ||
      (is_resize && (inputs.empty() || inputs.size() > 4)) ||
      outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
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

  const std::string mode = readNodeAttribute<std::string>(node, "mode").value_or("nearest");
  if (mode != "nearest") {
    return false;
  }

  if (is_resize) {
    const std::string coordinate_transformation_mode =
        readNodeAttribute<std::string>(node, "coordinate_transformation_mode").value_or("half_pixel");
    if (coordinate_transformation_mode != "asymmetric") {
      return false;
    }
    const std::string nearest_mode =
        readNodeAttribute<std::string>(node, "nearest_mode").value_or("round_prefer_floor");
    if (nearest_mode != "floor" && nearest_mode != "round_prefer_floor") {
      return false;
    }
  }

  int scale_h = 0;
  int scale_w = 0;
  if (!inferIntegerSpatialScale(x, y, &scale_h, &scale_w)) {
    return false;
  }
  if (scale_h != scale_w) {
    return false;
  }

  if (op_type == "Upsample" && inputs.size() == 1) {
    const auto scales = readNodeAttribute<std::vector<float>>(node, "scales");
    if (!scales.has_value() || scales->size() != 4) {
      return false;
    }
    if ((*scales)[0] != 1.0f || (*scales)[1] != 1.0f ||
        (*scales)[2] != static_cast<float>(scale_h) ||
        (*scales)[3] != static_cast<float>(scale_w)) {
      return false;
    }
  }

  if (op_type == "Upsample" && inputs.size() == 2) {
    const auto scales =
        readConstantInputArray<float>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    if (scales.has_value() &&
        (scales->size() != 4 || (*scales)[0] != 1.0f || (*scales)[1] != 1.0f ||
         (*scales)[2] != static_cast<float>(scale_h) || (*scales)[3] != static_cast<float>(scale_w))) {
      return false;
    }
  }

  if (is_resize && inputs.size() >= 3 && inputs[2] != nullptr) {
    const auto scales =
        readConstantInputArray<float>(node, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    if (scales.has_value() &&
        (scales->size() != 4 || (*scales)[0] != 1.0f || (*scales)[1] != 1.0f ||
         (*scales)[2] != static_cast<float>(scale_h) || (*scales)[3] != static_cast<float>(scale_w))) {
      return false;
    }
  }

  if (is_resize && inputs.size() >= 4 && inputs[3] != nullptr) {
    const auto sizes =
        readConstantInputArray<int64_t>(node, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    if (sizes.has_value() &&
        (sizes->size() != 4 || (*sizes)[0] != y.dims[0] || (*sizes)[1] != y.dims[1] ||
         (*sizes)[2] != y.dims[2] || (*sizes)[3] != y.dims[3])) {
      return false;
    }
  }

  if (!is_resize && inputs.size() == 1 && scale_h <= 0) {
    return false;
  }

  return true;
}

void CompileUpsampleAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  GGONNX_ASSERT(!inputs.empty() && inputs[0] != nullptr && outputs.size() == 1 && outputs[0] != nullptr,
                "Upsample/Resize must have one data input and one output");
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  int scale_h = 0;
  int scale_w = 0;
  GGONNX_ASSERT(inferIntegerSpatialScale(x, y, &scale_h, &scale_w),
                "Upsample/Resize must have static integer spatial scale");
  compiled_node->attrs = NodeDesc::UpsampleAttrs{
      .scale_w = scale_w,
      .scale_h = scale_h,
  };
}

EmitResult EmitUpsampleNode(ggml_context* ctx,
                            const NodeDesc& node,
                            const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(!node.inputs.empty() && node.outputs.size() == 1,
                "compiled Upsample/Resize node has invalid arity");
  const auto* attrs = std::get_if<NodeDesc::UpsampleAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Upsample node missing attributes");

  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled Upsample node missing GGML input");
  GGONNX_ASSERT(attrs->scale_w == attrs->scale_h,
                "GGML nearest upsample requires equal width/height scale");
  return EmitOutputs{ggml_upscale(ctx, x, attrs->scale_w, GGML_SCALE_MODE_NEAREST)};
}

// Identity: logical copy. We emit a view of the input so the result has a
// distinct ggml_tensor* (graph build asserts on reused output slots) while
// sharing storage and staying cheap.
bool IsSupportedIdentityNode(Ort::ConstNode node) {
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
  return true;
}

EmitResult EmitIdentityNode(ggml_context* ctx,
                            const NodeDesc& node,
                            const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled Identity node missing GGML input");
  // ggml_view_tensor() has op==NONE, so if the partition output is a view-only
  // chain the graph allocator skips the input buffer entirely. ggml_cont is a
  // real op and keeps the graph traversal honest.
  return EmitOutputs{ggml_cont(ctx, x)};
}

// ONNX Transpose: permutes dims by `perm` (default = reverse). GGML dim order is
// reversed vs ONNX, so the ggml axis that feeds output GGML axis j is
// R-1 - perm[R-1 - j]. Axes beyond the ONNX rank are padded identity.
bool IsSupportedTransposeNode(Ort::ConstNode node) {
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
  const size_t rank = in.dims.size();
  if (rank == 0 || rank != out.dims.size()) {
    return false;
  }
  if (const auto perm = readNodeAttribute<std::vector<int64_t>>(node, "perm")) {
    if (perm->size() != rank) return false;
    std::vector<int> seen(rank, 0);
    for (int64_t p : *perm) {
      if (p < 0 || static_cast<size_t>(p) >= rank) return false;
      if (seen[p]) return false;
      seen[p] = 1;
    }
  }
  return true;
}

void CompileTransposeAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto inputs = node.GetInputs();
  const size_t rank = getTensorMetadata(inputs[0]).dims.size();

  std::vector<int64_t> perm;
  if (const auto attr = readNodeAttribute<std::vector<int64_t>>(node, "perm")) {
    perm = *attr;
  } else {
    // ONNX default: reverse dims.
    perm.resize(rank);
    for (size_t i = 0; i < rank; ++i) perm[i] = static_cast<int64_t>(rank - 1 - i);
  }

  // Invert the ONNX permutation: inv[perm[i]] = i. Semantically inv maps an
  // input ONNX dim to the output ONNX dim it ends up at — which matches
  // ggml_permute's axis_j = "new position of input axis j".
  std::vector<int64_t> inv(rank);
  for (size_t i = 0; i < rank; ++i) {
    inv[static_cast<size_t>(perm[i])] = static_cast<int64_t>(i);
  }

  NodeDesc::TransposeAttrs attrs;
  // Identity for padded GGML axes beyond input rank.
  for (int i = 0; i < GGML_MAX_DIMS; ++i) attrs.ggml_perm[i] = i;
  for (size_t j = 0; j < rank; ++j) {
    const size_t onnx_in_dim = rank - 1 - j;
    const size_t onnx_out_dim = static_cast<size_t>(inv[onnx_in_dim]);
    attrs.ggml_perm[j] = static_cast<int>(rank - 1 - onnx_out_dim);
  }
  compiled_node->attrs = attrs;
}

EmitResult EmitTransposeNode(ggml_context* ctx,
                             const NodeDesc& node,
                             const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::TransposeAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Transpose node missing attributes");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled Transpose node missing GGML input");
  ggml_tensor* permuted = ggml_permute(ctx,
                                       x,
                                       attrs->ggml_perm[0],
                                       attrs->ggml_perm[1],
                                       attrs->ggml_perm[2],
                                       attrs->ggml_perm[3]);
  // ggml_permute returns a non-contiguous view; materialize so downstream ops
  // that assume contiguous input (reshape, concat, mul_mat) are happy.
  return EmitOutputs{ggml_cont(ctx, permuted)};
}

// ONNX Concat: join N inputs along `axis`. GGML's ggml_concat takes two tensors
// and a GGML-axis integer, so we translate ONNX axis -> (rank-1-axis) and fold
// left-to-right.
bool IsSupportedConcatNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.empty() || outputs.size() != 1 || node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (outputs[0] == nullptr) return false;
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  if (out.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || !rankSupportedByGGML(out)) {
    return false;
  }
  const size_t rank = out.dims.size();
  if (rank == 0) return false;

  const auto axis_attr = readNodeAttribute<int64_t>(node, "axis");
  if (!axis_attr) return false;
  int64_t axis = *axis_attr;
  if (axis < 0) axis += static_cast<int64_t>(rank);
  if (axis < 0 || axis >= static_cast<int64_t>(rank)) return false;

  for (Ort::ConstValueInfo input : inputs) {
    if (input == nullptr) return false;
    const TensorMetadata meta = getTensorMetadata(input);
    if (meta.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return false;
    if (meta.dims.size() != rank) return false;
    if (!rankSupportedByGGML(meta)) return false;
  }
  return true;
}

void CompileConcatAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto outputs = node.GetOutputs();
  const size_t rank = getTensorMetadata(outputs[0]).dims.size();
  const int64_t axis = readNodeAttribute<int64_t>(node, "axis").value_or(0);
  const int64_t normalized = axis < 0 ? axis + static_cast<int64_t>(rank) : axis;
  // Store as GGML axis index so emit is a pure translation.
  compiled_node->attrs = NodeDesc::AxisAttrs{.axis = static_cast<int64_t>(rank) - 1 - normalized};
}

EmitResult EmitConcatNode(ggml_context* ctx,
                          const NodeDesc& node,
                          const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::AxisAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Concat node missing attributes");
  GGONNX_ASSERT(!node.inputs.empty(), "compiled Concat node has no inputs");
  const int ggml_axis = static_cast<int>(attrs->axis);

  ggml_tensor* acc = values[node.inputs[0]];
  GGONNX_NOT_NULL(acc, "compiled Concat node missing GGML input 0");
  if (!ggml_is_contiguous(acc)) acc = ggml_cont(ctx, acc);

  for (size_t i = 1; i < node.inputs.size(); ++i) {
    ggml_tensor* rhs = values[node.inputs[i]];
    GGONNX_NOT_NULL(rhs, "compiled Concat node missing GGML input");
    if (!ggml_is_contiguous(rhs)) rhs = ggml_cont(ctx, rhs);
    acc = ggml_concat(ctx, acc, rhs, ggml_axis);
  }
  return EmitOutputs{acc};
}

// ONNX Slice (opset >= 10): data, starts, ends, axes?, steps?. We handle the
// common case where starts/ends/axes/steps are constant int64 initializers and
// all steps are 1 — that lets us lower the op to a single ggml_view_4d, which
// is a zero-copy aliased view of the source buffer.
bool IsSupportedSliceNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() < 3 || inputs.size() > 5 || outputs.size() != 1 ||
      node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || outputs[0] == nullptr) {
    return false;
  }

  const TensorMetadata data = getTensorMetadata(inputs[0]);
  const TensorMetadata out = getTensorMetadata(outputs[0]);
  if (data.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      out.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (!rankSupportedByGGML(data) || !rankSupportedByGGML(out)) {
    return false;
  }
  if (!shapeIsFullyStatic(data) || data.dims.size() == 0) {
    return false;
  }
  // We require constant int64 starts/ends (and axes/steps if present) so the
  // view can be materialized at compile time. Dynamic Slice falls back to CPU.
  const auto starts = readConstantInputArray<int64_t>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  const auto ends = readConstantInputArray<int64_t>(node, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  if (!starts || !ends || starts->size() != ends->size() || starts->empty()) {
    return false;
  }
  if (inputs.size() >= 4 && inputs[3] != nullptr) {
    const auto axes = readConstantInputArray<int64_t>(node, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    if (!axes || axes->size() != starts->size()) return false;
  }
  if (inputs.size() == 5 && inputs[4] != nullptr) {
    const auto steps = readConstantInputArray<int64_t>(node, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    if (!steps || steps->size() != starts->size()) return false;
    for (int64_t s : *steps) {
      if (s != 1) return false;
    }
  }
  return true;
}

void CompileSliceAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto inputs = node.GetInputs();
  const TensorMetadata data = getTensorMetadata(inputs[0]);
  const int64_t rank = static_cast<int64_t>(data.dims.size());

  const auto starts = readConstantInputArray<int64_t>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  const auto ends = readConstantInputArray<int64_t>(node, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  GGONNX_ASSERT(starts.has_value() && ends.has_value(),
                "Slice compile: starts/ends must be constant int64 initializers");

  std::vector<int64_t> axes;
  if (inputs.size() >= 4 && inputs[3] != nullptr) {
    const auto axes_opt = readConstantInputArray<int64_t>(node, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    GGONNX_ASSERT(axes_opt.has_value(), "Slice compile: axes must be a constant int64 initializer");
    axes = *axes_opt;
  } else {
    axes.resize(starts->size());
    for (size_t i = 0; i < starts->size(); ++i) axes[i] = static_cast<int64_t>(i);
  }

  NodeDesc::SliceAttrs attrs;
  attrs.ggml_starts.fill(0);
  // Default: each dim passes through unsliced.
  for (int64_t i = 0; i < rank; ++i) {
    attrs.ggml_ne[rank - 1 - i] = data.dims[i];
  }
  for (int i = rank; i < GGML_MAX_DIMS; ++i) {
    attrs.ggml_ne[i] = 1;
  }

  for (size_t k = 0; k < axes.size(); ++k) {
    int64_t onnx_axis = axes[k];
    if (onnx_axis < 0) onnx_axis += rank;
    GGONNX_ASSERT(onnx_axis >= 0 && onnx_axis < rank,
                  "Slice compile: axis out of range");
    const int64_t dim = data.dims[onnx_axis];

    // ONNX clamping rules: clamp start to [0, dim], clamp end to [0, dim].
    auto clamp = [](int64_t v, int64_t lo, int64_t hi) {
      return v < lo ? lo : (v > hi ? hi : v);
    };
    int64_t s = (*starts)[k];
    int64_t e = (*ends)[k];
    if (s < 0) s += dim;
    if (e < 0) e += dim;
    s = clamp(s, 0, dim);
    e = clamp(e, 0, dim);
    const int64_t length = e > s ? e - s : 0;

    const int ggml_axis = static_cast<int>(rank - 1 - onnx_axis);
    attrs.ggml_starts[ggml_axis] = s;
    attrs.ggml_ne[ggml_axis] = length;
  }

  compiled_node->attrs = attrs;
}

EmitResult EmitSliceNode(ggml_context* ctx,
                         const NodeDesc& node,
                         const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::SliceAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Slice node missing attributes");
  ggml_tensor* data = values[node.inputs[0]];
  GGONNX_NOT_NULL(data, "compiled Slice node missing GGML data input");

  // Inherit source strides — a step==1 slice is a plain rectangular view.
  const size_t offset_bytes =
      static_cast<size_t>(attrs->ggml_starts[0]) * data->nb[0] +
      static_cast<size_t>(attrs->ggml_starts[1]) * data->nb[1] +
      static_cast<size_t>(attrs->ggml_starts[2]) * data->nb[2] +
      static_cast<size_t>(attrs->ggml_starts[3]) * data->nb[3];

  ggml_tensor* view = ggml_view_4d(ctx, data,
                                   attrs->ggml_ne[0], attrs->ggml_ne[1],
                                   attrs->ggml_ne[2], attrs->ggml_ne[3],
                                   data->nb[1], data->nb[2], data->nb[3],
                                   offset_bytes);
  // Materialize because downstream kernels (and the graph-output copy path)
  // require contiguous memory. The view keeps the same dims but aliases the
  // source buffer — ggml_cont does the actual pack.
  return EmitOutputs{ggml_cont(ctx, view)};
}

// ONNX BatchNormalization (inference mode): 5 inputs, 1 output.
// Y = scale * (X - mean) / sqrt(var + eps) + bias, broadcast over channel axis.
// Supports ONNX rank 2..4. ONNX channel is always axis 1, which in ggml's
// reversed layout is ggml axis (rank-2). ORT usually folds BN into the
// preceding Conv bias, so this mostly matters for BN outside Conv (e.g. the
// trailing BN on arcface's flattened feature vector, or language models).
bool IsSupportedBatchNormNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 5 || outputs.size() != 1 ||
      node.GetImplicitInputs().size() != 0) {
    return false;
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i] == nullptr) return false;
  }
  if (outputs[0] == nullptr) return false;
  if (readNodeAttribute<int64_t>(node, "training_mode").value_or(0) != 0) {
    return false;
  }

  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      y.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (x.dims.size() < 2 || x.dims.size() > 4) return false;
  if (x.dims.size() != y.dims.size()) return false;
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(y)) return false;

  for (int i = 1; i < 5; ++i) {
    const TensorMetadata p = getTensorMetadata(inputs[i]);
    if (p.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return false;
    if (p.dims.size() != 1) return false;
  }
  return true;
}

void CompileBatchNormAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto inputs = node.GetInputs();
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  compiled_node->attrs = NodeDesc::BatchNormAttrs{
      .epsilon = readNodeAttribute<float>(node, "epsilon").value_or(1e-5f),
      .onnx_rank = static_cast<int>(x.dims.size()),
  };
}

EmitResult EmitBatchNormNode(ggml_context* ctx,
                             const NodeDesc& node,
                             const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_ASSERT(node.inputs.size() == 5 && node.outputs.size() == 1,
                "compiled BatchNormalization node has invalid arity");
  const auto* attrs = std::get_if<NodeDesc::BatchNormAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled BatchNormalization node missing attributes");

  ggml_tensor* x = values[node.inputs[0]];
  ggml_tensor* scale = values[node.inputs[1]];
  ggml_tensor* bias = values[node.inputs[2]];
  ggml_tensor* mean = values[node.inputs[3]];
  ggml_tensor* var = values[node.inputs[4]];
  GGONNX_NOT_NULL(x, "compiled BatchNormalization node missing GGML X input");
  GGONNX_NOT_NULL(scale, "compiled BatchNormalization node missing scale input");
  GGONNX_NOT_NULL(bias, "compiled BatchNormalization node missing bias input");
  GGONNX_NOT_NULL(mean, "compiled BatchNormalization node missing mean input");
  GGONNX_NOT_NULL(var, "compiled BatchNormalization node missing var input");

  // Place C at ggml axis (rank-2), 1s elsewhere. For rank 4 NCHW: ne[2]=C; for
  // rank 3 [N,C,L]: ne[1]=C; for rank 2 [N,C]: ne[0]=C. ggml tensors always
  // carry 4 ne slots, so a 4D reshape works for all ranks.
  const int channel_axis = attrs->onnx_rank - 2;
  GGONNX_ASSERT(channel_axis >= 0 && channel_axis < GGML_MAX_DIMS,
                "BatchNormalization has unexpected onnx_rank");
  const int64_t c = x->ne[channel_axis];
  int64_t ne[4] = {1, 1, 1, 1};
  ne[channel_axis] = c;
  ggml_tensor* scale_4d = ggml_reshape_4d(ctx, scale, ne[0], ne[1], ne[2], ne[3]);
  ggml_tensor* bias_4d = ggml_reshape_4d(ctx, bias, ne[0], ne[1], ne[2], ne[3]);
  ggml_tensor* mean_4d = ggml_reshape_4d(ctx, mean, ne[0], ne[1], ne[2], ne[3]);
  ggml_tensor* var_4d = ggml_reshape_4d(ctx, var, ne[0], ne[1], ne[2], ne[3]);

  // std = sqrt(var + eps); scale_bias(a, s, b) = s*a + b, so we can fold the
  // scalar epsilon add into the same op without a separate eps-filled tensor.
  ggml_tensor* std_tensor = ggml_sqrt(ctx, ggml_scale_bias(ctx, var_4d, 1.0f, attrs->epsilon));
  ggml_tensor* centered = ggml_sub(ctx, x, mean_4d);
  ggml_tensor* normed = ggml_div(ctx, centered, std_tensor);
  ggml_tensor* out = ggml_add(ctx, ggml_mul(ctx, normed, scale_4d), bias_4d);
  return EmitOutputs{out};
}

// ONNX Split (opset 2/11/13): produces N outputs by slicing `data` along `axis`.
// We handle the case where the split sizes are known at compile time — either
// via the `split` attribute (opset <13), the `split` input (opset 13+), or by
// equal division into `outputs.size()` chunks (the default).
bool IsSupportedSplitNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.empty() || inputs.size() > 2 || outputs.empty() ||
      node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr) return false;
  const TensorMetadata data = getTensorMetadata(inputs[0]);
  if (data.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return false;
  if (!rankSupportedByGGML(data) || data.dims.empty()) return false;

  const int64_t rank = static_cast<int64_t>(data.dims.size());
  int64_t axis = readNodeAttribute<int64_t>(node, "axis").value_or(0);
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) return false;
  if (data.dims[axis] < 0) return false;

  // Split sizes: prefer the `split` input (opset 13+), then the attribute
  // (opset 2/11), then equal division.
  std::vector<int64_t> split_sizes;
  if (inputs.size() == 2 && inputs[1] != nullptr) {
    const auto s = readConstantInputArray<int64_t>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    if (!s) return false;
    split_sizes = *s;
  } else if (const auto attr = readNodeAttribute<std::vector<int64_t>>(node, "split")) {
    split_sizes = *attr;
  } else {
    const int64_t n = static_cast<int64_t>(outputs.size());
    if (n <= 0 || data.dims[axis] % n != 0) return false;
    split_sizes.assign(n, data.dims[axis] / n);
  }
  if (split_sizes.size() != outputs.size()) return false;
  int64_t total = 0;
  for (int64_t s : split_sizes) {
    if (s < 0) return false;
    total += s;
  }
  if (total != data.dims[axis]) return false;

  for (Ort::ConstValueInfo out : outputs) {
    if (out == nullptr) return false;
    const TensorMetadata meta = getTensorMetadata(out);
    if (meta.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return false;
    if (!rankSupportedByGGML(meta)) return false;
  }
  return true;
}

void CompileSplitAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  const TensorMetadata data = getTensorMetadata(inputs[0]);
  const int64_t rank = static_cast<int64_t>(data.dims.size());
  int64_t axis = readNodeAttribute<int64_t>(node, "axis").value_or(0);
  if (axis < 0) axis += rank;

  NodeDesc::SplitAttrs attrs;
  attrs.ggml_axis = static_cast<int>(rank - 1 - axis);
  if (inputs.size() == 2 && inputs[1] != nullptr) {
    const auto s = readConstantInputArray<int64_t>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    GGONNX_ASSERT(s.has_value(), "Split compile: split input must be constant int64");
    attrs.lengths = *s;
  } else if (const auto a = readNodeAttribute<std::vector<int64_t>>(node, "split")) {
    attrs.lengths = *a;
  } else {
    const int64_t n = static_cast<int64_t>(outputs.size());
    attrs.lengths.assign(n, data.dims[axis] / n);
  }
  compiled_node->attrs = attrs;
}

EmitResult EmitSplitNode(ggml_context* ctx,
                         const NodeDesc& node,
                         const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::SplitAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled Split node missing attributes");
  ggml_tensor* data = values[node.inputs[0]];
  GGONNX_NOT_NULL(data, "compiled Split node missing GGML data input");

  const int axis = attrs->ggml_axis;
  EmitOutputs out;
  out.reserve(attrs->lengths.size());
  int64_t offset_elems = 0;
  int64_t ne[GGML_MAX_DIMS];
  for (int i = 0; i < GGML_MAX_DIMS; ++i) ne[i] = data->ne[i];
  for (int64_t length : attrs->lengths) {
    ne[axis] = length;
    const size_t offset_bytes = static_cast<size_t>(offset_elems) * data->nb[axis];
    ggml_tensor* view = ggml_view_4d(ctx, data,
                                     ne[0], ne[1], ne[2], ne[3],
                                     data->nb[1], data->nb[2], data->nb[3],
                                     offset_bytes);
    out.push_back(ggml_cont(ctx, view));
    offset_elems += length;
  }
  return out;
}

// ONNX ReduceMean (opset 11/13/18). We support the common case of reducing a
// contiguous suffix of the ONNX dims — e.g. axes=[2,3] on an [N,C,H,W] input —
// because that's what ggml_mean (which reduces only axis 0) can express after
// collapsing the trailing dims.
bool IsSupportedReduceMeanNode(Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.empty() || inputs.size() > 2 || outputs.size() != 1 ||
      node.GetImplicitInputs().size() != 0) {
    return false;
  }
  if (inputs[0] == nullptr || outputs[0] == nullptr) return false;
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const TensorMetadata y = getTensorMetadata(outputs[0]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      y.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (!rankSupportedByGGML(x) || !rankSupportedByGGML(y)) return false;
  if (!shapeIsFullyStatic(x)) return false;

  const int64_t rank = static_cast<int64_t>(x.dims.size());
  if (rank == 0) return false;

  // Collect axes. Prefer the `axes` input (opset 18+), then the attribute.
  std::vector<int64_t> axes;
  if (inputs.size() == 2 && inputs[1] != nullptr) {
    const auto v = readConstantInputArray<int64_t>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    if (!v) return false;
    axes = *v;
  } else if (const auto attr = readNodeAttribute<std::vector<int64_t>>(node, "axes")) {
    axes = *attr;
  } else {
    // Default: reduce all axes.
    axes.resize(rank);
    for (int64_t i = 0; i < rank; ++i) axes[i] = i;
  }
  if (axes.empty()) return false;

  std::vector<int64_t> normalized;
  normalized.reserve(axes.size());
  for (int64_t a : axes) {
    if (a < 0) a += rank;
    if (a < 0 || a >= rank) return false;
    normalized.push_back(a);
  }
  std::sort(normalized.begin(), normalized.end());
  for (size_t i = 1; i < normalized.size(); ++i) {
    if (normalized[i] == normalized[i - 1]) return false;
  }
  // Must be a contiguous suffix: [rank - k, rank - 1].
  const int64_t k = static_cast<int64_t>(normalized.size());
  for (int64_t i = 0; i < k; ++i) {
    if (normalized[i] != rank - k + i) return false;
  }
  return true;
}

void CompileReduceMeanAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  const auto inputs = node.GetInputs();
  const TensorMetadata x = getTensorMetadata(inputs[0]);
  const int64_t rank = static_cast<int64_t>(x.dims.size());

  std::vector<int64_t> axes;
  if (inputs.size() == 2 && inputs[1] != nullptr) {
    const auto v = readConstantInputArray<int64_t>(node, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    GGONNX_ASSERT(v.has_value(), "ReduceMean compile: axes input must be constant int64");
    axes = *v;
  } else if (const auto attr = readNodeAttribute<std::vector<int64_t>>(node, "axes")) {
    axes = *attr;
  } else {
    axes.resize(rank);
    for (int64_t i = 0; i < rank; ++i) axes[i] = i;
  }

  NodeDesc::ReduceAttrs attrs;
  attrs.trailing_count = static_cast<int>(axes.size());
  attrs.keepdims = readNodeAttribute<int64_t>(node, "keepdims").value_or(1) != 0;
  compiled_node->attrs = attrs;
}

EmitResult EmitReduceMeanNode(ggml_context* ctx,
                              const NodeDesc& node,
                              const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::ReduceAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled ReduceMean node missing attributes");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled ReduceMean node missing GGML input");

  // Collapse the trailing ONNX axes (= leading ggml axes) into ggml axis 0 so
  // ggml_mean reduces them in one shot.
  const int k = attrs->trailing_count;
  int64_t collapsed = 1;
  for (int i = 0; i < k; ++i) collapsed *= x->ne[i];
  int64_t keep[GGML_MAX_DIMS] = {1, 1, 1, 1};
  keep[0] = collapsed;
  for (int i = k; i < GGML_MAX_DIMS; ++i) keep[i - k + 1] = x->ne[i];

  ggml_tensor* src = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
  ggml_tensor* flat = ggml_reshape_4d(ctx, src, keep[0], keep[1], keep[2], keep[3]);
  ggml_tensor* reduced = ggml_mean(ctx, flat);  // ne[0] becomes 1

  if (attrs->keepdims) {
    // ggml_mean left the leading ggml axis at size 1; that corresponds to the
    // collapsed ONNX trailing axes. Expand back to k separate size-1 axes,
    // which for k in {1,2,3} fits in 4D.
    int64_t out_ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
    for (int i = 0; i < k; ++i) out_ne[i] = 1;
    for (int i = k; i < GGML_MAX_DIMS; ++i) out_ne[i] = reduced->ne[i - k + 1];
    return EmitOutputs{
        ggml_reshape_4d(ctx, reduced, out_ne[0], out_ne[1], out_ne[2], out_ne[3])};
  }
  // keepdims == 0: drop the leading size-1 axis by re-packing kept axes into
  // ggml positions [0..rank-k-1].
  int64_t out_ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
  for (int i = 0; i < GGML_MAX_DIMS - 1; ++i) out_ne[i] = reduced->ne[i + 1];
  return EmitOutputs{
      ggml_reshape_4d(ctx, reduced, out_ne[0], out_ne[1], out_ne[2], out_ne[3])};
}

// ONNX DepthToSpace: [N, C, H, W] -> [N, C/b^2, H*b, W*b]. Two modes:
//   DCR (default) reshapes C as [b, b, C/b^2] (row-block outer, col-block middle).
//   CRD            reshapes C as [C/b^2, b, b] (channel outer, then row, col).
// A faithful lowering needs a 5D intermediate (ggml max is 4D), so we split it
// into two 4D permute+reshape passes — one expands H, one expands W — by
// factoring the C axis into two at a time. With N == 1 the batch axis is free
// and both passes fit in 4D directly. For N > 1 we first pay an NCHW -> CNHW
// permute to fold N into the height axis, run the N == 1 shuffle on the fat
// height, then unfold and swap N/Cout back. The N == 1 fast path is kept
// separate — waifu2x / image SR hit it almost exclusively and we don't want
// to pay the extra conts there.
//
// KNOWN INEFFICIENCY: the N == 1 path emits two ggml_cont copies (one per
// pass, each materializing a permuted view before the following reshape can
// collapse two axes). Within 4D ggml ops this is the floor — the output index
// 'wb*b + cb' can't be encoded as a constant-stride axis over the input, so
// no single strided view can feed a lone cont. The N > 1 path stacks another
// two conts on top of that (input N<->C and output N<->Cout). A custom kernel
// (ggml_custom_4d with hand-rolled stride math) could collapse everything to
// a single pass but we're staying pure-ggml here; revisit if a profile pins a
// hot model on this op.

// Emit the 2-pass shuffle for a tensor with ne=[W, H, C, 1]; returns
// ne=[W*b, H*b, C/(b*b), 1]. Requires `src` to be contiguous.
static ggml_tensor* EmitDepthToSpaceShuffleBatch1(ggml_context* ctx,
                                                  ggml_tensor* src,
                                                  int b,
                                                  bool crd) {
  const int64_t W = src->ne[0];
  const int64_t H = src->ne[1];
  const int64_t C = src->ne[2];
  const int64_t Cout = C / (b * b);

  if (!crd) {
    // DCR: channel c = r*b*Cout + c_blk*Cout + cc.
    // Pass 1 — expand H. Factor C as (b_row outer, b_col*Cout inner).
    // ggml ne=[W, H, b_col*Cout, b_row] -> permute (b_row between H and W-inner) ->
    // ne=[W, b_row, H, b_col*Cout], then collapse axes 1,2 into H*b.
    ggml_tensor* v1 = ggml_reshape_4d(ctx, src, W, H, b * Cout, b);
    ggml_tensor* p1 = ggml_cont(ctx, ggml_permute(ctx, v1, 0, 2, 3, 1));
    ggml_tensor* r1 = ggml_reshape_4d(ctx, p1, W, H * b, b * Cout, 1);

    // Pass 2 — expand W. Factor b_col*Cout as (b_col outer, Cout inner).
    // ggml ne=[W, H*b, Cout, b_col] -> permute (b_col innermost) ->
    // ne=[b_col, W, H*b, Cout], then collapse axes 0,1 into W*b.
    ggml_tensor* v2 = ggml_reshape_4d(ctx, r1, W, H * b, Cout, b);
    ggml_tensor* p2 = ggml_cont(ctx, ggml_permute(ctx, v2, 1, 2, 3, 0));
    return ggml_reshape_4d(ctx, p2, W * b, H * b, Cout, 1);
  }

  // CRD: channel c = cc*b*b + r*b + c_blk.
  // Pass 1 — expand W. Factor C as (Cout*b_row outer, b_col inner).
  // ggml ne=[W, H, b_col, Cout*b_row] -> permute (b_col innermost) ->
  // ne=[b_col, W, H, Cout*b_row], then collapse axes 0,1 into W*b.
  ggml_tensor* v1 = ggml_reshape_4d(ctx, src, W, H, b, Cout * b);
  ggml_tensor* p1 = ggml_cont(ctx, ggml_permute(ctx, v1, 1, 2, 0, 3));
  ggml_tensor* r1 = ggml_reshape_4d(ctx, p1, W * b, H, Cout * b, 1);

  // Pass 2 — expand H. Factor Cout*b_row as (Cout outer, b_row inner).
  // ggml ne=[W*b, H, b_row, Cout] -> permute (b_row between H and W) ->
  // ne=[W*b, b_row, H, Cout], then collapse axes 1,2 into H*b.
  ggml_tensor* v2 = ggml_reshape_4d(ctx, r1, W * b, H, b, Cout);
  ggml_tensor* p2 = ggml_cont(ctx, ggml_permute(ctx, v2, 0, 2, 1, 3));
  return ggml_reshape_4d(ctx, p2, W * b, H * b, Cout, 1);
}

bool IsSupportedDepthToSpaceNode(Ort::ConstNode node) {
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
  if (in.dims.size() != 4 || out.dims.size() != 4) return false;
  if (!shapeIsFullyStatic(in.dims) || !shapeIsFullyStatic(out.dims)) return false;

  const auto blocksize = readNodeAttribute<int64_t>(node, "blocksize");
  if (!blocksize.has_value() || *blocksize < 1) return false;
  const int64_t b = *blocksize;
  if (in.dims[1] % (b * b) != 0) return false;

  const std::string mode = readNodeAttribute<std::string>(node, "mode").value_or("DCR");
  if (mode != "DCR" && mode != "CRD") return false;

  return true;
}

void CompileDepthToSpaceAttributes(Ort::ConstNode node, NodeDesc* compiled_node) {
  GGONNX_NOT_NULL(compiled_node, "compiled node output must not be null");
  NodeDesc::DepthToSpaceAttrs attrs;
  attrs.blocksize = static_cast<int>(*readNodeAttribute<int64_t>(node, "blocksize"));
  const std::string mode = readNodeAttribute<std::string>(node, "mode").value_or("DCR");
  attrs.crd = (mode == "CRD");
  compiled_node->attrs = attrs;
}

EmitResult EmitDepthToSpaceNode(ggml_context* ctx,
                                const NodeDesc& node,
                                const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::DepthToSpaceAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled DepthToSpace node missing attributes");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled DepthToSpace node missing GGML input");

  const int b = attrs->blocksize;
  const int64_t W = x->ne[0];
  const int64_t H = x->ne[1];
  const int64_t C = x->ne[2];
  const int64_t N = x->ne[3];
  GGONNX_ASSERT(C % (b * b) == 0, "DepthToSpace: C not divisible by blocksize^2");
  const int64_t Cout = C / (b * b);

  ggml_tensor* src = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);

  if (N == 1) {
    return EmitOutputs{EmitDepthToSpaceShuffleBatch1(ctx, src, b, attrs->crd)};
  }

  // N > 1: permute ONNX NCHW -> CNHW so N sits next to H in memory, then collapse
  // (N, H) into one axis and run the N==1 shuffle on the fat height. In ggml C is
  // at ne[2] and N at ne[3], so the permute swaps axes 2 and 3.
  ggml_tensor* cnhw = ggml_cont(ctx, ggml_permute(ctx, src, 0, 1, 3, 2));
  // ne=[W, H, N, C] contiguous -> ne=[W, N*H, C, 1]. N and H are adjacent with H
  // inner (stride W) and N outer (stride W*H), so the collapsed axis indexes as
  // n*H + h, which is exactly what the shuffle will later invert.
  ggml_tensor* folded = ggml_reshape_4d(ctx, cnhw, W, N * H, C, 1);

  ggml_tensor* shuffled = EmitDepthToSpaceShuffleBatch1(ctx, folded, b, attrs->crd);
  // shuffled ne=[W*b, N*H*b, Cout, 1]. The expanded height encodes n*H*b + h*b + r
  // = n*(H*b) + (h*b + r), so splitting it as (H*b inner, N outer) is a pure view.
  // The resulting ne=[W*b, H*b, N, Cout] means ONNX [Cout, N, H*b, W*b] — one more
  // permute (swap N <-> Cout) lands us at the expected ONNX [N, Cout, H*b, W*b].
  ggml_tensor* split = ggml_reshape_4d(ctx, shuffled, W * b, H * b, N, Cout);
  return EmitOutputs{ggml_cont(ctx, ggml_permute(ctx, split, 0, 1, 3, 2))};
}

// Synthetic op: fused Reshape->Transpose(perm=[0,1,3,2,4,5])->Reshape triple.
// Lifts the rank-6 intermediate outside of GGML's rank-4 cap by viewing the
// input as 4D ne=[d4*d5, d3, d2, d0*d1] (a pure reshape since memory is
// contiguous), swapping ggml axes 2 and 3 (= ONNX axes 2 and 3 in reversed
// layout), and reshaping to the final ONNX dims. One cont is required —
// unavoidable, since the middle two axes of the 6D view must be transposed.
EmitResult EmitWindowShuffleNode(ggml_context* ctx,
                                 const NodeDesc& node,
                                 const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::WindowShuffleAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled __WindowShuffle node missing attributes");
  GGONNX_ASSERT(node.inputs.size() == 1 && node.outputs.size() == 1,
                "compiled __WindowShuffle node has invalid arity");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled __WindowShuffle node missing GGML input");

  const int64_t d0 = attrs->onnx_rank6_dims[0];
  const int64_t d1 = attrs->onnx_rank6_dims[1];
  const int64_t d2 = attrs->onnx_rank6_dims[2];
  const int64_t d3 = attrs->onnx_rank6_dims[3];
  const int64_t d4 = attrs->onnx_rank6_dims[4];
  const int64_t d5 = attrs->onnx_rank6_dims[5];
  GGONNX_ASSERT(d0 > 0 && d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0 && d5 > 0,
                "WindowShuffle dims must be positive");

  ggml_tensor* src = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
  ggml_tensor* view = ggml_reshape_4d(ctx, src, d4 * d5, d3, d2, d0 * d1);
  ggml_tensor* permuted = ggml_permute(ctx, view, 0, 2, 1, 3);
  ggml_tensor* cont = ggml_cont(ctx, permuted);

  const std::array<int64_t, GGML_MAX_DIMS> target =
      ToPaddedGGMLDims(attrs->output_onnx_dims);
  ggml_tensor* out =
      ggml_reshape_4d(ctx, cont, target[0], target[1], target[2], target[3]);
  return EmitOutputs{out};
}

// Synthetic op: fused Reshape([B, M*M, 3C]->[B, M*M, 3, H, D]) ->
// Transpose(perm=[2,0,3,1,4]) -> Split(axis=0) -> 3x Squeeze(axes=[0]).
// The rank-5 intermediates never materialize: we reinterpret the packed input
// as rank-4 ne=[D, 3H, M*M, B], take Q/K/V as strided views at offsets
// {0, H*D, 2*H*D} scalars into that buffer, and swap the heads/tokens axes
// to land at the canonical attention layout ne=[D, M*M, H, B]. The final cont
// matches what a downstream ggml_mul_mat would already pay for.
EmitResult EmitQKVSplitNode(ggml_context* ctx,
                            const NodeDesc& node,
                            const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const auto* attrs = std::get_if<NodeDesc::QKVSplitAttrs>(&node.attrs);
  GGONNX_ASSERT(attrs != nullptr, "compiled __QKVSplit node missing attributes");
  GGONNX_ASSERT(node.inputs.size() == 1 && node.outputs.size() == 3,
                "compiled __QKVSplit node has invalid arity");
  ggml_tensor* x = values[node.inputs[0]];
  GGONNX_NOT_NULL(x, "compiled __QKVSplit node missing GGML input");

  const int64_t H = attrs->num_heads;
  const int64_t D = attrs->num_heads > 0 ? attrs->head_dim : 0;
  const int64_t T = attrs->num_tokens;
  const int64_t B = attrs->num_batch;
  GGONNX_ASSERT(H > 0 && D > 0 && T > 0 && B > 0,
                "QKVSplit dims must be positive");

  // Collapse the input to a contiguous rank-4 view [D, 3H, T, B]. The input is
  // produced by a Gemm/MatMul so it's normally contiguous; the cont guards
  // against a non-standard producer.
  ggml_tensor* src = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
  ggml_tensor* packed = ggml_reshape_4d(ctx, src, D, 3 * H, T, B);

  const size_t type_size = ggml_type_size(packed->type);
  const size_t nb1 = static_cast<size_t>(packed->nb[1]);
  const size_t nb2 = static_cast<size_t>(packed->nb[2]);
  const size_t nb3 = static_cast<size_t>(packed->nb[3]);
  // Q/K/V slice offsets along the packed `3H` axis.
  const std::array<size_t, 3> offsets{
      0,
      static_cast<size_t>(H) * D * type_size,
      2 * static_cast<size_t>(H) * D * type_size,
  };

  EmitOutputs out;
  out.reserve(3);
  for (size_t i = 0; i < 3; ++i) {
    // View of shape [D, H, T, B] starting at the i-th QKV chunk.
    ggml_tensor* view = ggml_view_4d(ctx, packed, D, H, T, B, nb1, nb2, nb3, offsets[i]);
    // Swap heads <-> tokens so the output is [D, T, H, B].
    ggml_tensor* permuted = ggml_permute(ctx, view, 0, 2, 1, 3);
    out.push_back(ggml_cont(ctx, permuted));
  }
  return out;
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
      {{"", "Max"}, {IsSupportedElementwiseBinaryOpNode, nullptr, EmitElementwiseBinaryNode}},
      {{"", "Min"}, {IsSupportedElementwiseBinaryOpNode, nullptr, EmitElementwiseBinaryNode}},
      {{"", "GRU"}, {IsSupportedGRUNode, CompileGRUAttributes, EmitGRUNode}},
      {{"", "LSTM"}, {IsSupportedLSTMNode, CompileLSTMAttributes, EmitLSTMNode}},
      {{"", "Relu"},     {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Sigmoid"},  {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Tanh"},     {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Neg"},      {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Abs"},      {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Sqrt"},     {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Exp"},      {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Log"},      {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Erf"},      {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Softplus"}, {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "Elu"},      {IsSupportedUnaryFloatNode, nullptr, EmitUnaryFloatNode}},
      {{"", "LeakyRelu"}, {IsSupportedLeakyReluNode, CompileLeakyReluAttributes, EmitLeakyReluNode}},
      {{"", "PRelu"},     {IsSupportedPReluNode, nullptr, EmitPReluNode}},
      {{"", "Clip"},      {IsSupportedClipNode, CompileClipAttributes, EmitClipNode}},
      {{"", "Softmax"},  {IsSupportedSoftmaxNode, CompileSoftmaxAttributes, EmitSoftmaxNode}},
      {{"", "MatMul"},   {IsSupportedMatMulNode, nullptr, EmitMatMulNode, MatMulConstantLayout}},
      {{"", "Conv"},     {IsSupportedConvNode, CompileConvAttributes, EmitConvNode, nullptr}},
      {{"", "ConvTranspose"}, {IsSupportedConvTransposeNode, CompileConvTransposeAttributes, EmitConvTransposeNode}},
      {{"", "Expand"},   {IsSupportedExpandNode, CompileExpandAttributes, EmitExpandNode}},
      {{"", "Gemm"},     {IsSupportedGemmNode, CompileGemmAttributes, EmitGemmNode, GemmConstantLayout}},
      {{"", "Reshape"},  {IsSupportedReshapeNode, CompileReshapeAttributes, EmitReshapeNode}},
      {{"", "Flatten"},  {IsSupportedFlattenNode, CompileFlattenAttributes, EmitReshapeNode}},
      {{"", "Squeeze"},  {IsSupportedSqueezeNode, CompileSqueezeAttributes, EmitReshapeNode}},
      {{"", "MaxPool"},       {IsSupportedPool2DNode, CompilePool2DAttributes, EmitPool2DNode}},
      {{"", "AveragePool"},   {IsSupportedPool2DNode, CompilePool2DAttributes, EmitPool2DNode}},
      {{"", "GlobalMaxPool"}, {IsSupportedGlobalPoolNode, CompileGlobalPoolAttributes, EmitPool2DNode}},
      {{"", "GlobalAveragePool"}, {IsSupportedGlobalPoolNode, CompileGlobalPoolAttributes, EmitPool2DNode}},
      {{"", "Pad"}, {IsSupportedPadNode, CompilePadAttributes, EmitPadNode}},
      {{"", "InstanceNormalization"},
       {IsSupportedInstanceNormalizationNode, CompileInstanceNormalizationAttributes, EmitInstanceNormalizationNode}},
      {{"", "Upsample"}, {IsSupportedUpsampleNode, CompileUpsampleAttributes, EmitUpsampleNode}},
      {{"", "Resize"}, {IsSupportedUpsampleNode, CompileUpsampleAttributes, EmitUpsampleNode}},
      {{"", "Identity"}, {IsSupportedIdentityNode, nullptr, EmitIdentityNode}},
      {{"", "Transpose"}, {IsSupportedTransposeNode, CompileTransposeAttributes, EmitTransposeNode}},
      {{"", "Concat"}, {IsSupportedConcatNode, CompileConcatAttributes, EmitConcatNode}},
      {{"", "Slice"}, {IsSupportedSliceNode, CompileSliceAttributes, EmitSliceNode}},
      {{"", "Split"}, {IsSupportedSplitNode, CompileSplitAttributes, EmitSplitNode}},
      {{"", "ReduceMean"},
       {IsSupportedReduceMeanNode, CompileReduceMeanAttributes, EmitReduceMeanNode}},
      {{"", "BatchNormalization"},
       {IsSupportedBatchNormNode, CompileBatchNormAttributes, EmitBatchNormNode}},
      {{"", "DepthToSpace"},
       {IsSupportedDepthToSpaceNode, CompileDepthToSpaceAttributes, EmitDepthToSpaceNode}},
      // Synthetic op injected by FusionPlan. `support` and `compile_attrs` are
      // null because the fusion machinery builds the NodeDesc directly and the
      // per-node support check is never called on __WindowShuffle.
      {{"", "__WindowShuffle"}, {nullptr, nullptr, EmitWindowShuffleNode}},
      {{"", "__QKVSplit"}, {nullptr, nullptr, EmitQKVSplitNode}},
  };

  auto it = ops_table.find({domain, op_type});
  if (it != ops_table.end()) {
    return &it->second;
  }
  return nullptr;
}
