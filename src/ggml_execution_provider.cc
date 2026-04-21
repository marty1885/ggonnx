#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include "inner/helpers.hpp"
#include "inner/ort_api_helpers.hpp"

#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <onnxruntime/onnxruntime_c_api.h>
#include <onnxruntime/onnxruntime_cxx_api.h>

namespace {

constexpr const char* kRegistrationName = "GGONNX";
constexpr const char* kEpName = "GGMLExecutionProvider";
constexpr const char* kVendorName = "nekko";
constexpr const char* kVersion = "0.0.1";

std::atomic<uint64_t> g_debug_graph_build_count{0};

using ggonnx::ort_internal::GetOrtApi;
using ggonnx::ort_internal::GetOrtEpApi;
using ggonnx::ort_internal::InitializeOrtApi;
using ggonnx::ort_internal::IsOrtApiInitialized;
using ggonnx::ort_internal::MakeStatus;
using ggonnx::ort_internal::THROW_ON_ERROR;
using ggonnx::ort_internal::WrapStatus;

struct GGMLFactory {
  OrtEpFactory iface{};
  std::string registered_name{kRegistrationName};
};

struct GGMLEp {
  OrtEp iface{};
  const OrtLogger* logger{};
  std::vector<std::string> selected_devices;
};

struct ValueDesc {
  std::string name;
  std::vector<int64_t> dims;
  bool is_graph_input{};
  bool is_graph_output{};
};

struct NodeDesc {
  std::string op_type;
  std::string domain;
  std::string name;
  std::vector<size_t> inputs;
  std::vector<size_t> outputs;
  std::string direction;
  int64_t hidden_size{};
  int64_t layout{};
  int64_t linear_before_reset{};
};

struct CompiledPartition {
  std::vector<ValueDesc> values;
  std::vector<size_t> graph_inputs;
  std::vector<size_t> graph_outputs;
  std::vector<NodeDesc> nodes;
};

struct ShapeKey {
  std::vector<std::vector<int64_t>> input_dims;

  bool operator==(const ShapeKey& other) const {
    return input_dims == other.input_dims;
  }

  bool operator!=(const ShapeKey& other) const {
    return !(*this == other);
  }
};

struct MaterializedGraph {
  ShapeKey key;
  ggml_context* ctx{};
  ggml_cgraph* graph{};
  std::vector<ggml_tensor*> values;
  std::vector<std::vector<int64_t>> value_dims;
  std::vector<ggml_tensor*> input_tensors;
  std::vector<ggml_tensor*> output_tensors;
};

struct GGMLComputeState {
  CompiledPartition partition;
  std::unique_ptr<MaterializedGraph> active_graph;
};

struct GGMLNodeComputeInfo {
  OrtNodeComputeInfo iface{};
  CompiledPartition partition;
};

using EmitOutputs = std::vector<ggml_tensor*>;
using EmitResult = std::optional<EmitOutputs>;
using EmitNodeFn = EmitResult (*)(ggml_context* ctx,
                                  const NodeDesc& node,
                                  const std::vector<ggml_tensor*>& values);
using CompileAttrsFn = void (*)(Ort::ConstNode node, NodeDesc* compiled_node);

struct OpDefinition {
  std::string_view domain;
  std::string_view op_type;
  bool (*support)(Ort::ConstNode node);
  CompileAttrsFn compile_attrs;
  EmitNodeFn emit;
};

GGMLFactory* AsFactory(OrtEpFactory* factory) {
  return reinterpret_cast<GGMLFactory*>(factory);
}

const GGMLFactory* AsFactory(const OrtEpFactory* factory) {
  return reinterpret_cast<const GGMLFactory*>(factory);
}

GGMLEp* AsEp(OrtEp* ep) {
  return reinterpret_cast<GGMLEp*>(ep);
}

const GGMLEp* AsEp(const OrtEp* ep) {
  return reinterpret_cast<const GGMLEp*>(ep);
}

GGMLNodeComputeInfo* AsNodeComputeInfo(OrtNodeComputeInfo* info) {
  return reinterpret_cast<GGMLNodeComputeInfo*>(info);
}

GGMLComputeState* AsComputeState(void* state) {
  return reinterpret_cast<GGMLComputeState*>(state);
}

constexpr size_t kOptionalValueAbsent = std::numeric_limits<size_t>::max();

std::string DescribeHardwareDevice(const OrtHardwareDevice* device) {
  GGONNX_NOT_NULL(device, "hardware device pointer must not be null");
  const char* vendor_ptr = GGONNX_NOT_NULL(GetOrtApi().HardwareDevice_Vendor(device),
                                          "hardware device vendor must not be null");
  std::string vendor = vendor_ptr;
  const auto device_type = GetOrtApi().HardwareDevice_Type(device);
  const auto vendor_id = GetOrtApi().HardwareDevice_VendorId(device);
  const auto device_id = GetOrtApi().HardwareDevice_DeviceId(device);

  return vendor + ":" + std::to_string(static_cast<int>(device_type)) + ":" +
         std::to_string(vendor_id) + ":" + std::to_string(device_id);
}

bool IsSupportedHardwareDevice(const OrtHardwareDevice* device) {
  GGONNX_NOT_NULL(device, "hardware device pointer must not be null");
  // Start with host execution only. This keeps the first milestone aligned with
  // differential debugging against ORT CPU before device-specific GGML backends.
  return GetOrtApi().HardwareDevice_Type(device) == OrtHardwareDeviceType_CPU;
}

struct TensorMetadata {
  ONNXTensorElementDataType element_type{};
  std::vector<int64_t> dims;
};

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

bool shapeIsFullyStatic(const TensorMetadata& tensor) {
  return std::all_of(tensor.dims.begin(), tensor.dims.end(), [](int64_t dim) { return dim >= 0; });
}

inline bool rankSupportedByGGML(const TensorMetadata& tensor) {
  return tensor.dims.size() <= GGML_MAX_DIMS;
}

bool shapeIsFullyStatic(const std::vector<int64_t>& dims) {
  return std::all_of(dims.begin(), dims.end(), [](int64_t dim) { return dim >= 0; });
}

void AssertShapeMatchesGGML(const std::vector<int64_t>& expected_onnx_dims,
                            const ggml_tensor* tensor,
                            const std::string& tensor_name) {
  GGONNX_NOT_NULL(tensor, "ggml tensor must not be null for shape check");
  const std::vector<int64_t> actual_onnx_dims = ToOnnxDims(tensor);
  if (actual_onnx_dims != expected_onnx_dims) {
    throw std::runtime_error("shape mismatch for tensor '" + tensor_name + "': ONNX " +
                             FormatDims(expected_onnx_dims) + " vs GGML " +
                             FormatDims(actual_onnx_dims));
  }
}

// broadcastSupportedByGGML
// @brief Returns true the requested broadcast in ONNX format is supported by GGML's
//   tensor broadcasting rules.
// @param lhs_dims The dimensions of the left-hand side tensor.
// @param rhs_dims The dimensions of the right-hand side tensor.
// @return True if the broadcast is supported, false otherwise.
bool broadcastSupportedByGGML(const std::vector<int64_t>& lhs_dims,
                              const std::vector<int64_t>& rhs_dims) {
  const std::array<int64_t, GGML_MAX_DIMS> lhs_ggml_dims = ToPaddedGGMLDims(lhs_dims);
  const std::array<int64_t, GGML_MAX_DIMS> rhs_ggml_dims = ToPaddedGGMLDims(rhs_dims);

  auto can_repeat = [](const std::array<int64_t, GGML_MAX_DIMS>& src,
                       const std::array<int64_t, GGML_MAX_DIMS>& dst) {
    for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
      // special path to support ONNX non-specified dimensions
      if (src[i] <= 0 || dst[i] <= 0) {
        continue;
      }
      if (dst[i] % src[i] != 0) {
        return false;
      }
    }
    return true;
  };

  return can_repeat(lhs_ggml_dims, rhs_ggml_dims);
}

std::vector<int64_t> inferBroadcastOutputDims(const std::vector<int64_t>& lhs_dims,
                                              const std::vector<int64_t>& rhs_dims) {
  const size_t output_rank = std::max(lhs_dims.size(), rhs_dims.size());
  std::vector<int64_t> output_dims(output_rank, 1);

  for (size_t i = 0; i < output_rank; ++i) {
    const size_t lhs_index = output_rank - i;
    const int64_t lhs_dim = i < lhs_dims.size() ? lhs_dims[lhs_dims.size() - 1 - i] : 1;
    const int64_t rhs_dim = i < rhs_dims.size() ? rhs_dims[rhs_dims.size() - 1 - i] : 1;
    if (lhs_dim == rhs_dim || lhs_dim == 1) {
      output_dims[lhs_index - 1] = rhs_dim;
      continue;
    }
    if (rhs_dim == 1) {
      output_dims[lhs_index - 1] = lhs_dim;
      continue;
    }
    throw std::runtime_error("runtime binary op shapes are not ONNX-broadcastable: " +
                             FormatDims(lhs_dims) + " and " + FormatDims(rhs_dims));
  }

  return output_dims;
}

bool IsSupportedElementwiseBinaryOpType(std::string_view op_type) {
  return op_type == "Add" || op_type == "Sub" || op_type == "Mul" || op_type == "Div";
}

template <typename T>
std::optional<T> readNodeAttribute(Ort::ConstNode node, const char* attribute_name) {
  Ort::ConstOpAttr attr{nullptr};
  auto status = node.GetAttributeByName(attribute_name, attr);
  if (status.IsOK()) {
    T value{};
    if constexpr (std::is_same_v<T, std::vector<std::string>>) {
      std::vector<std::string> values;
      Ort::ThrowOnError(attr.GetValueArray(values));
      return values;
    } else {
      Ort::ThrowOnError(attr.GetValue(value));
      return value;
    }
  }

  if (status.GetErrorCode() == ORT_INVALID_ARGUMENT) {
    return std::nullopt;
  }

  Ort::ThrowOnError(status);
  return std::nullopt;
}

bool IsSupportedElementwiseBinaryNode(Ort::ConstNode node, std::string_view op_type) {
  if (!IsSupportedElementwiseBinaryOpType(op_type)) {
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
  compiled_node->direction = readNodeAttribute<std::string>(node, "direction").value_or("forward");
  GGONNX_ASSERT(compiled_node->direction == "forward", "only forward GRU direction is supported");
  const auto hidden_size = readNodeAttribute<int64_t>(node, "hidden_size");
  GGONNX_ASSERT(hidden_size.has_value() && *hidden_size > 0,
                "GRU hidden_size attribute must be present and positive");
  compiled_node->hidden_size = *hidden_size;
  compiled_node->layout = readNodeAttribute<int64_t>(node, "layout").value_or(0);
  compiled_node->linear_before_reset =
      readNodeAttribute<int64_t>(node, "linear_before_reset").value_or(0);
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
  GGONNX_ASSERT(node.hidden_size > 0, "compiled GRU node must have positive hidden_size");

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
  const int64_t hidden_size = node.hidden_size;
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
    outputs.push_back(ggml_cont_3d(ctx, h_t, hidden_size, batch_size, 1));
  }
  return outputs;
}

const OpDefinition* FindOpDefinition(std::string_view domain, std::string_view op_type) {
  static const std::array<OpDefinition, 5> kOps = {{
      {std::string_view(), "Add", IsSupportedElementwiseBinaryOpNode, nullptr, EmitElementwiseBinaryNode},
      {std::string_view(), "Sub", IsSupportedElementwiseBinaryOpNode, nullptr, EmitElementwiseBinaryNode},
      {std::string_view(), "Mul", IsSupportedElementwiseBinaryOpNode, nullptr, EmitElementwiseBinaryNode},
      {std::string_view(), "Div", IsSupportedElementwiseBinaryOpNode, nullptr, EmitElementwiseBinaryNode},
      {std::string_view(), "GRU", IsSupportedGRUNode, CompileGRUAttributes, EmitGRUNode},
  }};

  for (const OpDefinition& op : kOps) {
    if (op.domain == domain && op.op_type == op_type) {
      return &op;
    }
  }

  return nullptr;
}

bool IsNodeSupported(Ort::ConstNode node) {
  const std::string op_type = node.GetOperatorType();
  const std::string domain = node.GetDomain();
  const OpDefinition* op = FindOpDefinition(domain, op_type);
  return op != nullptr && op->support(node);
}

size_t ElementCount(const std::vector<int64_t>& dims) {
  size_t count = 1;
  for (const int64_t dim : dims) {
    GGONNX_ASSERT(dim >= 0, "tensor element count requested for dynamic or invalid shape");
    count *= static_cast<size_t>(dim);
  }
  return count;
}

size_t EstimatePartitionTensorCount(const CompiledPartition& partition) {
  size_t count = partition.values.size() + partition.nodes.size();
  return std::max<size_t>(count, 8);
}

size_t EstimateInitialGraphDataBytes(const CompiledPartition& partition,
                                     const std::vector<TensorMetadata>& input_metadata) {
  GGONNX_ASSERT(input_metadata.size() == partition.graph_inputs.size(),
                "runtime input metadata does not match compiled partition");
  size_t bytes = 0;

  for (const TensorMetadata& meta : input_metadata) {
    GGONNX_ASSERT(meta.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                  "GGONNX runtime input must be float32");
    GGONNX_ASSERT(shapeIsFullyStatic(meta.dims), "runtime input shapes must be concrete");
    bytes += ElementCount(meta.dims) * sizeof(float);
  }

  // Leave plenty of headroom for intermediates and outputs. GGML will do the final
  // graph-level planning; this context size only needs to be safely sufficient.
  bytes *= std::max<size_t>(partition.nodes.size() + partition.graph_outputs.size(), 4);
  return std::max<size_t>(bytes, 256 * 1024);
}

CompiledPartition CompilePartition(const OrtGraph* graph) {
  GGONNX_NOT_NULL(graph, "graph must not be null");
  const Ort::ConstGraph ort_graph{graph};
  CompiledPartition partition;
  std::unordered_map<std::string, size_t> value_ids;

  auto ensure_value = [&](Ort::ConstValueInfo value_info) -> size_t {
    const std::string name = value_info.GetName();
    const TensorMetadata metadata = getTensorMetadata(value_info);
    GGONNX_ASSERT(metadata.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            "compiled partition currently supports only float32 tensors");
    GGONNX_ASSERT(rankSupportedByGGML(metadata),
                  "compiled partition requires tensor rank <= " + std::to_string(GGML_MAX_DIMS) +
                      ", got " + std::to_string(metadata.dims.size()) + " for value '" + name + "'");
    const auto it = value_ids.find(name);
    if (it != value_ids.end()) {
      partition.values[it->second].dims = metadata.dims;
      return it->second;
    }

    const size_t id = partition.values.size();
    value_ids.emplace(name, id);
    ValueDesc value;
    value.name = name;
    value.dims = metadata.dims;
    partition.values.push_back(std::move(value));
    return id;
  };

  const auto graph_inputs = ort_graph.GetInputs();
  for (Ort::ConstValueInfo input : graph_inputs) {
    GGONNX_ASSERT(input != nullptr, "graph input metadata must not be null");
    const size_t id = ensure_value(input);
    partition.values[id].is_graph_input = true;
    partition.graph_inputs.push_back(id);
  }

  const auto graph_outputs = ort_graph.GetOutputs();
  for (Ort::ConstValueInfo output : graph_outputs) {
    GGONNX_ASSERT(output != nullptr, "graph output metadata must not be null");
    const size_t id = ensure_value(output);
    partition.values[id].is_graph_output = true;
    partition.graph_outputs.push_back(id);
  }

  for (Ort::ConstNode node : ort_graph.GetNodes()) {
    GGONNX_ASSERT(node != nullptr, "graph node must not be null");
    if (!IsNodeSupported(node)) {
      throw std::runtime_error("GGONNX Compile received an unsupported partition");
    }

    const auto node_inputs = node.GetInputs();
    const auto node_outputs = node.GetOutputs();

    NodeDesc compiled_node;
    compiled_node.op_type = node.GetOperatorType();
    compiled_node.domain = node.GetDomain();
    compiled_node.name = node.GetName();
    const OpDefinition* op = FindOpDefinition(compiled_node.domain, compiled_node.op_type);
    GGONNX_ASSERT(op != nullptr, "compiled partition could not locate supported op definition");
    if (op->compile_attrs != nullptr) {
      op->compile_attrs(node, &compiled_node);
    }

    for (Ort::ConstValueInfo input : node_inputs) {
      if (input == nullptr) {
        compiled_node.inputs.push_back(kOptionalValueAbsent);
        continue;
      }
      compiled_node.inputs.push_back(ensure_value(input));
    }
    for (Ort::ConstValueInfo output : node_outputs) {
      if (output == nullptr) {
        compiled_node.outputs.push_back(kOptionalValueAbsent);
        continue;
      }
      compiled_node.outputs.push_back(ensure_value(output));
    }

    partition.nodes.push_back(std::move(compiled_node));
  }

  GGONNX_ASSERT(!partition.nodes.empty(), "compiled partition must contain at least one node");
  GGONNX_ASSERT(!partition.graph_outputs.empty(), "compiled partition must contain at least one graph output");
  return partition;
}

ShapeKey MakeShapeKey(const std::vector<TensorMetadata>& input_metadata) {
  ShapeKey key;
  key.input_dims.reserve(input_metadata.size());
  for (const TensorMetadata& meta : input_metadata) {
    GGONNX_ASSERT(shapeIsFullyStatic(meta.dims),
                  "runtime input shapes must be concrete before GGML execution");
    key.input_dims.push_back(meta.dims);
  }
  return key;
}

ggml_tensor* CreateTensorForOnnxShape(ggml_context* ctx, const std::vector<int64_t>& dims) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  if (dims.empty()) {
    return ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  }

  const std::array<int64_t, GGML_MAX_DIMS> ggml_dims = ToGGMLDims(dims);
  return ggml_new_tensor(ctx, GGML_TYPE_F32, static_cast<int>(dims.size()), ggml_dims.data());
}

void CopyInputDataToTensor(const OrtValue* input_value,
                           const std::vector<int64_t>& expected_dims,
                           const std::string& tensor_name,
                           ggml_tensor* tensor) {
  GGONNX_NOT_NULL(input_value, "graph input value must not be null");
  GGONNX_NOT_NULL(tensor, "cached GGML input tensor must not be null");
  const TensorMetadata meta = getTensorMetadata(Ort::ConstValue{input_value});
  if (meta.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("GGONNX graph input must be float32");
  }

  const void* input_data = nullptr;
  THROW_ON_ERROR(GetOrtApi().GetTensorData(input_value, &input_data));
  GGONNX_NOT_NULL(input_data, "ORT returned null tensor data for graph input");

  GGONNX_ASSERT(meta.dims == expected_dims,
                "runtime input shape mismatch for tensor '" + tensor_name + "': expected " +
                    FormatDims(expected_dims) + ", got " + FormatDims(meta.dims));
  const size_t element_count = ElementCount(meta.dims);
  const size_t bytes = element_count * sizeof(float);
  AssertShapeMatchesGGML(meta.dims, tensor, tensor_name);
  std::memcpy(tensor->data, input_data, bytes);
}

EmitResult EmitNode(ggml_context* ctx,
                    const NodeDesc& node,
                    const std::vector<ggml_tensor*>& values) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  const OpDefinition* op = FindOpDefinition(node.domain, node.op_type);
  GGONNX_ASSERT(op != nullptr && op->emit != nullptr,
                "compiled partition contains op without GGML emitter");
  return op->emit(ctx, node, values);
}

void DestroyMaterializedGraph(std::unique_ptr<MaterializedGraph>& graph) {
  if (graph != nullptr && graph->ctx != nullptr) {
    ggml_free(graph->ctx);
  }
  graph.reset();
}

std::unique_ptr<MaterializedGraph> BuildMaterializedGraph(const CompiledPartition& partition,
                                                          ShapeKey key,
                                                          const std::vector<TensorMetadata>& input_metadata) {
  const size_t mem_size =
      EstimatePartitionTensorCount(partition) * ggml_tensor_overhead() +
      ggml_graph_overhead() +
      EstimateInitialGraphDataBytes(partition, input_metadata);

  ggml_init_params params{};
  params.mem_size = mem_size;
  params.mem_buffer = nullptr;
  params.no_alloc = false;

  ggml_context* ctx = ggml_init(params);
  if (ctx == nullptr) {
    throw std::runtime_error("ggml_init failed");
  }

  try {
    auto graph_state = std::make_unique<MaterializedGraph>();
    graph_state->key = std::move(key);
    graph_state->ctx = ctx;
    graph_state->values.resize(partition.values.size(), nullptr);
    graph_state->value_dims.resize(partition.values.size());
    graph_state->input_tensors.resize(partition.graph_inputs.size(), nullptr);
    graph_state->output_tensors.resize(partition.graph_outputs.size(), nullptr);

    for (size_t i = 0; i < partition.graph_inputs.size(); ++i) {
      const size_t value_id = partition.graph_inputs[i];
      const TensorMetadata& meta = input_metadata[i];
      const ValueDesc& value = partition.values[value_id];
      GGONNX_ASSERT(meta.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                    "GGONNX runtime input must be float32");
      GGONNX_ASSERT(broadcastSupportedByGGML(value.dims, meta.dims),
                    "runtime input shape mismatch for tensor '" + value.name + "': declared " +
                        FormatDims(value.dims) + ", got " + FormatDims(meta.dims));
      GGONNX_ASSERT(shapeIsFullyStatic(meta.dims),
                    "runtime input shapes must be concrete for tensor '" + value.name + "'");

      ggml_tensor* input_tensor = CreateTensorForOnnxShape(ctx, meta.dims);
      GGONNX_NOT_NULL(input_tensor, "failed to allocate cached GGML input tensor");
      AssertShapeMatchesGGML(meta.dims, input_tensor, partition.values[value_id].name);
      graph_state->values[value_id] = input_tensor;
      graph_state->value_dims[value_id] = meta.dims;
      graph_state->input_tensors[i] = input_tensor;
    }

    for (const NodeDesc& node : partition.nodes) {
      EmitResult emitted_outputs = EmitNode(ctx, node, graph_state->values);
      GGONNX_ASSERT(emitted_outputs.has_value(),
                    "EmitNode() returned unsupported for a compiled node");

      size_t emitted_index = 0;
      for (size_t output_id : node.outputs) {
        if (output_id == kOptionalValueAbsent) {
          continue;
        }
        GGONNX_ASSERT(emitted_index < emitted_outputs->size(), "compiled node emitted too few outputs");
        GGONNX_ASSERT(output_id < graph_state->values.size(), "compiled node output index out of range");
        GGONNX_ASSERT(graph_state->values[output_id] == nullptr,
                      "compiled node output slot already populated");
        ggml_tensor* node_output = (*emitted_outputs)[emitted_index++];
        const std::vector<int64_t> output_dims = ToOnnxDims(node_output);
        GGONNX_ASSERT(broadcastSupportedByGGML(partition.values[output_id].dims, output_dims),
                      "emitted output shape mismatch for tensor '" + partition.values[output_id].name +
                          "': declared " + FormatDims(partition.values[output_id].dims) + ", got " +
                          FormatDims(output_dims));
        graph_state->values[output_id] = node_output;
        graph_state->value_dims[output_id] = std::move(output_dims);
      }
      GGONNX_ASSERT(emitted_index == emitted_outputs->size(), "compiled node emitted too many outputs");
    }

    graph_state->graph = ggml_new_graph(ctx);
    GGONNX_NOT_NULL(graph_state->graph, "ggml_new_graph failed");
    for (size_t i = 0; i < partition.graph_outputs.size(); ++i) {
      const size_t output_id = partition.graph_outputs[i];
      GGONNX_ASSERT(output_id < graph_state->values.size(), "graph output index out of range");
      ggml_tensor* output = graph_state->values[output_id];
      GGONNX_NOT_NULL(output, "compiled partition output was not materialized");
      ggml_build_forward_expand(graph_state->graph, output);
      graph_state->output_tensors[i] = output;
    }

    g_debug_graph_build_count.fetch_add(1, std::memory_order_relaxed);
    return graph_state;
  } catch (...) {
    ggml_free(ctx);
    throw;
  }
}

void ExecutePartitionWithGGML(GGMLComputeState& state, OrtKernelContext* kernel_context) {
  const CompiledPartition& partition = state.partition;
  GGONNX_NOT_NULL(kernel_context, "kernel context must not be null");
  GGONNX_ASSERT(!partition.graph_inputs.empty(), "compiled partition must contain graph inputs");
  GGONNX_ASSERT(!partition.graph_outputs.empty(), "compiled partition must contain graph outputs");
  GGONNX_ASSERT(!partition.nodes.empty(), "compiled partition must contain nodes");
  size_t num_inputs = 0;
  size_t num_outputs = 0;
  THROW_ON_ERROR(GetOrtApi().KernelContext_GetInputCount(kernel_context, &num_inputs));
  THROW_ON_ERROR(GetOrtApi().KernelContext_GetOutputCount(kernel_context, &num_outputs));
  if (num_inputs != partition.graph_inputs.size() || num_outputs != partition.graph_outputs.size()) {
    throw std::runtime_error("kernel IO count does not match compiled partition");
  }

  std::vector<const OrtValue*> input_values(num_inputs, nullptr);
  std::vector<TensorMetadata> input_metadata;
  input_metadata.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    THROW_ON_ERROR(GetOrtApi().KernelContext_GetInput(kernel_context, i, &input_values[i]));
    if (input_values[i] == nullptr) {
      throw std::runtime_error("compiled partition received null input");
    }

    int is_tensor = 0;
    THROW_ON_ERROR(GetOrtApi().IsTensor(input_values[i], &is_tensor));
    if (!is_tensor) {
      throw std::runtime_error("compiled partition input is not a tensor");
    }

    input_metadata.push_back(getTensorMetadata(Ort::ConstValue{input_values[i]}));
  }

  const ShapeKey shape_key = MakeShapeKey(input_metadata);
  if (state.active_graph == nullptr || state.active_graph->key != shape_key) {
    DestroyMaterializedGraph(state.active_graph);
    state.active_graph = BuildMaterializedGraph(partition, shape_key, input_metadata);
  }

  GGONNX_NOT_NULL(state.active_graph.get(), "active GGML graph must not be null");
  for (size_t i = 0; i < partition.graph_inputs.size(); ++i) {
    const size_t input_id = partition.graph_inputs[i];
    CopyInputDataToTensor(input_values[i],
                          state.active_graph->value_dims[input_id],
                          partition.values[input_id].name,
                          state.active_graph->input_tensors[i]);
  }

  if (ggml_graph_compute_with_ctx(state.active_graph->ctx, state.active_graph->graph, 1) != GGML_STATUS_SUCCESS) {
    throw std::runtime_error("ggml_graph_compute_with_ctx failed");
  }

  for (size_t i = 0; i < partition.graph_outputs.size(); ++i) {
    const size_t output_id = partition.graph_outputs[i];
    const std::vector<int64_t>& output_dims = state.active_graph->value_dims[output_id];
    OrtValue* output_value = nullptr;
    THROW_ON_ERROR(GetOrtApi().KernelContext_GetOutput(
        kernel_context, i, output_dims.data(), output_dims.size(), &output_value));
    if (output_value == nullptr) {
      throw std::runtime_error("failed to allocate ORT output");
    }

    void* output_data = nullptr;
    THROW_ON_ERROR(GetOrtApi().GetTensorMutableData(output_value, &output_data));
    GGONNX_NOT_NULL(output_data, "ORT returned null output buffer");

    ggml_tensor* output_tensor = state.active_graph->output_tensors[i];
    GGONNX_NOT_NULL(output_tensor, "missing cached GGML output tensor");
    AssertShapeMatchesGGML(output_dims, output_tensor, partition.values[output_id].name);
    GGONNX_ASSERT(ggml_is_contiguous(output_tensor),
                  "GGML output tensor must be contiguous before copy for '" +
                      partition.values[output_id].name + "'");

    const size_t bytes = ggml_nbytes(output_tensor);
    GGONNX_ASSERT(bytes == ElementCount(output_dims) * sizeof(float),
                  "GGML output byte size mismatch for '" + partition.values[output_id].name + "'");
    std::memcpy(output_data, output_tensor->data, bytes);
  }
}

OrtStatus* NodeComputeInfoCreateState(OrtNodeComputeInfo* this_ptr,
                                      OrtNodeComputeContext* /*compute_context*/,
                                      void** compute_state) noexcept {
  return WrapStatus([&] {
    GGONNX_NOT_NULL(this_ptr, "node compute info must not be null");
    GGONNX_NOT_NULL(compute_state, "compute_state output must not be null");
    auto* info = AsNodeComputeInfo(this_ptr);
    auto state = std::make_unique<GGMLComputeState>();
    state->partition = info->partition;
    *compute_state = state.release();
  });
}

OrtStatus* NodeComputeInfoCompute(OrtNodeComputeInfo* /*this_ptr*/,
                                  void* compute_state,
                                  OrtKernelContext* kernel_context) noexcept {
  return WrapStatus([&] {
    GGONNX_NOT_NULL(compute_state, "compute state must not be null");
    auto* state = AsComputeState(compute_state);
    ExecutePartitionWithGGML(*state, kernel_context);
  });
}

void NodeComputeInfoReleaseState(OrtNodeComputeInfo* /*this_ptr*/, void* compute_state) noexcept {
  if (compute_state == nullptr) {
    return;
  }
  auto* state = AsComputeState(compute_state);
  DestroyMaterializedGraph(state->active_graph);
  delete state;
}

std::unique_ptr<GGMLNodeComputeInfo> BuildNodeComputeInfo(CompiledPartition partition) {
  auto info = std::make_unique<GGMLNodeComputeInfo>();
  info->iface.ort_version_supported = ORT_API_VERSION;
  info->iface.CreateState = NodeComputeInfoCreateState;
  info->iface.Compute = NodeComputeInfoCompute;
  info->iface.ReleaseState = NodeComputeInfoReleaseState;
  info->partition = std::move(partition);
  return info;
}

const char* FactoryGetName(const OrtEpFactory* /*this_ptr*/) noexcept {
  return kEpName;
}

const char* FactoryGetVendor(const OrtEpFactory* /*this_ptr*/) noexcept {
  return kVendorName;
}

uint32_t FactoryGetVendorId(const OrtEpFactory* /*this_ptr*/) noexcept {
  return 0;
}

const char* FactoryGetVersion(const OrtEpFactory* /*this_ptr*/) noexcept {
  return kVersion;
}

OrtStatus* FactoryGetSupportedDevices(OrtEpFactory* this_ptr,
                                      const OrtHardwareDevice* const* devices,
                                      size_t num_devices,
                                      OrtEpDevice** ep_devices,
                                      size_t max_ep_devices,
                                      size_t* num_ep_devices) noexcept {
  return WrapStatus([&] {
    GGONNX_NOT_NULL(this_ptr, "factory must not be null");
    GGONNX_NOT_NULL(ep_devices, "ep_devices output array must not be null");
    GGONNX_NOT_NULL(num_ep_devices, "num_ep_devices output must not be null");
    auto* factory = AsFactory(this_ptr);
    size_t count = 0;

    for (size_t i = 0; i < num_devices; ++i) {
      const OrtHardwareDevice* device = devices[i];
      if (!IsSupportedHardwareDevice(device)) {
        continue;
      }
      if (count >= max_ep_devices) {
        throw std::runtime_error("ORT provided too few EP device slots");
      }

      THROW_ON_ERROR(GetOrtEpApi().CreateEpDevice(&factory->iface, device, nullptr, nullptr, &ep_devices[count]));
      ++count;
    }

    *num_ep_devices = count;
  });
}

OrtStatus* EpGetCapability(OrtEp* /*this_ptr*/,
                           const OrtGraph* graph,
                           OrtEpGraphSupportInfo* graph_support_info) noexcept {
  return WrapStatus([&] {
    GGONNX_NOT_NULL(graph_support_info, "graph support info must not be null");
    const Ort::ConstGraph ort_graph{graph};

    std::vector<const OrtNode*> supported_nodes;
    for (Ort::ConstNode node : ort_graph.GetNodes()) {
      if (!IsNodeSupported(node)) {
        continue;
      }
      supported_nodes.push_back(node);
    }

    if (!supported_nodes.empty()) {
      THROW_ON_ERROR(GetOrtEpApi().EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info, supported_nodes.data(), supported_nodes.size(), nullptr));
    }
  });
}

OrtStatus* EpCompile(OrtEp* /*this_ptr*/,
                     const OrtGraph** graphs,
                     const OrtNode** /*fused_nodes*/,
                     size_t count,
                     OrtNodeComputeInfo** node_compute_infos,
                     OrtNode** /*ep_context_nodes*/) noexcept {
  return WrapStatus([&] {
    GGONNX_NOT_NULL(graphs, "graphs array must not be null");
    GGONNX_NOT_NULL(node_compute_infos, "node_compute_infos output array must not be null");
    for (size_t i = 0; i < count; ++i) {
      GGONNX_NOT_NULL(graphs[i], "graph entry must not be null");
      auto compute_info = BuildNodeComputeInfo(CompilePartition(graphs[i]));
      node_compute_infos[i] = &compute_info.release()->iface;
    }
  });
}

void EpReleaseNodeComputeInfos(OrtEp* /*this_ptr*/,
                               OrtNodeComputeInfo** node_compute_infos,
                               size_t num_node_compute_infos) noexcept {
  if (node_compute_infos == nullptr) {
    return;
  }
  for (size_t i = 0; i < num_node_compute_infos; ++i) {
    if (node_compute_infos[i] == nullptr) {
      continue;
    }
    delete AsNodeComputeInfo(node_compute_infos[i]);
  }
}

OrtStatus* EpGetPreferredDataLayout(OrtEp* /*this_ptr*/, OrtEpDataLayout* preferred_data_layout) noexcept {
  if (preferred_data_layout == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "preferred_data_layout output must not be null");
  }
  *preferred_data_layout = OrtEpDataLayout_NCHW;
  return nullptr;
}

OrtStatus* EpOnRunStart(OrtEp* /*this_ptr*/, const OrtRunOptions* /*run_options*/) noexcept {
  return nullptr;
}

OrtStatus* EpOnRunEnd(OrtEp* /*this_ptr*/, const OrtRunOptions* /*run_options*/, bool /*sync_stream*/) noexcept {
  return nullptr;
}

OrtStatus* EpCreateAllocator(OrtEp* /*this_ptr*/,
                             const OrtMemoryInfo* /*memory_info*/,
                             OrtAllocator** allocator) noexcept {
  if (allocator == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "allocator output must not be null");
  }
  *allocator = nullptr;
  return nullptr;
}

OrtStatus* EpCreateSyncStreamForDevice(OrtEp* /*this_ptr*/,
                                       const OrtMemoryDevice* /*memory_device*/,
                                       OrtSyncStreamImpl** stream) noexcept {
  if (stream == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "stream output must not be null");
  }
  *stream = nullptr;
  return nullptr;
}

const char* EpGetCompiledModelCompatibilityInfo(OrtEp* /*this_ptr*/, const OrtGraph* /*graph*/) noexcept {
  return "ggonnx-scaffold";
}

OrtStatus* EpGetKernelRegistry(OrtEp* /*this_ptr*/, const OrtKernelRegistry** kernel_registry) noexcept {
  if (kernel_registry == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "kernel_registry output must not be null");
  }
  *kernel_registry = nullptr;
  return nullptr;
}

OrtStatus* EpIsConcurrentRunSupported(OrtEp* /*this_ptr*/, bool* is_supported) noexcept {
  if (is_supported == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "is_supported output must not be null");
  }
  *is_supported = false;
  return nullptr;
}

const char* EpGetName(const OrtEp* /*this_ptr*/) noexcept {
  return kEpName;
}

OrtStatus* FactoryCreateEp(OrtEpFactory* /*this_ptr*/,
                           const OrtHardwareDevice* const* devices,
                           const OrtKeyValuePairs* const* /*ep_metadata_pairs*/,
                           size_t num_devices,
                           const OrtSessionOptions* /*session_options*/,
                           const OrtLogger* logger,
                           OrtEp** ep) noexcept {
  return WrapStatus([&] {
    GGONNX_NOT_NULL(devices, "devices array must not be null");
    GGONNX_NOT_NULL(ep, "ep output must not be null");
    auto impl = std::make_unique<GGMLEp>();
    impl->iface.ort_version_supported = ORT_API_VERSION;
    impl->iface.GetName = EpGetName;
    impl->iface.GetCapability = EpGetCapability;
    impl->iface.Compile = EpCompile;
    impl->iface.ReleaseNodeComputeInfos = EpReleaseNodeComputeInfos;
    impl->iface.GetPreferredDataLayout = EpGetPreferredDataLayout;
    impl->iface.ShouldConvertDataLayoutForOp = nullptr;
    impl->iface.SetDynamicOptions = nullptr;
    impl->iface.OnRunStart = EpOnRunStart;
    impl->iface.OnRunEnd = EpOnRunEnd;
    impl->iface.CreateAllocator = EpCreateAllocator;
    impl->iface.CreateSyncStreamForDevice = EpCreateSyncStreamForDevice;
    impl->iface.GetCompiledModelCompatibilityInfo = EpGetCompiledModelCompatibilityInfo;
    impl->iface.GetKernelRegistry = EpGetKernelRegistry;
    impl->iface.IsConcurrentRunSupported = EpIsConcurrentRunSupported;
    impl->logger = logger;

    for (size_t i = 0; i < num_devices; ++i) {
      impl->selected_devices.push_back(DescribeHardwareDevice(devices[i]));
    }

    *ep = &impl.release()->iface;
  });
}

void FactoryReleaseEp(OrtEpFactory* /*this_ptr*/, OrtEp* ep) noexcept {
  if (ep == nullptr) {
    return;
  }
  delete AsEp(ep);
}

OrtStatus* FactoryValidateCompiledModelCompatibilityInfo(OrtEpFactory* /*this_ptr*/,
                                                         const OrtHardwareDevice* const* /*devices*/,
                                                         size_t /*num_devices*/,
                                                         const char* compatibility_info,
                                                         OrtCompiledModelCompatibility* compatibility) noexcept {
  if (compatibility_info == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "compatibility_info must not be null");
  }
  if (compatibility == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "compatibility output must not be null");
  }
  *compatibility = std::strcmp(compatibility_info, "ggonnx-scaffold") == 0
                       ? OrtCompiledModelCompatibility_EP_SUPPORTED_OPTIMAL
                       : OrtCompiledModelCompatibility_EP_UNSUPPORTED;
  return nullptr;
}

OrtStatus* FactoryCreateAllocator(OrtEpFactory* /*this_ptr*/,
                                  const OrtMemoryInfo* /*memory_info*/,
                                  const OrtKeyValuePairs* /*allocator_options*/,
                                  OrtAllocator** allocator) noexcept {
  if (allocator == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "allocator output must not be null");
  }
  *allocator = nullptr;
  return nullptr;
}

void FactoryReleaseAllocator(OrtEpFactory* /*this_ptr*/, OrtAllocator* /*allocator*/) noexcept {}

OrtStatus* FactoryCreateDataTransfer(OrtEpFactory* /*this_ptr*/, OrtDataTransferImpl** data_transfer) noexcept {
  if (data_transfer == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "data_transfer output must not be null");
  }
  *data_transfer = nullptr;
  return nullptr;
}

bool FactoryIsStreamAware(const OrtEpFactory* /*this_ptr*/) noexcept {
  return false;
}

OrtStatus* FactoryCreateSyncStreamForDevice(OrtEpFactory* /*this_ptr*/,
                                            const OrtMemoryDevice* /*memory_device*/,
                                            const OrtKeyValuePairs* /*stream_options*/,
                                            OrtSyncStreamImpl** stream) noexcept {
  if (stream == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "stream output must not be null");
  }
  *stream = nullptr;
  return nullptr;
}

OrtStatus* FactoryGetHardwareDeviceIncompatibilityDetails(OrtEpFactory* /*this_ptr*/,
                                                          const OrtHardwareDevice* /*hw*/,
                                                          OrtDeviceEpIncompatibilityDetails* /*details*/) noexcept {
  return nullptr;
}

OrtStatus* FactoryCreateExternalResourceImporterForDevice(
    OrtEpFactory* /*this_ptr*/,
    const OrtEpDevice* /*ep_device*/,
    OrtExternalResourceImporterImpl** out_importer) noexcept {
  if (out_importer == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "external resource importer output must not be null");
  }
  *out_importer = nullptr;
  return MakeStatus(ORT_NOT_IMPLEMENTED, "external resource import is not supported");
}

OrtStatus* FactoryGetNumCustomOpDomains(OrtEpFactory* /*this_ptr*/, size_t* num_domains) noexcept {
  if (num_domains == nullptr) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "num_domains output must not be null");
  }
  *num_domains = 0;
  return nullptr;
}

OrtStatus* FactoryGetCustomOpDomains(OrtEpFactory* /*this_ptr*/,
                                     OrtCustomOpDomain** /*domains*/,
                                     size_t num_domains) noexcept {
  if (num_domains != 0) {
    return MakeStatus(ORT_INVALID_ARGUMENT, "GGONNX does not provide custom op domains");
  }
  return nullptr;
}

std::unique_ptr<GGMLFactory> BuildFactory(const char* registered_name, const OrtApiBase* ort_api_base) {
  InitializeOrtApi(ort_api_base);

  auto factory = std::make_unique<GGMLFactory>();
  factory->registered_name = registered_name != nullptr ? registered_name : kRegistrationName;
  factory->iface.ort_version_supported = ORT_API_VERSION;
  factory->iface.GetName = FactoryGetName;
  factory->iface.GetVendor = FactoryGetVendor;
  factory->iface.GetSupportedDevices = FactoryGetSupportedDevices;
  factory->iface.CreateEp = FactoryCreateEp;
  factory->iface.ReleaseEp = FactoryReleaseEp;
  factory->iface.GetVendorId = FactoryGetVendorId;
  factory->iface.GetVersion = FactoryGetVersion;
  factory->iface.ValidateCompiledModelCompatibilityInfo = FactoryValidateCompiledModelCompatibilityInfo;
  factory->iface.CreateAllocator = FactoryCreateAllocator;
  factory->iface.ReleaseAllocator = FactoryReleaseAllocator;
  factory->iface.CreateDataTransfer = FactoryCreateDataTransfer;
  factory->iface.IsStreamAware = FactoryIsStreamAware;
  factory->iface.CreateSyncStreamForDevice = FactoryCreateSyncStreamForDevice;
  factory->iface.GetHardwareDeviceIncompatibilityDetails = FactoryGetHardwareDeviceIncompatibilityDetails;
  factory->iface.CreateExternalResourceImporterForDevice = FactoryCreateExternalResourceImporterForDevice;
  factory->iface.GetNumCustomOpDomains = FactoryGetNumCustomOpDomains;
  factory->iface.GetCustomOpDomains = FactoryGetCustomOpDomains;
  return factory;
}

}  // namespace

extern "C" OrtStatus* CreateEpFactories(const char* registered_name,
                                        const OrtApiBase* ort_api_base,
                                        const OrtLogger* /*default_logger*/,
                                        OrtEpFactory** factories,
                                        size_t max_factories,
                                        size_t* num_factories) noexcept {
  if (ort_api_base != nullptr && !IsOrtApiInitialized()) {
    InitializeOrtApi(ort_api_base);
  }

  return WrapStatus([&] {
    if (max_factories == 0) {
      throw std::runtime_error("max_factories must be greater than zero");
    }
    GGONNX_NOT_NULL(factories, "factories output array must not be null");
    GGONNX_NOT_NULL(num_factories, "num_factories output must not be null");
    auto factory = BuildFactory(registered_name, ort_api_base);
    factories[0] = &factory.release()->iface;
    *num_factories = 1;
  });
}

extern "C" OrtStatus* ReleaseEpFactory(OrtEpFactory* factory) noexcept {
  if (factory == nullptr) {
    return nullptr;
  }
  delete AsFactory(factory);
  return nullptr;
}

extern "C" uint64_t GGONNX_DebugGetGraphBuildCount() noexcept {
  return g_debug_graph_build_count.load(std::memory_order_relaxed);
}

extern "C" void GGONNX_DebugResetGraphBuildCount() noexcept {
  g_debug_graph_build_count.store(0, std::memory_order_relaxed);
}
