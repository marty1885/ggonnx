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
using ggonnx::ort_internal::ThrowOnError;
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

struct ResolvedPartition {
  std::vector<std::vector<int64_t>> value_dims;
};

struct ShapeKey {
  std::vector<std::vector<int64_t>> input_dims;
};

struct MaterializedGraph {
  ShapeKey key;
  ResolvedPartition resolved;
  ggml_context* ctx{};
  ggml_cgraph* graph{};
  std::vector<ggml_tensor*> values;
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

TensorMetadata GetTensorMetadata(const OrtValueInfo* value_info) {
  GGONNX_NOT_NULL(value_info, "value info must not be null");
  const OrtTypeInfo* type_info = nullptr;
  ThrowOnError(GetOrtApi().GetValueInfoTypeInfo(value_info, &type_info));
  GGONNX_NOT_NULL(type_info, "ORT returned null type info for value");

  const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  ThrowOnError(GetOrtApi().CastTypeInfoToTensorInfo(type_info, &tensor_info));
  GGONNX_NOT_NULL(tensor_info, "value is not a tensor");

  TensorMetadata result;
  ThrowOnError(GetOrtApi().GetTensorElementType(tensor_info, &result.element_type));

  size_t rank = 0;
  ThrowOnError(GetOrtApi().GetDimensionsCount(tensor_info, &rank));
  result.dims.resize(rank);
  if (rank != 0) {
    ThrowOnError(GetOrtApi().GetDimensions(tensor_info, result.dims.data(), rank));
  }

  return result;
}

TensorMetadata GetTensorMetadata(const OrtValue* value) {
  GGONNX_NOT_NULL(value, "OrtValue must not be null");
  OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  ThrowOnError(GetOrtApi().GetTensorTypeAndShape(value, &tensor_info));
  GGONNX_NOT_NULL(tensor_info, "ORT returned null tensor type info");

  TensorMetadata result;
  ThrowOnError(GetOrtApi().GetTensorElementType(tensor_info, &result.element_type));

  size_t rank = 0;
  ThrowOnError(GetOrtApi().GetDimensionsCount(tensor_info, &rank));
  result.dims.resize(rank);
  if (rank != 0) {
    ThrowOnError(GetOrtApi().GetDimensions(tensor_info, result.dims.data(), rank));
  }

  GetOrtApi().ReleaseTensorTypeAndShapeInfo(tensor_info);
  return result;
}

std::string GetValueName(const OrtValueInfo* value_info) {
  GGONNX_NOT_NULL(value_info, "value info must not be null");
  const char* name = nullptr;
  ThrowOnError(GetOrtApi().GetValueInfoName(value_info, &name));
  return name != nullptr ? name : "";
}

std::string GetNodeName(const OrtNode* node) {
  GGONNX_NOT_NULL(node, "node must not be null");
  const char* name = nullptr;
  ThrowOnError(GetOrtApi().Node_GetName(node, &name));
  return name != nullptr ? name : "";
}

std::string GetNodeOperatorType(const OrtNode* node) {
  GGONNX_NOT_NULL(node, "node must not be null");
  const char* op_type = nullptr;
  ThrowOnError(GetOrtApi().Node_GetOperatorType(node, &op_type));
  GGONNX_NOT_NULL(op_type, "ORT returned null operator type");
  return op_type;
}

std::string GetNodeDomain(const OrtNode* node) {
  GGONNX_NOT_NULL(node, "node must not be null");
  const char* domain = nullptr;
  ThrowOnError(GetOrtApi().Node_GetDomain(node, &domain));
  GGONNX_NOT_NULL(domain, "ORT returned null domain");
  return domain;
}

bool HasFullyStaticShape(const TensorMetadata& tensor) {
  return std::all_of(tensor.dims.begin(), tensor.dims.end(), [](int64_t dim) { return dim >= 0; });
}

bool HasSupportedGGMLRank(const TensorMetadata& tensor) {
  return tensor.dims.size() <= GGML_MAX_DIMS;
}

bool HasFullyDefinedDims(const std::vector<int64_t>& dims) {
  return std::all_of(dims.begin(), dims.end(), [](int64_t dim) { return dim >= 0; });
}

bool AreShapesCompatible(const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] >= 0 && rhs[i] >= 0 && lhs[i] != rhs[i]) {
      return false;
    }
  }

  return true;
}

std::string FormatDims(const std::vector<int64_t>& dims) {
  std::string result = "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i != 0) {
      result += ", ";
    }
    result += std::to_string(dims[i]);
  }
  result += "]";
  return result;
}

std::array<int64_t, GGML_MAX_DIMS> ToGGMLDims(const std::vector<int64_t>& onnx_dims) {
  GGONNX_ASSERT(onnx_dims.size() <= GGML_MAX_DIMS,
                "ONNX tensor rank exceeds GGML maximum rank of " + std::to_string(GGML_MAX_DIMS));
  std::array<int64_t, GGML_MAX_DIMS> ggml_dims{};
  for (size_t i = 0; i < onnx_dims.size(); ++i) {
    ggml_dims[i] = onnx_dims[onnx_dims.size() - 1 - i];
  }
  return ggml_dims;
}

std::array<int64_t, GGML_MAX_DIMS> ToPaddedGGMLDims(const std::vector<int64_t>& onnx_dims) {
  GGONNX_ASSERT(onnx_dims.size() <= GGML_MAX_DIMS,
                "ONNX tensor rank exceeds GGML maximum rank of " + std::to_string(GGML_MAX_DIMS));
  std::array<int64_t, GGML_MAX_DIMS> ggml_dims;
  ggml_dims.fill(1);
  for (size_t i = 0; i < onnx_dims.size(); ++i) {
    ggml_dims[i] = onnx_dims[onnx_dims.size() - 1 - i];
  }
  return ggml_dims;
}

std::vector<int64_t> ToOnnxDims(const ggml_tensor* tensor) {
  GGONNX_NOT_NULL(tensor, "ggml tensor must not be null");
  if (ggml_is_scalar(tensor)) {
    return {};
  }

  const int rank = ggml_n_dims(tensor);
  GGONNX_ASSERT(rank >= 1 && rank <= GGML_MAX_DIMS, "ggml tensor has invalid rank");

  std::vector<int64_t> onnx_dims;
  onnx_dims.reserve(static_cast<size_t>(rank));
  for (int i = rank - 1; i >= 0; --i) {
    onnx_dims.push_back(tensor->ne[i]);
  }
  return onnx_dims;
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

bool ShapeKeysMatch(const ShapeKey& lhs, const ShapeKey& rhs) {
  return lhs.input_dims == rhs.input_dims;
}

void GetNodeIoMetadata(const OrtNode* node,
                       std::vector<const OrtValueInfo*>* inputs,
                       std::vector<const OrtValueInfo*>* outputs) {
  GGONNX_NOT_NULL(node, "node must not be null");
  GGONNX_NOT_NULL(inputs, "node input metadata output must not be null");
  GGONNX_NOT_NULL(outputs, "node output metadata output must not be null");

  size_t num_inputs = 0;
  size_t num_outputs = 0;
  ThrowOnError(GetOrtApi().Node_GetNumInputs(node, &num_inputs));
  ThrowOnError(GetOrtApi().Node_GetNumOutputs(node, &num_outputs));

  inputs->resize(num_inputs);
  outputs->resize(num_outputs);
  if (num_inputs != 0) {
    ThrowOnError(GetOrtApi().Node_GetInputs(node, inputs->data(), inputs->size()));
  }
  if (num_outputs != 0) {
    ThrowOnError(GetOrtApi().Node_GetOutputs(node, outputs->data(), outputs->size()));
  }
}

// IsBroadcastSupportedByGGML
// @brief Returns true the requested broadcast in ONNX format is supported by GGML's
//   tensor broadcasting rules.
// @param lhs_dims The dimensions of the left-hand side tensor.
// @param rhs_dims The dimensions of the right-hand side tensor.
// @return True if the broadcast is supported, false otherwise.
bool IsBroadcastSupportedByGGML(const std::vector<int64_t>& lhs_dims,
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

std::vector<int64_t> InferBroadcastOutputDims(const std::vector<int64_t>& lhs_dims,
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

const OrtOpAttr* FindNodeAttribute(const OrtNode* node, const char* attribute_name) {
  GGONNX_NOT_NULL(node, "node must not be null");
  GGONNX_NOT_NULL(attribute_name, "attribute name must not be null");
  const OrtOpAttr* attr = nullptr;
  OrtStatus* status = GetOrtApi().Node_GetAttributeByName(node, attribute_name, &attr);
  if (status == nullptr) {
    return attr;
  }

  const OrtErrorCode code = GetOrtApi().GetErrorCode(status);
  GetOrtApi().ReleaseStatus(status);
  if (code == ORT_INVALID_ARGUMENT) {
    return nullptr;
  }

  throw std::runtime_error("failed to query node attribute '" + std::string(attribute_name) + "'");
}

template <typename T>
inline constexpr bool FalseValue = false;

template <typename T>
std::optional<T> ReadNodeAttribute(const OrtNode* node, const char* attribute_name) {
  const OrtOpAttr* attr = FindNodeAttribute(node, attribute_name);
  if (attr == nullptr) {
    return std::nullopt;
  }

  if constexpr (std::is_same_v<T, std::string>) {
    size_t total_size = 0;
    ThrowOnError(GetOrtApi().ReadOpAttr(attr, ORT_OP_ATTR_STRING, nullptr, 0, &total_size));
    std::vector<char> buffer(total_size);
    ThrowOnError(
        GetOrtApi().ReadOpAttr(attr, ORT_OP_ATTR_STRING, buffer.data(), buffer.size(), &total_size));
    return std::string(buffer.data(), total_size);
  }
  else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
    size_t total_size = 0;
    ThrowOnError(
        GetOrtApi().ReadOpAttr(attr, ORT_OP_ATTR_STRINGS, nullptr, 0, &total_size));
    std::vector<char> buffer(total_size);
    ThrowOnError(
        GetOrtApi().ReadOpAttr(attr, ORT_OP_ATTR_STRINGS, buffer.data(), buffer.size(), &total_size));

    std::vector<std::string> values;
    const char* current = buffer.data();
    const char* end = current + total_size;
    while (current < end) {
      values.emplace_back(current);
      current += values.back().size() + 1;
    }
    throw std::runtime_error("This implementation is weird.... I don't trust it");
    return values;
  }
  else if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, float>) {
    constexpr OrtOpAttrType kType =
        std::is_same_v<T, int64_t> ? ORT_OP_ATTR_INT : ORT_OP_ATTR_FLOAT;
    T value{};
    size_t bytes_read = 0;
    ThrowOnError(GetOrtApi().ReadOpAttr(attr, kType, &value, sizeof(T), &bytes_read));
    GGONNX_ASSERT(bytes_read == sizeof(T),
                  "attribute '" + std::string(attribute_name) + "' returned unexpected size");
    return value;
  }
  else {
    static_assert(FalseValue<T>, "unsupported node attribute type");
  }
}

bool IsSupportedElementwiseBinaryNode(const OrtNode* node, std::string_view op_type) {
  GGONNX_NOT_NULL(node, "node must not be null");
  if (!IsSupportedElementwiseBinaryOpType(op_type)) {
    return false;
  }

  size_t num_inputs = 0;
  size_t num_outputs = 0;
  size_t num_implicit_inputs = 0;
  ThrowOnError(GetOrtApi().Node_GetNumInputs(node, &num_inputs));
  ThrowOnError(GetOrtApi().Node_GetNumOutputs(node, &num_outputs));
  ThrowOnError(GetOrtApi().Node_GetNumImplicitInputs(node, &num_implicit_inputs));
  if (num_inputs != 2 || num_outputs != 1 || num_implicit_inputs != 0) {
    return false;
  }

  std::vector<const OrtValueInfo*> inputs;
  std::vector<const OrtValueInfo*> outputs;
  GetNodeIoMetadata(node, &inputs, &outputs);
  GGONNX_ASSERT(inputs[0] != nullptr && inputs[1] != nullptr && outputs[0] != nullptr,
                "ORT returned null binary op input/output metadata");

  const TensorMetadata lhs = GetTensorMetadata(inputs[0]);
  const TensorMetadata rhs = GetTensorMetadata(inputs[1]);
  const TensorMetadata out = GetTensorMetadata(outputs[0]);

  if (lhs.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      rhs.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      out.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (!HasSupportedGGMLRank(lhs) || !HasSupportedGGMLRank(rhs) || !HasSupportedGGMLRank(out)) {
    return false;
  }
  if (!IsBroadcastSupportedByGGML(lhs.dims, rhs.dims)) {
    return false;
  }
  if (!AreShapesCompatible(lhs.dims, out.dims)) {
    return false;
  }

  return true;
}

bool IsSupportedGRUNode(const OrtNode* node) {
  GGONNX_NOT_NULL(node, "node must not be null");

  size_t num_inputs = 0;
  size_t num_outputs = 0;
  size_t num_implicit_inputs = 0;
  ThrowOnError(GetOrtApi().Node_GetNumInputs(node, &num_inputs));
  ThrowOnError(GetOrtApi().Node_GetNumOutputs(node, &num_outputs));
  ThrowOnError(GetOrtApi().Node_GetNumImplicitInputs(node, &num_implicit_inputs));
  if (num_inputs < 3 || num_outputs == 0 || num_implicit_inputs != 0) {
    return false;
  }

  std::string direction = ReadNodeAttribute<std::string>(node, "direction").value_or("forward");
  if (direction != "forward") {
    return false;
  }

  const int64_t layout = ReadNodeAttribute<int64_t>(node, "layout").value_or(0);
  if (layout != 0) {
    return false;
  }

  const int64_t linear_before_reset =
      ReadNodeAttribute<int64_t>(node, "linear_before_reset").value_or(0);
  if (linear_before_reset != 0) {
    return false;
  }

  if (ReadNodeAttribute<float>(node, "clip").has_value()) {
    return false;
  }

  if (const auto activations = ReadNodeAttribute<std::vector<std::string>>(node, "activations")) {
    if (activations->size() != 2 || (*activations)[0] != "Sigmoid" || (*activations)[1] != "Tanh") {
      return false;
    }
  }

  std::vector<const OrtValueInfo*> inputs;
  std::vector<const OrtValueInfo*> outputs;
  GetNodeIoMetadata(node, &inputs, &outputs);
  if (inputs.size() < 3 || inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr) {
    return false;
  }
  if (inputs.size() > 4 && inputs[4] != nullptr) {
    return false;
  }

  const TensorMetadata x = GetTensorMetadata(inputs[0]);
  const TensorMetadata w = GetTensorMetadata(inputs[1]);
  const TensorMetadata r = GetTensorMetadata(inputs[2]);
  if (x.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      w.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      r.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return false;
  }
  if (x.dims.size() != 3 || w.dims.size() != 3 || r.dims.size() != 3) {
    return false;
  }
  if (!HasSupportedGGMLRank(x) || !HasSupportedGGMLRank(w) || !HasSupportedGGMLRank(r)) {
    return false;
  }

  const auto hidden_size = ReadNodeAttribute<int64_t>(node, "hidden_size");
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
    const TensorMetadata b = GetTensorMetadata(inputs[3]);
    if (b.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || b.dims.size() != 2 || b.dims[0] != 1 ||
        b.dims[1] != *hidden_size * 6) {
      return false;
    }
  }

  if (inputs.size() > 5 && inputs[5] != nullptr) {
    const TensorMetadata initial_h = GetTensorMetadata(inputs[5]);
    if (initial_h.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || initial_h.dims.size() != 3 ||
        initial_h.dims[0] != 1 || initial_h.dims[2] != *hidden_size) {
      return false;
    }
    if (x.dims[1] >= 0 && initial_h.dims[1] >= 0 && x.dims[1] != initial_h.dims[1]) {
      return false;
    }
  }

  for (const OrtValueInfo* output : outputs) {
    if (output == nullptr) {
      continue;
    }
    const TensorMetadata meta = GetTensorMetadata(output);
    if (meta.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || !HasSupportedGGMLRank(meta)) {
      return false;
    }
  }

  return true;
}

void ResolveElementwiseBinaryNode(const CompiledPartition& partition,
                                  const NodeDesc& node,
                                  const std::vector<std::vector<int64_t>>& value_dims,
                                  const std::vector<bool>& is_resolved,
                                  std::vector<std::vector<int64_t>>* resolved_dims,
                                  std::vector<bool>* resolved_flags) {
  GGONNX_NOT_NULL(resolved_dims, "resolved dims output must not be null");
  GGONNX_NOT_NULL(resolved_flags, "resolved flags output must not be null");
  GGONNX_ASSERT(node.inputs.size() == 2 && node.outputs.size() == 1,
                "compiled binary op node has invalid arity");
  const size_t lhs_id = node.inputs[0];
  const size_t rhs_id = node.inputs[1];
  const size_t out_id = node.outputs[0];
  GGONNX_ASSERT(is_resolved[lhs_id] && is_resolved[rhs_id],
                "compiled binary op node inputs must be resolved before execution");

  const std::vector<int64_t>& lhs_dims = value_dims[lhs_id];
  const std::vector<int64_t>& rhs_dims = value_dims[rhs_id];
  const std::vector<int64_t> output_dims = InferBroadcastOutputDims(lhs_dims, rhs_dims);
  GGONNX_ASSERT(IsBroadcastSupportedByGGML(lhs_dims, output_dims),
                "GGML cannot realize broadcast from lhs " + FormatDims(lhs_dims) + " to output " +
                    FormatDims(output_dims));
  GGONNX_ASSERT(IsBroadcastSupportedByGGML(rhs_dims, output_dims),
                "GGML cannot realize broadcast from rhs " + FormatDims(rhs_dims) + " to output " +
                    FormatDims(output_dims));
  GGONNX_ASSERT(AreShapesCompatible(partition.values[out_id].dims, output_dims),
                "resolved output shape mismatch for tensor '" + partition.values[out_id].name + "'");
  (*resolved_dims)[out_id] = output_dims;
  (*resolved_flags)[out_id] = true;
}

void ResolveGRUNode(const CompiledPartition& partition,
                    const NodeDesc& node,
                    const std::vector<std::vector<int64_t>>& value_dims,
                    const std::vector<bool>& is_resolved,
                    std::vector<std::vector<int64_t>>* resolved_dims,
                    std::vector<bool>* resolved_flags) {
  GGONNX_NOT_NULL(resolved_dims, "resolved dims output must not be null");
  GGONNX_NOT_NULL(resolved_flags, "resolved flags output must not be null");
  GGONNX_ASSERT(node.inputs.size() >= 3, "compiled GRU node has invalid input arity");

  const size_t x_id = node.inputs[0];
  const size_t w_id = node.inputs[1];
  const size_t r_id = node.inputs[2];
  GGONNX_ASSERT(is_resolved[x_id] && is_resolved[w_id] && is_resolved[r_id],
                "compiled GRU node required inputs must be resolved before execution");

  const std::vector<int64_t>& x_dims = value_dims[x_id];
  const std::vector<int64_t>& w_dims = value_dims[w_id];
  const std::vector<int64_t>& r_dims = value_dims[r_id];
  GGONNX_ASSERT(x_dims.size() == 3 && w_dims.size() == 3 && r_dims.size() == 3,
                "resolved GRU node inputs must be rank-3 tensors");
  GGONNX_ASSERT(w_dims[0] == 1 && r_dims[0] == 1,
                "resolved GRU node currently supports only forward direction");
  GGONNX_ASSERT(w_dims[1] == node.hidden_size * 3 && r_dims[1] == node.hidden_size * 3,
                "resolved GRU node weight shapes do not match hidden size");
  GGONNX_ASSERT(x_dims[2] == w_dims[2],
                "resolved GRU node input size mismatch between X and W");
  GGONNX_ASSERT(r_dims[2] == node.hidden_size,
                "resolved GRU recurrent weights must match hidden size");

  if (node.inputs.size() > 3 && node.inputs[3] != kOptionalValueAbsent) {
    const size_t b_id = node.inputs[3];
    GGONNX_ASSERT(is_resolved[b_id], "compiled GRU bias input must be resolved before execution");
    GGONNX_ASSERT(value_dims[b_id] == std::vector<int64_t>({1, node.hidden_size * 6}),
                  "resolved GRU bias shape mismatch");
  }

  if (node.inputs.size() > 5 && node.inputs[5] != kOptionalValueAbsent) {
    const size_t initial_h_id = node.inputs[5];
    GGONNX_ASSERT(is_resolved[initial_h_id], "compiled GRU initial_h input must be resolved before execution");
    GGONNX_ASSERT(value_dims[initial_h_id].size() == 3 && value_dims[initial_h_id][0] == 1 &&
                      value_dims[initial_h_id][1] == x_dims[1] &&
                      value_dims[initial_h_id][2] == node.hidden_size,
                  "resolved GRU initial_h shape mismatch");
  }

  const std::vector<int64_t> y_dims = {x_dims[0], 1, x_dims[1], node.hidden_size};
  const std::vector<int64_t> y_h_dims = {1, x_dims[1], node.hidden_size};

  if (!node.outputs.empty() && node.outputs[0] != kOptionalValueAbsent) {
    const size_t y_id = node.outputs[0];
    GGONNX_ASSERT(AreShapesCompatible(partition.values[y_id].dims, y_dims),
                  "resolved GRU Y shape mismatch for tensor '" + partition.values[y_id].name + "'");
    (*resolved_dims)[y_id] = y_dims;
    (*resolved_flags)[y_id] = true;
  }

  if (node.outputs.size() > 1 && node.outputs[1] != kOptionalValueAbsent) {
    const size_t y_h_id = node.outputs[1];
    GGONNX_ASSERT(AreShapesCompatible(partition.values[y_h_id].dims, y_h_dims),
                  "resolved GRU Y_h shape mismatch for tensor '" + partition.values[y_h_id].name + "'");
    (*resolved_dims)[y_h_id] = y_h_dims;
    (*resolved_flags)[y_h_id] = true;
  }
}

bool IsNodeSupported(const OrtNode* node) {
  GGONNX_NOT_NULL(node, "node must not be null");
  const std::string op_type = GetNodeOperatorType(node);
  const std::string domain = GetNodeDomain(node);
  const std::string_view op_type_view(op_type);
  const std::string_view domain_view(domain);

  if (!(domain_view == std::string_view())) {
    return false;
  }

  if(IsSupportedElementwiseBinaryNode(node, op_type_view)) {
      return true;
  }

  if(op_type_view == "GRU" && IsSupportedGRUNode(node)) {
      return true;
  }

  return false;
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

size_t EstimatePartitionDataBytes(const std::vector<std::vector<int64_t>>& value_dims) {
  GGONNX_ASSERT(!value_dims.empty(), "resolved partition must contain at least one value");
  size_t bytes = 0;
  for (const std::vector<int64_t>& dims : value_dims) {
    bytes += ElementCount(dims) * sizeof(float);
  }

  return std::max<size_t>(bytes * 8, 256 * 1024);
}

CompiledPartition CompilePartition(const OrtGraph* graph) {
  GGONNX_NOT_NULL(graph, "graph must not be null");
  CompiledPartition partition;
  std::unordered_map<std::string, size_t> value_ids;

  auto ensure_value = [&](const OrtValueInfo* value_info) -> size_t {
    const std::string name = GetValueName(value_info);
    const TensorMetadata metadata = GetTensorMetadata(value_info);
    GGONNX_ASSERT(metadata.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            "compiled partition currently supports only float32 tensors");
    GGONNX_ASSERT(HasSupportedGGMLRank(metadata),
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

  size_t num_inputs = 0;
  ThrowOnError(GetOrtApi().Graph_GetNumInputs(graph, &num_inputs));
  std::vector<const OrtValueInfo*> graph_inputs(num_inputs);
  if (num_inputs != 0) {
    ThrowOnError(GetOrtApi().Graph_GetInputs(graph, graph_inputs.data(), graph_inputs.size()));
  }

  for (const OrtValueInfo* input : graph_inputs) {
    GGONNX_NOT_NULL(input, "graph input metadata must not be null");
    const size_t id = ensure_value(input);
    partition.values[id].is_graph_input = true;
    partition.graph_inputs.push_back(id);
  }

  size_t num_outputs = 0;
  ThrowOnError(GetOrtApi().Graph_GetNumOutputs(graph, &num_outputs));
  std::vector<const OrtValueInfo*> graph_outputs(num_outputs);
  if (num_outputs != 0) {
    ThrowOnError(GetOrtApi().Graph_GetOutputs(graph, graph_outputs.data(), graph_outputs.size()));
  }

  for (const OrtValueInfo* output : graph_outputs) {
    GGONNX_NOT_NULL(output, "graph output metadata must not be null");
    const size_t id = ensure_value(output);
    partition.values[id].is_graph_output = true;
    partition.graph_outputs.push_back(id);
  }

  size_t num_nodes = 0;
  ThrowOnError(GetOrtApi().Graph_GetNumNodes(graph, &num_nodes));
  std::vector<const OrtNode*> nodes(num_nodes);
  if (num_nodes != 0) {
    ThrowOnError(GetOrtApi().Graph_GetNodes(graph, nodes.data(), nodes.size()));
  }

  for (const OrtNode* node : nodes) {
    GGONNX_NOT_NULL(node, "graph node must not be null");
    if (!IsNodeSupported(node)) {
      throw std::runtime_error("GGONNX Compile received an unsupported partition");
    }

    std::vector<const OrtValueInfo*> node_inputs;
    std::vector<const OrtValueInfo*> node_outputs;
    GetNodeIoMetadata(node, &node_inputs, &node_outputs);

    NodeDesc compiled_node;
    compiled_node.op_type = GetNodeOperatorType(node);
    compiled_node.domain = GetNodeDomain(node);
    compiled_node.name = GetNodeName(node);

    if (compiled_node.op_type == "GRU") {
      compiled_node.direction = ReadNodeAttribute<std::string>(node, "direction").value_or("forward");
      GGONNX_ASSERT(compiled_node.direction == "forward", "only forward GRU direction is supported");
      const auto hidden_size = ReadNodeAttribute<int64_t>(node, "hidden_size");
      GGONNX_ASSERT(hidden_size.has_value() && *hidden_size > 0,
                    "GRU hidden_size attribute must be present and positive");
      compiled_node.hidden_size = *hidden_size;
      compiled_node.layout = ReadNodeAttribute<int64_t>(node, "layout").value_or(0);
      compiled_node.linear_before_reset =
          ReadNodeAttribute<int64_t>(node, "linear_before_reset").value_or(0);
    }

    for (const OrtValueInfo* input : node_inputs) {
      if (input == nullptr) {
        compiled_node.inputs.push_back(kOptionalValueAbsent);
        continue;
      }
      compiled_node.inputs.push_back(ensure_value(input));
    }
    for (const OrtValueInfo* output : node_outputs) {
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
    GGONNX_ASSERT(HasFullyDefinedDims(meta.dims),
                  "runtime input shapes must be concrete before GGML execution");
    key.input_dims.push_back(meta.dims);
  }
  return key;
}

ResolvedPartition ResolvePartitionShapes(const CompiledPartition& partition,
                                         const std::vector<TensorMetadata>& input_metadata) {
  GGONNX_ASSERT(input_metadata.size() == partition.graph_inputs.size(),
                "runtime input metadata does not match compiled partition");

  ResolvedPartition resolved;
  resolved.value_dims.resize(partition.values.size());
  std::vector<bool> is_resolved(partition.values.size(), false);

  for (size_t i = 0; i < input_metadata.size(); ++i) {
    const size_t value_id = partition.graph_inputs[i];
    const TensorMetadata& meta = input_metadata[i];
    const ValueDesc& value = partition.values[value_id];

    GGONNX_ASSERT(meta.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                  "GGONNX runtime input must be float32");
    GGONNX_ASSERT(AreShapesCompatible(value.dims, meta.dims),
                  "runtime input shape mismatch for tensor '" + value.name + "': declared " +
                      FormatDims(value.dims) + ", got " + FormatDims(meta.dims));
    GGONNX_ASSERT(HasFullyDefinedDims(meta.dims),
                  "runtime input shapes must be concrete for tensor '" + value.name + "'");

    resolved.value_dims[value_id] = meta.dims;
    is_resolved[value_id] = true;
  }

  for (const NodeDesc& node : partition.nodes) {
    const std::string_view op_type(node.op_type);
    const std::string_view domain(node.domain);
    if (domain.empty() && IsSupportedElementwiseBinaryOpType(op_type)) {
      ResolveElementwiseBinaryNode(partition,
                                   node,
                                   resolved.value_dims,
                                   is_resolved,
                                   &resolved.value_dims,
                                   &is_resolved);
      continue;
    }
    if (domain.empty() && op_type == "GRU") {
      ResolveGRUNode(partition,
                     node,
                     resolved.value_dims,
                     is_resolved,
                     &resolved.value_dims,
                     &is_resolved);
      continue;
    }

    GGONNX_ABORT("GGONNX fatal: Unsupported ONNX op in ResolvePartitionShapes");
  }

  for (size_t output_id : partition.graph_outputs) {
    GGONNX_ASSERT(output_id < is_resolved.size() && is_resolved[output_id],
                  "compiled partition output shape was not resolved");
  }

  return resolved;
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
  const TensorMetadata meta = GetTensorMetadata(input_value);
  if (meta.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("GGONNX graph input must be float32");
  }

  const void* input_data = nullptr;
  ThrowOnError(GetOrtApi().GetTensorData(input_value, &input_data));
  GGONNX_NOT_NULL(input_data, "ORT returned null tensor data for graph input");

  GGONNX_ASSERT(meta.dims == expected_dims,
                "runtime input shape mismatch for tensor '" + tensor_name + "': expected " +
                    FormatDims(expected_dims) + ", got " + FormatDims(meta.dims));
  const size_t element_count = ElementCount(meta.dims);
  const size_t bytes = element_count * sizeof(float);
  AssertShapeMatchesGGML(meta.dims, tensor, tensor_name);
  std::memcpy(tensor->data, input_data, bytes);
}

ggml_tensor* EmitNode(ggml_context* ctx,
                      const NodeDesc& node,
                      const std::vector<ggml_tensor*>& values,
                      const std::vector<std::vector<int64_t>>& resolved_value_dims,
                      std::vector<ggml_tensor*>* node_outputs) {
  GGONNX_NOT_NULL(ctx, "ggml context must not be null");
  GGONNX_NOT_NULL(node_outputs, "node outputs must not be null");
  const std::string_view op_type(node.op_type);
  const std::string_view domain(node.domain);
  if (domain.empty() && IsSupportedElementwiseBinaryOpType(op_type)) {
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

    if (op_type == "Add") {
      node_outputs->push_back(ggml_add(ctx, lhs, rhs));
      return node_outputs->back();
    }
    if (op_type == "Sub") {
      node_outputs->push_back(ggml_sub(ctx, lhs, rhs));
      return node_outputs->back();
    }
    if (op_type == "Mul") {
      node_outputs->push_back(ggml_mul(ctx, lhs, rhs));
      return node_outputs->back();
    }
    if (op_type == "Div") {
      node_outputs->push_back(ggml_div(ctx, lhs, rhs));
      return node_outputs->back();
    }
  }

  if (domain.empty() && op_type == "GRU") {
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

    if (!node.outputs.empty() && node.outputs[0] != kOptionalValueAbsent) {
      node_outputs->push_back(ggml_cont(ctx, y));
    }
    if (node.outputs.size() > 1 && node.outputs[1] != kOptionalValueAbsent) {
      node_outputs->push_back(ggml_cont_3d(ctx, h_t, hidden_size, batch_size, 1));
    }
    return node_outputs->empty() ? nullptr : node_outputs->front();
  }

  GGONNX_ABORT("GGONNX fatal: Unsupported ONNX op in EmitNode");
}

void DestroyMaterializedGraph(std::unique_ptr<MaterializedGraph>& graph) {
  if (graph != nullptr && graph->ctx != nullptr) {
    ggml_free(graph->ctx);
  }
  graph.reset();
}

std::unique_ptr<MaterializedGraph> BuildMaterializedGraph(const CompiledPartition& partition,
                                                          ShapeKey key,
                                                          ResolvedPartition resolved) {
  const size_t mem_size =
      EstimatePartitionTensorCount(partition) * ggml_tensor_overhead() +
      ggml_graph_overhead() +
      EstimatePartitionDataBytes(resolved.value_dims);

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
    graph_state->resolved = std::move(resolved);
    graph_state->ctx = ctx;
    graph_state->values.resize(partition.values.size(), nullptr);
    graph_state->input_tensors.resize(partition.graph_inputs.size(), nullptr);
    graph_state->output_tensors.resize(partition.graph_outputs.size(), nullptr);

    for (size_t i = 0; i < partition.graph_inputs.size(); ++i) {
      const size_t value_id = partition.graph_inputs[i];
      ggml_tensor* input_tensor = CreateTensorForOnnxShape(ctx, graph_state->resolved.value_dims[value_id]);
      GGONNX_NOT_NULL(input_tensor, "failed to allocate cached GGML input tensor");
      AssertShapeMatchesGGML(graph_state->resolved.value_dims[value_id],
                             input_tensor,
                             partition.values[value_id].name);
      graph_state->values[value_id] = input_tensor;
      graph_state->input_tensors[i] = input_tensor;
    }

    for (const NodeDesc& node : partition.nodes) {
      std::vector<ggml_tensor*> emitted_outputs;
      GGONNX_NOT_NULL(EmitNode(ctx,
                               node,
                               graph_state->values,
                               graph_state->resolved.value_dims,
                               &emitted_outputs),
                      "EmitNode() returned nullptr, unsupported output?");

      size_t emitted_index = 0;
      for (size_t output_id : node.outputs) {
        if (output_id == kOptionalValueAbsent) {
          continue;
        }
        GGONNX_ASSERT(emitted_index < emitted_outputs.size(), "compiled node emitted too few outputs");
        GGONNX_ASSERT(output_id < graph_state->values.size(), "compiled node output index out of range");
        GGONNX_ASSERT(graph_state->values[output_id] == nullptr,
                      "compiled node output slot already populated");
        ggml_tensor* node_output = emitted_outputs[emitted_index++];
        AssertShapeMatchesGGML(graph_state->resolved.value_dims[output_id],
                               node_output,
                               partition.values[output_id].name);
        graph_state->values[output_id] = node_output;
      }
      GGONNX_ASSERT(emitted_index == emitted_outputs.size(), "compiled node emitted too many outputs");
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
  ThrowOnError(GetOrtApi().KernelContext_GetInputCount(kernel_context, &num_inputs));
  ThrowOnError(GetOrtApi().KernelContext_GetOutputCount(kernel_context, &num_outputs));
  if (num_inputs != partition.graph_inputs.size() || num_outputs != partition.graph_outputs.size()) {
    throw std::runtime_error("kernel IO count does not match compiled partition");
  }

  std::vector<const OrtValue*> input_values(num_inputs, nullptr);
  std::vector<TensorMetadata> input_metadata;
  input_metadata.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    ThrowOnError(GetOrtApi().KernelContext_GetInput(kernel_context, i, &input_values[i]));
    if (input_values[i] == nullptr) {
      throw std::runtime_error("compiled partition received null input");
    }

    int is_tensor = 0;
    ThrowOnError(GetOrtApi().IsTensor(input_values[i], &is_tensor));
    if (!is_tensor) {
      throw std::runtime_error("compiled partition input is not a tensor");
    }

    input_metadata.push_back(GetTensorMetadata(input_values[i]));
  }

  const ShapeKey shape_key = MakeShapeKey(input_metadata);
  if (state.active_graph == nullptr || !ShapeKeysMatch(state.active_graph->key, shape_key)) {
    DestroyMaterializedGraph(state.active_graph);
    state.active_graph = BuildMaterializedGraph(
        partition, shape_key, ResolvePartitionShapes(partition, input_metadata));
  }

  GGONNX_NOT_NULL(state.active_graph.get(), "active GGML graph must not be null");
  for (size_t i = 0; i < partition.graph_inputs.size(); ++i) {
    const size_t input_id = partition.graph_inputs[i];
    CopyInputDataToTensor(input_values[i],
                          state.active_graph->resolved.value_dims[input_id],
                          partition.values[input_id].name,
                          state.active_graph->input_tensors[i]);
  }

  if (ggml_graph_compute_with_ctx(state.active_graph->ctx, state.active_graph->graph, 1) != GGML_STATUS_SUCCESS) {
    throw std::runtime_error("ggml_graph_compute_with_ctx failed");
  }

  for (size_t i = 0; i < partition.graph_outputs.size(); ++i) {
    const size_t output_id = partition.graph_outputs[i];
    const std::vector<int64_t>& output_dims = state.active_graph->resolved.value_dims[output_id];
    OrtValue* output_value = nullptr;
    ThrowOnError(GetOrtApi().KernelContext_GetOutput(
        kernel_context, i, output_dims.data(), output_dims.size(), &output_value));
    if (output_value == nullptr) {
      throw std::runtime_error("failed to allocate ORT output");
    }

    void* output_data = nullptr;
    ThrowOnError(GetOrtApi().GetTensorMutableData(output_value, &output_data));
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

      ThrowOnError(GetOrtEpApi().CreateEpDevice(&factory->iface, device, nullptr, nullptr, &ep_devices[count]));
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
    size_t num_nodes = 0;
    ThrowOnError(GetOrtApi().Graph_GetNumNodes(graph, &num_nodes));
    std::vector<const OrtNode*> nodes(num_nodes);
    if (num_nodes != 0) {
      ThrowOnError(GetOrtApi().Graph_GetNodes(graph, nodes.data(), nodes.size()));
    }

    std::vector<const OrtNode*> supported_nodes;
    for (const OrtNode* node : nodes) {
      if (!IsNodeSupported(node)) {
        continue;
      }
      supported_nodes.push_back(node);
    }

    if (!supported_nodes.empty()) {
      ThrowOnError(GetOrtEpApi().EpGraphSupportInfo_AddNodesToFuse(
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
