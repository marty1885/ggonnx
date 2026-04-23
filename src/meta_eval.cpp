#include "meta_eval.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <optional>
#include <stdexcept>

#include "inner/helpers.hpp"
#include "inner/ort_api_helpers.hpp"

namespace {

using ggonnx::ort_internal::GetOrtApi;
using ggonnx::ort_internal::THROW_ON_ERROR;

constexpr size_t kMaxFoldedElements = 256;
// Raw initializers live on disk/in ORT memory already; loading them into the
// constants map is a bounded duplication, not an unbounded compute. The
// tighter kMaxFoldedElements still guards derived folds (Tile/Expand/etc.)
// so they can't synthesize huge tensors.
constexpr size_t kMaxInitializerElements = 1 << 20;  // 1M elements (~4 MB f32)

size_t ByteWidth(ONNXTensorElementDataType element_type) {
  switch (element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return sizeof(float);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return sizeof(uint16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return sizeof(int32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return sizeof(int64_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return sizeof(uint8_t);
    default:
      throw std::runtime_error("unsupported compile-time constant dtype");
  }
}

size_t ElementCount(const std::vector<int64_t>& dims) {
  size_t count = 1;
  for (int64_t dim : dims) {
    GGONNX_ASSERT(dim >= 0, "compile-time constant has invalid dynamic shape");
    count *= static_cast<size_t>(dim);
  }
  return count;
}

bool IsFoldableDType(ONNXTensorElementDataType element_type) {
  return element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT   ||
         element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
         element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
         element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
}

bool FitsFoldLimit(const std::vector<int64_t>& dims) {
  return ElementCount(dims) <= kMaxFoldedElements;
}

bool FitsInitializerLimit(const std::vector<int64_t>& dims) {
  return ElementCount(dims) <= kMaxInitializerElements;
}

template <typename T>
std::vector<T> ReadTypedData(const ConstantTensor& tensor) {
  const size_t count = ElementCount(tensor.dims);
  GGONNX_ASSERT(tensor.data.size() == count * sizeof(T),
                "compile-time constant byte size mismatch");
  std::vector<T> out(count);
  if (!out.empty()) {
    std::memcpy(out.data(), tensor.data.data(), tensor.data.size());
  }
  return out;
}

template <typename T>
ConstantTensor MakeConstant(std::vector<int64_t> dims, std::vector<T> values,
                            ONNXTensorElementDataType element_type) {
  GGONNX_ASSERT(ElementCount(dims) == values.size(),
                "compile-time constant element count mismatch");
  ConstantTensor tensor;
  tensor.element_type = element_type;
  tensor.dims = std::move(dims);
  tensor.data.resize(values.size() * sizeof(T));
  if (!values.empty()) {
    std::memcpy(tensor.data.data(), values.data(), tensor.data.size());
  }
  return tensor;
}

ConstantTensor LoadInitializer(Ort::ConstValue value) {
  const TensorMetadata meta = getTensorMetadata(value);
  GGONNX_ASSERT(IsFoldableDType(meta.element_type),
                "unsupported initializer dtype for compile-time folding");
  GGONNX_ASSERT(FitsInitializerLimit(meta.dims),
                "initializer too large for compile-time folding");
  const size_t bytes = ElementCount(meta.dims) * ByteWidth(meta.element_type);
  const void* src = nullptr;
  THROW_ON_ERROR(GetOrtApi().GetTensorData(value, &src));
  GGONNX_NOT_NULL(src, "ORT returned null tensor data for initializer");

  ConstantTensor tensor;
  tensor.element_type = meta.element_type;
  tensor.dims = meta.dims;
  tensor.data.resize(bytes);
  if (bytes > 0) {
    std::memcpy(tensor.data.data(), src, bytes);
  }
  return tensor;
}

std::optional<ConstantTensor> LookupConstant(const ConstantValueMap& constants, Ort::ConstValueInfo value) {
  if (value == nullptr) return std::nullopt;
  auto it = constants.find(value.GetName());
  if (it == constants.end()) return std::nullopt;
  return it->second;
}

std::optional<std::vector<int64_t>> ReadConstantInt64Input(const ConstantValueMap& constants,
                                                           Ort::ConstNode node,
                                                           size_t input_idx) {
  const auto inputs = node.GetInputs();
  if (input_idx >= inputs.size() || inputs[input_idx] == nullptr) return std::nullopt;
  const auto tensor_opt = LookupConstant(constants, inputs[input_idx]);
  if (!tensor_opt || tensor_opt->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return std::nullopt;
  }
  return ReadTypedData<int64_t>(*tensor_opt);
}

std::optional<bool> ReadConstantBool(Ort::ConstValueInfo value_info) {
  if (value_info == nullptr || !value_info.IsConstantInitializer()) return std::nullopt;
  Ort::ConstValue value{nullptr};
  Ort::ThrowOnError(value_info.GetInitializer(value));
  if (value == nullptr) return std::nullopt;
  const TensorMetadata meta = getTensorMetadata(value);
  if (meta.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) return std::nullopt;
  GGONNX_ASSERT(ElementCount(meta.dims) == 1, "bool constant must be scalar");
  const void* src = nullptr;
  THROW_ON_ERROR(GetOrtApi().GetTensorData(value, &src));
  GGONNX_NOT_NULL(src, "ORT returned null bool tensor data");
  return *static_cast<const bool*>(src);
}

template <typename T>
std::vector<int64_t> MakeStrides(const std::vector<T>& dims) {
  std::vector<int64_t> strides(dims.size(), 1);
  for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * static_cast<int64_t>(dims[static_cast<size_t>(i + 1)]);
  }
  return strides;
}

template <typename T>
ConstantTensor CastConstant(const ConstantTensor& input, ONNXTensorElementDataType to_type) {
  const size_t count = ElementCount(input.dims);
  if (input.element_type == to_type) return input;

  if (to_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    std::vector<int32_t> out(count);
    if (input.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      const auto values = ReadTypedData<int64_t>(input);
      for (size_t i = 0; i < count; ++i) out[i] = static_cast<int32_t>(values[i]);
    } else if (input.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      const auto values = ReadTypedData<float>(input);
      for (size_t i = 0; i < count; ++i) out[i] = static_cast<int32_t>(values[i]);
    } else {
      throw std::runtime_error("unsupported cast source dtype");
    }
    return MakeConstant(std::vector<int64_t>(input.dims), std::move(out), to_type);
  }
  if (to_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    std::vector<int64_t> out(count);
    if (input.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      const auto values = ReadTypedData<int32_t>(input);
      for (size_t i = 0; i < count; ++i) out[i] = static_cast<int64_t>(values[i]);
    } else if (input.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      const auto values = ReadTypedData<float>(input);
      for (size_t i = 0; i < count; ++i) out[i] = static_cast<int64_t>(values[i]);
    } else {
      throw std::runtime_error("unsupported cast source dtype");
    }
    return MakeConstant(std::vector<int64_t>(input.dims), std::move(out), to_type);
  }
  if (to_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    std::vector<float> out(count);
    if (input.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      const auto values = ReadTypedData<int32_t>(input);
      for (size_t i = 0; i < count; ++i) out[i] = static_cast<float>(values[i]);
    } else if (input.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      const auto values = ReadTypedData<int64_t>(input);
      for (size_t i = 0; i < count; ++i) out[i] = static_cast<float>(values[i]);
    } else {
      throw std::runtime_error("unsupported cast source dtype");
    }
    return MakeConstant(std::vector<int64_t>(input.dims), std::move(out), to_type);
  }
  throw std::runtime_error("unsupported cast destination dtype");
}

template <typename T>
ConstantTensor EvalSliceTyped(const ConstantTensor& data,
                              const std::vector<int64_t>& starts,
                              const std::vector<int64_t>& ends,
                              std::vector<int64_t> axes,
                              const std::vector<int64_t>& steps,
                              ONNXTensorElementDataType element_type) {
  const int64_t rank = static_cast<int64_t>(data.dims.size());
  GGONNX_ASSERT(rank >= 0, "invalid slice rank");
  if (axes.empty()) {
    axes.resize(starts.size());
    for (size_t i = 0; i < starts.size(); ++i) axes[i] = static_cast<int64_t>(i);
  }

  std::vector<int64_t> begin = std::vector<int64_t>(data.dims.size(), 0);
  std::vector<int64_t> stride = std::vector<int64_t>(data.dims.size(), 1);
  std::vector<int64_t> out_dims = data.dims;

  auto clamp = [](int64_t value, int64_t lo, int64_t hi) {
    return std::max(lo, std::min(hi, value));
  };

  for (size_t i = 0; i < axes.size(); ++i) {
    int64_t axis = axes[i];
    if (axis < 0) axis += rank;
    GGONNX_ASSERT(axis >= 0 && axis < rank, "slice axis out of range");
    const int64_t dim = data.dims[static_cast<size_t>(axis)];
    const int64_t step = steps.empty() ? 1 : steps[i];
    GGONNX_ASSERT(step != 0, "slice step must not be zero");

    int64_t start = starts[i];
    int64_t end = ends[i];
    if (step > 0) {
      if (start < 0) start += dim;
      if (end < 0) end += dim;
      start = clamp(start, 0, dim);
      end = clamp(end, 0, dim);
      out_dims[static_cast<size_t>(axis)] = end > start ? (end - start + step - 1) / step : 0;
    } else {
      if (start < 0) start += dim;
      if (end < 0) end += dim;
      start = clamp(start, -1, dim - 1);
      end = clamp(end, -1, dim - 1);
      out_dims[static_cast<size_t>(axis)] = start > end ? (start - end + (-step) - 1) / (-step) : 0;
    }
    begin[static_cast<size_t>(axis)] = start;
    stride[static_cast<size_t>(axis)] = step;
  }

  GGONNX_ASSERT(FitsFoldLimit(out_dims), "slice output too large for compile-time folding");
  const auto input = ReadTypedData<T>(data);
  const size_t out_count = ElementCount(out_dims);
  std::vector<T> out(out_count);
  const auto in_strides = MakeStrides(data.dims);
  const auto out_strides = MakeStrides(out_dims);
  std::vector<int64_t> coord(out_dims.size(), 0);

  for (size_t linear = 0; linear < out_count; ++linear) {
    int64_t input_linear = 0;
    for (size_t axis = 0; axis < out_dims.size(); ++axis) {
      const int64_t idx = begin[axis] + coord[axis] * stride[axis];
      input_linear += idx * in_strides[axis];
    }
    out[linear] = input[static_cast<size_t>(input_linear)];
    for (int axis = static_cast<int>(coord.size()) - 1; axis >= 0; --axis) {
      coord[static_cast<size_t>(axis)]++;
      if (coord[static_cast<size_t>(axis)] < out_dims[static_cast<size_t>(axis)]) break;
      coord[static_cast<size_t>(axis)] = 0;
    }
  }
  return MakeConstant(std::move(out_dims), std::move(out), element_type);
}

std::optional<ConstantTensor> EvalShape(const ConstantValueMap& /*constants*/, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 1 || inputs[0] == nullptr) return std::nullopt;
  const TensorMetadata meta = getTensorMetadata(inputs[0]);
  // ORT nested subgraphs may omit shape metadata entirely; GetShape() then
  // comes back as [] even for non-scalar tensors. Treat [] as "unknown" here
  // instead of folding Shape(x) to an empty constant.
  if (meta.dims.empty()) return std::nullopt;
  if (!shapeIsFullyStatic(meta)) return std::nullopt;
  std::vector<int64_t> values = meta.dims;
  return MakeConstant<int64_t>({static_cast<int64_t>(values.size())}, std::move(values),
                               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
}

std::optional<ConstantTensor> EvalCast(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 1 || inputs[0] == nullptr) return std::nullopt;
  const auto input = LookupConstant(constants, inputs[0]);
  if (!input) return std::nullopt;
  Ort::ConstOpAttr attr{nullptr};
  const Ort::Status status = node.GetAttributeByName("to", attr);
  if (!status.IsOK() || attr == nullptr) return std::nullopt;
  int64_t to = 0;
  Ort::ThrowOnError(attr.GetValue(to));
  return CastConstant<int64_t>(*input, static_cast<ONNXTensorElementDataType>(to));
}

std::optional<ConstantTensor> EvalIdentity(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 1 || inputs[0] == nullptr) return std::nullopt;
  return LookupConstant(constants, inputs[0]);
}

std::optional<ConstantTensor> EvalConcat(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.empty()) return std::nullopt;
  std::vector<ConstantTensor> tensors;
  tensors.reserve(inputs.size());
  for (auto input : inputs) {
    if (input == nullptr) return std::nullopt;
    const auto tensor = LookupConstant(constants, input);
    if (!tensor) return std::nullopt;
    tensors.push_back(*tensor);
  }
  const ONNXTensorElementDataType element_type = tensors[0].element_type;
  for (const auto& tensor : tensors) {
    if (tensor.element_type != element_type) return std::nullopt;
  }

  Ort::ConstOpAttr attr{nullptr};
  const Ort::Status status = node.GetAttributeByName("axis", attr);
  if (!status.IsOK() || attr == nullptr) return std::nullopt;
  int64_t axis = 0;
  Ort::ThrowOnError(attr.GetValue(axis));

  const std::vector<int64_t>& base_shape = tensors[0].dims;
  if (axis < 0) axis += static_cast<int64_t>(base_shape.size());
  if (axis < 0 || axis >= static_cast<int64_t>(base_shape.size())) return std::nullopt;
  std::vector<int64_t> out_dims = base_shape;
  out_dims[static_cast<size_t>(axis)] = 0;
  for (const auto& tensor : tensors) {
    if (tensor.dims.size() != base_shape.size()) return std::nullopt;
    for (size_t i = 0; i < tensor.dims.size(); ++i) {
      if (i == static_cast<size_t>(axis)) continue;
      if (tensor.dims[i] != base_shape[i]) return std::nullopt;
    }
    out_dims[static_cast<size_t>(axis)] += tensor.dims[static_cast<size_t>(axis)];
  }
  if (!FitsFoldLimit(out_dims)) return std::nullopt;

  if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    std::vector<int32_t> out;
    for (const auto& tensor : tensors) {
      auto values = ReadTypedData<int32_t>(tensor);
      out.insert(out.end(), values.begin(), values.end());
    }
    return MakeConstant(std::move(out_dims), std::move(out), element_type);
  }
  if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    std::vector<int64_t> out;
    for (const auto& tensor : tensors) {
      auto values = ReadTypedData<int64_t>(tensor);
      out.insert(out.end(), values.begin(), values.end());
    }
    return MakeConstant(std::move(out_dims), std::move(out), element_type);
  }
  if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    std::vector<float> out;
    for (const auto& tensor : tensors) {
      auto values = ReadTypedData<float>(tensor);
      out.insert(out.end(), values.begin(), values.end());
    }
    return MakeConstant(std::move(out_dims), std::move(out), element_type);
  }
  return std::nullopt;
}

std::optional<ConstantTensor> EvalUnsqueeze(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.empty() || inputs[0] == nullptr) return std::nullopt;
  const auto input = LookupConstant(constants, inputs[0]);
  if (!input) return std::nullopt;

  std::optional<std::vector<int64_t>> axes;
  if (inputs.size() >= 2) axes = ReadConstantInt64Input(constants, node, 1);
  if (!axes.has_value()) {
    Ort::ConstOpAttr attr{nullptr};
    const Ort::Status status = node.GetAttributeByName("axes", attr);
    if (!status.IsOK() || attr == nullptr) return std::nullopt;
    axes.emplace();
    Ort::ThrowOnError(attr.GetValueArray(*axes));
  }

  std::vector<int64_t> out_dims = input->dims;
  std::sort(axes->begin(), axes->end());
  const int64_t out_rank = static_cast<int64_t>(out_dims.size() + axes->size());
  for (int64_t axis : *axes) {
    if (axis < 0) axis += out_rank;
    if (axis < 0 || axis > static_cast<int64_t>(out_dims.size())) return std::nullopt;
    out_dims.insert(out_dims.begin() + axis, 1);
  }
  ConstantTensor out = *input;
  out.dims = std::move(out_dims);
  return out;
}

std::optional<ConstantTensor> EvalSqueeze(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.empty() || inputs[0] == nullptr) return std::nullopt;
  const auto input = LookupConstant(constants, inputs[0]);
  if (!input) return std::nullopt;

  std::vector<int64_t> out_dims = input->dims;
  std::optional<std::vector<int64_t>> axes;
  if (inputs.size() >= 2 && inputs[1] != nullptr) axes = ReadConstantInt64Input(constants, node, 1);
  if (!axes.has_value()) {
    Ort::ConstOpAttr attr{nullptr};
    const Ort::Status status = node.GetAttributeByName("axes", attr);
    if (status.IsOK() && attr != nullptr) {
      axes.emplace();
      Ort::ThrowOnError(attr.GetValueArray(*axes));
    }
  }

  if (axes.has_value()) {
    std::vector<int64_t> normalized = *axes;
    std::sort(normalized.rbegin(), normalized.rend());
    for (int64_t axis : normalized) {
      if (axis < 0) axis += static_cast<int64_t>(out_dims.size());
      if (axis < 0 || axis >= static_cast<int64_t>(out_dims.size())) return std::nullopt;
      if (out_dims[static_cast<size_t>(axis)] != 1) return std::nullopt;
      out_dims.erase(out_dims.begin() + axis);
    }
  } else {
    out_dims.erase(std::remove(out_dims.begin(), out_dims.end(), 1), out_dims.end());
  }

  ConstantTensor out = *input;
  out.dims = std::move(out_dims);
  return out;
}

template <typename T>
std::optional<ConstantTensor> EvalBinaryTyped(const ConstantTensor& lhs,
                                              const ConstantTensor& rhs,
                                              const std::string& op_type,
                                              ONNXTensorElementDataType element_type) {
  if (lhs.element_type != element_type || rhs.element_type != element_type) return std::nullopt;
  const size_t lhs_count = ElementCount(lhs.dims);
  const size_t rhs_count = ElementCount(rhs.dims);
  const bool lhs_scalar = lhs_count == 1;
  const bool rhs_scalar = rhs_count == 1;
  if (!lhs_scalar && !rhs_scalar && lhs.dims != rhs.dims) return std::nullopt;
  const std::vector<int64_t> out_dims = lhs_scalar ? rhs.dims : lhs.dims;
  const size_t out_count = std::max(lhs_count, rhs_count);
  if (!FitsFoldLimit(out_dims)) return std::nullopt;
  const auto lhs_values = ReadTypedData<T>(lhs);
  const auto rhs_values = ReadTypedData<T>(rhs);
  std::vector<T> out(out_count);
  for (size_t i = 0; i < out_count; ++i) {
    const T a = lhs_values[lhs_scalar ? 0 : i];
    const T b = rhs_values[rhs_scalar ? 0 : i];
    if (op_type == "Add") out[i] = a + b;
    else if (op_type == "Sub") out[i] = a - b;
    else if (op_type == "Mul") out[i] = a * b;
    else if (op_type == "Div") out[i] = a / b;
    else return std::nullopt;
  }
  return MakeConstant(std::vector<int64_t>(out_dims), std::move(out), element_type);
}

std::optional<ConstantTensor> EvalBinary(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 2 || inputs[0] == nullptr || inputs[1] == nullptr) return std::nullopt;
  const auto lhs = LookupConstant(constants, inputs[0]);
  const auto rhs = LookupConstant(constants, inputs[1]);
  if (!lhs || !rhs || lhs->element_type != rhs->element_type) return std::nullopt;
  const std::string op_type = node.GetOperatorType();
  switch (lhs->element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return EvalBinaryTyped<int32_t>(*lhs, *rhs, op_type, lhs->element_type);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return EvalBinaryTyped<int64_t>(*lhs, *rhs, op_type, lhs->element_type);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return EvalBinaryTyped<float>(*lhs, *rhs, op_type, lhs->element_type);
    default:
      return std::nullopt;
  }
}

std::optional<ConstantTensor> EvalReshape(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 2 || inputs[0] == nullptr || inputs[1] == nullptr) return std::nullopt;
  const auto data = LookupConstant(constants, inputs[0]);
  const auto shape = LookupConstant(constants, inputs[1]);
  if (!data || !shape || shape->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) return std::nullopt;
  const auto target_shape_values = ReadTypedData<int64_t>(*shape);
  std::vector<int64_t> out_dims = target_shape_values;
  int64_t infer_axis = -1;
  int64_t known_product = 1;
  for (size_t i = 0; i < out_dims.size(); ++i) {
    if (out_dims[i] == 0) out_dims[i] = data->dims[i];
    if (out_dims[i] == -1) {
      if (infer_axis >= 0) return std::nullopt;
      infer_axis = static_cast<int64_t>(i);
      continue;
    }
    known_product *= out_dims[i];
  }
  const int64_t total = static_cast<int64_t>(ElementCount(data->dims));
  if (infer_axis >= 0) {
    GGONNX_ASSERT(known_product != 0, "reshape known product must be non-zero");
    out_dims[static_cast<size_t>(infer_axis)] = total / known_product;
  }
  if (!FitsFoldLimit(out_dims)) return std::nullopt;
  if (ElementCount(out_dims) != ElementCount(data->dims)) return std::nullopt;
  ConstantTensor out = *data;
  out.dims = std::move(out_dims);
  return out;
}

std::optional<ConstantTensor> EvalSlice(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() < 3 || inputs[0] == nullptr) return std::nullopt;
  const auto data = LookupConstant(constants, inputs[0]);
  if (!data) return std::nullopt;
  const auto starts = ReadConstantInt64Input(constants, node, 1);
  const auto ends = ReadConstantInt64Input(constants, node, 2);
  if (!starts || !ends || starts->size() != ends->size()) return std::nullopt;
  std::vector<int64_t> axes;
  if (inputs.size() >= 4 && inputs[3] != nullptr) {
    const auto axes_opt = ReadConstantInt64Input(constants, node, 3);
    if (!axes_opt) return std::nullopt;
    axes = *axes_opt;
  }
  std::vector<int64_t> steps;
  if (inputs.size() >= 5 && inputs[4] != nullptr) {
    const auto steps_opt = ReadConstantInt64Input(constants, node, 4);
    if (!steps_opt) return std::nullopt;
    steps = *steps_opt;
  }
  switch (data->element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return EvalSliceTyped<int32_t>(*data, *starts, *ends, std::move(axes), steps, data->element_type);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return EvalSliceTyped<int64_t>(*data, *starts, *ends, std::move(axes), steps, data->element_type);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return EvalSliceTyped<float>(*data, *starts, *ends, std::move(axes), steps, data->element_type);
    default:
      return std::nullopt;
  }
}

template <typename T>
ConstantTensor EvalTileTyped(const ConstantTensor& data,
                             const std::vector<int64_t>& repeats,
                             ONNXTensorElementDataType element_type) {
  GGONNX_ASSERT(repeats.size() == data.dims.size(),
                "tile repeats rank must match input rank");
  std::vector<int64_t> out_dims(data.dims.size(), 0);
  for (size_t i = 0; i < repeats.size(); ++i) {
    GGONNX_ASSERT(repeats[i] >= 0, "tile repeats must be non-negative");
    out_dims[i] = data.dims[i] * repeats[i];
  }
  GGONNX_ASSERT(FitsFoldLimit(out_dims), "tile output too large for compile-time folding");

  const auto input = ReadTypedData<T>(data);
  const size_t out_count = ElementCount(out_dims);
  std::vector<T> out(out_count);
  const auto in_strides = MakeStrides(data.dims);
  std::vector<int64_t> coord(out_dims.size(), 0);

  for (size_t linear = 0; linear < out_count; ++linear) {
    int64_t input_linear = 0;
    for (size_t axis = 0; axis < out_dims.size(); ++axis) {
      const int64_t src_idx = data.dims[axis] == 0 ? 0 : (coord[axis] % data.dims[axis]);
      input_linear += src_idx * in_strides[axis];
    }
    out[linear] = input[static_cast<size_t>(input_linear)];
    for (int axis = static_cast<int>(coord.size()) - 1; axis >= 0; --axis) {
      coord[static_cast<size_t>(axis)]++;
      if (coord[static_cast<size_t>(axis)] < out_dims[static_cast<size_t>(axis)]) break;
      coord[static_cast<size_t>(axis)] = 0;
    }
  }
  return MakeConstant(std::move(out_dims), std::move(out), element_type);
}

std::optional<ConstantTensor> EvalTile(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 2 || inputs[0] == nullptr || inputs[1] == nullptr) return std::nullopt;
  const auto data = LookupConstant(constants, inputs[0]);
  if (!data) return std::nullopt;
  const auto repeats = ReadConstantInt64Input(constants, node, 1);
  if (!repeats) return std::nullopt;
  switch (data->element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return EvalTileTyped<int32_t>(*data, *repeats, data->element_type);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return EvalTileTyped<int64_t>(*data, *repeats, data->element_type);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return EvalTileTyped<float>(*data, *repeats, data->element_type);
    default:
      return std::nullopt;
  }
}

std::optional<std::vector<ConstantTensor>> EvalArangeLoop(const ConstantValueMap& constants,
                                                          Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  const auto outputs = node.GetOutputs();
  if (inputs.size() != 3 || outputs.size() != 2) return std::nullopt;
  if (inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr) return std::nullopt;

  const auto trip_count_tensor = LookupConstant(constants, inputs[0]);
  const auto start_tensor = LookupConstant(constants, inputs[2]);
  const auto cond = ReadConstantBool(inputs[1]);
  if (!trip_count_tensor || !start_tensor || !cond || !*cond) return std::nullopt;
  if (trip_count_tensor->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) return std::nullopt;
  if (ElementCount(trip_count_tensor->dims) != 1 || ElementCount(start_tensor->dims) != 1) return std::nullopt;
  const int64_t trip_count = ReadTypedData<int64_t>(*trip_count_tensor)[0];
  if (trip_count < 0 || trip_count > static_cast<int64_t>(kMaxFoldedElements)) return std::nullopt;

  Ort::ConstGraph body{nullptr};
  for (const auto& subgraph : node.GetSubgraphs()) {
    if (subgraph.attr_name == "body") {
      body = subgraph.sub_graph;
      break;
    }
  }
  if (body == nullptr) return std::nullopt;
  const auto body_inputs = body.GetInputs();
  const auto body_outputs = body.GetOutputs();
  const auto body_nodes = body.GetNodes();
  if (body_inputs.size() != 3 || body_outputs.size() != 3 || body_nodes.size() != 3) return std::nullopt;

  Ort::ConstNode add_node{nullptr};
  bool saw_prev_identity = false;
  bool saw_cond_identity = false;
  for (Ort::ConstNode body_node : body_nodes) {
    const std::string op_type = body_node.GetOperatorType();
    if (op_type == "Add") {
      add_node = body_node;
    } else if (op_type == "Identity") {
      const auto node_inputs = body_node.GetInputs();
      const auto node_outputs = body_node.GetOutputs();
      if (node_inputs.size() != 1 || node_outputs.size() != 1 || node_inputs[0] == nullptr || node_outputs[0] == nullptr) {
        return std::nullopt;
      }
      if (std::string(node_inputs[0].GetName()) == body_inputs[1].GetName() &&
          std::string(node_outputs[0].GetName()) == body_outputs[0].GetName()) {
        saw_cond_identity = true;
      } else if (std::string(node_inputs[0].GetName()) == body_inputs[2].GetName() &&
                 std::string(node_outputs[0].GetName()) == body_outputs[2].GetName()) {
        saw_prev_identity = true;
      } else {
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
  }
  if (add_node == nullptr || !saw_prev_identity || !saw_cond_identity) return std::nullopt;

  const auto add_inputs = add_node.GetInputs();
  const auto add_outputs = add_node.GetOutputs();
  if (add_inputs.size() != 2 || add_outputs.size() != 1 || add_outputs[0] == nullptr ||
      std::string(add_outputs[0].GetName()) != body_outputs[1].GetName()) {
    return std::nullopt;
  }

  std::optional<ConstantTensor> delta_tensor;
  bool saw_prev = false;
  for (size_t i = 0; i < 2; ++i) {
    if (add_inputs[i] == nullptr) return std::nullopt;
    if (std::string(add_inputs[i].GetName()) == body_inputs[2].GetName()) {
      saw_prev = true;
      continue;
    }
    delta_tensor = LookupConstant(constants, add_inputs[i]);
  }
  if (!saw_prev || !delta_tensor) return std::nullopt;
  if (delta_tensor->element_type != start_tensor->element_type ||
      ElementCount(delta_tensor->dims) != 1) {
    return std::nullopt;
  }

  if (start_tensor->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t start = ReadTypedData<int32_t>(*start_tensor)[0];
    const int32_t delta = ReadTypedData<int32_t>(*delta_tensor)[0];
    std::vector<int32_t> scan(static_cast<size_t>(trip_count));
    for (int64_t i = 0; i < trip_count; ++i) scan[static_cast<size_t>(i)] = start + static_cast<int32_t>(i) * delta;
    const int32_t final_value = start + static_cast<int32_t>(trip_count) * delta;
    return std::vector<ConstantTensor>{
        MakeConstant<int32_t>({}, {final_value}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
        MakeConstant<int32_t>({trip_count}, std::move(scan), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
    };
  }
  if (start_tensor->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    const int64_t start = ReadTypedData<int64_t>(*start_tensor)[0];
    const int64_t delta = ReadTypedData<int64_t>(*delta_tensor)[0];
    std::vector<int64_t> scan(static_cast<size_t>(trip_count));
    for (int64_t i = 0; i < trip_count; ++i) scan[static_cast<size_t>(i)] = start + i * delta;
    const int64_t final_value = start + trip_count * delta;
    return std::vector<ConstantTensor>{
        MakeConstant<int64_t>({}, {final_value}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
        MakeConstant<int64_t>({trip_count}, std::move(scan), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
    };
  }
  return std::nullopt;
}

std::optional<ConstantTensor> EvalConstant(const ConstantValueMap& /*constants*/,
                                            Ort::ConstNode node) {
  // Constant ops usually carry a `value` tensor attribute; other attr forms
  // (value_float, value_int, value_ints, ...) are rare in practice, so we only
  // handle the tensor form here.
  Ort::ConstOpAttr attr{nullptr};
  const Ort::Status status = node.GetAttributeByName("value", attr);
  if (!status.IsOK() || attr == nullptr) return std::nullopt;
  Ort::Value value{nullptr};
  const Ort::Status vstatus = attr.GetTensorAttributeAsOrtValue(value);
  if (!vstatus.IsOK() || !value) return std::nullopt;
  const auto tshape = value.GetTensorTypeAndShapeInfo();
  const ONNXTensorElementDataType et = tshape.GetElementType();
  if (!IsFoldableDType(et)) return std::nullopt;
  const std::vector<int64_t> dims = tshape.GetShape();
  if (!FitsFoldLimit(dims)) return std::nullopt;
  const size_t count = ElementCount(dims);
  const size_t elem = ByteWidth(et);
  const void* src = nullptr;
  THROW_ON_ERROR(GetOrtApi().GetTensorData(value, &src));
  GGONNX_NOT_NULL(src, "ORT returned null Constant tensor data");
  ConstantTensor out;
  out.element_type = et;
  out.dims = dims;
  out.data.resize(count * elem);
  if (count > 0) std::memcpy(out.data.data(), src, count * elem);
  return out;
}

std::optional<ConstantTensor> EvalTranspose(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 1 || inputs[0] == nullptr) return std::nullopt;
  const auto data = LookupConstant(constants, inputs[0]);
  if (!data) return std::nullopt;
  const int64_t rank = static_cast<int64_t>(data->dims.size());
  std::vector<int64_t> perm(rank);
  Ort::ConstOpAttr attr{nullptr};
  const Ort::Status status = node.GetAttributeByName("perm", attr);
  if (status.IsOK() && attr != nullptr) {
    Ort::ThrowOnError(attr.GetValueArray(perm));
    if (static_cast<int64_t>(perm.size()) != rank) return std::nullopt;
  } else {
    for (int64_t i = 0; i < rank; ++i) perm[static_cast<size_t>(i)] = rank - 1 - i;
  }
  std::vector<int64_t> out_dims(rank);
  for (int64_t i = 0; i < rank; ++i) {
    const int64_t p = perm[static_cast<size_t>(i)];
    if (p < 0 || p >= rank) return std::nullopt;
    out_dims[static_cast<size_t>(i)] = data->dims[static_cast<size_t>(p)];
  }
  if (!FitsFoldLimit(out_dims)) return std::nullopt;

  const size_t elem = ByteWidth(data->element_type);
  const size_t count = ElementCount(out_dims);
  const auto in_strides = MakeStrides(data->dims);
  const auto out_strides = MakeStrides(out_dims);

  ConstantTensor out;
  out.element_type = data->element_type;
  out.dims = out_dims;
  out.data.resize(count * elem);
  std::vector<int64_t> coord(rank, 0);
  for (size_t linear = 0; linear < count; ++linear) {
    int64_t src_linear = 0;
    for (int64_t i = 0; i < rank; ++i) {
      src_linear += coord[static_cast<size_t>(i)] * in_strides[static_cast<size_t>(perm[static_cast<size_t>(i)])];
    }
    std::memcpy(out.data.data() + linear * elem,
                data->data.data() + static_cast<size_t>(src_linear) * elem, elem);
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      coord[static_cast<size_t>(axis)]++;
      if (coord[static_cast<size_t>(axis)] < out_dims[static_cast<size_t>(axis)]) break;
      coord[static_cast<size_t>(axis)] = 0;
    }
  }
  return out;
}

std::optional<ConstantTensor> EvalGather(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 2 || inputs[0] == nullptr || inputs[1] == nullptr) return std::nullopt;
  const auto data = LookupConstant(constants, inputs[0]);
  const auto indices = LookupConstant(constants, inputs[1]);
  if (!data || !indices) return std::nullopt;
  if (indices->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 &&
      indices->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    return std::nullopt;
  }
  const int64_t data_rank = static_cast<int64_t>(data->dims.size());
  if (data_rank == 0) return std::nullopt;

  int64_t axis = 0;
  if (auto attr = [&]() -> std::optional<int64_t> {
        Ort::ConstOpAttr a{nullptr};
        const Ort::Status s = node.GetAttributeByName("axis", a);
        if (!s.IsOK() || a == nullptr) return std::nullopt;
        int64_t v = 0;
        Ort::ThrowOnError(a.GetValue(v));
        return v;
      }()) {
    axis = *attr;
  }
  if (axis < 0) axis += data_rank;
  if (axis < 0 || axis >= data_rank) return std::nullopt;

  std::vector<int64_t> idx_values;
  if (indices->element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    idx_values = ReadTypedData<int64_t>(*indices);
  } else {
    const auto v32 = ReadTypedData<int32_t>(*indices);
    idx_values.assign(v32.begin(), v32.end());
  }
  const int64_t axis_len = data->dims[static_cast<size_t>(axis)];
  for (int64_t& idx : idx_values) {
    if (idx < 0) idx += axis_len;
    if (idx < 0 || idx >= axis_len) return std::nullopt;
  }

  std::vector<int64_t> out_dims;
  out_dims.reserve(data->dims.size() + indices->dims.size() - 1);
  for (int64_t i = 0; i < axis; ++i) out_dims.push_back(data->dims[static_cast<size_t>(i)]);
  for (int64_t d : indices->dims) out_dims.push_back(d);
  for (int64_t i = axis + 1; i < data_rank; ++i) out_dims.push_back(data->dims[static_cast<size_t>(i)]);
  if (!FitsFoldLimit(out_dims)) return std::nullopt;

  // Element stride along the gather axis.
  int64_t inner = 1;
  for (int64_t i = axis + 1; i < data_rank; ++i) inner *= data->dims[static_cast<size_t>(i)];
  int64_t outer = 1;
  for (int64_t i = 0; i < axis; ++i) outer *= data->dims[static_cast<size_t>(i)];
  const size_t elem_size = ByteWidth(data->element_type);
  const size_t inner_bytes = static_cast<size_t>(inner) * elem_size;

  ConstantTensor out;
  out.element_type = data->element_type;
  out.dims = std::move(out_dims);
  out.data.resize(ElementCount(out.dims) * elem_size);
  uint8_t* dst = out.data.data();
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t k : idx_values) {
      const size_t src_offset =
          (static_cast<size_t>(o) * static_cast<size_t>(axis_len) + static_cast<size_t>(k)) *
          inner_bytes;
      std::memcpy(dst, data->data.data() + src_offset, inner_bytes);
      dst += inner_bytes;
    }
  }
  return out;
}

std::optional<ConstantTensor> EvalConstantOfShape(const ConstantValueMap& constants,
                                                  Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 1 || inputs[0] == nullptr) return std::nullopt;
  const auto shape_tensor = LookupConstant(constants, inputs[0]);
  if (!shape_tensor || shape_tensor->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    return std::nullopt;
  }
  const auto shape_values = ReadTypedData<int64_t>(*shape_tensor);
  for (int64_t d : shape_values) {
    if (d < 0) return std::nullopt;
  }
  if (!FitsFoldLimit(shape_values)) return std::nullopt;

  // Default value is float 0. An optional "value" attribute supplies a single-
  // element tensor whose dtype determines the output dtype.
  ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  std::vector<uint8_t> scalar_bytes(sizeof(float), 0);

  Ort::ConstOpAttr attr{nullptr};
  const Ort::Status status = node.GetAttributeByName("value", attr);
  if (status.IsOK() && attr != nullptr) {
    Ort::Value value{nullptr};
    const Ort::Status v_status = attr.GetTensorAttributeAsOrtValue(value);
    if (v_status.IsOK() && value) {
      const auto tshape = value.GetTensorTypeAndShapeInfo();
      const ONNXTensorElementDataType et = tshape.GetElementType();
      if (!IsFoldableDType(et)) return std::nullopt;
      element_type = et;
      const void* src = nullptr;
      THROW_ON_ERROR(GetOrtApi().GetTensorData(value, &src));
      GGONNX_NOT_NULL(src, "ORT returned null ConstantOfShape value data");
      const size_t elem = ByteWidth(element_type);
      scalar_bytes.assign(static_cast<const uint8_t*>(src),
                          static_cast<const uint8_t*>(src) + elem);
    }
  }

  const size_t elem_size = ByteWidth(element_type);
  const size_t count = ElementCount(shape_values);
  ConstantTensor out;
  out.element_type = element_type;
  out.dims = shape_values;
  out.data.resize(count * elem_size);
  for (size_t i = 0; i < count; ++i) {
    std::memcpy(out.data.data() + i * elem_size, scalar_bytes.data(), elem_size);
  }
  return out;
}

template <typename T>
std::optional<ConstantTensor> EvalEqualTyped(const ConstantTensor& lhs, const ConstantTensor& rhs) {
  const size_t lhs_count = ElementCount(lhs.dims);
  const size_t rhs_count = ElementCount(rhs.dims);
  const bool lhs_scalar = lhs_count == 1;
  const bool rhs_scalar = rhs_count == 1;
  if (!lhs_scalar && !rhs_scalar && lhs.dims != rhs.dims) return std::nullopt;
  const std::vector<int64_t> out_dims = lhs_scalar ? rhs.dims : lhs.dims;
  const size_t out_count = std::max(lhs_count, rhs_count);
  if (!FitsFoldLimit(out_dims)) return std::nullopt;
  const auto a = ReadTypedData<T>(lhs);
  const auto b = ReadTypedData<T>(rhs);
  std::vector<uint8_t> out(out_count);
  for (size_t i = 0; i < out_count; ++i) {
    out[i] = a[lhs_scalar ? 0 : i] == b[rhs_scalar ? 0 : i] ? 1u : 0u;
  }
  ConstantTensor tensor;
  tensor.element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  tensor.dims = out_dims;
  tensor.data.resize(out.size());
  std::memcpy(tensor.data.data(), out.data(), out.size());
  return tensor;
}

std::optional<ConstantTensor> EvalEqual(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 2 || inputs[0] == nullptr || inputs[1] == nullptr) return std::nullopt;
  const auto lhs = LookupConstant(constants, inputs[0]);
  const auto rhs = LookupConstant(constants, inputs[1]);
  if (!lhs || !rhs || lhs->element_type != rhs->element_type) return std::nullopt;
  switch (lhs->element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return EvalEqualTyped<int32_t>(*lhs, *rhs);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return EvalEqualTyped<int64_t>(*lhs, *rhs);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return EvalEqualTyped<float>(*lhs, *rhs);
    default: return std::nullopt;
  }
}

std::optional<ConstantTensor> EvalWhere(const ConstantValueMap& constants, Ort::ConstNode node) {
  const auto inputs = node.GetInputs();
  if (inputs.size() != 3 || inputs[0] == nullptr || inputs[1] == nullptr || inputs[2] == nullptr) {
    return std::nullopt;
  }
  const auto cond = LookupConstant(constants, inputs[0]);
  const auto lhs = LookupConstant(constants, inputs[1]);
  const auto rhs = LookupConstant(constants, inputs[2]);
  if (!cond || !lhs || !rhs) return std::nullopt;
  if (cond->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) return std::nullopt;
  if (lhs->element_type != rhs->element_type) return std::nullopt;
  // Limit broadcasting: all three equal-shape, or any scalar.
  const size_t c = ElementCount(cond->dims);
  const size_t l = ElementCount(lhs->dims);
  const size_t r = ElementCount(rhs->dims);
  const size_t out_count = std::max({c, l, r});
  std::vector<int64_t> out_dims;
  if (c == out_count) out_dims = cond->dims;
  else if (l == out_count) out_dims = lhs->dims;
  else out_dims = rhs->dims;
  auto pick = [&](size_t i, const ConstantTensor& t, size_t n) -> size_t {
    return n == 1 ? 0 : i;
  };
  if (!FitsFoldLimit(out_dims)) return std::nullopt;

  const size_t elem = ByteWidth(lhs->element_type);
  ConstantTensor out;
  out.element_type = lhs->element_type;
  out.dims = out_dims;
  out.data.resize(out_count * elem);
  for (size_t i = 0; i < out_count; ++i) {
    const uint8_t cval = cond->data[pick(i, *cond, c)];
    const uint8_t* src = (cval ? lhs->data.data() + pick(i, *lhs, l) * elem
                               : rhs->data.data() + pick(i, *rhs, r) * elem);
    std::memcpy(out.data.data() + i * elem, src, elem);
  }
  return out;
}

std::optional<ConstantTensor> TryFoldNode(const ConstantValueMap& constants, Ort::ConstNode node) {
  const std::string op_type = node.GetOperatorType();
  if (op_type == "Constant") return EvalConstant(constants, node);
  if (op_type == "Shape") return EvalShape(constants, node);
  if (op_type == "Cast") return EvalCast(constants, node);
  if (op_type == "Identity") return EvalIdentity(constants, node);
  if (op_type == "Slice") return EvalSlice(constants, node);
  if (op_type == "Unsqueeze") return EvalUnsqueeze(constants, node);
  if (op_type == "Squeeze") return EvalSqueeze(constants, node);
  if (op_type == "Concat") return EvalConcat(constants, node);
  if (op_type == "Reshape") return EvalReshape(constants, node);
  if (op_type == "Tile") return EvalTile(constants, node);
  if (op_type == "Add" || op_type == "Sub" || op_type == "Mul" || op_type == "Div") {
    return EvalBinary(constants, node);
  }
  if (op_type == "Transpose") return EvalTranspose(constants, node);
  if (op_type == "Gather") return EvalGather(constants, node);
  if (op_type == "ConstantOfShape") return EvalConstantOfShape(constants, node);
  if (op_type == "Equal") return EvalEqual(constants, node);
  if (op_type == "Where") return EvalWhere(constants, node);
  return std::nullopt;
}

}  // namespace

std::string NodeKey(Ort::ConstNode node) {
  const auto outputs = node.GetOutputs();
  for (auto output : outputs) {
    if (output != nullptr) return output.GetName();
  }
  const std::string name = node.GetName();
  if (!name.empty()) return name;
  return node.GetOperatorType();
}

namespace {

std::optional<std::vector<int64_t>> ReadTransposePerm(Ort::ConstNode node) {
  Ort::ConstOpAttr attr{nullptr};
  const Ort::Status status = node.GetAttributeByName("perm", attr);
  if (!status.IsOK() || attr == nullptr) return std::nullopt;
  std::vector<int64_t> values;
  Ort::ThrowOnError(attr.GetValueArray(values));
  return values;
}

// Resolve a Reshape target tensor (with 0 and -1 placeholders) against the
// input's concrete dims. Returns nullopt if resolution would leave dynamic
// entries. Element-count conservation is not enforced here — the ORT graph
// already guarantees it at export time for valid models.
std::optional<std::vector<int64_t>> ResolveReshapeTarget(const std::vector<int64_t>& target,
                                                         const std::vector<int64_t>& input_dims) {
  std::vector<int64_t> out = target;
  int64_t infer_axis = -1;
  int64_t known = 1;
  int64_t total = 1;
  for (int64_t d : input_dims) {
    if (d < 0) return std::nullopt;
    total *= d;
  }
  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] == 0) {
      if (i >= input_dims.size() || input_dims[i] < 0) return std::nullopt;
      out[i] = input_dims[i];
    }
    if (out[i] == -1) {
      if (infer_axis >= 0) return std::nullopt;
      infer_axis = static_cast<int64_t>(i);
      continue;
    }
    if (out[i] <= 0) return std::nullopt;
    known *= out[i];
  }
  if (infer_axis >= 0) {
    if (known == 0) return std::nullopt;
    out[static_cast<size_t>(infer_axis)] = total / known;
  }
  return out;
}

// Detects any Reshape(4D->XD)->Transpose->Reshape(XD->4D) triple where X >
// GGML_MAX_DIMS and the permutation coalesces to ≤GGML_MAX_DIMS axis groups.
//
// Coalescing rule: consecutive ONNX input axes i and i+1 can be merged into
// one group when they remain adjacent in the output with the same order, i.e.
// inv_perm[i+1] == inv_perm[i]+1. We greedily merge maximal such runs, giving
// N_groups groups. If N_groups ≤ GGML_MAX_DIMS each group is flattened to one
// dim and the inter-group reorder becomes a plain rank-4 ggml_permute.
//
// The GGML permutation formula (for GGML axis j, fastest-varying first):
//   ONNX output group out_g = N_groups-1-j
//   ONNX input group  in_g  = coalesced_perm[out_g]
//   ggml_perm[j] = N_groups-1 - coalesced_perm[N_groups-1-j]
void DetectShuffleTripleFusions(const Ort::ConstGraph& ort_graph,
                                const ConstantValueMap& constants,
                                FusionPlan& plan) {
  std::unordered_map<std::string, Ort::ConstNode> producer;
  std::unordered_map<std::string, int> consumer_count;
  const auto nodes = ort_graph.GetNodes();
  for (Ort::ConstNode n : nodes) {
    for (Ort::ConstValueInfo out : n.GetOutputs()) {
      if (out != nullptr) producer[std::string(out.GetName())] = n;
    }
    for (Ort::ConstValueInfo in : n.GetInputs()) {
      if (in != nullptr) consumer_count[std::string(in.GetName())]++;
    }
  }
  for (Ort::ConstValueInfo out : ort_graph.GetOutputs()) {
    if (out != nullptr) consumer_count[std::string(out.GetName())]++;
  }

  std::unordered_set<std::string> claimed;
  auto claim = [&](const std::string& key) {
    return claimed.insert(key).second;
  };

  for (Ort::ConstNode trans : nodes) {
    if (trans.GetOperatorType() != "Transpose") continue;
    const auto perm_opt = ReadTransposePerm(trans);
    if (!perm_opt) continue;
    const std::vector<int64_t>& perm = *perm_opt;
    const int X = static_cast<int>(perm.size());
    if (X <= GGML_MAX_DIMS) continue;  // individual ops already handle rank ≤4

    const auto trans_inputs = trans.GetInputs();
    const auto trans_outputs = trans.GetOutputs();
    if (trans_inputs.size() != 1 || trans_outputs.size() != 1) continue;
    if (trans_inputs[0] == nullptr || trans_outputs[0] == nullptr) continue;

    const std::string r1_out = std::string(trans_inputs[0].GetName());
    auto prod_it = producer.find(r1_out);
    if (prod_it == producer.end()) continue;
    Ort::ConstNode r1 = prod_it->second;
    if (r1.GetOperatorType() != "Reshape") continue;
    if (consumer_count[r1_out] != 1) continue;

    const std::string trans_out_name = std::string(trans_outputs[0].GetName());
    if (consumer_count[trans_out_name] != 1) continue;

    Ort::ConstNode r2{nullptr};
    for (Ort::ConstNode n : nodes) {
      for (Ort::ConstValueInfo in : n.GetInputs()) {
        if (in != nullptr && std::string(in.GetName()) == trans_out_name) {
          r2 = n;
          break;
        }
      }
      if (r2 != nullptr) break;
    }
    if (r2 == nullptr || r2.GetOperatorType() != "Reshape") continue;

    const auto r1_inputs = r1.GetInputs();
    const auto r2_inputs = r2.GetInputs();
    const auto r2_outputs = r2.GetOutputs();
    if (r1_inputs.size() != 2 || r2_inputs.size() != 2 || r2_outputs.size() != 1) continue;
    if (r1_inputs[0] == nullptr || r1_inputs[1] == nullptr ||
        r2_inputs[0] == nullptr || r2_inputs[1] == nullptr || r2_outputs[0] == nullptr) {
      continue;
    }

    // r1 input must be fully static with rank ≤ GGML_MAX_DIMS.
    const TensorMetadata r1_in_meta = getTensorMetadata(r1_inputs[0]);
    if (!shapeIsFullyStatic(r1_in_meta) ||
        r1_in_meta.dims.size() > static_cast<size_t>(GGML_MAX_DIMS)) continue;

    // r1 target must resolve to exactly rank X (the intermediate rank).
    auto r1_target = ReadConstantInt64Input(constants, r1, 1);
    if (!r1_target || static_cast<int>(r1_target->size()) != X) continue;
    auto r1_resolved = ResolveReshapeTarget(*r1_target, r1_in_meta.dims);
    if (!r1_resolved || static_cast<int>(r1_resolved->size()) != X) continue;
    bool any_nonpos = false;
    for (int64_t d : *r1_resolved) { if (d <= 0) { any_nonpos = true; break; } }
    if (any_nonpos) continue;

    // Compute inv_perm.
    std::vector<int> inv_perm(X);
    for (int j = 0; j < X; ++j) inv_perm[static_cast<int>(perm[j])] = j;

    // Greedily merge consecutive input axes that remain adjacent post-transpose.
    // Each group is a contiguous run [start..end) of input axes.
    struct Group { int start; int end; };
    std::vector<Group> groups;
    groups.push_back({0, 1});
    for (int i = 1; i < X; ++i) {
      if (inv_perm[i] == inv_perm[i - 1] + 1) {
        groups.back().end = i + 1;
      } else {
        groups.push_back({i, i + 1});
      }
    }
    const int N_groups = static_cast<int>(groups.size());
    if (N_groups > GGML_MAX_DIMS) continue;

    // Compute group product dims (ONNX order, slowest first).
    std::vector<int64_t> group_onnx_dims(N_groups);
    for (int g = 0; g < N_groups; ++g) {
      int64_t prod = 1;
      for (int ax = groups[g].start; ax < groups[g].end; ++ax)
        prod *= (*r1_resolved)[ax];
      group_onnx_dims[g] = prod;
    }

    // Determine where each input group lands in the output.
    // The first axis of group g goes to output position inv_perm[groups[g].start].
    // Sort groups by that position to get the coalesced ONNX perm:
    //   coalesced_perm[out_g] = in_g (input group that produces output group out_g).
    std::vector<int> group_out_pos(N_groups);
    for (int g = 0; g < N_groups; ++g) group_out_pos[g] = inv_perm[groups[g].start];
    std::vector<int> order(N_groups);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return group_out_pos[a] < group_out_pos[b]; });
    // order[out_g] = in_g
    std::vector<int> coalesced_perm(N_groups);
    for (int out_g = 0; out_g < N_groups; ++out_g) coalesced_perm[out_g] = order[out_g];

    // Convert to GGML axis order (reversed from ONNX, padded to GGML_MAX_DIMS).
    // ggml_perm[j] = N_groups-1 - coalesced_perm[N_groups-1-j]
    NodeDesc::GenericShuffleAttrs attrs;
    attrs.grouped_ggml_dims.fill(1);
    attrs.ggml_perm = {0, 1, 2, 3};
    for (int g = 0; g < N_groups; ++g) {
      int ggml_axis = N_groups - 1 - g;  // ONNX group g → GGML axis (N-1-g)
      attrs.grouped_ggml_dims[ggml_axis] = group_onnx_dims[g];
    }
    for (int j = 0; j < N_groups; ++j) {
      attrs.ggml_perm[j] = N_groups - 1 - coalesced_perm[N_groups - 1 - j];
    }

    // Compute r2's input dims from the transposed intermediate shape.
    std::vector<int64_t> r2_in_dims(X);
    for (int j = 0; j < X; ++j) r2_in_dims[j] = (*r1_resolved)[static_cast<int>(perm[j])];

    auto r2_target = ReadConstantInt64Input(constants, r2, 1);
    std::optional<std::vector<int64_t>> r2_resolved;
    if (r2_target) {
      r2_resolved = ResolveReshapeTarget(*r2_target, r2_in_dims);
    } else {
      const TensorMetadata r2_out_meta = getTensorMetadata(r2_outputs[0]);
      if (shapeIsFullyStatic(r2_out_meta)) r2_resolved = r2_out_meta.dims;
    }
    if (!r2_resolved || r2_resolved->empty() ||
        r2_resolved->size() > static_cast<size_t>(GGML_MAX_DIMS)) continue;
    for (int64_t d : *r2_resolved) { if (d <= 0) { r2_resolved.reset(); break; } }
    if (!r2_resolved) continue;

    auto prod_fn = [](const std::vector<int64_t>& v) {
      int64_t p = 1; for (int64_t d : v) p *= d; return p;
    };
    if (prod_fn(*r1_resolved) != prod_fn(r1_in_meta.dims) ||
        prod_fn(*r2_resolved) != prod_fn(*r1_resolved)) {
      continue;
    }

    const std::string r1_key = NodeKey(r1);
    const std::string r2_key = NodeKey(r2);
    const std::string anchor_key = NodeKey(trans);
    if (!claim(r1_key) || !claim(r2_key) || !claim(anchor_key)) continue;

    attrs.output_onnx_dims = *r2_resolved;

    FusionPlan::AnchorIO io;
    io.input_values.push_back(std::string(r1_inputs[0].GetName()));
    io.output_values.push_back(std::string(r2_outputs[0].GetName()));
    io.anchor_node_name = std::string(trans.GetName());

    plan.generic_shuffle_anchors.emplace(anchor_key, std::move(attrs));
    plan.anchor_io.emplace(anchor_key, std::move(io));
    plan.consumed_nodes.insert(r1_key);
    plan.consumed_nodes.insert(r2_key);
  }
}

std::optional<int64_t> ReadInt64Attr(Ort::ConstNode node, const char* name) {
  Ort::ConstOpAttr attr{nullptr};
  const Ort::Status status = node.GetAttributeByName(name, attr);
  if (!status.IsOK() || attr == nullptr) return std::nullopt;
  int64_t value{};
  Ort::ThrowOnError(attr.GetValue(value));
  return value;
}

bool IsNodeCompilable(Ort::ConstNode node, const ConstantValueMap& constants) {
  return get_node_support(node, &constants).has_value();
}

void DetectQKVSplitFusions(const Ort::ConstGraph& ort_graph,
                           const ConstantValueMap& constants,
                           const std::unordered_set<std::string>& folded_nodes,
                           FusionPlan& plan) {
  // Matches the pre-L2 attention QKV split that ORT's CPU EP would later fuse
  // into a Split kernel. Our EP sees the raw form:
  //   Reshape(x=[..., 3C] -> [B*nw, M*M, 3, heads, head_dim]) ->
  //     Transpose(perm=[2,0,3,1,4]) ->
  //     3x Gather(axis=0, scalar index in {0,1,2})
  // Every rank-5 intermediate has exactly one consumer, so the fusion is safe
  // to discard them. Emits three rank-4 ne=[head_dim, M*M, heads, B*nw].
  std::unordered_map<std::string, Ort::ConstNode> producer;
  std::unordered_map<std::string, int> consumer_count;
  const auto nodes = ort_graph.GetNodes();
  for (Ort::ConstNode n : nodes) {
    for (Ort::ConstValueInfo out : n.GetOutputs()) {
      if (out != nullptr) producer[std::string(out.GetName())] = n;
    }
    for (Ort::ConstValueInfo in : n.GetInputs()) {
      if (in != nullptr) consumer_count[std::string(in.GetName())]++;
    }
  }
  for (Ort::ConstValueInfo out : ort_graph.GetOutputs()) {
    if (out != nullptr) consumer_count[std::string(out.GetName())]++;
  }

  // Topological index per node key — used to verify the fusion span has no
  // intervening unsupported node that would force ORT to split the partition.
  std::unordered_map<std::string, size_t> index_by_key;
  for (size_t i = 0; i < nodes.size(); ++i) {
    index_by_key.emplace(NodeKey(nodes[i]), i);
  }

  std::unordered_set<std::string> claimed;
  auto claim = [&](const std::string& key) { return claimed.insert(key).second; };

  for (Ort::ConstNode trans : nodes) {
    if (trans.GetOperatorType() != "Transpose") continue;
    const auto perm = ReadTransposePerm(trans);
    const std::vector<int64_t> expected_perm = {2, 0, 3, 1, 4};
    if (!perm || *perm != expected_perm) continue;

    const auto trans_inputs = trans.GetInputs();
    const auto trans_outputs = trans.GetOutputs();
    if (trans_inputs.size() != 1 || trans_outputs.size() != 1) continue;
    if (trans_inputs[0] == nullptr || trans_outputs[0] == nullptr) continue;

    // Upstream Reshape producing the rank-5 [B*nw, M*M, 3, heads, head_dim].
    const std::string r_out_name = std::string(trans_inputs[0].GetName());
    if (consumer_count[r_out_name] != 1) continue;
    auto r_it = producer.find(r_out_name);
    if (r_it == producer.end()) continue;
    Ort::ConstNode reshape = r_it->second;
    if (reshape.GetOperatorType() != "Reshape") continue;

    const auto r_inputs = reshape.GetInputs();
    const auto r_outputs = reshape.GetOutputs();
    if (r_inputs.size() != 2 || r_outputs.size() != 1) continue;
    if (r_inputs[0] == nullptr || r_outputs[0] == nullptr) continue;

    const TensorMetadata r_in_meta = getTensorMetadata(r_inputs[0]);
    if (!shapeIsFullyStatic(r_in_meta)) continue;

    std::vector<int64_t> r_target_dims;
    if (auto r_target = ReadConstantInt64Input(constants, reshape, 1)) {
      auto resolved = ResolveReshapeTarget(*r_target, r_in_meta.dims);
      if (!resolved) continue;
      r_target_dims = *resolved;
    } else {
      const TensorMetadata r_out_meta = getTensorMetadata(r_outputs[0]);
      if (!shapeIsFullyStatic(r_out_meta)) continue;
      r_target_dims = r_out_meta.dims;
    }
    if (r_target_dims.size() != 5) continue;
    for (int64_t d : r_target_dims) {
      if (d <= 0) { r_target_dims.clear(); break; }
    }
    if (r_target_dims.empty()) continue;
    if (r_target_dims[2] != 3) continue;

    // Downstream: exactly three consumers of the Transpose output, each a
    // Gather(axis=0) with a scalar int64 index in {0,1,2} — one per Q/K/V.
    const std::string trans_out_name = std::string(trans_outputs[0].GetName());
    if (consumer_count[trans_out_name] != 3) continue;

    std::array<Ort::ConstNode, 3> gathers{Ort::ConstNode{nullptr}, Ort::ConstNode{nullptr},
                                          Ort::ConstNode{nullptr}};
    bool seen[3] = {false, false, false};
    bool ok = true;
    for (Ort::ConstNode n : nodes) {
      bool consumes = false;
      for (Ort::ConstValueInfo in : n.GetInputs()) {
        if (in != nullptr && std::string(in.GetName()) == trans_out_name) {
          consumes = true;
          break;
        }
      }
      if (!consumes) continue;
      if (n.GetOperatorType() != "Gather") { ok = false; break; }
      const int64_t gaxis = ReadInt64Attr(n, "axis").value_or(0);
      if (gaxis != 0) { ok = false; break; }
      const auto g_inputs = n.GetInputs();
      const auto g_outputs = n.GetOutputs();
      if (g_inputs.size() != 2 || g_outputs.size() != 1) { ok = false; break; }
      const auto idx = ReadConstantInt64Input(constants, n, 1);
      if (!idx || idx->size() != 1) { ok = false; break; }
      const int64_t which = (*idx)[0];
      if (which < 0 || which > 2 || seen[which]) { ok = false; break; }
      seen[which] = true;
      gathers[static_cast<size_t>(which)] = n;
    }
    if (!ok || !seen[0] || !seen[1] || !seen[2]) continue;

    const std::string r_key = NodeKey(reshape);
    const std::string t_key = NodeKey(trans);
    const std::string g0_key = NodeKey(gathers[0]);
    const std::string g1_key = NodeKey(gathers[1]);
    const std::string g2_key = NodeKey(gathers[2]);

    // Partition safety: the fusion members (R, T, G0, G1, G2) must end up in
    // the same fused partition. ORT's grouping is topological, so the span
    // from min to max index must contain only nodes that ggml will keep in
    // the partition — i.e. compilable (supported), folded at compile time, or
    // already in the fusion itself. An unsupported node (e.g. a Softmax we
    // haven't implemented) sitting between G0 and G2 would split the
    // partition and leave the rank-5 Transpose_2 output exposed as a cross-
    // partition boundary, which ensure_value then rejects.
    const std::array<std::string, 5> member_keys{r_key, t_key, g0_key, g1_key, g2_key};
    std::unordered_set<std::string> member_set(member_keys.begin(), member_keys.end());
    size_t lo = std::numeric_limits<size_t>::max();
    size_t hi = 0;
    for (const std::string& k : member_keys) {
      const auto it = index_by_key.find(k);
      if (it == index_by_key.end()) { lo = 1; hi = 0; break; }
      lo = std::min(lo, it->second);
      hi = std::max(hi, it->second);
    }
    if (lo > hi) continue;
    bool span_ok = true;
    for (size_t i = lo; i <= hi; ++i) {
      const std::string key = NodeKey(nodes[i]);
      if (member_set.count(key) > 0) continue;
      if (folded_nodes.count(key) > 0) continue;
      // Nodes claimed by an earlier fusion (anchors or consumed) stay in the
      // partition just like compilable ones.
      if (plan.consumed_nodes.count(key) > 0) continue;
      if (plan.generic_shuffle_anchors.count(key) > 0) continue;
      if (plan.qkv_split_anchors.count(key) > 0) continue;
      if (plan.window_mask_add_anchors.count(key) > 0) continue;
      if (IsNodeCompilable(nodes[i], constants)) continue;
      span_ok = false;
      break;
    }
    if (!span_ok) continue;

    if (!claim(r_key) || !claim(t_key) || !claim(g0_key) || !claim(g1_key) || !claim(g2_key)) {
      continue;
    }

    NodeDesc::QKVSplitAttrs attrs;
    attrs.num_batch = r_target_dims[0];
    attrs.num_tokens = r_target_dims[1];
    attrs.num_heads = r_target_dims[3];
    attrs.head_dim = r_target_dims[4];

    // Anchor on the Transpose — it's the unique node in the pattern (the
    // Gathers don't share keys). Synthesizing outputs in Q/K/V order requires
    // indexing the `gathers` array by the scalar index value.
    const std::string anchor_key = t_key;

    FusionPlan::AnchorIO io;
    io.input_values.push_back(std::string(r_inputs[0].GetName()));
    for (size_t i = 0; i < 3; ++i) {
      io.output_values.push_back(std::string(gathers[i].GetOutputs()[0].GetName()));
    }
    io.anchor_node_name = std::string(trans.GetName());

    plan.qkv_split_anchors.emplace(anchor_key, std::move(attrs));
    plan.anchor_io.emplace(anchor_key, std::move(io));
    plan.consumed_nodes.insert(r_key);
    plan.consumed_nodes.insert(g0_key);
    plan.consumed_nodes.insert(g1_key);
    plan.consumed_nodes.insert(g2_key);
  }
}

void DetectWindowMaskAddFusions(const Ort::ConstGraph& ort_graph,
                                const ConstantValueMap& constants,
                                const std::unordered_set<std::string>& folded_nodes,
                                FusionPlan& plan) {
  // Matches the SW-MSA mask-apply round-trip:
  //   Reshape(X: rank<=4 -> rank>4) -> Add(X', constant_mask) -> Reshape(back)
  // where the trailing Reshape's output shape equals the leading Reshape's
  // input shape. Because the mask is a compile-time constant, we can drop its
  // leading size-1 ONNX axes to rank-reduce it to <=4 without moving any
  // bytes; the Add then becomes a plain rank-4 broadcast add directly on X.
  std::unordered_map<std::string, Ort::ConstNode> producer;
  std::unordered_map<std::string, int> consumer_count;
  const auto nodes = ort_graph.GetNodes();
  for (Ort::ConstNode n : nodes) {
    for (Ort::ConstValueInfo out : n.GetOutputs()) {
      if (out != nullptr) producer[std::string(out.GetName())] = n;
    }
    for (Ort::ConstValueInfo in : n.GetInputs()) {
      if (in != nullptr) consumer_count[std::string(in.GetName())]++;
    }
  }
  for (Ort::ConstValueInfo out : ort_graph.GetOutputs()) {
    if (out != nullptr) consumer_count[std::string(out.GetName())]++;
  }

  std::unordered_map<std::string, size_t> index_by_key;
  for (size_t i = 0; i < nodes.size(); ++i) {
    index_by_key.emplace(NodeKey(nodes[i]), i);
  }

  std::unordered_set<std::string> claimed;
  auto claim = [&](const std::string& key) { return claimed.insert(key).second; };

  for (Ort::ConstNode add : nodes) {
    if (add.GetOperatorType() != "Add") continue;
    const auto add_inputs = add.GetInputs();
    const auto add_outputs = add.GetOutputs();
    if (add_inputs.size() != 2 || add_outputs.size() != 1) continue;
    if (add_inputs[0] == nullptr || add_inputs[1] == nullptr || add_outputs[0] == nullptr) continue;

    // One side must be the rank-5 Reshape output, the other a constant mask.
    // The mask can be in either slot — probe both orderings.
    for (size_t score_slot = 0; score_slot < 2; ++score_slot) {
      const size_t mask_slot = 1 - score_slot;
      Ort::ConstValueInfo score_vi = add_inputs[score_slot];
      Ort::ConstValueInfo mask_vi = add_inputs[mask_slot];

      // Mask must be a tracked compile-time constant.
      const std::string mask_name = std::string(mask_vi.GetName());
      auto mask_it = constants.find(mask_name);
      if (mask_it == constants.end()) continue;
      if (mask_it->second.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) continue;
      std::vector<int64_t> mask_dims = mask_it->second.dims;

      // Leading Reshape must produce the score side.
      const std::string lead_out_name = std::string(score_vi.GetName());
      if (consumer_count[lead_out_name] != 1) continue;
      auto prod_it = producer.find(lead_out_name);
      if (prod_it == producer.end()) continue;
      Ort::ConstNode lead = prod_it->second;
      if (lead.GetOperatorType() != "Reshape") continue;
      const auto lead_inputs = lead.GetInputs();
      if (lead_inputs.size() < 1 || lead_inputs[0] == nullptr) continue;

      const TensorMetadata lead_in_meta = getTensorMetadata(lead_inputs[0]);
      if (!shapeIsFullyStatic(lead_in_meta)) continue;
      if (!rankSupportedByGGML(lead_in_meta)) continue;  // X must already fit in ggml

      const TensorMetadata lead_out_meta = getTensorMetadata(score_vi);
      if (!shapeIsFullyStatic(lead_out_meta)) continue;
      if (rankSupportedByGGML(lead_out_meta)) continue;  // fusion only pays off at rank>4

      // Trailing Reshape must be the single consumer of the Add's output and
      // must land back at X's shape.
      const std::string add_out_name = std::string(add_outputs[0].GetName());
      if (consumer_count[add_out_name] != 1) continue;
      Ort::ConstNode trail{nullptr};
      for (Ort::ConstNode n : nodes) {
        for (Ort::ConstValueInfo in : n.GetInputs()) {
          if (in != nullptr && std::string(in.GetName()) == add_out_name) { trail = n; break; }
        }
        if (trail != nullptr) break;
      }
      if (trail == nullptr || trail.GetOperatorType() != "Reshape") continue;
      const auto trail_outputs = trail.GetOutputs();
      if (trail_outputs.size() != 1 || trail_outputs[0] == nullptr) continue;
      const TensorMetadata trail_out_meta = getTensorMetadata(trail_outputs[0]);
      if (!shapeIsFullyStatic(trail_out_meta)) continue;
      if (trail_out_meta.dims != lead_in_meta.dims) continue;

      // Rank-reduce the mask by dropping leading size-1 ONNX axes (these are
      // pure broadcast axes — dropping them preserves element ordering).
      std::vector<int64_t> reduced = mask_dims;
      while (reduced.size() > 4 && !reduced.empty() && reduced.front() == 1) {
        reduced.erase(reduced.begin());
      }
      if (reduced.size() > 4) continue;
      if (!broadcastSupportedByGGML(reduced, lead_in_meta.dims)) continue;

      // Partition-safety span check: only members, compilable nodes, and
      // folded nodes may sit between the three members in topological order.
      const std::string lead_key = NodeKey(lead);
      const std::string anchor_key = NodeKey(add);
      const std::string trail_key = NodeKey(trail);
      std::unordered_set<std::string> member_set{lead_key, anchor_key, trail_key};
      size_t lo = std::numeric_limits<size_t>::max(), hi = 0;
      bool ok_keys = true;
      for (const std::string& k : member_set) {
        auto it = index_by_key.find(k);
        if (it == index_by_key.end()) { ok_keys = false; break; }
        lo = std::min(lo, it->second);
        hi = std::max(hi, it->second);
      }
      if (!ok_keys) continue;
      bool span_ok = true;
      for (size_t i = lo; i <= hi; ++i) {
        const std::string key = NodeKey(nodes[i]);
        if (member_set.count(key) > 0) continue;
        if (folded_nodes.count(key) > 0) continue;
        if (IsNodeCompilable(nodes[i], constants)) continue;
        span_ok = false;
        break;
      }
      if (!span_ok) continue;

      if (!claim(lead_key) || !claim(anchor_key) || !claim(trail_key)) continue;

      FusionPlan::AnchorIO io;
      io.input_values.push_back(std::string(lead_inputs[0].GetName()));
      io.input_values.push_back(mask_name);
      io.output_values.push_back(std::string(trail_outputs[0].GetName()));
      io.anchor_node_name = std::string(add.GetName());

      plan.constant_override_dims.emplace(mask_name, reduced);
      plan.window_mask_add_anchors.emplace(anchor_key, std::move(reduced));
      plan.anchor_io.emplace(anchor_key, std::move(io));
      plan.consumed_nodes.insert(lead_key);
      plan.consumed_nodes.insert(trail_key);
      break;  // don't probe the swapped ordering once we've fused
    }
  }
}

// Reads the pads vector from a Pad node (attribute or constant input[1]).
static std::optional<std::vector<int64_t>> ReadPadNodePads(
    Ort::ConstNode node, const ConstantValueMap& constants) {
  // opset 9: pads is an int64 attribute.
  Ort::ConstOpAttr attr{nullptr};
  if (node.GetAttributeByName("pads", attr).IsOK() && attr != nullptr) {
    std::vector<int64_t> v;
    if (attr.GetValueArray(v).IsOK()) return v;
  }
  // opset 11+: pads is input[1].
  const auto inputs = node.GetInputs();
  if (inputs.size() >= 2 && inputs[1] != nullptr) {
    auto it = constants.find(std::string(inputs[1].GetName()));
    if (it != constants.end() &&
        it->second.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      const size_t count = it->second.data.size() / sizeof(int64_t);
      std::vector<int64_t> v(count);
      std::memcpy(v.data(), it->second.data.data(), it->second.data.size());
      return v;
    }
  }
  return std::nullopt;
}

// Folds zero-constant Pad → Conv pairs: the Pad's H/W padding is added to the
// Conv's p0/p1 and the Pad node is eliminated. Applies only when the Pad is
// constant-mode with value 0, pads only H/W symmetrically, and the Conv has
// no padding of its own and has exactly one consumer of the Pad's output.
void DetectPadConvFusions(const Ort::ConstGraph& ort_graph,
                          const ConstantValueMap& constants,
                          FusionPlan& plan) {
  // Build value-name → producing node map.
  std::unordered_map<std::string, Ort::ConstNode> producer;
  for (Ort::ConstNode n : ort_graph.GetNodes()) {
    for (Ort::ConstValueInfo out : n.GetOutputs()) {
      if (out != nullptr) producer.emplace(std::string(out.GetName()), n);
    }
  }

  // Build a use-count map for all value names.
  std::unordered_map<std::string, int> use_count;
  for (Ort::ConstNode n : ort_graph.GetNodes()) {
    for (Ort::ConstValueInfo inp : n.GetInputs()) {
      if (inp != nullptr) ++use_count[std::string(inp.GetName())];
    }
  }

  for (Ort::ConstNode conv : ort_graph.GetNodes()) {
    const std::string op = conv.GetOperatorType();
    if (op != "Conv") continue;
    const std::string dom = conv.GetDomain();
    if (!dom.empty() && dom != "ai.onnx") continue;

    const auto conv_inputs = conv.GetInputs();
    if (conv_inputs.empty() || conv_inputs[0] == nullptr) continue;

    // Conv must have no explicit padding (VALID or zero pads).
    const std::string auto_pad =
        [&]() -> std::string {
          Ort::ConstOpAttr a{nullptr};
          if (!conv.GetAttributeByName("auto_pad", a).IsOK() || a == nullptr) return "NOTSET";
          std::string s;
          a.GetValue(s);
          return s;
        }();
    if (auto_pad != "NOTSET" && auto_pad != "VALID") continue;
    bool conv_has_nonzero_pads = false;
    {
      Ort::ConstOpAttr a{nullptr};
      if (conv.GetAttributeByName("pads", a).IsOK() && a != nullptr) {
        std::vector<int64_t> cp;
        if (a.GetValueArray(cp).IsOK()) {
          for (int64_t p : cp) if (p != 0) { conv_has_nonzero_pads = true; break; }
        }
      }
    }
    if (conv_has_nonzero_pads) continue;

    // Find the Pad node producing the Conv's data input.
    const std::string data_input_name = conv_inputs[0].GetName();
    auto pit = producer.find(data_input_name);
    if (pit == producer.end()) continue;
    Ort::ConstNode pad = pit->second;
    if (std::string(pad.GetOperatorType()) != "Pad") continue;
    const std::string pad_dom = pad.GetDomain();
    if (!pad_dom.empty() && pad_dom != "ai.onnx") continue;

    // Pad must be constant mode with value 0.
    std::string mode = "constant";
    {
      Ort::ConstOpAttr a{nullptr};
      if (pad.GetAttributeByName("mode", a).IsOK() && a != nullptr) a.GetValue(mode);
    }
    if (mode != "constant") continue;

    const auto pad_inputs = pad.GetInputs();
    if (pad_inputs.size() >= 3 && pad_inputs[2] != nullptr) {
      auto it = constants.find(std::string(pad_inputs[2].GetName()));
      if (it == constants.end()) continue;
      if (it->second.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        float val = 0.f;
        std::memcpy(&val, it->second.data.data(), sizeof(float));
        if (val != 0.f) continue;
      }
    }

    // Read the pads vector; must be 4D (8 elements).
    const auto pads = ReadPadNodePads(pad, constants);
    if (!pads.has_value() || pads->size() != 8) continue;

    // No N or C padding, symmetric H and W padding only.
    if ((*pads)[0] != 0 || (*pads)[1] != 0 || (*pads)[4] != 0 || (*pads)[5] != 0) continue;
    if ((*pads)[2] != (*pads)[6] || (*pads)[3] != (*pads)[7]) continue;
    const int h_pad = static_cast<int>((*pads)[2]);
    const int w_pad = static_cast<int>((*pads)[3]);

    // The Pad output must be consumed only by this Conv (use_count == 1).
    if (use_count[data_input_name] != 1) continue;

    // Don't absorb a Pad that another fusion already claimed.
    const std::string pad_key = NodeKey(pad);
    if (plan.consumed_nodes.count(pad_key) > 0) continue;

    plan.consumed_nodes.insert(pad_key);
    FusionPlan::AbsorbedPad absorbed;
    absorbed.p0 = w_pad;
    absorbed.p1 = h_pad;
    absorbed.data_input_name = std::string(pad_inputs[0].GetName());
    plan.absorbed_pads.emplace(NodeKey(conv), std::move(absorbed));
  }
}

}  // namespace

MetaAnalysis AnalyzeCompileTimeConstants(const OrtGraph* graph) {
  GGONNX_NOT_NULL(graph, "graph must not be null");
  const Ort::ConstGraph ort_graph{graph};
  MetaAnalysis analysis;

  for (Ort::ConstValueInfo input : ort_graph.GetInputs()) {
    if (input == nullptr || !input.IsConstantInitializer()) continue;
    Ort::ConstValue value{nullptr};
    Ort::ThrowOnError(input.GetInitializer(value));
    if (value == nullptr) continue;
    const TensorMetadata meta = getTensorMetadata(value);
    if (!IsFoldableDType(meta.element_type) || !FitsInitializerLimit(meta.dims)) continue;
    analysis.constants.emplace(input.GetName(), LoadInitializer(value));
  }

  for (Ort::ConstValueInfo init : ort_graph.GetInitializers()) {
    if (init == nullptr || !init.IsConstantInitializer()) continue;
    Ort::ConstValue value{nullptr};
    Ort::ThrowOnError(init.GetInitializer(value));
    if (value == nullptr) continue;
    const TensorMetadata meta = getTensorMetadata(value);
    if (!IsFoldableDType(meta.element_type) || !FitsInitializerLimit(meta.dims)) continue;
    analysis.constants.emplace(init.GetName(), LoadInitializer(value));
  }

  for (Ort::ConstNode node : ort_graph.GetNodes()) {
    const auto outputs = node.GetOutputs();
    if (outputs.empty() || outputs[0] == nullptr) continue;
    try {
      if (node.GetOperatorType() == "Loop") {
        const auto folded_outputs = EvalArangeLoop(analysis.constants, node);
        if (folded_outputs.has_value()) {
          GGONNX_ASSERT(folded_outputs->size() == outputs.size(),
                        "folded Loop output count mismatch");
          analysis.folded_nodes.insert(NodeKey(node));
          for (size_t i = 0; i < outputs.size(); ++i) {
            if (outputs[i] == nullptr) continue;
            analysis.constants[outputs[i].GetName()] = (*folded_outputs)[i];
          }
          continue;
        }
      }
      const auto folded = TryFoldNode(analysis.constants, node);
      if (!folded.has_value()) continue;
      analysis.folded_nodes.insert(NodeKey(node));
      for (Ort::ConstValueInfo output : outputs) {
        if (output == nullptr) continue;
        analysis.constants[output.GetName()] = *folded;
      }
    } catch (...) {
      continue;
    }
  }
  DetectShuffleTripleFusions(ort_graph, analysis.constants, analysis.fusions);
  // WindowMaskAdd runs before QKVSplit: the mask-apply rank-5 bubble sits
  // between the QKV Gathers in topological order, so fusing it out first
  // lets QKVSplit's span check clear the (now much shorter) span.
  DetectWindowMaskAddFusions(ort_graph, analysis.constants, analysis.folded_nodes,
                             analysis.fusions);
  DetectQKVSplitFusions(ort_graph, analysis.constants, analysis.folded_nodes, analysis.fusions);
  DetectPadConvFusions(ort_graph, analysis.constants, analysis.fusions);
  return analysis;
}
