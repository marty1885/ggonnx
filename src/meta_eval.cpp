#include "meta_eval.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include "inner/helpers.hpp"
#include "inner/ort_api_helpers.hpp"

namespace {

using ggonnx::ort_internal::GetOrtApi;
using ggonnx::ort_internal::THROW_ON_ERROR;

constexpr size_t kMaxFoldedElements = 256;

size_t ByteWidth(ONNXTensorElementDataType element_type) {
  switch (element_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return sizeof(float);
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
  return element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
         element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
         element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
}

bool FitsFoldLimit(const std::vector<int64_t>& dims) {
  return ElementCount(dims) <= kMaxFoldedElements;
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
  GGONNX_ASSERT(FitsFoldLimit(meta.dims),
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

void DetectWindowShuffleFusions(const Ort::ConstGraph& ort_graph,
                                const ConstantValueMap& constants,
                                FusionPlan& plan) {
  // Producer map (output value name -> node) and consumer counts (value name ->
  // number of consumers across this graph). We only fuse when the Reshape1
  // output and Transpose output are each consumed by exactly one downstream
  // node, so their rank-6 tensors can be discarded safely.
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
  // Graph outputs count as external consumers — a fusion-internal value named
  // as a graph output must stay materialized.
  for (Ort::ConstValueInfo out : ort_graph.GetOutputs()) {
    if (out != nullptr) consumer_count[std::string(out.GetName())]++;
  }

  std::unordered_set<std::string> claimed;
  auto claim = [&](const std::string& key) {
    return claimed.insert(key).second;
  };

  for (Ort::ConstNode trans : nodes) {
    if (trans.GetOperatorType() != "Transpose") continue;
    const auto perm = ReadTransposePerm(trans);
    if (!perm || perm->size() != 6) continue;
    const std::vector<int64_t> expected = {0, 1, 3, 2, 4, 5};
    if (*perm != expected) continue;

    const auto trans_inputs = trans.GetInputs();
    const auto trans_outputs = trans.GetOutputs();
    if (trans_inputs.size() != 1 || trans_outputs.size() != 1) continue;
    if (trans_inputs[0] == nullptr || trans_outputs[0] == nullptr) continue;

    const std::string r1_out = std::string(trans_inputs[0].GetName());
    auto prod_it = producer.find(r1_out);
    if (prod_it == producer.end()) continue;
    Ort::ConstNode r1 = prod_it->second;
    if (r1.GetOperatorType() != "Reshape") continue;
    if (consumer_count[r1_out] != 1) continue;  // the Transpose must be the only consumer

    const std::string trans_out_name = std::string(trans_outputs[0].GetName());
    if (consumer_count[trans_out_name] != 1) continue;

    // There must be exactly one downstream consumer, and it must be a Reshape.
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
    if (r2 == nullptr) continue;
    if (r2.GetOperatorType() != "Reshape") continue;

    const auto r1_inputs = r1.GetInputs();
    const auto r2_inputs = r2.GetInputs();
    const auto r2_outputs = r2.GetOutputs();
    if (r1_inputs.size() != 2 || r2_inputs.size() != 2 || r2_outputs.size() != 1) continue;
    if (r1_inputs[0] == nullptr || r1_inputs[1] == nullptr ||
        r2_inputs[0] == nullptr || r2_inputs[1] == nullptr || r2_outputs[0] == nullptr) {
      continue;
    }

    // Fully-static input shape feeding Reshape1 (needed to resolve 0/-1 in
    // the target shape and to validate element-count).
    const TensorMetadata r1_in_meta = getTensorMetadata(r1_inputs[0]);
    if (!shapeIsFullyStatic(r1_in_meta)) continue;

    auto r1_target = ReadConstantInt64Input(constants, r1, 1);
    if (!r1_target || r1_target->size() != 6) continue;
    auto r1_resolved = ResolveReshapeTarget(*r1_target, r1_in_meta.dims);
    if (!r1_resolved || r1_resolved->size() != 6) continue;
    for (int64_t d : *r1_resolved) {
      if (d <= 0) { r1_resolved.reset(); break; }
    }
    if (!r1_resolved) continue;

    // Reshape2's intermediate input shape is the Transpose output: swap dims
    // 2 and 3 of the rank-6 tensor.
    std::vector<int64_t> r2_input_dims = *r1_resolved;
    std::swap(r2_input_dims[2], r2_input_dims[3]);

    auto r2_target = ReadConstantInt64Input(constants, r2, 1);
    std::optional<std::vector<int64_t>> r2_resolved;
    if (r2_target) {
      r2_resolved = ResolveReshapeTarget(*r2_target, r2_input_dims);
    } else {
      // Fall back to ORT's shape inference on the output value.
      const TensorMetadata r2_out_meta = getTensorMetadata(r2_outputs[0]);
      if (shapeIsFullyStatic(r2_out_meta)) r2_resolved = r2_out_meta.dims;
    }
    if (!r2_resolved || r2_resolved->empty()) continue;
    for (int64_t d : *r2_resolved) {
      if (d <= 0) { r2_resolved.reset(); break; }
    }
    if (!r2_resolved) continue;

    // Element count must be conserved across the triple.
    auto prod = [](const std::vector<int64_t>& v) {
      int64_t p = 1; for (int64_t d : v) p *= d; return p;
    };
    if (prod(*r1_resolved) != prod(r1_in_meta.dims) ||
        prod(*r2_resolved) != prod(*r1_resolved)) {
      continue;
    }

    const std::string r1_key = NodeKey(r1);
    const std::string r2_key = NodeKey(r2);
    const std::string anchor_key = NodeKey(trans);
    if (!claim(r1_key) || !claim(r2_key) || !claim(anchor_key)) continue;

    NodeDesc::WindowShuffleAttrs attrs;
    for (size_t i = 0; i < 6; ++i) attrs.onnx_rank6_dims[i] = (*r1_resolved)[i];
    attrs.output_onnx_dims = *r2_resolved;

    FusionPlan::AnchorIO io;
    io.input_value = std::string(r1_inputs[0].GetName());
    io.output_value = std::string(r2_outputs[0].GetName());
    io.anchor_node_name = std::string(trans.GetName());

    plan.window_shuffle_anchors.emplace(anchor_key, std::move(attrs));
    plan.anchor_io.emplace(anchor_key, std::move(io));
    plan.consumed_nodes.insert(r1_key);
    plan.consumed_nodes.insert(r2_key);
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
    if (!IsFoldableDType(meta.element_type) || !FitsFoldLimit(meta.dims)) continue;
    analysis.constants.emplace(input.GetName(), LoadInitializer(value));
  }

  for (Ort::ConstValueInfo init : ort_graph.GetInitializers()) {
    if (init == nullptr || !init.IsConstantInitializer()) continue;
    Ort::ConstValue value{nullptr};
    Ort::ThrowOnError(init.GetInitializer(value));
    if (value == nullptr) continue;
    const TensorMetadata meta = getTensorMetadata(value);
    if (!IsFoldableDType(meta.element_type) || !FitsFoldLimit(meta.dims)) continue;
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
  DetectWindowShuffleFusions(ort_graph, analysis.constants, analysis.fusions);
  return analysis;
}
