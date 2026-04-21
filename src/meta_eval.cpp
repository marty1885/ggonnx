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
         element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
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

std::optional<ConstantTensor> TryFoldNode(const ConstantValueMap& constants, Ort::ConstNode node) {
  const std::string op_type = node.GetOperatorType();
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
  return analysis;
}
