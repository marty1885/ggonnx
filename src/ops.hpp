#pragma once

#include <ggml.h>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "inner/helpers.hpp"

inline constexpr size_t kOptionalValueAbsent = std::numeric_limits<size_t>::max();

struct TensorMetadata {
  ONNXTensorElementDataType element_type{};
  std::vector<int64_t> dims;
};

struct ConstantTensor {
  ONNXTensorElementDataType element_type{};
  std::vector<int64_t> dims;
  std::vector<uint8_t> data;
};

using ConstantValueMap = std::unordered_map<std::string, ConstantTensor>;

TensorMetadata getTensorMetadata(Ort::ConstValueInfo value_info);
TensorMetadata getTensorMetadata(Ort::ConstValue value);
void SetActiveCompileTimeConstants(const ConstantValueMap* constants);
const ConstantValueMap* GetActiveCompileTimeConstants();

inline bool shapeIsFullyStatic(const TensorMetadata& tensor) {
  return std::all_of(tensor.dims.begin(), tensor.dims.end(), [](int64_t dim) { return dim >= 0; });
}

inline bool rankSupportedByGGML(const TensorMetadata& tensor) {
  return tensor.dims.size() <= GGML_MAX_DIMS;
}

inline bool shapeIsFullyStatic(const std::vector<int64_t>& dims) {
  return std::all_of(dims.begin(), dims.end(), [](int64_t dim) { return dim >= 0; });
}

inline void AssertShapeMatchesGGML(const std::vector<int64_t>& expected_onnx_dims,
                            const ggml_tensor* tensor,
                            const std::string& tensor_name) {
  GGONNX_NOT_NULL(tensor, "ggml tensor must not be null for shape check");
  const std::vector<int64_t> actual_onnx_dims = ToOnnxDims(tensor);
  if (ToPaddedGGMLDims(actual_onnx_dims) != ToPaddedGGMLDims(expected_onnx_dims)) {
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
inline bool broadcastSupportedByGGML(const std::vector<int64_t>& lhs_dims,
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

struct NodeDesc {
  struct NoAttrs {};
  struct GRUAttrs {
    int64_t hidden_size{};
  };
  struct AlphaAttrs {
    float alpha{};
  };
  struct AxisAttrs {
    int64_t axis{};
  };
  struct Conv2DAttrs {
    int s0{1}, s1{1};
    int p0{0}, p1{0};
    int d0{1}, d1{1};
    bool is_depthwise{false};
  };
  struct ClipAttrs {
    float min{-std::numeric_limits<float>::infinity()};
    float max{std::numeric_limits<float>::infinity()};
  };
  struct GemmAttrs {
    float alpha{1.0f};
    float beta{1.0f};
    bool trans_a{false};
    bool trans_b{false};
  };
  struct ReshapeAttrs {
    std::vector<int64_t> onnx_dims;  // fully static target shape
  };
  struct Pool2DAttrs {
    ggml_op_pool op{GGML_OP_POOL_MAX};
    bool is_global{false};  // kernel derived from input spatial dims at emit time
    int k0{1}, k1{1};
    int s0{1}, s1{1};
    int p0{0}, p1{0};
  };
  struct PadAttrs {
    int pad_w_left{0};
    int pad_w_right{0};
    int pad_h_top{0};
    int pad_h_bottom{0};
  };
  struct InstanceNormAttrs {
    float epsilon{1e-5f};
  };
  struct BatchNormAttrs {
    float epsilon{1e-5f};
    // ONNX input rank (2..4 supported). The channel axis in ONNX is always 1,
    // which in ggml's reversed layout sits at ggml axis (rank-2). Stored at
    // compile time because the emit-time tensor alone can't distinguish "rank
    // 4 with N=1, H=1" from "rank 2".
    int onnx_rank{4};
  };
  struct UpsampleAttrs {
    int scale_w{1};
    int scale_h{1};
  };
  struct TransposeAttrs {
    // Permutation expressed in GGML axis order (padded to GGML_MAX_DIMS). Axis j
    // of the output is taken from axis ggml_perm[j] of the input — the same
    // convention ggml_permute() uses.
    std::array<int, GGML_MAX_DIMS> ggml_perm{0, 1, 2, 3};
  };
  struct SliceAttrs {
    // Slice is implemented as a ggml_view with step == 1. Both arrays are in
    // padded GGML axis order: untouched dims get start=0 and full size.
    std::array<int64_t, GGML_MAX_DIMS> ggml_starts{0, 0, 0, 0};
    std::array<int64_t, GGML_MAX_DIMS> ggml_ne{1, 1, 1, 1};
  };

  using Attrs = std::variant<NoAttrs,
                             GRUAttrs,
                             AlphaAttrs,
                             AxisAttrs,
                             Conv2DAttrs,
                             ClipAttrs,
                             GemmAttrs,
                             ReshapeAttrs,
                             Pool2DAttrs,
                             PadAttrs,
                             InstanceNormAttrs,
                             BatchNormAttrs,
                             UpsampleAttrs,
                             TransposeAttrs,
                             SliceAttrs>;

  std::string op_type;
  std::string domain;
  std::string name;
  std::vector<size_t> inputs;
  std::vector<size_t> outputs;
  Attrs attrs{};
};

using EmitOutputs = std::vector<ggml_tensor*>;
using EmitResult = std::optional<EmitOutputs>;
using EmitNodeFn = EmitResult (*)(ggml_context* ctx,
                                  const NodeDesc& node,
                                  const std::vector<ggml_tensor*>& values);
using CompileAttrsFn = void (*)(Ort::ConstNode node, NodeDesc* compiled_node);

// Layout hint for constant-initializer inputs materialized at compile time.
// AS_IS: ggml ne = reversed ONNX dims, data byte-identical.
// MATMUL_WEIGHT_TRANSPOSED: swap the last two ONNX dims before storing. For a
//   [K, N] ONNX weight this yields ggml ne=[K, N] with contiguous data, which
//   is what ggml_mul_mat(weight, activation) wants — so the runtime transpose
//   + cont vanishes. Extends naturally to batched [..., K, N].
enum class ConstantLayout {
  AS_IS,
  MATMUL_WEIGHT_TRANSPOSED,
};

// Per-op hook: given a compiled node and an input index that is a constant
// initializer, return the desired layout. Nullable on OpDefinition; nullptr
// means "always AS_IS".
using ConstantLayoutFn = ConstantLayout (*)(const NodeDesc& node, size_t input_idx);

struct OpDefinition {
  bool (*support)(Ort::ConstNode node);
  CompileAttrsFn compile_attrs;
  EmitNodeFn emit;
  ConstantLayoutFn constant_layout;
};

const OpDefinition* FindOpDefinition(std::string_view domain, std::string_view op_type);
