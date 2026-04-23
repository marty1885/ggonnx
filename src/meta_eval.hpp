#pragma once

#include <limits>
#include <string>
#include <unordered_set>

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "ops.hpp"

// Sentinel stored in folded INT64/INT32 tensors to mark a dimension that is
// dynamic (unknown at compile time). Such "constants" must never be used as
// compile-time GGML values — treat them as runtime inputs instead.
inline constexpr int64_t kUnknownShapeDimSentinel =
    static_cast<int64_t>(std::numeric_limits<int32_t>::min());

// Returns true if a folded constant tensor contains the dynamic sentinel.
bool ConstantContainsSentinel(const ConstantTensor& tensor);

struct MetaAnalysis {
  ConstantValueMap constants;
  std::unordered_set<std::string> folded_nodes;
  FusionPlan fusions;
};

MetaAnalysis AnalyzeCompileTimeConstants(const OrtGraph* graph);
std::string NodeKey(Ort::ConstNode node);
