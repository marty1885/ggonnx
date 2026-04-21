#pragma once

#include <string>
#include <unordered_set>

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "ops.hpp"

struct MetaAnalysis {
  ConstantValueMap constants;
  std::unordered_set<std::string> folded_nodes;
};

MetaAnalysis AnalyzeCompileTimeConstants(const OrtGraph* graph);
std::string NodeKey(Ort::ConstNode node);
