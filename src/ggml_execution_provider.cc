#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "inner/helpers.hpp"
#include "inner/ort_api_helpers.hpp"
#include "meta_eval.hpp"
#include "ops.hpp"

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
  ONNXTensorElementDataType element_type{};
  std::vector<int64_t> dims;
  bool is_graph_input{};
  bool is_graph_output{};
};

struct CompiledPartition {
  std::vector<ValueDesc> values;
  std::vector<size_t> graph_inputs;
  std::vector<size_t> graph_outputs;
  // Kernel-input-index → value_id. Non-float kernel inputs (e.g. Reshape's
  // int64 shape tensor) are represented as kOptionalValueAbsent and ignored
  // at runtime — their semantic role was already consumed at compile time.
  std::vector<size_t> kernel_input_to_value;
  std::vector<NodeDesc> nodes;
  // Persistent ggml context holding pre-materialized constant initializers.
  // Entries in `constants` are indexed by value_id; nullptr means "not a
  // constant". Constants never appear in graph_inputs — their data is snapshotted
  // at compile time, with per-op layout hints (e.g. MatMul B is pre-transposed).
  ggml_backend_t backend{nullptr};
  ggml_context* constant_ctx{nullptr};
  ggml_backend_buffer_t constant_buffer{nullptr};
  std::vector<ggml_tensor*> constants;
  std::vector<std::optional<ConstantTensor>> folded_constants;

  CompiledPartition() = default;
  CompiledPartition(const CompiledPartition&) = delete;
  CompiledPartition& operator=(const CompiledPartition&) = delete;
  CompiledPartition(CompiledPartition&& other) noexcept { *this = std::move(other); }
  CompiledPartition& operator=(CompiledPartition&& other) noexcept {
    if (this == &other) return *this;
    if (constant_buffer != nullptr) ggml_backend_buffer_free(constant_buffer);
    if (constant_ctx != nullptr) ggml_free(constant_ctx);
    if (backend != nullptr) ggml_backend_free(backend);
    values = std::move(other.values);
    graph_inputs = std::move(other.graph_inputs);
    graph_outputs = std::move(other.graph_outputs);
    kernel_input_to_value = std::move(other.kernel_input_to_value);
    nodes = std::move(other.nodes);
    backend = other.backend;
    constant_ctx = other.constant_ctx;
    constant_buffer = other.constant_buffer;
    constants = std::move(other.constants);
    other.backend = nullptr;
    other.constant_ctx = nullptr;
    other.constant_buffer = nullptr;
    return *this;
  }
  ~CompiledPartition() {
    if (constant_buffer != nullptr) ggml_backend_buffer_free(constant_buffer);
    if (constant_ctx != nullptr) ggml_free(constant_ctx);
    if (backend != nullptr) ggml_backend_free(backend);
  }
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
  ggml_gallocr_t gallocr{};
  std::vector<ggml_tensor*> values;
  std::vector<std::vector<int64_t>> value_dims;
  std::vector<ggml_tensor*> input_tensors;
  std::vector<ggml_tensor*> output_tensors;
};

struct GGMLComputeState {
  const CompiledPartition* partition{nullptr};  // owned by GGMLNodeComputeInfo
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

bool isNodeSupported(Ort::ConstNode node) {
  const std::string op_type = node.GetOperatorType();
  const std::string domain = node.GetDomain();
  const OpDefinition* op = FindOpDefinition(domain, op_type);
  return op != nullptr && op->support(node);
}

size_t elementCount(const std::vector<int64_t>& dims) {
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

size_t EstimateGraphMetadataBytes(const CompiledPartition& partition) {
  // Metadata-only graph context. Tensor payloads are allocated via GGML backend
  // buffers/gallocr, not from the ggml_context arena.
  const size_t tensor_count = partition.values.size() + partition.nodes.size() * 4 +
                              partition.graph_inputs.size() + partition.graph_outputs.size() + 16;
  return tensor_count * ggml_tensor_overhead() +
         ggml_graph_overhead() +
         64 * 1024;
}

// Byte count for a constant value given its ONNX dims (float32 only).
size_t ConstantByteSize(const std::vector<int64_t>& dims) {
  return elementCount(dims) * sizeof(float);
}

// Copy ONNX float data into a ggml tensor according to `layout`. For AS_IS,
// bytes are identical to ONNX (ggml col-major reinterpretation already flips
// the dim order). For MATMUL_WEIGHT_TRANSPOSED, we physically transpose the
// last two ONNX dims, so the resulting ggml tensor has ne[0]=K and is ready
// to feed as the first arg of ggml_mul_mat.
void MaterializeConstantData(const float* src,
                             const std::vector<int64_t>& onnx_dims,
                             ConstantLayout layout,
                             ggml_tensor* dst) {
  if (layout == ConstantLayout::AS_IS) {
    std::memcpy(dst->data, src, ConstantByteSize(onnx_dims));
    return;
  }
  GGONNX_ASSERT(onnx_dims.size() >= 2,
                "MATMUL_WEIGHT_TRANSPOSED layout requires rank >= 2");
  // Transpose the last two ONNX dims. For ONNX shape [..., K, N], produce
  // data in ONNX shape [..., N, K], which under ggml's reversed interpretation
  // becomes ne=[K, N, ...] — exactly what ggml_mul_mat wants.
  const int64_t K = onnx_dims[onnx_dims.size() - 2];
  const int64_t N = onnx_dims[onnx_dims.size() - 1];
  int64_t batch = 1;
  for (size_t i = 0; i + 2 < onnx_dims.size(); ++i) batch *= onnx_dims[i];
  float* out = static_cast<float*>(dst->data);
  // TODO: Optimize this aweful slow transpose
  for (int64_t b = 0; b < batch; ++b) {
    const float* src_mat = src + b * K * N;
    float* dst_mat = out + b * K * N;
    for (int64_t k = 0; k < K; ++k) {
      for (int64_t n = 0; n < N; ++n) {
        dst_mat[n * K + k] = src_mat[k * N + n];
      }
    }
  }
}

CompiledPartition CompilePartition(const OrtGraph* graph) {
  GGONNX_NOT_NULL(graph, "graph must not be null");
  const Ort::ConstGraph ort_graph{graph};
  MetaAnalysis meta_analysis = AnalyzeCompileTimeConstants(graph);
  SetActiveCompileTimeConstants(&meta_analysis.constants);
  CompiledPartition partition;
  struct ActiveConstantsGuard {
    ~ActiveConstantsGuard() { SetActiveCompileTimeConstants(nullptr); }
  } active_constants_guard;

  partition.backend = ggml_backend_cpu_init();
  GGONNX_NOT_NULL(partition.backend, "ggml_backend_cpu_init failed");
  std::unordered_map<std::string, size_t> value_ids;
  ConstantValueMap constants_by_name = meta_analysis.constants;
  // value_id -> chosen layout. Populated during node walking; first writer wins,
  // later uses that disagree force AS_IS and the op falls back to runtime transpose.
  std::unordered_map<size_t, ConstantLayout> constant_layout_by_id;
  auto record_constant_use = [&](size_t value_id, ConstantLayout layout) {
    auto [it, inserted] = constant_layout_by_id.emplace(value_id, layout);
    if (!inserted && it->second != layout) {
      it->second = ConstantLayout::AS_IS;
    }
  };

  auto ensure_value = [&](Ort::ConstValueInfo value_info) -> size_t {
    const std::string name = value_info.GetName();
    const TensorMetadata metadata = getTensorMetadata(value_info);
    if (metadata.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      GGONNX_ASSERT(rankSupportedByGGML(metadata),
                    "compiled partition requires tensor rank <= " + std::to_string(GGML_MAX_DIMS) +
                        ", got " + std::to_string(metadata.dims.size()) + " for value '" + name + "'");
    }
    const auto it = value_ids.find(name);
    if (it != value_ids.end()) {
      partition.values[it->second].element_type = metadata.element_type;
      partition.values[it->second].dims = metadata.dims;
      return it->second;
    }

    const size_t id = partition.values.size();
    value_ids.emplace(name, id);
    ValueDesc value;
    value.name = name;
    value.element_type = metadata.element_type;
    value.dims = metadata.dims;
    partition.values.push_back(std::move(value));
    return id;
  };

  const auto graph_inputs = ort_graph.GetInputs();
  for (Ort::ConstValueInfo input : graph_inputs) {
    GGONNX_ASSERT(input != nullptr, "graph input metadata must not be null");
    // Non-float graph inputs (e.g. Reshape's int64 shape initializer lifted by
    // ORT into the subgraph boundary) are consumed at compile time via
    // attributes; they are not ggml runtime values.
    const TensorMetadata meta = getTensorMetadata(input);
    if (meta.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      partition.kernel_input_to_value.push_back(kOptionalValueAbsent);
      continue;
    }
    // Constant initializers: materialize once at compile time. ORT still passes
    // them as kernel inputs so we reserve a kernel slot but ignore the value at
    // runtime — the compile-time copy in partition.constants is authoritative.
    if (constants_by_name.count(std::string(input.GetName())) > 0) {
      ensure_value(input);  // reserve value_id so node walking can reference it
      partition.kernel_input_to_value.push_back(kOptionalValueAbsent);
      continue;
    }
    const size_t id = ensure_value(input);
    partition.values[id].is_graph_input = true;
    partition.graph_inputs.push_back(id);
    partition.kernel_input_to_value.push_back(id);
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
    if (meta_analysis.folded_nodes.count(NodeKey(node)) > 0) {
      for (Ort::ConstValueInfo output : node.GetOutputs()) {
        if (output != nullptr) {
          ensure_value(output);
        }
      }
      continue;
    }
    if (!isNodeSupported(node)) {
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

    for (size_t input_idx = 0; input_idx < node_inputs.size(); ++input_idx) {
      Ort::ConstValueInfo input = node_inputs[input_idx];
      if (input == nullptr) {
        compiled_node.inputs.push_back(kOptionalValueAbsent);
        continue;
      }
      // Non-float inputs (e.g. Reshape's int64 shape input) are consumed at
      // compile time via attributes/output metadata and have no ggml value.
      const TensorMetadata meta = getTensorMetadata(input);
      if (meta.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        compiled_node.inputs.push_back(kOptionalValueAbsent);
        continue;
      }
      const size_t id = ensure_value(input);
      compiled_node.inputs.push_back(id);
      if (constants_by_name.count(std::string(input.GetName())) > 0) {
        const ConstantLayout layout = (op->constant_layout != nullptr)
            ? op->constant_layout(compiled_node, input_idx)
            : ConstantLayout::AS_IS;
        record_constant_use(id, layout);
      }
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

  GGONNX_ASSERT(!partition.graph_outputs.empty(), "compiled partition must contain at least one graph output");
  partition.folded_constants.resize(partition.values.size());
  for (size_t id = 0; id < partition.values.size(); ++id) {
    auto it = constants_by_name.find(partition.values[id].name);
    if (it != constants_by_name.end()) {
      partition.folded_constants[id] = it->second;
    }
  }

  // Materialize detected constants into a persistent ggml context. Sized
  // tightly: sum of constant data bytes + per-tensor overhead + slack.
  if (!constant_layout_by_id.empty()) {
    size_t constant_bytes = 0;
    for (const auto& [id, layout] : constant_layout_by_id) {
      constant_bytes += ConstantByteSize(partition.values[id].dims);
    }
    const size_t mem_size =
        constant_bytes +
        (constant_layout_by_id.size() + 4) * ggml_tensor_overhead() +
        4 * 1024;

    ggml_init_params p{};
    p.mem_size = mem_size;
    p.mem_buffer = nullptr;
    p.no_alloc = true;
    partition.constant_ctx = ggml_init(p);
    GGONNX_NOT_NULL(partition.constant_ctx, "ggml_init failed for constants");

    partition.constants.assign(partition.values.size(), nullptr);
    for (const auto& [id, layout] : constant_layout_by_id) {
      const ValueDesc& value = partition.values[id];
      auto it = constants_by_name.find(value.name);
      GGONNX_ASSERT(it != constants_by_name.end(), "constant tensor lost during compile");
      GGONNX_ASSERT(it->second.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                    "GGML constants must be float32");

      std::vector<int64_t> tensor_dims = value.dims;
      if (layout == ConstantLayout::MATMUL_WEIGHT_TRANSPOSED && tensor_dims.size() >= 2) {
        std::swap(tensor_dims[tensor_dims.size() - 2], tensor_dims[tensor_dims.size() - 1]);
      }
      // Scalar ONNX constants (rank 0) are represented in ggml as a rank-1 size-1 tensor.
      std::vector<int64_t> ggml_src_dims = tensor_dims.empty() ? std::vector<int64_t>{1} : tensor_dims;
      const std::array<int64_t, GGML_MAX_DIMS> ggml_dims = ToGGMLDims(ggml_src_dims);
      ggml_tensor* t = ggml_new_tensor(partition.constant_ctx, GGML_TYPE_F32,
                                       static_cast<int>(ggml_src_dims.size()),
                                       ggml_dims.data());
      GGONNX_NOT_NULL(t, "ggml_new_tensor failed for constant");
      partition.constants[id] = t;
    }

    partition.constant_buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(partition.constant_ctx,
                                                 ggml_backend_get_default_buffer_type(partition.backend));
    GGONNX_NOT_NULL(partition.constant_buffer, "ggml_backend_alloc_ctx_tensors_from_buft failed for constants");
    ggml_backend_buffer_set_usage(partition.constant_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (const auto& [id, layout] : constant_layout_by_id) {
      ggml_tensor* t = partition.constants[id];
      GGONNX_NOT_NULL(t, "constant tensor missing backend allocation");
      const ValueDesc& value = partition.values[id];
      auto it = constants_by_name.find(value.name);
      GGONNX_ASSERT(it != constants_by_name.end(), "constant tensor lost during backend upload");
      GGONNX_ASSERT(it->second.element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                    "GGML constants must be float32");

      std::vector<float> materialized(elementCount(value.dims));
      MaterializeConstantData(reinterpret_cast<const float*>(it->second.data.data()), value.dims, layout, t);
      std::memcpy(materialized.data(), t->data, materialized.size() * sizeof(float));
      ggml_backend_tensor_set(t, materialized.data(), 0, materialized.size() * sizeof(float));
    }
  }

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
  const size_t element_count = elementCount(meta.dims);
  const size_t bytes = element_count * sizeof(float);
  AssertShapeMatchesGGML(meta.dims, tensor, tensor_name);
  ggml_backend_tensor_set(tensor, input_data, 0, bytes);
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
  if (graph != nullptr) {
    if (graph->gallocr != nullptr) {
      ggml_gallocr_free(graph->gallocr);
    }
    if (graph->ctx != nullptr) {
      ggml_free(graph->ctx);
    }
  }
  graph.reset();
}

std::unique_ptr<MaterializedGraph> BuildMaterializedGraph(const CompiledPartition& partition,
                                                          ShapeKey key,
                                                          const std::vector<TensorMetadata>& input_metadata) {
  const size_t mem_size =
      EstimatePartitionTensorCount(partition) * ggml_tensor_overhead() +
      ggml_graph_overhead() +
      EstimateGraphMetadataBytes(partition);

  ggml_init_params params{};
  params.mem_size = mem_size;
  params.mem_buffer = nullptr;
  params.no_alloc = true;

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

    // Preload compile-time constants into the value table. value_dims keeps
    // the declared (logical) ONNX shape even if the constant was physically
    // pre-transposed — emits that use these tensors inspect ne[] directly.
    for (size_t id = 0; id < partition.constants.size(); ++id) {
      ggml_tensor* c = partition.constants[id];
      if (c == nullptr) continue;
      graph_state->values[id] = c;
      graph_state->value_dims[id] = partition.values[id].dims;
    }

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
      ggml_set_input(input_tensor);
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
        const std::vector<int64_t> emitted_dims = ToOnnxDims(node_output);
        const std::vector<int64_t>& declared_dims = partition.values[output_id].dims;
        GGONNX_ASSERT(broadcastSupportedByGGML(declared_dims, emitted_dims),
                      "emitted output shape mismatch for tensor '" + partition.values[output_id].name +
                          "': declared " + FormatDims(declared_dims) + ", got " +
                          FormatDims(emitted_dims));
        graph_state->values[output_id] = node_output;
        // Preserve the declared ONNX rank even when some dims are dynamic: GGML
        // strips trailing unit dims (and only carries 4), so a declared shape
        // like (1, S, V) would otherwise collapse to (S, V) when taken straight
        // from the ggml tensor. Merge right-aligned — declared wins when
        // concrete, emitted fills in dynamic slots.
        if (shapeIsFullyStatic(declared_dims)) {
          graph_state->value_dims[output_id] = declared_dims;
        } else {
          std::vector<int64_t> merged(declared_dims.size(), 1);
          const size_t off = declared_dims.size() >= emitted_dims.size()
                                 ? declared_dims.size() - emitted_dims.size()
                                 : 0;
          for (size_t i = 0; i < merged.size(); ++i) {
            if (declared_dims[i] >= 0) {
              merged[i] = declared_dims[i];
            } else if (i >= off) {
              merged[i] = emitted_dims[i - off];
            }
          }
          graph_state->value_dims[output_id] = std::move(merged);
        }
      }
      GGONNX_ASSERT(emitted_index == emitted_outputs->size(), "compiled node emitted too many outputs");
    }

    bool has_runtime_output = false;
    for (size_t i = 0; i < partition.graph_outputs.size(); ++i) {
      const size_t output_id = partition.graph_outputs[i];
      GGONNX_ASSERT(output_id < graph_state->values.size(), "graph output index out of range");
      ggml_tensor* output = graph_state->values[output_id];
      if (output == nullptr) {
        GGONNX_ASSERT(partition.folded_constants.size() > output_id &&
                          partition.folded_constants[output_id].has_value(),
                      "compiled partition output was not materialized: " +
                          partition.values[output_id].name);
        graph_state->value_dims[output_id] = partition.values[output_id].dims;
        continue;
      }
      if (!has_runtime_output) {
        graph_state->graph = ggml_new_graph(ctx);
        GGONNX_NOT_NULL(graph_state->graph, "ggml_new_graph failed");
        has_runtime_output = true;
      }
      ggml_set_output(output);
      ggml_build_forward_expand(graph_state->graph, output);
      graph_state->output_tensors[i] = output;
    }

    if (has_runtime_output) {
      graph_state->gallocr =
          ggml_gallocr_new(ggml_backend_get_default_buffer_type(partition.backend));
      GGONNX_NOT_NULL(graph_state->gallocr, "ggml_gallocr_new failed");
      GGONNX_ASSERT(ggml_gallocr_alloc_graph(graph_state->gallocr, graph_state->graph),
                    "ggml_gallocr_alloc_graph failed");
    }

    g_debug_graph_build_count.fetch_add(1, std::memory_order_relaxed);
    return graph_state;
  } catch (...) {
    ggml_free(ctx);
    throw;
  }
}

void ExecutePartitionWithGGML(GGMLComputeState& state, OrtKernelContext* kernel_context) {
  GGONNX_NOT_NULL(state.partition, "compute state missing compiled partition");
  const CompiledPartition& partition = *state.partition;
  GGONNX_NOT_NULL(kernel_context, "kernel context must not be null");
  GGONNX_ASSERT(!partition.graph_outputs.empty(), "compiled partition must contain graph outputs");
  size_t num_inputs = 0;
  size_t num_outputs = 0;
  THROW_ON_ERROR(GetOrtApi().KernelContext_GetInputCount(kernel_context, &num_inputs));
  THROW_ON_ERROR(GetOrtApi().KernelContext_GetOutputCount(kernel_context, &num_outputs));
  if (num_inputs != partition.kernel_input_to_value.size() ||
      num_outputs != partition.graph_outputs.size()) {
    throw std::runtime_error("kernel IO count does not match compiled partition");
  }

  std::vector<const OrtValue*> float_input_values;
  std::vector<TensorMetadata> input_metadata;
  float_input_values.reserve(partition.graph_inputs.size());
  input_metadata.reserve(partition.graph_inputs.size());
  for (size_t i = 0; i < num_inputs; ++i) {
    const OrtValue* value = nullptr;
    THROW_ON_ERROR(GetOrtApi().KernelContext_GetInput(kernel_context, i, &value));
    if (value == nullptr) {
      throw std::runtime_error("compiled partition received null input");
    }

    int is_tensor = 0;
    THROW_ON_ERROR(GetOrtApi().IsTensor(value, &is_tensor));
    if (!is_tensor) {
      throw std::runtime_error("compiled partition input is not a tensor");
    }

    if (partition.kernel_input_to_value[i] == kOptionalValueAbsent) {
      continue;  // compile-time-only input, ignore at runtime
    }
    float_input_values.push_back(value);
    input_metadata.push_back(getTensorMetadata(Ort::ConstValue{value}));
  }

  const ShapeKey shape_key = MakeShapeKey(input_metadata);
  if (state.active_graph == nullptr || state.active_graph->key != shape_key) {
    DestroyMaterializedGraph(state.active_graph);
    state.active_graph = BuildMaterializedGraph(partition, shape_key, input_metadata);
  }

  GGONNX_NOT_NULL(state.active_graph.get(), "active GGML graph must not be null");
  for (size_t i = 0; i < partition.graph_inputs.size(); ++i) {
    const size_t input_id = partition.graph_inputs[i];
    CopyInputDataToTensor(float_input_values[i],
                          state.active_graph->value_dims[input_id],
                          partition.values[input_id].name,
                          state.active_graph->input_tensors[i]);
  }

  if (state.active_graph->graph != nullptr &&
      ggml_backend_graph_compute(partition.backend, state.active_graph->graph) != GGML_STATUS_SUCCESS) {
    throw std::runtime_error("ggml_backend_graph_compute failed");
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

    if (partition.folded_constants.size() > output_id && partition.folded_constants[output_id].has_value()) {
      const ConstantTensor& constant = *partition.folded_constants[output_id];
      const size_t bytes = constant.data.size();
      if (bytes > 0) {
        std::memcpy(output_data, constant.data.data(), bytes);
      }
      continue;
    }

    ggml_tensor* output_tensor = state.active_graph->output_tensors[i];
    GGONNX_NOT_NULL(output_tensor, "missing cached GGML output tensor");
    AssertShapeMatchesGGML(output_dims, output_tensor, partition.values[output_id].name);
    GGONNX_ASSERT(ggml_is_contiguous(output_tensor),
                  "GGML output tensor must be contiguous before copy for '" +
                      partition.values[output_id].name + "'");

    const size_t bytes = ggml_nbytes(output_tensor);
    GGONNX_ASSERT(bytes == elementCount(output_dims) * sizeof(float),
                  "GGML output byte size mismatch for '" + partition.values[output_id].name + "'");
    ggml_backend_tensor_get(output_tensor, output_data, 0, bytes);
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
    state->partition = &info->partition;
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
    const MetaAnalysis meta_analysis = AnalyzeCompileTimeConstants(graph);
    SetActiveCompileTimeConstants(&meta_analysis.constants);
    struct ActiveConstantsGuard {
      ~ActiveConstantsGuard() { SetActiveCompileTimeConstants(nullptr); }
    } active_constants_guard;

    // Partition by walking nodes in topological order (how ORT yields them) and
    // sealing the current fuse group whenever we hit an unsupported node. Any
    // directed path between two supported nodes must pass through every node
    // that lies strictly between them in topological order, so an unsupported
    // node on such a path is guaranteed to split them into different groups
    // Because ONNX wants that and will complain if not in that way

    // Build the set of value names that are "live at this graph level" — either
    // consumed as a direct or implicit input by some node in this graph, or
    // named as a graph output. A node whose outputs are only consumed inside a
    // deeper nested subgraph (via implicit capture of the inner If/Loop/Scan
    // body) is NOT visible here: ORT's fused-node output computation only looks
    // at direct consumption at the current level, so including such a node in
    // a fuse group produces a partition with zero graph outputs and Compile
    // rejects it. Filter those nodes out of our capability.
    std::unordered_set<std::string> live_names;
    for (Ort::ConstNode node : ort_graph.GetNodes()) {
      for (Ort::ConstValueInfo v : node.GetInputs()) {
        if (v != nullptr) live_names.insert(std::string(v.GetName()));
      }
      for (Ort::ConstValueInfo v : node.GetImplicitInputs()) {
        if (v != nullptr) live_names.insert(std::string(v.GetName()));
      }
    }
    for (Ort::ConstValueInfo v : ort_graph.GetOutputs()) {
      if (v != nullptr) live_names.insert(std::string(v.GetName()));
    }
    auto has_visible_output = [&](Ort::ConstNode node) {
      for (Ort::ConstValueInfo v : node.GetOutputs()) {
        if (v == nullptr) continue;
        if (live_names.count(std::string(v.GetName())) > 0) return true;
      }
      return false;
    };

    std::vector<const OrtNode*> current_group;
    bool current_group_has_runtime_node = false;
    auto flush_group = [&] {
      if (current_group.empty()) return;
      if (!current_group_has_runtime_node) {
        current_group.clear();
        current_group_has_runtime_node = false;
        return;
      }
      THROW_ON_ERROR(GetOrtEpApi().EpGraphSupportInfo_AddNodesToFuse(
          graph_support_info, current_group.data(), current_group.size(), nullptr));
      current_group.clear();
      current_group_has_runtime_node = false;
    };

    for (Ort::ConstNode node : ort_graph.GetNodes()) {
      const bool supported = isNodeSupported(node) && has_visible_output(node);
      if (supported || meta_analysis.folded_nodes.count(NodeKey(node)) > 0) {
        current_group.push_back(node);
        current_group_has_runtime_node = current_group_has_runtime_node || supported;
      } else {
        flush_group();
      }
    }
    flush_group();
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
