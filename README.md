# ggonnx

Out-of-tree ONNX Runtime Execution Provider that lowers ONNX graph partitions onto GGML.

> [!NOTE]
> Running on GGML backends does not automatically mean high performance. Espicially as GGML is focused on transformers and less on Conv nets. However, this porject still provides a viable path for custom hardware already with GGML support to run ONNX models without modification.

## Build and testing

```bash
mkdir build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_TYPE=Release -DLLAMA_CPP_ROOT=/path/to/your/local/llama.cpp
make -j
```

To test

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_runtime.py --ep-library build/libggonnx_ep.so
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_integration.py --ep-library build/libggonnx_ep.so
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_onnx_node_conformance.py --ep-library build/libggonnx_ep.so
```

By default the Execution Provider will attempt to initialize and use GGML GPU devices before falling back to GGML's CPU backend and ONNX's other backends.
