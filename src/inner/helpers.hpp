#pragma once

#include <onnxruntime/onnxruntime_c_api.h>

#include <stdexcept>
#include <string>

[[noreturn]] static void GGONNX_ABORT(const char* message) noexcept {
  fprintf(stderr, "%s\n", message);
  std::abort();
}

static void GGONNX_ASSERT(bool condition, const std::string& message) {
  if (condition) {
    return;
  }
  throw std::runtime_error(message);
}

template <typename T>
T* GGONNX_NOT_NULL(T* ptr, const std::string& message) {
  if (ptr == nullptr) {
    throw std::runtime_error(message);
  }
  return ptr;
}

template <typename T>
const T* GGONNX_NOT_NULL(const T* ptr, const std::string& message) {
  if (ptr == nullptr) {
    throw std::runtime_error(message);
  }
  return ptr;
}
