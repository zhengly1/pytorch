/*
 * ULP (Unit in the Last Place) utility functions for CUTLASS GEMM
 * Provides functions to perturb floating point values by 1 ULP downward
 */

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

namespace cutlass {
namespace gemm {
namespace thread {

// ULP perturbation for different data types
template<typename T>
CUTLASS_HOST_DEVICE
T perturb_ulp_down(T value) {
  return value; // Default: no perturbation
}

// Specialization for float
template<>
CUTLASS_HOST_DEVICE
float perturb_ulp_down<float>(float value) {
  if (value == 0.0f) return value;
  
  union {
    float f;
    uint32_t i;
  } u;
  u.f = value;
  
  // Perturb downward by 1 ULP
  if (value > 0.0f) {
    u.i--;
  } else {
    u.i++;
  }
  
  return u.f;
}

// Specialization for half precision
template<>
CUTLASS_HOST_DEVICE
cutlass::half_t perturb_ulp_down<cutlass::half_t>(cutlass::half_t value) {
  if (static_cast<float>(value) == 0.0f) return value;
  
  uint16_t bits = reinterpret_cast<const uint16_t&>(value);
  
  // Perturb downward by 1 ULP
  if (static_cast<float>(value) > 0.0f) {
    bits--;
  } else {
    bits++;
  }
  
  return reinterpret_cast<const cutlass::half_t&>(bits);
}

// Specialization for bfloat16
template<>
CUTLASS_HOST_DEVICE
cutlass::bfloat16_t perturb_ulp_down<cutlass::bfloat16_t>(cutlass::bfloat16_t value) {
  if (static_cast<float>(value) == 0.0f) return value;
  
  uint16_t bits = reinterpret_cast<const uint16_t&>(value);
  
  // Perturb downward by 1 ULP
  if (static_cast<float>(value) > 0.0f) {
    bits--;
  } else {
    bits++;
  }
  
  return reinterpret_cast<const cutlass::bfloat16_t&>(bits);
}

} // namespace thread
} // namespace gemm
} // namespace cutlass