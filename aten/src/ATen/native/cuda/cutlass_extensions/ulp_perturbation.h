/*
 * Copyright (c) 2024 
 * ULP (Unit in the Last Place) Perturbation utilities for CUTLASS GEMM
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <c10/util/floating_point_utils.h>
#include <cmath>  // for isnan, isinf

namespace cutlass {
namespace epilogue {
namespace thread {

/// ULP perturbation utilities for different data types
template<typename T>
CUTLASS_DEVICE
T ulp_perturb_down(T value) {
    // Default implementation - no perturbation for unsupported types
    return value;
}

/// Specialization for float (32-bit)
template<>
CUTLASS_DEVICE
float ulp_perturb_down<float>(float value) {
    // Handle special cases
    if (isnan(value) || isinf(value) || value == 0.0f) {
        return value;
    }
    
    // Reinterpret as integer for bit manipulation using portable c10 functions
    uint32_t bits = c10::detail::fp32_to_bits(value);
    
    // For positive numbers, decrement the mantissa
    // For negative numbers, increment the mantissa (to move towards zero)
    if (value > 0.0f) {
        bits--;
    } else {
        bits++;
    }
    
    return c10::detail::fp32_from_bits(bits);
}

/// Specialization for half (16-bit)
template<>
CUTLASS_DEVICE
cutlass::half_t ulp_perturb_down<cutlass::half_t>(cutlass::half_t value) {
    // Handle special cases
    __half val = static_cast<__half>(value);
    if (__hisnan(val) || __hisinf(val) || __heq(val, __float2half(0.0f))) {
        return value;
    }
    
    // Reinterpret as integer for bit manipulation
    uint16_t bits = __half_as_ushort(val);
    
    // For positive numbers, decrement the mantissa
    // For negative numbers, increment the mantissa (to move towards zero)
    if (__hgt(val, __float2half(0.0f))) {
        bits--;
    } else {
        bits++;
    }
    
    return cutlass::half_t(__ushort_as_half(bits));
}

/// Specialization for bfloat16
template<>
CUTLASS_DEVICE
cutlass::bfloat16_t ulp_perturb_down<cutlass::bfloat16_t>(cutlass::bfloat16_t value) {
    // Handle special cases
    if (isnan(static_cast<float>(value)) || isinf(static_cast<float>(value)) || value == cutlass::bfloat16_t(0.0f)) {
        return value;
    }
    
    // Convert to uint16 for bit manipulation
    uint16_t bits = value.raw();
    
    // For positive numbers, decrement the mantissa
    // For negative numbers, increment the mantissa (to move towards zero)
    if (value > cutlass::bfloat16_t(0.0f)) {
        bits--;
    } else {
        bits++;
    }
    
    cutlass::bfloat16_t result;
    result = cutlass::bfloat16_t::bitcast(bits);
    return result;
}

} // namespace thread
} // namespace epilogue
} // namespace cutlass