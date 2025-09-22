# ULP-Aware CUTLASS GEMM Implementation

This implementation provides a custom CUTLASS GEMM that applies ULP (Unit in the Last Place) perturbation to the left operand before each addition operation during matrix multiplication.

## Overview

The implementation replaces the cuBLAS calls in PyTorch's `addmm_out_cuda_impl` function with a custom CUTLASS GEMM kernel that:

1. **Applies ULP perturbation**: Before each accumulation operation, the left operand (partial accumulation result) is perturbed downward by 1 ULP
2. **Maintains compatibility**: Works with existing PyTorch API without breaking changes
3. **Supports multiple data types**: Float32, Float16 (Half), and BFloat16
4. **Uses environment variable control**: Can be enabled/disabled via `PYTORCH_USE_ULP_GEMM=1`

## Files Added/Modified

### New Files:
- `aten/src/ATen/native/cuda/cutlass_extensions/ulp_utils.h` - ULP perturbation utilities
- `aten/src/ATen/native/cuda/cutlass_extensions/epilogue/ulp_linear_combination.h` - Custom CUTLASS epilogue
- `aten/src/ATen/native/cuda/cutlass_extensions/gemm/kernel/ulp_gemm.h` - ULP-aware GEMM kernel
- `aten/src/ATen/native/cuda/cutlass_extensions/ulp_gemm_interface.h` - High-level interface
- `aten/src/ATen/native/cuda/cutlass_extensions/ulp_gemm_interface.cu` - Interface implementation
- `aten/src/ATen/native/cuda/UlpGemm.h` - PyTorch wrapper header
- `aten/src/ATen/native/cuda/UlpGemm.cpp` - PyTorch wrapper implementation
- `aten/src/ATen/native/cuda/test_ulp_gemm.cpp` - Test suite

### Modified Files:
- `aten/src/ATen/native/cuda/Blas.cpp` - Integrated ULP GEMM into `addmm_out_cuda_impl`

## Architecture

```
PyTorch addmm_out_cuda_impl
    ↓
ULP GEMM availability check
    ↓
ULP GEMM dispatch (UlpGemm.cpp)
    ↓
CUTLASS interface (ulp_gemm_interface.cu)
    ↓
CUTLASS Device GEMM with ULP Epilogue
    ↓
ULP Linear Combination Epilogue
    ↓
ULP perturbation utilities
```

## Usage

### Enable ULP GEMM:
```bash
export PYTORCH_USE_ULP_GEMM=1
```

### Requirements:
- CUDA 12.5+ (as specified in problem statement)
- GCC 10+ (as specified in problem statement)
- GPU with compute capability 8.0+ (for Tensor Core operations)
- CUTLASS library (should be included with PyTorch CUDA builds)

### Example:
```python
import torch
import os

# Enable ULP GEMM
os.environ['PYTORCH_USE_ULP_GEMM'] = '1'

# Create test tensors
mat1 = torch.randn(128, 256, device='cuda', dtype=torch.float32)
mat2 = torch.randn(256, 128, device='cuda', dtype=torch.float32)
bias = torch.randn(128, device='cuda', dtype=torch.float32)

# This will use ULP GEMM if available
result = torch.addmm(bias, mat1, mat2)
```

## Implementation Details

### ULP Perturbation Algorithm:
```cpp
template<typename T>
CUTLASS_HOST_DEVICE
T perturb_ulp_down(T value) {
  if (value == 0.0f) return value;
  
  union {
    T f;
    uint32_t i;  // or uint16_t for half/bfloat16
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
```

### Integration Point:
The ULP GEMM is integrated into `addmm_out_cuda_impl` and will be used when:
1. `PYTORCH_USE_ULP_GEMM=1` environment variable is set
2. GPU has compute capability 8.0+
3. Input tensors are float32, float16, or bfloat16
4. No activation function is specified (activation functions still fall back to cuBLAS + post-processing)
5. Matrix dimensions meet minimum requirements (>= 16 in each dimension)

### Performance Considerations:
- ULP perturbation adds minimal computational overhead
- May be slightly slower than cuBLAS due to custom epilogue
- Memory usage is equivalent to standard GEMM operations
- Tensor Core acceleration is maintained where applicable

## Testing

Run the test suite to validate functionality:
```bash
# Build PyTorch with CUDA support
python setup.py develop

# Run ULP GEMM tests (requires CUDA-capable device)
cd aten/src/ATen/native/cuda
g++ -std=c++17 -I<pytorch_include_dirs> -I<cuda_include_dirs> \
    test_ulp_gemm.cpp -lgtest -lgtest_main -o test_ulp_gemm
./test_ulp_gemm
```

## Limitations

1. **Activation Functions**: When activation functions (RELU, GELU) are specified, the implementation falls back to cuBLAS + post-processing to maintain compatibility
2. **Complex Numbers**: Not currently supported for complex data types
3. **Architecture Requirements**: Requires modern GPU architecture (compute capability 8.0+)
4. **Matrix Size**: Small matrices (< 16 in any dimension) fall back to cuBLAS for efficiency

## Build Integration

To build with ULP GEMM support:

1. Ensure CUTLASS is available (typically included with PyTorch CUDA builds)
2. Build PyTorch with CUDA support:
   ```bash
   export CUDA_HOME=/usr/local/cuda-12.5
   export CC=gcc-10
   export CXX=g++-10
   python setup.py develop
   ```

## Verification

To verify ULP perturbation is working:
```python
import torch
import os

# Test without ULP
os.environ['PYTORCH_USE_ULP_GEMM'] = '0'
result1 = torch.addmm(bias, mat1, mat2)

# Test with ULP
os.environ['PYTORCH_USE_ULP_GEMM'] = '1' 
result2 = torch.addmm(bias, mat1, mat2)

# Results should be different due to ULP perturbation
print("Results are different:", not torch.allclose(result1, result2, rtol=1e-6))
print("Max difference:", torch.max(torch.abs(result1 - result2)).item())
```

## Future Enhancements

1. Support for activation functions in ULP epilogue
2. Complex number support
3. Batch GEMM support with ULP perturbation
4. Configurable ULP perturbation direction (up/down)
5. Integration with PyTorch's tunable GEMM framework