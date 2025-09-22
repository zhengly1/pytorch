/*
 * Implementation of ULP-aware CUTLASS GEMM interface
 */

#include "ulp_gemm_interface.h"
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/thread/activation.h>

namespace {

template<typename T>
cutlass::Status dispatch_ulp_gemm(
    bool transpose_a,
    bool transpose_b,
    int m,
    int n,
    int k,
    float alpha,
    const T* a_ptr,
    int lda,
    const T* b_ptr,
    int ldb,
    float beta,
    T* c_ptr,
    int ldc,
    cudaStream_t stream) {

  // Define the GEMM operation
  using GemmDevice = cutlass::gemm::device::UlpGemmDevice<T, T, T>;

  // Setup problem size
  cutlass::gemm::GemmCoord problem_size(m, n, k);

  // Define tensor refs based on transpose flags
  cutlass::TensorRef<T const, cutlass::layout::RowMajor> tensor_a;
  cutlass::TensorRef<T const, cutlass::layout::RowMajor> tensor_b;
  
  if (transpose_a) {
    tensor_a = cutlass::TensorRef<T const, cutlass::layout::RowMajor>(a_ptr, lda);
  } else {
    tensor_a = cutlass::TensorRef<T const, cutlass::layout::RowMajor>(a_ptr, lda);
  }
  
  if (transpose_b) {
    tensor_b = cutlass::TensorRef<T const, cutlass::layout::RowMajor>(b_ptr, ldb);
  } else {
    tensor_b = cutlass::TensorRef<T const, cutlass::layout::RowMajor>(b_ptr, ldb);
  }

  cutlass::TensorRef<T, cutlass::layout::RowMajor> tensor_c(c_ptr, ldc);
  cutlass::TensorRef<T, cutlass::layout::RowMajor> tensor_d(c_ptr, ldc);

  // Create epilogue parameters
  typename GemmDevice::EpilogueOutputOp::Params epilogue_params(alpha, beta);

  // Setup GEMM arguments
  typename GemmDevice::Arguments arguments{
    problem_size,
    tensor_a,
    tensor_b,
    tensor_c,
    tensor_d,
    epilogue_params
  };

  // Get workspace size
  size_t workspace_size = GemmDevice::get_workspace_size(arguments);
  
  // Allocate workspace if needed
  void* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    if (cudaMalloc(&workspace_ptr, workspace_size) != cudaSuccess) {
      return cutlass::Status::kErrorMemoryAllocation;
    }
  }

  // Initialize the GEMM
  cutlass::Status status = GemmDevice::initialize(arguments, workspace_ptr, stream);
  if (status != cutlass::Status::kSuccess) {
    if (workspace_ptr) cudaFree(workspace_ptr);
    return status;
  }

  // Run the GEMM
  status = GemmDevice::run(arguments, workspace_ptr, stream);
  
  // Clean up workspace
  if (workspace_ptr) {
    cudaFree(workspace_ptr);
  }

  return status;
}

} // anonymous namespace

extern "C" {

int ulp_gemm_f32(
    bool transpose_a,
    bool transpose_b,
    int m,
    int n,
    int k,
    float alpha,
    const float* a_ptr,
    int lda,
    const float* b_ptr,
    int ldb,
    float beta,
    float* c_ptr,
    int ldc,
    cudaStream_t stream) {

  cutlass::Status status = dispatch_ulp_gemm<float>(
    transpose_a, transpose_b, m, n, k,
    alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc, stream
  );

  return static_cast<int>(status);
}

int ulp_gemm_f16(
    bool transpose_a,
    bool transpose_b,
    int m,
    int n,
    int k,
    float alpha,
    const cutlass::half_t* a_ptr,
    int lda,
    const cutlass::half_t* b_ptr,
    int ldb,
    float beta,
    cutlass::half_t* c_ptr,
    int ldc,
    cudaStream_t stream) {

  cutlass::Status status = dispatch_ulp_gemm<cutlass::half_t>(
    transpose_a, transpose_b, m, n, k,
    alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc, stream
  );

  return static_cast<int>(status);
}

int ulp_gemm_bf16(
    bool transpose_a,
    bool transpose_b,
    int m,
    int n,
    int k,
    float alpha,
    const cutlass::bfloat16_t* a_ptr,
    int lda,
    const cutlass::bfloat16_t* b_ptr,
    int ldb,
    float beta,
    cutlass::bfloat16_t* c_ptr,
    int ldc,
    cudaStream_t stream) {

  cutlass::Status status = dispatch_ulp_gemm<cutlass::bfloat16_t>(
    transpose_a, transpose_b, m, n, k,
    alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc, stream
  );

  return static_cast<int>(status);
}

} // extern "C"