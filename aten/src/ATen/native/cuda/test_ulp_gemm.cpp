/*
 * Test and example usage for ULP-aware CUTLASS GEMM
 * This demonstrates how to enable and test the ULP perturbation functionality
 */

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>

#include "UlpGemm.h"

// Helper function to create test tensors
at::Tensor create_test_tensor(std::vector<int64_t> sizes, at::ScalarType dtype, bool cuda = true) {
  auto options = at::TensorOptions().dtype(dtype);
  if (cuda) {
    options = options.device(at::kCUDA);
  }
  return at::randn(sizes, options);
}

// Basic test for ULP GEMM availability check
TEST(UlpGemmTest, AvailabilityCheck) {
  if (!at::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }
  
  // Test with different data types
  auto mat1_f32 = create_test_tensor({64, 128}, at::kFloat);
  auto mat2_f32 = create_test_tensor({128, 64}, at::kFloat);
  
  auto mat1_f16 = create_test_tensor({64, 128}, at::kHalf);
  auto mat2_f16 = create_test_tensor({128, 64}, at::kHalf);
  
  // These should pass basic availability checks if environment is set up
  // Note: Actual availability depends on environment variable PYTORCH_USE_ULP_GEMM=1
  bool f32_available = at::native::ulp_gemm::is_ulp_gemm_available(mat1_f32, mat2_f32);
  bool f16_available = at::native::ulp_gemm::is_ulp_gemm_available(mat1_f16, mat2_f16);
  
  // Test device support
  bool device_supported = at::native::ulp_gemm::is_device_supported();
  
  std::cout << "ULP GEMM Float32 available: " << (f32_available ? "Yes" : "No") << std::endl;
  std::cout << "ULP GEMM Float16 available: " << (f16_available ? "Yes" : "No") << std::endl;
  std::cout << "Device supported: " << (device_supported ? "Yes" : "No") << std::endl;
}

// Test ULP GEMM functionality if available
TEST(UlpGemmTest, BasicFunctionality) {
  if (!at::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }
  
  // Set environment variable to enable ULP GEMM
  setenv("PYTORCH_USE_ULP_GEMM", "1", 1);
  
  int64_t M = 64, N = 64, K = 64;
  
  auto mat1 = create_test_tensor({M, K}, at::kFloat);
  auto mat2 = create_test_tensor({K, N}, at::kFloat);
  auto bias = create_test_tensor({N}, at::kFloat);
  
  if (!at::native::ulp_gemm::is_ulp_gemm_available(mat1, mat2)) {
    GTEST_SKIP() << "ULP GEMM not available for current configuration";
  }
  
  // Test addmm with ULP GEMM enabled
  auto result_ulp = at::addmm(bias, mat1, mat2, 1.0, 1.0);
  
  // Disable ULP GEMM and compare with standard implementation
  setenv("PYTORCH_USE_ULP_GEMM", "0", 1);
  auto result_standard = at::addmm(bias, mat1, mat2, 1.0, 1.0);
  
  // Results should be different due to ULP perturbation
  bool results_different = !at::allclose(result_ulp, result_standard, 1e-6, 1e-6);
  
  EXPECT_TRUE(results_different) << "ULP perturbation should produce different results";
  
  // But the difference should be small (within reasonable bounds)
  auto diff = at::abs(result_ulp - result_standard);
  auto max_diff = at::max(diff).item<float>();
  
  EXPECT_LT(max_diff, 1e-3) << "ULP perturbation difference too large: " << max_diff;
  
  std::cout << "Maximum difference between ULP and standard GEMM: " << max_diff << std::endl;
}

// Performance comparison test
TEST(UlpGemmTest, PerformanceComparison) {
  if (!at::cuda::is_available()) {
    GTEST_SKIP() << "CUDA not available";
  }
  
  setenv("PYTORCH_USE_ULP_GEMM", "1", 1);
  
  int64_t M = 512, N = 512, K = 512;
  
  auto mat1 = create_test_tensor({M, K}, at::kFloat);
  auto mat2 = create_test_tensor({K, N}, at::kFloat);
  auto bias = create_test_tensor({N}, at::kFloat);
  
  if (!at::native::ulp_gemm::is_ulp_gemm_available(mat1, mat2)) {
    GTEST_SKIP() << "ULP GEMM not available for current configuration";
  }
  
  // Warm up
  for (int i = 0; i < 5; ++i) {
    auto result = at::addmm(bias, mat1, mat2, 1.0, 1.0);
    at::cuda::synchronize();
  }
  
  // Time ULP GEMM
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10; ++i) {
    auto result = at::addmm(bias, mat1, mat2, 1.0, 1.0);
  }
  at::cuda::synchronize();
  auto end = std::chrono::high_resolution_clock::now();
  auto ulp_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  
  // Time standard GEMM
  setenv("PYTORCH_USE_ULP_GEMM", "0", 1);
  
  // Warm up
  for (int i = 0; i < 5; ++i) {
    auto result = at::addmm(bias, mat1, mat2, 1.0, 1.0);
    at::cuda::synchronize();
  }
  
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10; ++i) {
    auto result = at::addmm(bias, mat1, mat2, 1.0, 1.0);
  }
  at::cuda::synchronize();
  end = std::chrono::high_resolution_clock::now();
  auto standard_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  
  std::cout << "ULP GEMM time: " << ulp_time << " microseconds" << std::endl;
  std::cout << "Standard GEMM time: " << standard_time << " microseconds" << std::endl;
  std::cout << "Performance ratio: " << static_cast<double>(ulp_time) / standard_time << std::endl;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  
  std::cout << "ULP GEMM Test Suite" << std::endl;
  std::cout << "===================" << std::endl;
  std::cout << "To enable ULP GEMM, set environment variable: PYTORCH_USE_ULP_GEMM=1" << std::endl;
  std::cout << "Requires CUDA-capable device with compute capability 8.0+" << std::endl;
  std::cout << std::endl;
  
  return RUN_ALL_TESTS();
}