// Roofline Model Generator for GPU
//
// This program empirically generates a roofline curve by testing kernels
// with different arithmetic intensities (FLOP/Byte ratio).
//
// Roofline Model:
// - X-axis: Arithmetic Intensity (FLOP/Byte)
// - Y-axis: Performance (GFLOPS)
// - Shows the relationship between memory bandwidth and compute throughput

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>

// CUDA error checking macro
#define CUDA_CHECK(x) \
    do { \
        cudaError_t err = (x); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

////////////////////////////////////////////////////////////////////////////////
// Kernels with different arithmetic intensities

// Memory-bound kernel: 1 FLOP per load/store (2 memory ops)
// AI = 1 FLOP / 8 bytes = 0.125 FLOP/Byte
template<int ITERATIONS>
__global__ void kernel_ai_low(const float *__restrict__ in, float *__restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = in[idx];  // 1 load (4 bytes)
        #pragma unroll
        for (int i = 0; i < ITERATIONS; i++) {
            x = x + 1.0f;  // 1 FLOP
        }
        out[idx] = x;  // 1 store (4 bytes)
    }
    // AI = ITERATIONS FLOP / 8 bytes = ITERATIONS/8 FLOP/Byte
}

// Medium arithmetic intensity: Multiple FMAs per memory access
// AI = 2*N FLOP / 8 bytes
template<int NUM_OPS>
__global__ void kernel_ai_medium(const float *__restrict__ in, float *__restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = in[idx];  // 1 load
        #pragma unroll
        for (int i = 0; i < NUM_OPS; i++) {
            x = x * 1.01f + 0.99f;  // 2 FLOPs (FMA)
        }
        out[idx] = x;  // 1 store
    }
    // AI = 2*NUM_OPS FLOP / 8 bytes
}

// High arithmetic intensity: Many operations per memory access
template<int NUM_OPS>
__global__ void kernel_ai_high(const float *__restrict__ in, float *__restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = in[idx];
        float y = x;
        #pragma unroll
        for (int i = 0; i < NUM_OPS; i++) {
            x = x * y + x;  // 2 FLOPs
            y = y * x + y;  // 2 FLOPs
        }
        out[idx] = x + y;  // 1 FLOP
    }
    // AI = (4*NUM_OPS + 1) FLOP / 8 bytes
}

// Very high arithmetic intensity: Nested loops
template<int OUTER_OPS, int INNER_OPS>
__global__ void kernel_ai_very_high(const float *__restrict__ in, float *__restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = in[idx];
        #pragma unroll
        for (int i = 0; i < OUTER_OPS; i++) {
            #pragma unroll
            for (int j = 0; j < INNER_OPS; j++) {
                x = x * 1.001f + 0.999f;  // 2 FLOPs
            }
        }
        out[idx] = x;
    }
    // AI = 2*OUTER_OPS*INNER_OPS FLOP / 8 bytes
}

////////////////////////////////////////////////////////////////////////////////
// Benchmark helper

struct BenchmarkResult {
    double arithmetic_intensity;  // FLOP/Byte
    double gflops;               // Achieved GFLOPS
    double time_ms;              // Execution time in ms
};

template<typename KernelFunc>
BenchmarkResult benchmark_kernel(
    KernelFunc kernel,
    const float *d_in,
    float *d_out,
    int N,
    int total_flops,
    int memory_bytes,
    dim3 grid,
    dim3 block) {

    constexpr int NUM_WARMUP = 3;
    constexpr int NUM_ITERATIONS = 10;

    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++) {
        kernel<<<grid, block>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    std::vector<double> times;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        kernel<<<grid, block>>>(d_in, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        double time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        times.push_back(time_ns);
    }

    std::sort(times.begin(), times.end());
    double best_time_s = times[0] / 1e9;

    BenchmarkResult result;
    result.arithmetic_intensity = (double)total_flops / memory_bytes;
    result.gflops = total_flops / best_time_s / 1e9;
    result.time_ms = best_time_s * 1000.0;

    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Main

int main() {
    // Get GPU properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Peak Memory Bandwidth: " << prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2 / 1e6
              << " GB/s" << std::endl;

    // Calculate theoretical peak FLOPS (rough estimate)
    // P100: 56 SMs, 64 FP32 cores/SM, ~1.3 GHz boost clock
    double clock_ghz = prop.clockRate / 1e6;
    int cores_per_sm = 64;  // For Pascal (P100)
    double peak_gflops = prop.multiProcessorCount * cores_per_sm * clock_ghz * 2;  // *2 for FMA
    std::cout << "Estimated Peak GFLOPS: " << peak_gflops << std::endl;
    std::cout << std::endl;

    // Problem size
    constexpr int N = 16 * 1024 * 1024;  // 16M elements
    constexpr int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Allocate memory
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_in, 1, N * sizeof(float)));

    std::vector<BenchmarkResult> results;

    std::cout << "Arithmetic Intensity (FLOP/Byte), Performance (GFLOPS), Time (ms)" << std::endl;

    // Test different arithmetic intensities

    // Very low AI: 1 FLOP
    {
        auto result = benchmark_kernel(
            kernel_ai_low<1>,
            d_in, d_out, N,
            N * 1,           // total FLOPs
            N * 2 * 4,       // memory bytes (1 load + 1 store)
            grid, block
        );
        results.push_back(result);
        std::cout << result.arithmetic_intensity << ", " << result.gflops << ", " << result.time_ms << std::endl;
    }

    // Low AI: 2, 4, 8, 16 FLOPs
    for (int ops : {2, 4, 8, 16}) {
        BenchmarkResult result;
        switch(ops) {
            case 2:
                result = benchmark_kernel(kernel_ai_low<2>, d_in, d_out, N, N * 2, N * 2 * 4, grid, block);
                break;
            case 4:
                result = benchmark_kernel(kernel_ai_low<4>, d_in, d_out, N, N * 4, N * 2 * 4, grid, block);
                break;
            case 8:
                result = benchmark_kernel(kernel_ai_low<8>, d_in, d_out, N, N * 8, N * 2 * 4, grid, block);
                break;
            case 16:
                result = benchmark_kernel(kernel_ai_low<16>, d_in, d_out, N, N * 16, N * 2 * 4, grid, block);
                break;
        }
        results.push_back(result);
        std::cout << result.arithmetic_intensity << ", " << result.gflops << ", " << result.time_ms << std::endl;
    }

    // Medium AI: 32, 64, 128 FMA operations
    for (int ops : {16, 32, 64}) {
        BenchmarkResult result;
        switch(ops) {
            case 16:
                result = benchmark_kernel(kernel_ai_medium<16>, d_in, d_out, N, N * 2 * 16, N * 2 * 4, grid, block);
                break;
            case 32:
                result = benchmark_kernel(kernel_ai_medium<32>, d_in, d_out, N, N * 2 * 32, N * 2 * 4, grid, block);
                break;
            case 64:
                result = benchmark_kernel(kernel_ai_medium<64>, d_in, d_out, N, N * 2 * 64, N * 2 * 4, grid, block);
                break;
        }
        results.push_back(result);
        std::cout << result.arithmetic_intensity << ", " << result.gflops << ", " << result.time_ms << std::endl;
    }

    // High AI: 128, 256, 512 operations
    for (int ops : {64, 128, 256}) {
        BenchmarkResult result;
        switch(ops) {
            case 64:
                result = benchmark_kernel(kernel_ai_high<64>, d_in, d_out, N, N * (4 * 64 + 1), N * 2 * 4, grid, block);
                break;
            case 128:
                result = benchmark_kernel(kernel_ai_high<128>, d_in, d_out, N, N * (4 * 128 + 1), N * 2 * 4, grid, block);
                break;
            case 256:
                result = benchmark_kernel(kernel_ai_high<256>, d_in, d_out, N, N * (4 * 256 + 1), N * 2 * 4, grid, block);
                break;
        }
        results.push_back(result);
        std::cout << result.arithmetic_intensity << ", " << result.gflops << ", " << result.time_ms << std::endl;
    }

    // Very high AI: Nested loops
    {
        auto result = benchmark_kernel(
            kernel_ai_very_high<32, 32>,
            d_in, d_out, N,
            N * 2 * 32 * 32,  // total FLOPs
            N * 2 * 4,        // memory bytes
            grid, block
        );
        results.push_back(result);
        std::cout << result.arithmetic_intensity << ", " << result.gflops << ", " << result.time_ms << std::endl;
    }

    {
        auto result = benchmark_kernel(
            kernel_ai_very_high<64, 32>,
            d_in, d_out, N,
            N * 2 * 64 * 32,
            N * 2 * 4,
            grid, block
        );
        results.push_back(result);
        std::cout << result.arithmetic_intensity << ", " << result.gflops << ", " << result.time_ms << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    std::cout << std::endl;
    std::cout << "=== Roofline Analysis ===" << std::endl;
    std::cout << "Memory bandwidth bound region: Low AI (< ~10 FLOP/Byte)" << std::endl;
    std::cout << "Compute bound region: High AI (> ~10 FLOP/Byte)" << std::endl;
    std::cout << std::endl;
    std::cout << "To visualize: plot AI (x-axis) vs GFLOPS (y-axis)" << std::endl;
    std::cout << "Expected shape: Linear rise (memory-bound) → plateau (compute-bound)" << std::endl;

    return 0;
}
