%%writefile mem-latency.cu
// Memory Latency Measurement Kernels
//
// This file contains various CUDA kernels to measure the latency of
// memory accesses at different levels of the memory hierarchy:
// - L1 cache latency measurement
// - L2 cache latency measurement
// - Global memory latency measurement

#include <cuda_runtime.h>
#include <iostream>

using data_type = float;

////////////////////////////////////////////////////////////////////////////////
// L1 Cache Memory Latency

__global__ void l1_mem_latency(
    unsigned long *time_start,
    unsigned long *time_end,
    data_type *array_1,
    data_type *array_2) {
    unsigned long start_time, end_time;
    unsigned long addr = (unsigned long)array_2;
    unsigned long result;

    // Warm up L1 cache - load the data into L1
    asm volatile(
        "ld.global.ca.u64 %0, [%1];\n\t"
        : "=l"(result)
        : "l"(addr)
    );

    // Measure L1 cache latency with dependent loads from same address
    asm volatile(
        "mov.u64 %0, %%clock64;\n\t"  // Start timing

        // Dependent chain: each load must complete before the next
        "ld.global.ca.u64 %1, [%2];\n\t"
        "add.u64 %1, %1, 0;\n\t"  // Create dependency
        "ld.global.ca.u64 %1, [%2];\n\t"
        "add.u64 %1, %1, 0;\n\t"
        "ld.global.ca.u64 %1, [%2];\n\t"
        "add.u64 %1, %1, 0;\n\t"
        "ld.global.ca.u64 %1, [%2];\n\t"
        "add.u64 %1, %1, 0;\n\t"
        "ld.global.ca.u64 %1, [%2];\n\t"
        "add.u64 %1, %1, 0;\n\t"
        "ld.global.ca.u64 %1, [%2];\n\t"
        "add.u64 %1, %1, 0;\n\t"
        "ld.global.ca.u64 %1, [%2];\n\t"
        "add.u64 %1, %1, 0;\n\t"
        "ld.global.ca.u64 %1, [%2];\n\t"
        "add.u64 %1, %1, 0;\n\t"
        "ld.global.ca.u64 %1, [%2];\n\t"
        "add.u64 %1, %1, 0;\n\t"
        "ld.global.ca.u64 %1, [%2];\n\t"

        "mov.u64 %3, %%clock64;\n\t"  // End timing
        : "=l"(start_time), "=l"(result), "+l"(addr), "=l"(end_time)
    );

    *time_start = start_time;
    *time_end = end_time;
    *((unsigned long *)array_1) = result;
}

////////////////////////////////////////////////////////////////////////////////
// L2 Cache Memory Latency

__global__ void l2_mem_latency(
    unsigned long *time_start,
    unsigned long *time_end,
    data_type *array_1,
    data_type *array_2) {
    unsigned long start_time, end_time;
    unsigned long addr = (unsigned long)array_2;
    unsigned long result;

    // Warm up L2 cache (bypass L1)
    asm volatile(
        "ld.global.cg.u64 %0, [%1];\n\t"
        : "=l"(result)
        : "l"(addr)
    );

    asm volatile("membar.gl;\n\t" ::: "memory");

    // Measure L2 cache latency with dependent loads from same address
    asm volatile(
        "mov.u64 %0, %%clock64;\n\t"  // Start timing

        // Dependent chain: each load must complete before the next
        "ld.global.cg.u64 %1, [%2];\n\t"
        "add.u64 %1, %1, 0;\n\t"
        "ld.global.cg.u64 %1, [%2];\n\t"
        "add.u64 %1, %1, 0;\n\t"
        "ld.global.cg.u64 %1, [%2];\n\t"
        "add.u64 %1, %1, 0;\n\t"
        "ld.global.cg.u64 %1, [%2];\n\t"
        "add.u64 %1, %1, 0;\n\t"
        "ld.global.cg.u64 %1, [%2];\n\t"
        "add.u64 %1, %1, 0;\n\t"
        "ld.global.cg.u64 %1, [%2];\n\t"
        "add.u64 %1, %1, 0;\n\t"
        "ld.global.cg.u64 %1, [%2];\n\t"
        "add.u64 %1, %1, 0;\n\t"
        "ld.global.cg.u64 %1, [%2];\n\t"
        "add.u64 %1, %1, 0;\n\t"
        "ld.global.cg.u64 %1, [%2];\n\t"
        "add.u64 %1, %1, 0;\n\t"
        "ld.global.cg.u64 %1, [%2];\n\t"

        "mov.u64 %3, %%clock64;\n\t"  // End timing
        : "=l"(start_time), "=l"(result), "+l"(addr), "=l"(end_time)
    );

    *time_start = start_time;
    *time_end = end_time;
    *((unsigned long *)array_1) = result;
}

////////////////////////////////////////////////////////////////////////////////
// Global Memory Latency

__global__ void global_mem_latency(
    unsigned long *time_start,
    unsigned long *time_end,
    data_type *array_1,
    data_type *array_2) {
    unsigned long start_time, end_time;
    unsigned long ptr = (unsigned long)array_2;

    asm volatile("membar.gl;\n\t" ::: "memory");

    // Measure latency with pointer chasing
    asm volatile(
        "mov.u64 %0, %%clock64;\n\t"  // Start timing

        // TODO: Fill in 10 global memory loads using ld.global.cv.u64
        // Each load should use the result from the previous load as the address
        "ld.global.cv.u64 %1, [%1];\n\t"
        "ld.global.cv.u64 %1, [%1];\n\t"
        "ld.global.cv.u64 %1, [%1];\n\t"
        "ld.global.cv.u64 %1, [%1];\n\t"
        "ld.global.cv.u64 %1, [%1];\n\t"
        "ld.global.cv.u64 %1, [%1];\n\t"
        "ld.global.cv.u64 %1, [%1];\n\t"
        "ld.global.cv.u64 %1, [%1];\n\t"
        "ld.global.cv.u64 %1, [%1];\n\t"
        "ld.global.cv.u64 %1, [%1];\n\t"

        "mov.u64 %2, %%clock64;\n\t"  // End timing
        : "=l"(start_time), "+l"(ptr), "=l"(end_time)
    );

    *time_start = start_time;
    *time_end = end_time;
    *((unsigned long *)array_1) = ptr;
}

////////////////////////////////////////////////////////////////////////////////
// Memory Bandwidth Measurement Kernels
//
// Latency measurement uses dependent chains to force serial execution,
// while bandwidth measurement requires maximizing parallelism.
// Key differences:
// - Latency: Each load depends on the previous result (serial)
// - Bandwidth: All loads are independent (parallel)

// L1 Cache Bandwidth Measurement (improved version, based on Chips and Cheese)
// Data size: 4-16 KB (ensure it fits in L1)
// Use vectorized reads + multiple independent accumulators to maximize bandwidth
__global__ void l1_mem_bandwidth(
    data_type *data,
    data_type *output,
    int num_elements,
    int iterations_per_thread
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Use multiple independent accumulators to increase instruction-level parallelism (ILP)
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    // num_elements must be a power of 2, use bitwise operation for masking
    int mask = num_elements - 1;

    // Multiple iterations, reading 4 values per iteration
    // Note: For cache tests, we "want" to repeatedly access the same small dataset
    // Don't use scattered access, since the goal is to measure cache hit bandwidth
    #pragma unroll 1  // Prevent excessive unrolling
    for (int iter = 0; iter < iterations_per_thread; iter += 4) {
        // 4 independent load addresses (leverage Pascal's dual-issue capability)
        int idx0 = (tid + (iter + 0) * total_threads) & mask;
        int idx1 = (tid + (iter + 1) * total_threads) & mask;
        int idx2 = (tid + (iter + 2) * total_threads) & mask;
        int idx3 = (tid + (iter + 3) * total_threads) & mask;

        // 4 independent load operations
        float val0, val1, val2, val3;
        asm volatile(
            "ld.global.ca.f32 %0, [%1];\n\t"
            : "=f"(val0)
            : "l"(&data[idx0])
        );
        asm volatile(
            "ld.global.ca.f32 %0, [%1];\n\t"
            : "=f"(val1)
            : "l"(&data[idx1])
        );
        asm volatile(
            "ld.global.ca.f32 %0, [%1];\n\t"
            : "=f"(val2)
            : "l"(&data[idx2])
        );
        asm volatile(
            "ld.global.ca.f32 %0, [%1];\n\t"
            : "=f"(val3)
            : "l"(&data[idx3])
        );

        // Accumulate into different accumulators
        sum0 += val0;
        sum1 += val1;
        sum2 += val2;
        sum3 += val3;
    }

    // Merge results and prevent compiler optimization
    float final_sum = sum0 + sum1 + sum2 + sum3;
    if (tid == 0) {
        output[0] = final_sum;
    }
}

// L2 Cache Bandwidth Measurement
// Data size: > 128KB but < 48MB (larger than L1, smaller than L2)
// Use ld.global.cg instruction (cache only in L2, bypass L1)
__global__ void l2_mem_bandwidth(
    data_type *data,
    data_type *output,
    int num_elements,
    int iterations_per_thread
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    float sum = 0.0f;

    // num_elements must be a power of 2, use bitwise operation instead of modulo
    int mask = num_elements - 1;

    for (int iter = 0; iter < iterations_per_thread; iter++) {
        // Use bitwise & instead of expensive modulo operation %
        int idx = (tid + iter * total_threads) & mask;

        // Use PTX inline assembly to ensure bypass L1, use only L2
        float val;
        asm volatile(
            "ld.global.cg.f32 %0, [%1];\n\t"
            : "=f"(val)
            : "l"(&data[idx])
        );
        sum += val;
    }

    if (tid == 0) {
        output[0] = sum;
    }
}

// DRAM Bandwidth Measurement
// Data size: >> 48MB (much larger than L2 capacity)
// Use ld.global.cv instruction (no caching, read from DRAM every time)
__global__ void dram_mem_bandwidth(
    data_type *data,
    data_type *output,
    int num_elements,
    int iterations_per_thread
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    float sum = 0.0f;

    // num_elements must be a power of 2, use bitwise operation instead of modulo
    int mask = num_elements - 1;

    for (int iter = 0; iter < iterations_per_thread; iter++) {
        // Use bitwise & instead of expensive modulo operation %
        int idx = (tid + iter * total_threads) & mask;

        // Use PTX inline assembly to ensure no caching
        float val;
        asm volatile(
            "ld.global.cv.f32 %0, [%1];\n\t"
            : "=f"(val)
            : "l"(&data[idx])
        );
        sum += val;
    }

    if (tid == 0) {
        output[0] = sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

// CUDA error checking macro
#define CUDA_CHECK(x) \
    do { \
        cudaError_t err = (x); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error " << static_cast<int>(err) << " (" \
                      << cudaGetErrorString(err) << ") at " << __FILE__ << ":" \
                      << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// Macro to run kernel and print timing results
#define run_kernel_and_print(kernel_name, num_loads) \
    do { \
        unsigned long h_time_start, h_time_end; \
        kernel_name<<<1, 1>>>(d_time_start, d_time_end, array_1, array_2); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
        CUDA_CHECK(cudaMemcpy( \
            &h_time_start, \
            d_time_start, \
            sizeof(unsigned long), \
            cudaMemcpyDeviceToHost)); \
        CUDA_CHECK(cudaMemcpy( \
            &h_time_end, \
            d_time_end, \
            sizeof(unsigned long), \
            cudaMemcpyDeviceToHost)); \
        unsigned long long total_cycles = h_time_end - h_time_start; \
        unsigned long long latency = total_cycles / (num_loads); \
        std::cout << #kernel_name " latency = \t" << latency \
                  << " cycles (total: " << total_cycles \
                  << " cycles for " << (num_loads) << " loads)" << std::endl; \
    } while (0)

// Bandwidth test helper function
void test_bandwidth(
    const char* cache_level,
    void (*kernel)(data_type*, data_type*, int, int),
    int num_elements,
    int iterations_per_thread,
    int num_blocks,
    int threads_per_block,
    float gpu_clock_ghz
) {
    data_type *d_data = nullptr;
    data_type *d_output = nullptr;

    // Allocate memory
    CUDA_CHECK(cudaMalloc(&d_data, num_elements * sizeof(data_type)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(data_type)));

    // Initialize data (avoid uninitialized memory issues)
    CUDA_CHECK(cudaMemset(d_data, 0, num_elements * sizeof(data_type)));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(data_type)));

    // Warm-up run (ensure data is loaded into cache)
    kernel<<<num_blocks, threads_per_block>>>(d_data, d_output, num_elements, iterations_per_thread);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start time
    CUDA_CHECK(cudaEventRecord(start));

    // Execute kernel
    kernel<<<num_blocks, threads_per_block>>>(d_data, d_output, num_elements, iterations_per_thread);

    // Record end time
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate execution time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Calculate bandwidth
    long long total_threads_count = (long long)num_blocks * threads_per_block;
    long long total_loads = total_threads_count * iterations_per_thread;
    long long total_bytes = total_loads * sizeof(data_type);
    double bandwidth_gb_s = (total_bytes / 1e9) / (milliseconds / 1000.0);

    std::cout << cache_level << " Bandwidth Test:" << std::endl;
    std::cout << "  Data size: " << (num_elements * sizeof(data_type) / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Number of threads: " << total_threads_count << std::endl;
    std::cout << "  Iterations per thread: " << iterations_per_thread << std::endl;
    std::cout << "  Total read amount: " << (total_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "  Execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "  Measured bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
    std::cout << std::endl;

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    // Initialize CUDA and allocate device memory
    unsigned long *d_time_start = nullptr;
    unsigned long *d_time_end = nullptr;
    data_type *array_1 = nullptr;
    data_type *array_2 = nullptr;

    unsigned long host_init_time = 0ull;

    CUDA_CHECK(cudaMalloc(&d_time_start, sizeof(unsigned long)));
    CUDA_CHECK(cudaMalloc(&d_time_end, sizeof(unsigned long)));
    CUDA_CHECK(cudaMalloc(&array_1, sizeof(unsigned long) * 60));
    CUDA_CHECK(cudaMalloc(&array_2, sizeof(unsigned long) * 60));

    // Create a pointer that points to itself
    // This ensures pointer chasing always stays in the same cache line
    unsigned long self_ptr = (unsigned long)array_2;
    CUDA_CHECK(cudaMemcpy(array_2, &self_ptr, sizeof(unsigned long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_time_start,
        &host_init_time,
        sizeof(unsigned long),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_time_end,
        &host_init_time,
        sizeof(unsigned long),
        cudaMemcpyHostToDevice));

    std::cout << "========================================" << std::endl;
    std::cout << "Memory Latency Measurement" << std::endl;
    std::cout << "========================================" << std::endl;
    run_kernel_and_print(global_mem_latency, 10);
    run_kernel_and_print(l2_mem_latency, 10);
    run_kernel_and_print(l1_mem_latency, 10);
    std::cout << std::endl;

    // Clean up latency test memory
    CUDA_CHECK(cudaFree(d_time_start));
    CUDA_CHECK(cudaFree(d_time_end));
    CUDA_CHECK(cudaFree(array_1));
    CUDA_CHECK(cudaFree(array_2));

    std::cout << "========================================" << std::endl;
    std::cout << "Memory Bandwidth Measurement" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // ============================================================
    // Tesla P100 Specifications:
    // - 56 SMs
    // - Per SM: 64 KB L1/Shared Memory (configurable)
    // - L2 Cache: 4 MB (shared across entire GPU)
    // - DRAM: 16 GB HBM2, bandwidth ~720 GB/s
    // - L2 bandwidth: ~2-3 TB/s
    // ============================================================

    float gpu_clock_ghz = 1.5f;  // Tesla P100 ~1.3-1.5 GHz

    // Test parameter configuration
    int num_blocks = 56 * 2;      // Use multiple blocks to fully utilize all SMs
    int threads_per_block = 256;  // 256 threads per block
    int total_threads = num_blocks * threads_per_block;  // 28,672 threads

    std::cout << "Tesla P100 Specs: 56 SMs, L1 total capacity ~3.5 MB, L2 capacity 4 MB" << std::endl;
    std::cout << std::endl;

    // L1 Cache Bandwidth Test (based on Chips and Cheese method)
    // Strategy: minimal working set (4 KB) + many iterations + multiple independent reads
    // Data size must be power of 2 to use bitwise operation instead of modulo
    // 4 KB ensures data fits entirely in L1 cache
    int l1_elements = 1 << 10;  // 2^10 = 1,024 floats = 4 KB
    int l1_iterations = 80000;  // Many iterations, must be multiple of 4
    std::cout << "L1 Test (improved): Data " << (l1_elements * 4.0 / 1024) << " KB, iterations " << l1_iterations << std::endl;
    std::cout << "  Using multiple independent accumulators + leverage Pascal dual-issue capability" << std::endl;
    test_bandwidth(
        "L1 Cache",
        l1_mem_bandwidth,
        l1_elements,
        l1_iterations,
        num_blocks,
        threads_per_block,
        gpu_clock_ghz
    );

    // L2 Cache Bandwidth Test
    // Strategy: moderate dataset (close to L2 capacity but > L1) + moderate iterations
    int l2_elements = 1 << 20;  // 2^20 = 1,048,576 floats = 4 MB (close to L2's 4 MB)
    int l2_iterations = 5000;
    std::cout << "L2 Test: Data " << (l2_elements * 4.0 / 1024 / 1024) << " MB, iterations " << l2_iterations << std::endl;
    test_bandwidth(
        "L2 Cache",
        l2_mem_bandwidth,
        l2_elements,
        l2_iterations,
        num_blocks,
        threads_per_block,
        gpu_clock_ghz
    );

    // DRAM Bandwidth Test
    // Strategy: very large dataset (>> L2 capacity) + fewer iterations
    // Tesla P100's DRAM bandwidth ~720 GB/s
    int dram_elements = 1 << 26;  // 2^26 = 67,108,864 floats = 256 MB (>> 4 MB L2)
    int dram_iterations = 500;  // Fewer iterations since dataset is already large
    std::cout << "DRAM Test: Data " << (dram_elements * 4.0 / 1024 / 1024) << " MB, iterations " << dram_iterations << std::endl;
    test_bandwidth(
        "DRAM",
        dram_mem_bandwidth,
        dram_elements,
        dram_iterations,
        num_blocks,
        threads_per_block,
        gpu_clock_ghz
    );

    std::cout << "========================================" << std::endl;
    std::cout << "Test Complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
