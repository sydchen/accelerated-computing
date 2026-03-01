%%writefile fma_latency.cu
// FMA Latency Measurement Kernels
//
// This file contains various CUDA kernels to measure the latency of
// fused multiply-add (FMA) operations under different execution patterns:
// - Basic latency measurement
// - Interleaved execution (ILP)
// - Non-interleaved execution (sequential chains)

#include <cuda_runtime.h>
#include <iostream>

using data_type = float;

// Inline assembly macro to read GPU cycle counter
#define clock_cycle() \
    ({ \
        unsigned long long ret; \
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(ret)); \
        ret; \
    })

////////////////////////////////////////////////////////////////////////////////
// Basic FMA Latency

__global__ void
fma_latency(data_type *n, unsigned long long *d_start, unsigned long long *d_end) {
    unsigned long long start_time = clock_cycle();
    unsigned long long end_time = clock_cycle();
    data_type x = *n;
    // Memory fence to ensure that the reads are done.
    __threadfence();
    start_time = clock_cycle();

    /// <--- /your code here --->
    // Chain dependent FMA operations to measure latency
    // x = x * x + x creates dependency chain and generates FFMA instruction
    #pragma unroll
    for (int i = 0; i < 100; i++) {
        x = x * x + x;  // FFMA: multiply x by x, then add x
    }

    end_time = clock_cycle();

    *n = x;
    *d_start = start_time;
    *d_end = end_time;
}

////////////////////////////////////////////////////////////////////////////////
// FMA Latency + Instruction Level Parallelism (Interleaved)

__global__ void fma_latency_interleaved(
    data_type *n,
    unsigned long long *d_start,
    unsigned long long *d_end) {
    unsigned long long start_time = clock_cycle();
    unsigned long long end_time = clock_cycle();

    data_type x = *n;
    data_type y = *n;
    // Memory fence to ensure that the reads are done.
    __threadfence();

    start_time = clock_cycle();

    /// <--- /your code here --->
    // Explicitly interleave two independent FMA chains to exploit ILP
    #pragma unroll
    for (int i = 0; i < 100; i++) {
        x = x * x + x;  // Chain 1 (FFMA)
        y = y * y + y;  // Chain 2 (independent of chain 1, FFMA)
    }

    end_time = clock_cycle();

    *n = x + y;
    *d_start = start_time;
    *d_end = end_time;
}

////////////////////////////////////////////////////////////////////////////////
// FMA Latency + Sequential Execution (No Interleaving)

__global__ void fma_latency_no_interleave(
    data_type *n,
    unsigned long long *d_start,
    unsigned long long *d_end) {

    unsigned long long start_time = clock_cycle();
    unsigned long long end_time = clock_cycle();

    data_type x = *n;
    data_type y = *n;
    // Memory fence to ensure that the reads are done.
    __threadfence();

    start_time = clock_cycle();

    /// <--- /your code here --->
    // Two independent chains, but not explicitly interleaved
    // Let the compiler/hardware exploit ILP automatically
    #pragma unroll
    for (int i = 0; i < 100; i++) {
        x = x * x + x;  // Complete chain 1 first (FFMA)
    }
    #pragma unroll
    for (int i = 0; i < 100; i++) {
        y = y * y + y;  // Then complete chain 2 (FFMA)
    }

    end_time = clock_cycle();

    *n = x + y;
    *d_start = start_time;
    *d_end = end_time;
}

////////////////////////////////////////////////////////////////////////////////
// Additional Instruction Latency Tests

// Integer ADD latency (IADD)
__global__ void iadd_latency(
    int *n,
    unsigned long long *d_start,
    unsigned long long *d_end) {
    unsigned long long start_time = clock_cycle();
    unsigned long long end_time = clock_cycle();
    int x = *n;
    __threadfence();
    start_time = clock_cycle();

    #pragma unroll
    for (int i = 0; i < 100; i++) {
        x = x + x;  // IADD: dependent chain
    }

    end_time = clock_cycle();
    *n = x;
    *d_start = start_time;
    *d_end = end_time;
}

// Integer MUL latency (IMUL)
__global__ void imul_latency(
    int *n,
    unsigned long long *d_start,
    unsigned long long *d_end) {
    unsigned long long start_time = clock_cycle();
    unsigned long long end_time = clock_cycle();
    int x = *n;
    __threadfence();
    start_time = clock_cycle();

    #pragma unroll
    for (int i = 0; i < 100; i++) {
        x = x * 3 + 1;  // IMUL + IADD
    }

    end_time = clock_cycle();
    *n = x;
    *d_start = start_time;
    *d_end = end_time;
}

// Float ADD latency (FADD)
__global__ void fadd_latency(
    data_type *n,
    unsigned long long *d_start,
    unsigned long long *d_end) {
    unsigned long long start_time = clock_cycle();
    unsigned long long end_time = clock_cycle();
    data_type x = *n;
    __threadfence();
    start_time = clock_cycle();

    #pragma unroll
    for (int i = 0; i < 100; i++) {
        x = x + x;  // FADD: dependent chain
    }

    end_time = clock_cycle();
    *n = x;
    *d_start = start_time;
    *d_end = end_time;
}

// Float MUL latency (FMUL)
__global__ void fmul_latency(
    data_type *n,
    unsigned long long *d_start,
    unsigned long long *d_end) {
    unsigned long long start_time = clock_cycle();
    unsigned long long end_time = clock_cycle();
    data_type x = *n;
    __threadfence();
    start_time = clock_cycle();

    #pragma unroll
    for (int i = 0; i < 100; i++) {
        x = x * x;  // FMUL: dependent chain
    }

    end_time = clock_cycle();
    *n = x;
    *d_start = start_time;
    *d_end = end_time;
}

// Float DIV latency (FDIV)
__global__ void fdiv_latency(
    data_type *n,
    unsigned long long *d_start,
    unsigned long long *d_end) {
    unsigned long long start_time = clock_cycle();
    unsigned long long end_time = clock_cycle();
    data_type x = *n;
    __threadfence();
    start_time = clock_cycle();

    #pragma unroll
    for (int i = 0; i < 100; i++) {
        x = 1.0f / (x + 0.1f);  // FDIV: dependent chain (add small value to avoid div by zero)
    }

    end_time = clock_cycle();
    *n = x;
    *d_start = start_time;
    *d_end = end_time;
}

// Float SQRT latency (FSQRT)
__global__ void fsqrt_latency(
    data_type *n,
    unsigned long long *d_start,
    unsigned long long *d_end) {
    unsigned long long start_time = clock_cycle();
    unsigned long long end_time = clock_cycle();
    data_type x = *n;
    __threadfence();
    start_time = clock_cycle();

    #pragma unroll
    for (int i = 0; i < 100; i++) {
        x = sqrtf(x + 1.0f);  // FSQRT: dependent chain
    }

    end_time = clock_cycle();
    *n = x;
    *d_start = start_time;
    *d_end = end_time;
}

// Global memory load latency
__global__ void global_mem_latency(
    data_type *arr,
    int *indices,
    unsigned long long *d_start,
    unsigned long long *d_end) {
    unsigned long long start_time = clock_cycle();
    unsigned long long end_time = clock_cycle();

    __threadfence();
    start_time = clock_cycle();

    // Pointer-chasing pattern to measure memory latency
    // Each load depends on the previous load result
    int idx = 0;
    #pragma unroll
    for (int i = 0; i < 100; i++) {
        idx = indices[idx];  // Load next index from memory
    }

    end_time = clock_cycle();
    *d_start = start_time;
    *d_end = end_time;
    arr[0] = (data_type)idx;  // Write result to prevent optimization
}

// Shared memory load latency
__global__ void shared_mem_latency(
    data_type *result,
    unsigned long long *d_start,
    unsigned long long *d_end) {
    __shared__ int s_indices[128];

    // Initialize shared memory with pointer-chasing pattern
    for (int i = threadIdx.x; i < 128; i += blockDim.x) {
        s_indices[i] = (i + 1) % 128;
    }
    __syncthreads();

    unsigned long long start_time = clock_cycle();
    unsigned long long end_time = clock_cycle();

    start_time = clock_cycle();

    // Pointer-chasing in shared memory
    int idx = 0;
    #pragma unroll
    for (int i = 0; i < 100; i++) {
        idx = s_indices[idx];
    }

    end_time = clock_cycle();

    if (threadIdx.x == 0) {
        *d_start = start_time;
        *d_end = end_time;
        *result = (data_type)idx;
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

// Macro to run kernel and print timing results.
#define run_kernel_and_print(kernel, d_n, d_start, d_end) \
    do { \
        unsigned long long h_time_start = 0ull, h_time_end = 0ull; \
        data_type result = 0.0f; \
\
        kernel<<<1, 1>>>(d_n, d_start, d_end); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
        CUDA_CHECK(cudaMemcpy( \
            &h_time_start, \
            d_start, \
            sizeof(unsigned long long), \
            cudaMemcpyDeviceToHost)); \
        CUDA_CHECK(cudaMemcpy( \
            &h_time_end, \
            d_end, \
            sizeof(unsigned long long), \
            cudaMemcpyDeviceToHost)); \
        CUDA_CHECK(cudaMemcpy(&result, d_n, sizeof(data_type), cudaMemcpyDeviceToHost)); \
\
        std::cout << "Latency of " << #kernel \
                  << " code snippet = " << (h_time_end - h_time_start) << " cycles" \
                  << std::endl; \
    } while (0)

int main() {
    data_type *d_n = nullptr;
    int *d_n_int = nullptr;
    unsigned long long *d_start = nullptr;
    unsigned long long *d_end = nullptr;

    data_type host_val = 4.0f;
    int host_val_int = 3;
    unsigned long long host_init_time = 0ull;

    CUDA_CHECK(cudaMalloc(&d_n, sizeof(data_type)));
    CUDA_CHECK(cudaMalloc(&d_n_int, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_start, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_end, sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(d_n, &host_val, sizeof(data_type), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_n_int, &host_val_int, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_start,
        &host_init_time,
        sizeof(unsigned long long),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_end,
        &host_init_time,
        sizeof(unsigned long long),
        cudaMemcpyHostToDevice));

    // Test FFMA latency
    std::cout << "\n=== FFMA Latency Tests (x*x+x) ===" << std::endl;
    run_kernel_and_print(fma_latency, d_n, d_start, d_end);
    run_kernel_and_print(fma_latency_interleaved, d_n, d_start, d_end);
    run_kernel_and_print(fma_latency_no_interleave, d_n, d_start, d_end);

    // Test integer arithmetic latency
    std::cout << "\n=== Integer Arithmetic Latency Tests ===" << std::endl;
    run_kernel_and_print(iadd_latency, d_n_int, d_start, d_end);
    run_kernel_and_print(imul_latency, d_n_int, d_start, d_end);

    // Test float arithmetic latency
    std::cout << "\n=== Float Arithmetic Latency Tests ===" << std::endl;
    run_kernel_and_print(fadd_latency, d_n, d_start, d_end);
    run_kernel_and_print(fmul_latency, d_n, d_start, d_end);
    run_kernel_and_print(fdiv_latency, d_n, d_start, d_end);
    run_kernel_and_print(fsqrt_latency, d_n, d_start, d_end);

    // Test memory subsystem latency
    std::cout << "\n=== Memory Subsystem Latency Tests ===" << std::endl;

    // Global memory latency test (pointer chasing)
    constexpr int mem_size = 1024;
    int *d_indices = nullptr;
    data_type *d_arr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_indices, mem_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_arr, mem_size * sizeof(data_type)));

    // Initialize pointer-chasing pattern on host
    int *h_indices = new int[mem_size];
    for (int i = 0; i < mem_size; i++) {
        h_indices[i] = (i + 1) % mem_size;  // Circular linked list
    }
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices, mem_size * sizeof(int), cudaMemcpyHostToDevice));
    delete[] h_indices;

    unsigned long long h_time_start = 0ull, h_time_end = 0ull;
    data_type result = 0.0f;

    global_mem_latency<<<1, 1>>>(d_arr, d_indices, d_start, d_end);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_time_start, d_start, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_time_end, d_end, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&result, d_arr, sizeof(data_type), cudaMemcpyDeviceToHost));
    std::cout << "Latency of global_mem_latency code snippet = "
              << (h_time_end - h_time_start) << " cycles" << std::endl;

    // Shared memory latency test
    shared_mem_latency<<<1, 32>>>(d_n, d_start, d_end);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_time_start, d_start, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_time_end, d_end, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&result, d_n, sizeof(data_type), cudaMemcpyDeviceToHost));
    std::cout << "Latency of shared_mem_latency code snippet = "
              << (h_time_end - h_time_start) << " cycles" << std::endl;

    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    CUDA_CHECK(cudaFree(d_n));
    CUDA_CHECK(cudaFree(d_n_int));
    CUDA_CHECK(cudaFree(d_start));
    CUDA_CHECK(cudaFree(d_end));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_arr));

    return 0;
}
