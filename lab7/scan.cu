%%writefile scan.cu
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Utility Functions

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

template <typename Op>
void print_array(
    size_t n,
    typename Op::Data const *x // allowed to be either a CPU or GPU pointer
);

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Already Written)

template <typename Op>
void scan_cpu(size_t n, typename Op::Data const *x, typename Op::Data *out) {
    using Data = typename Op::Data;
    Data accumulator = Op::identity();
    for (size_t i = 0; i < n; i++) {
        accumulator = Op::combine(accumulator, x[i]);
        out[i] = accumulator;
    }
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace scan_gpu {

/* GPU Kernels */

// Generic shuffle wrapper - handles both native types and structs
template <typename T>
__device__ __forceinline__ T shfl_up_generic(T val, int offset, unsigned int mask = 0xffffffff) {
    // Default: use byte-wise shuffle for arbitrary types
    T result;
    const int num_words = (sizeof(T) + sizeof(int) - 1) / sizeof(int);
    int* val_ptr = reinterpret_cast<int*>(&val);
    int* result_ptr = reinterpret_cast<int*>(&result);

    #pragma unroll
    for (int i = 0; i < num_words; i++) {
        result_ptr[i] = __shfl_up_sync(mask, val_ptr[i], offset);
    }
    return result;
}

// Specialization for native types (more efficient)
template <>
__device__ __forceinline__ uint32_t shfl_up_generic<uint32_t>(uint32_t val, int offset, unsigned int mask) {
    return __shfl_up_sync(mask, val, offset);
}

template <>
__device__ __forceinline__ int shfl_up_generic<int>(int val, int offset, unsigned int mask) {
    return __shfl_up_sync(mask, val, offset);
}

template <>
__device__ __forceinline__ float shfl_up_generic<float>(float val, int offset, unsigned int mask) {
    return __shfl_up_sync(mask, val, offset);
}

// Warp-level inclusive scan using shuffle (no sync needed)
template <typename Op>
__device__ __forceinline__ typename Op::Data warp_scan(typename Op::Data val) {
    using Data = typename Op::Data;

    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        Data temp = shfl_up_generic<Data>(val, offset);
        if ((threadIdx.x & 31) >= offset) {
            val = Op::combine(temp, val);
        }
    }
    return val;
}

// Optimized block-level scan kernel using warp shuffle
template <typename Op>
__global__ void scan_block_kernel(
    size_t n,
    typename Op::Data const *input,
    typename Op::Data *output,
    typename Op::Data *block_aggregates
) {
    using Data = typename Op::Data;

    // Shared memory for warp sums (only need 8 elements for 256 threads)
    __shared__ Data warp_sums[8];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid & 31;

    // Load with bounds check
    Data val = (gid < n) ? input[gid] : Op::identity();

    // Phase 1: Warp-level scan using shuffle (no __syncthreads needed!)
    val = warp_scan<Op>(val);

    // Store each warp's last value to shared memory
    if (lane_id == 31) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // Phase 2: Scan warp sums (only first warp does this)
    if (warp_id == 0 && lane_id < 8) {
        Data warp_sum = warp_sums[lane_id];

        // Mini scan on 8 elements using shuffle
        #pragma unroll
        for (int offset = 1; offset < 8; offset *= 2) {
            Data temp = shfl_up_generic<Data>(warp_sum, offset, 0xff);
            if (lane_id >= offset) {
                warp_sum = Op::combine(temp, warp_sum);
            }
        }
        warp_sums[lane_id] = warp_sum;
    }
    __syncthreads();

    // Phase 3: Add warp prefix to each element (except first warp)
    if (warp_id > 0) {
        val = Op::combine(warp_sums[warp_id - 1], val);
    }

    // Write output
    if (gid < n) {
        output[gid] = val;
    }

    // Store block aggregate (last valid element)
    if (tid == blockDim.x - 1) {
        block_aggregates[blockIdx.x] = val;
    }
}

// Fix-up kernel to add block prefixes
template <typename Op>
__global__ void scan_fixup_kernel(
    size_t n,
    typename Op::Data *data,
    typename Op::Data const *block_prefixes
) {
    using Data = typename Op::Data;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n && blockIdx.x > 0) {
        data[gid] = Op::combine(block_prefixes[blockIdx.x - 1], data[gid]);
    }
}

// Returns desired size of scratch buffer in bytes.
template <typename Op> size_t get_workspace_size(size_t n) {
    using Data = typename Op::Data;
    constexpr int BLOCK_SIZE = 256;

    size_t total_size = 0;
    size_t current_level = n;

    // Calculate space needed for all levels of block aggregates
    while (current_level > BLOCK_SIZE) {
        size_t num_blocks = (current_level + BLOCK_SIZE - 1) / BLOCK_SIZE;
        total_size += num_blocks * sizeof(Data);
        current_level = num_blocks;
    }

    // Add space for temporary output buffer
    total_size += n * sizeof(Data);

    return total_size;
}

// 'launch_scan'
//
// Input:
//
//   'n': Number of elements in the input array 'x'.
//
//   'x': Input array in GPU memory. The 'launch_scan' function is allowed to
//   overwrite the contents of this buffer.
//
//   'workspace': Scratch buffer in GPU memory. The size of the scratch buffer
//   in bytes is determined by 'get_workspace_size<Op>(n)'.
//
// Output:
//
//   Returns a pointer to GPU memory which will contain the results of the scan
//   after all launched kernels have completed. Must be either a pointer to the
//   'x' buffer or to an offset within the 'workspace' buffer.
//
//   The contents of the output array should be "partial reductions" of the
//   input; each element 'i' of the output array should be given by:
//
//     output[i] = Op::combine(x[0], x[1], ..., x[i])
//
//   where 'Op::combine(...)' of more than two arguments is defined in terms of
//   repeatedly combining pairs of arguments. Note that 'Op::combine' is
//   guaranteed to be associative, but not necessarily commutative, so
//
//        Op::combine(a, b, c)              // conceptual notation; not real C++
//     == Op::combine(a, Op::combine(b, c)) // real C++
//     == Op::combine(Op::combine(a, b), c) // real C++
//
//  but we don't necessarily have
//
//    Op::combine(a, b) == Op::combine(b, a) // not true in general!
//
template <typename Op>
typename Op::Data *launch_scan(
    size_t n,
    typename Op::Data *x, // pointer to GPU memory
    void *workspace       // pointer to GPU memory
) {
    using Data = typename Op::Data;
    constexpr int BLOCK_SIZE = 256;

    if (n == 0) {
        return x;
    }

    // Calculate number of blocks
    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Set up workspace pointers
    Data *block_aggregates = reinterpret_cast<Data*>(workspace);
    Data *output = x;  // We'll scan in-place

    // Phase 1: Block-level scan (uses static shared memory internally)
    scan_block_kernel<Op><<<num_blocks, BLOCK_SIZE>>>(
        n, x, output, block_aggregates
    );

    // Phase 2: Recursively scan block aggregates if needed
    if (num_blocks > 1) {
        // Calculate workspace for recursive call
        void *recursive_workspace = reinterpret_cast<char*>(block_aggregates) +
                                   num_blocks * sizeof(Data);

        // Recursively scan the block aggregates
        Data *scanned_prefixes = launch_scan<Op>(
            num_blocks,
            block_aggregates,
            recursive_workspace
        );

        // Phase 3: Fix-up - add block prefixes to each block
        scan_fixup_kernel<Op><<<num_blocks, BLOCK_SIZE>>>(
            n, output, scanned_prefixes
        );
    }

    return output;
}

} // namespace scan_gpu

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

struct DebugRange {
    uint32_t lo;
    uint32_t hi;

    static constexpr uint32_t INVALID = 0xffffffff;

    static __host__ __device__ __forceinline__ DebugRange invalid() {
        return {INVALID, INVALID};
    }

    __host__ __device__ __forceinline__ bool operator==(const DebugRange &other) const {
        return lo == other.lo && hi == other.hi;
    }

    __host__ __device__ __forceinline__ bool operator!=(const DebugRange &other) const {
        return !(*this == other);
    }

    __host__ __device__ bool is_empty() const { return lo == hi; }

    __host__ __device__ bool is_valid() const { return lo != INVALID; }

    std::string to_string() const {
        if (lo == INVALID) {
            return "INVALID";
        } else {
            return std::to_string(lo) + ":" + std::to_string(hi);
        }
    }
};

struct DebugRangeConcatOp {
    using Data = DebugRange;

    static __host__ __device__ __forceinline__ Data identity() { return {0, 0}; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        if (a.is_empty()) {
            return b;
        } else if (b.is_empty()) {
            return a;
        } else if (a.is_valid() && b.is_valid() && a.hi == b.lo) {
            return {a.lo, b.hi};
        } else {
            return Data::invalid();
        }
    }

    static std::string to_string(Data d) { return d.to_string(); }
};

struct SumOp {
    using Data = uint32_t;

    static __host__ __device__ __forceinline__ Data identity() { return 0; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        return a + b;
    }

    static std::string to_string(Data d) { return std::to_string(d); }
};

constexpr size_t max_print_array_output = 1025;
static thread_local size_t total_print_array_output = 0;

template <typename Op> void print_array(size_t n, typename Op::Data const *x) {
    using Data = typename Op::Data;

    // copy 'x' from device to host if necessary
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, x));
    auto x_host_buf = std::vector<Data>();
    Data const *x_host_ptr = nullptr;
    if (attr.type == cudaMemoryTypeDevice) {
        x_host_buf.resize(n);
        x_host_ptr = x_host_buf.data();
        CUDA_CHECK(
            cudaMemcpy(x_host_buf.data(), x, n * sizeof(Data), cudaMemcpyDeviceToHost));
    } else {
        x_host_ptr = x;
    }

    if (total_print_array_output >= max_print_array_output) {
        return;
    }

    printf("[\n");
    for (size_t i = 0; i < n; i++) {
        auto s = Op::to_string(x_host_ptr[i]);
        printf("  [%zu] = %s,\n", i, s.c_str());
        total_print_array_output++;
        if (total_print_array_output > max_print_array_output) {
            printf("  ... (output truncated)\n");
            break;
        }
    }
    printf("]\n");

    if (total_print_array_output >= max_print_array_output) {
        printf("(Reached maximum limit on 'print_array' output; skipping further calls "
               "to 'print_array')\n");
    }

    total_print_array_output++;
}

template <typename Reset, typename F>
double benchmark_ms(double target_time_ms, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        f();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms);
    }
    return best_time_ms;
}

struct Results {
    double time_ms;
    double bandwidth_gb_per_sec;
};

enum class Mode {
    TEST,
    BENCHMARK,
};

template <typename Op>
Results run_config(Mode mode, std::vector<typename Op::Data> const &x) {
    // Allocate buffers
    using Data = typename Op::Data;
    size_t n = x.size();
    size_t workspace_size = scan_gpu::get_workspace_size<Op>(n);
    Data *x_gpu;
    Data *workspace_gpu;
    CUDA_CHECK(cudaMalloc(&x_gpu, n * sizeof(Data)));
    CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
    CUDA_CHECK(cudaMemcpy(x_gpu, x.data(), n * sizeof(Data), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));

    // Test correctness
    auto expected = std::vector<Data>(n);
    scan_cpu<Op>(n, x.data(), expected.data());
    auto out_gpu = scan_gpu::launch_scan<Op>(n, x_gpu, workspace_gpu);
    if (out_gpu == nullptr) {
        printf("'launch_scan' function not yet implemented (returned nullptr)\n");
        exit(1);
    }
    auto actual = std::vector<Data>(n);
    CUDA_CHECK(
        cudaMemcpy(actual.data(), out_gpu, n * sizeof(Data), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        if (actual.at(i) != expected.at(i)) {
            auto actual_str = Op::to_string(actual.at(i));
            auto expected_str = Op::to_string(expected.at(i));
            printf(
                "Mismatch at position %zu: %s != %s\n",
                i,
                actual_str.c_str(),
                expected_str.c_str());
            if (n <= 128) {
                printf("Input:\n");
                print_array<Op>(n, x.data());
                printf("\nExpected:\n");
                print_array<Op>(n, expected.data());
                printf("\nActual:\n");
                print_array<Op>(n, actual.data());
            }
            exit(1);
        }
    }
    if (mode == Mode::TEST) {
        return {0.0, 0.0};
    }

    // Benchmark
    double target_time_ms = 200.0;
    double time_ms = benchmark_ms(
        target_time_ms,
        [&]() {
            CUDA_CHECK(
                cudaMemcpy(x_gpu, x.data(), n * sizeof(Data), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
        },
        [&]() { scan_gpu::launch_scan<Op>(n, x_gpu, workspace_gpu); });
    double bytes_processed = n * sizeof(Data) * 2;
    double bandwidth_gb_per_sec = bytes_processed / time_ms / 1e6;

    // Cleanup
    CUDA_CHECK(cudaFree(x_gpu));
    CUDA_CHECK(cudaFree(workspace_gpu));

    return {time_ms, bandwidth_gb_per_sec};
}

std::vector<DebugRange> gen_debug_ranges(uint32_t n) {
    auto ranges = std::vector<DebugRange>();
    for (uint32_t i = 0; i < n; ++i) {
        ranges.push_back({i, i + 1});
    }
    return ranges;
}

template <typename Rng> std::vector<uint32_t> gen_random_data(Rng &rng, uint32_t n) {
    auto uniform = std::uniform_int_distribution<uint32_t>(0, 100);
    auto data = std::vector<uint32_t>();
    for (uint32_t i = 0; i < n; ++i) {
        data.push_back(uniform(rng));
    }
    return data;
}

template <typename Op, typename GenData>
void run_tests(std::vector<uint32_t> const &sizes, GenData &&gen_data) {
    for (auto size : sizes) {
        auto data = gen_data(size);
        printf("  Testing size %8u\n", size);
        run_config<Op>(Mode::TEST, data);
        printf("  OK\n\n");
    }
}

int main(int argc, char const *const *argv) {
    auto correctness_sizes = std::vector<uint32_t>{
        16,
        10,
        128,
        100,
        1024,
        1000,
        1 << 20,
        1'000'000,
        16 << 20,
        64 << 20,
    };

    auto rng = std::mt19937(0xCA7CAFE);

    printf("Correctness:\n\n");
    printf("Testing scan operation: debug range concatenation\n\n");
    run_tests<DebugRangeConcatOp>(correctness_sizes, gen_debug_ranges);
    printf("Testing scan operation: integer sum\n\n");
    run_tests<SumOp>(correctness_sizes, [&](uint32_t n) {
        return gen_random_data(rng, n);
    });

    printf("Performance:\n\n");

    size_t n = 64 << 20;
    auto data = gen_random_data(rng, n);

    printf("Benchmarking scan operation: integer sum, size %zu\n\n", n);

    // Warmup
    run_config<SumOp>(Mode::BENCHMARK, data);
    // Benchmark
    auto results = run_config<SumOp>(Mode::BENCHMARK, data);
    printf("  Time: %.2f ms\n", results.time_ms);
    printf("  Throughput: %.2f GB/s\n", results.bandwidth_gb_per_sec);

    return 0;
}
