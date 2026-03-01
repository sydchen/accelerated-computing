%%writefile rle_decompress.cu
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
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

////////////////////////////////////////////////////////////////////////////////
// Simple Caching GPU Memory Allocator

class GpuAllocCache {
  public:
    GpuAllocCache() = default;

    ~GpuAllocCache();

    GpuAllocCache(GpuAllocCache const &) = delete;
    GpuAllocCache &operator=(GpuAllocCache const &) = delete;
    GpuAllocCache(GpuAllocCache &&) = delete;
    GpuAllocCache &operator=(GpuAllocCache &&) = delete;

    void *alloc(size_t size);
    void reset();

  private:
    void *buffer = nullptr;
    size_t capacity = 0;
    bool active = false;
};

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Already Written)

void rle_decompress_cpu(
    uint32_t compressed_count,
    char const *compressed_data,
    uint32_t const *compressed_lengths,
    std::vector<char> &raw) {
    raw.clear();
    for (uint32_t i = 0; i < compressed_count; i++) {
        char c = compressed_data[i];
        uint32_t run_length = compressed_lengths[i];
        for (uint32_t j = 0; j < run_length; j++) {
            raw.push_back(c);
        }
    }
}

struct Decompressed {
    uint32_t count;
    char const *data; // pointer to GPU memory
};

/// <--- your code here --->

// Define SumOp for scan operations
struct SumOp {
    using Data = uint32_t;

    static __host__ __device__ __forceinline__ Data identity() { return 0; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        return a + b;
    }
};

// Include scan implementation (optimized with warp shuffle)
namespace scan_gpu {

// Warp-level scan using shuffle (no shared memory needed for warp)
template <typename Op>
__device__ __forceinline__ typename Op::Data warp_scan(typename Op::Data val) {
    using Data = typename Op::Data;
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        Data temp = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x & 31) >= offset) {
            val = Op::combine(temp, val);
        }
    }
    return val;
}

template <typename Op>
__global__ void scan_block_kernel(
    size_t n,
    typename Op::Data const *input,
    typename Op::Data *output,
    typename Op::Data *block_aggregates
) {
    using Data = typename Op::Data;
    __shared__ Data warp_sums[8];  // For 256 threads = 8 warps

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid & 31;

    Data val = (gid < n) ? input[gid] : Op::identity();

    // Phase 1: Warp-level scan (no syncthreads needed!)
    val = warp_scan<Op>(val);

    // Store warp's last value
    if (lane_id == 31) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // Phase 2: Scan warp sums (only first warp)
    if (warp_id == 0 && lane_id < 8) {
        Data warp_sum = warp_sums[lane_id];
        #pragma unroll
        for (int offset = 1; offset < 8; offset *= 2) {
            Data temp = __shfl_up_sync(0xff, warp_sum, offset);
            if (lane_id >= offset) {
                warp_sum = Op::combine(temp, warp_sum);
            }
        }
        warp_sums[lane_id] = warp_sum;
    }
    __syncthreads();

    // Phase 3: Add warp prefix
    if (warp_id > 0) {
        val = Op::combine(warp_sums[warp_id - 1], val);
    }

    if (gid < n) {
        output[gid] = val;
    }

    if (tid == blockDim.x - 1) {
        block_aggregates[blockIdx.x] = val;
    }
}

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

template <typename Op>
size_t get_workspace_size(size_t n) {
    using Data = typename Op::Data;
    constexpr int BLOCK_SIZE = 256;

    size_t total_size = 0;
    size_t current_level = n;

    // Calculate space for block aggregates at each level
    while (current_level > BLOCK_SIZE) {
        size_t num_blocks = (current_level + BLOCK_SIZE - 1) / BLOCK_SIZE;
        total_size += num_blocks * sizeof(Data);
        current_level = num_blocks;
    }

    // Add space for final level (small enough for single block)
    if (current_level > 1) {
        total_size += current_level * sizeof(Data);
    }

    return total_size;
}

template <typename Op>
typename Op::Data *launch_scan(
    size_t n,
    typename Op::Data *x,
    void *workspace
) {
    using Data = typename Op::Data;
    constexpr int BLOCK_SIZE = 256;

    if (n == 0) {
        return x;
    }

    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    Data *block_aggregates = reinterpret_cast<Data*>(workspace);
    Data *output = x;

    // No dynamic shared memory needed - using warp shuffle + static shared
    scan_block_kernel<Op><<<num_blocks, BLOCK_SIZE>>>(
        n, x, output, block_aggregates
    );

    if (num_blocks > 1) {
        void *recursive_workspace = reinterpret_cast<char*>(block_aggregates) +
                                   num_blocks * sizeof(Data);
        Data *scanned_prefixes = launch_scan<Op>(
            num_blocks,
            block_aggregates,
            recursive_workspace
        );

        scan_fixup_kernel<Op><<<num_blocks, BLOCK_SIZE>>>(
            n, output, scanned_prefixes
        );
    }

    return output;
}

} // namespace scan_gpu

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace rle_gpu {

/* GPU Kernels */

// Convert inclusive scan to exclusive scan
__global__ void inclusive_to_exclusive_kernel(
    uint32_t n,
    uint32_t const *inclusive,
    uint32_t *exclusive
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        exclusive[i] = (i == 0) ? 0 : inclusive[i - 1];
    }
}

// Approach 1: Each thread expands one run (good for many runs)
// Optimized with vectorized writes
__global__ void expand_runs_by_run_kernel(
    uint32_t num_runs,
    char const *compressed_values,
    uint32_t const *inclusive_offsets,  // Inclusive scan of lengths
    char *decompressed
) {
    int run_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (run_id >= num_runs) return;

    // Get start and end positions for this run
    uint32_t start = (run_id == 0) ? 0 : inclusive_offsets[run_id - 1];
    uint32_t end = inclusive_offsets[run_id];
    char value = compressed_values[run_id];
    uint32_t len = end - start;

    if (len == 0) return;

    // Create a 4-byte pattern for vectorized writes
    uint32_t pattern = (uint32_t)(unsigned char)value;
    pattern = pattern | (pattern << 8) | (pattern << 16) | (pattern << 24);

    uint32_t i = start;

    // Handle unaligned prefix (write bytes until 4-byte aligned)
    while (i < end && (i & 3) != 0) {
        decompressed[i++] = value;
    }

    // Vectorized writes (4 bytes at a time)
    uint32_t *dst32 = reinterpret_cast<uint32_t*>(decompressed + i);
    uint32_t remaining = end - i;
    uint32_t vec_count = remaining / 4;

    for (uint32_t j = 0; j < vec_count; j++) {
        dst32[j] = pattern;
    }
    i += vec_count * 4;

    // Handle remaining bytes
    while (i < end) {
        decompressed[i++] = value;
    }
}

// Approach 2: Each thread handles one output with vectorized writes
// Good for few runs with long lengths - uses binary search but writes 4 bytes at a time
__global__ void expand_runs_by_output_vec_kernel(
    uint32_t total_output_size,
    uint32_t num_runs,
    char const *compressed_values,
    uint32_t const *inclusive_offsets,  // Inclusive scan of lengths
    char *decompressed
) {
    // Each thread handles 4 consecutive bytes
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_idx = vec_idx * 4;

    if (out_idx >= total_output_size) return;

    // Binary search to find which run this output belongs to
    int left = 0;
    int right = num_runs - 1;
    while (left < right) {
        int mid = (left + right) / 2;
        if (inclusive_offsets[mid] <= out_idx) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    char value = compressed_values[left];
    uint32_t run_end = inclusive_offsets[left];

    // Check if we can write 4 bytes from the same run
    if (out_idx + 4 <= run_end) {
        // All 4 bytes belong to same run - vectorized write
        uint32_t pattern = (uint32_t)(unsigned char)value;
        pattern = pattern | (pattern << 8) | (pattern << 16) | (pattern << 24);

        if ((out_idx & 3) == 0 && out_idx + 4 <= total_output_size) {
            *reinterpret_cast<uint32_t*>(decompressed + out_idx) = pattern;
        } else {
            // Unaligned or near end - write byte by byte
            for (int i = 0; i < 4 && out_idx + i < total_output_size; i++) {
                decompressed[out_idx + i] = value;
            }
        }
    } else {
        // Bytes span multiple runs - write byte by byte with search
        for (int i = 0; i < 4 && out_idx + i < total_output_size; i++) {
            int pos = out_idx + i;
            // Advance to next run if needed
            while (left < num_runs - 1 && inclusive_offsets[left] <= pos) {
                left++;
            }
            decompressed[pos] = compressed_values[left];
        }
    }
}

// 'launch_rle_decompress'
//
// Input:
//
//   'compressed_count': Number of runs in the compressed data.
//
//   'compressed_data': Array of size 'compressed_count' in GPU memory,
//   containing the byte value for each run.
//
//   'compressed_lengths': Array of size 'compressed_count' in GPU memory,
//    containing the length of each run.
//
//   'workspace_alloc_1', 'workspace_alloc_2': 'GpuAllocCache' objects each of
//   which can be used to allocate a single GPU buffer of arbitrary size.
//
// Output:
//
//   Returns a 'Decompressed' struct containing the following:
//
//     'count': Number of bytes in the decompressed data.
//
//     'data': Pointer to the decompressed data in GPU memory. May point to a
//     buffer allocated using 'workspace_alloc_1' or 'workspace_alloc_2'.
//
Decompressed launch_rle_decompress(
    uint32_t compressed_count,
    char const *compressed_data,
    uint32_t const *compressed_lengths,
    GpuAllocCache &workspace_alloc_1,
    GpuAllocCache &workspace_alloc_2) {

    if (compressed_count == 0) {
        return {0, nullptr};
    }

    constexpr int BLOCK_SIZE = 256;

    // Step 1: Allocate workspace for scan + lengths buffer
    size_t scan_ws_size = scan_gpu::get_workspace_size<SumOp>(compressed_count);
    size_t lengths_size = compressed_count * sizeof(uint32_t);
    void *workspace = workspace_alloc_1.alloc(scan_ws_size + lengths_size);

    uint32_t *lengths_copy = reinterpret_cast<uint32_t*>(workspace);
    void *scan_ws = reinterpret_cast<char*>(workspace) + lengths_size;

    // Step 2: Copy lengths for in-place scan
    CUDA_CHECK(cudaMemcpy(lengths_copy, compressed_lengths,
                          lengths_size, cudaMemcpyDeviceToDevice));

    // Step 3: Perform inclusive scan on lengths
    uint32_t *inclusive_offsets = scan_gpu::launch_scan<SumOp>(
        compressed_count,
        lengths_copy,
        scan_ws
    );

    // Step 4: Get total decompressed size (last element of inclusive scan)
    uint32_t total_size;
    CUDA_CHECK(cudaMemcpy(&total_size, inclusive_offsets + compressed_count - 1,
                          sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Step 5: Allocate output buffer
    char *decompressed = reinterpret_cast<char*>(
        workspace_alloc_2.alloc(total_size)
    );

    // Step 6: Choose expansion strategy based on parallelism
    // - Many runs: use "by run" kernel (plenty of parallelism)
    // - Few runs: use "by output" kernel (need more threads)
    if (compressed_count >= 8192) {
        // Enough parallelism with by_run kernel
        int num_blocks = (compressed_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        expand_runs_by_run_kernel<<<num_blocks, BLOCK_SIZE>>>(
            compressed_count, compressed_data,
            inclusive_offsets, decompressed
        );
    } else {
        // Need more parallelism - use by_output with vectorized writes
        // Each thread handles 4 bytes
        uint32_t num_vec_elements = (total_size + 3) / 4;
        int num_blocks = (num_vec_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        expand_runs_by_output_vec_kernel<<<num_blocks, BLOCK_SIZE>>>(
            total_size, compressed_count, compressed_data,
            inclusive_offsets, decompressed
        );
    }

    return {total_size, decompressed};
}

} // namespace rle_gpu

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

GpuAllocCache::~GpuAllocCache() {
    if (buffer) {
        CUDA_CHECK(cudaFree(buffer));
    }
}

void *GpuAllocCache::alloc(size_t size) {
    if (active) {
        printf("Error: GpuAllocCache::alloc called while active\n");
        exit(1);
    }

    if (size > capacity) {
        if (buffer) {
            CUDA_CHECK(cudaFree(buffer));
        }
        CUDA_CHECK(cudaMalloc(&buffer, size));
        CUDA_CHECK(cudaMemset(buffer, 0, size));
        capacity = size;
    }

    return buffer;
}

void GpuAllocCache::reset() {
    if (active) {
        CUDA_CHECK(cudaMemset(buffer, 0, capacity));
    }
    active = false;
}

void rle_compress_cpu(
    uint32_t raw_count,
    char const *raw,
    std::vector<char> &compressed_data,
    std::vector<uint32_t> &compressed_lengths) {
    compressed_data.clear();
    compressed_lengths.clear();

    uint32_t i = 0;
    while (i < raw_count) {
        char c = raw[i];
        uint32_t run_length = 1;
        i++;
        while (i < raw_count && raw[i] == c) {
            run_length++;
            i++;
        }
        compressed_data.push_back(c);
        compressed_lengths.push_back(run_length);
    }
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
};

enum class Mode {
    TEST,
    BENCHMARK,
};

Results run_config(Mode mode, std::vector<char> const &original_raw) {
    // Compress data
    auto compressed_data = std::vector<char>();
    auto compressed_lengths = std::vector<uint32_t>();
    rle_compress_cpu(
        original_raw.size(),
        original_raw.data(),
        compressed_data,
        compressed_lengths);

    // Allocate buffers
    char *compressed_data_gpu;
    uint32_t *compressed_lengths_gpu;
    CUDA_CHECK(cudaMalloc(&compressed_data_gpu, compressed_data.size()));
    CUDA_CHECK(cudaMalloc(
        &compressed_lengths_gpu,
        compressed_lengths.size() * sizeof(uint32_t)));
    auto workspace_alloc_1 = GpuAllocCache();
    auto workspace_alloc_2 = GpuAllocCache();

    // Copy input data to GPU
    CUDA_CHECK(cudaMemcpy(
        compressed_data_gpu,
        compressed_data.data(),
        compressed_data.size(),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        compressed_lengths_gpu,
        compressed_lengths.data(),
        compressed_lengths.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice));

    auto reset = [&]() {
        workspace_alloc_1.reset();
        workspace_alloc_2.reset();
    };

    auto f = [&]() {
        rle_gpu::launch_rle_decompress(
            compressed_data.size(),
            compressed_data_gpu,
            compressed_lengths_gpu,
            workspace_alloc_1,
            workspace_alloc_2);
    };

    // Test correctness
    auto decompressed = rle_gpu::launch_rle_decompress(
        compressed_data.size(),
        compressed_data_gpu,
        compressed_lengths_gpu,
        workspace_alloc_1,
        workspace_alloc_2);
    std::vector<char> raw(decompressed.count);
    CUDA_CHECK(cudaMemcpy(
        raw.data(),
        decompressed.data,
        decompressed.count,
        cudaMemcpyDeviceToHost));

    bool correct = true;
    if (raw.size() != original_raw.size()) {
        printf("Mismatch in decompressed size:\n");
        printf("  Expected: %zu\n", original_raw.size());
        printf("  Actual:   %zu\n", raw.size());
        correct = false;
    }
    if (correct) {
        for (size_t i = 0; i < raw.size(); i++) {
            if (raw[i] != original_raw[i]) {
                printf("Mismatch in decompressed data at index %zu:\n", i);
                printf(
                    "  Expected: 0x%02x\n",
                    static_cast<unsigned char>(original_raw[i]));
                printf("  Actual:   0x%02x\n", static_cast<unsigned char>(raw[i]));
                correct = false;
                break;
            }
        }
    }

    if (!correct) {
        if (original_raw.size() <= 1024) {
            printf("\nInput:\n");
            for (size_t i = 0; i < compressed_data.size(); i++) {
                printf(
                    "  [%4zu] = data: 0x%02x, length: %u\n",
                    i,
                    static_cast<unsigned char>(compressed_data.at(i)),
                    compressed_lengths.at(i));
            }
            printf("\nExpected:\n");
            for (size_t i = 0; i < original_raw.size(); i++) {
                printf(
                    "  [%4zu] = 0x%02x\n",
                    i,
                    static_cast<unsigned char>(original_raw[i]));
            }
            printf("\nActual:\n");
            if (raw.size() == 0) {
                printf("  (empty)\n");
            }
            for (size_t i = 0; i < raw.size(); i++) {
                printf("  [%4zu] = 0x%02x\n", i, static_cast<unsigned char>(raw[i]));
            }
        }
        exit(1);
    }

    if (mode == Mode::TEST) {
        return {};
    }

    // Benchmark
    double target_time_ms = 1000.0;
    double time_ms = benchmark_ms(target_time_ms, reset, f);

    // Cleanup
    CUDA_CHECK(cudaFree(compressed_data_gpu));
    CUDA_CHECK(cudaFree(compressed_lengths_gpu));

    return {time_ms};
}

template <typename Rng> std::vector<char> generate_test_data(uint32_t size, Rng &rng) {
    auto random_byte = std::uniform_int_distribution<int32_t>(
        std::numeric_limits<char>::min(),
        std::numeric_limits<char>::max());
    constexpr uint32_t alphabet_size = 4;
    auto alphabet = std::vector<char>();
    for (uint32_t i = 0; i < alphabet_size; i++) {
        alphabet.push_back(random_byte(rng));
    }
    auto random_symbol = std::uniform_int_distribution<uint32_t>(0, alphabet_size - 1);
    auto data = std::vector<char>();
    for (uint32_t i = 0; i < size; i++) {
        data.push_back(alphabet.at(random_symbol(rng)));
    }
    return data;
}

template <typename Rng>
std::vector<char> generate_sparse(uint32_t size, uint32_t nonzero_count, Rng &rng) {
    auto data = std::vector<char>(size, 0);
    auto random_index = std::uniform_int_distribution<uint32_t>(0, size - 1);
    auto random_byte = std::uniform_int_distribution<int32_t>(
        std::numeric_limits<char>::min(),
        std::numeric_limits<char>::max());
    for (uint32_t i = 0; i < nonzero_count; i++) {
        data.at(random_index(rng)) = random_byte(rng);
    }
    char fill = random_byte(rng);
    for (uint32_t i = 0; i < size; i++) {
        if (data.at(i) == 0) {
            data.at(i) = fill;
        } else {
            fill = random_byte(rng);
        }
    }
    return data;
}

int main(int argc, char const *const *argv) {
    auto rng = std::mt19937(0xCA7CAFE);

    auto test_sizes = std::vector<uint32_t>{
        16,
        10,
        128,
        100,
        1 << 10,
        1000,
        1 << 20,
        1'000'000,
        16 << 20,
    };

    printf("Correctness:\n\n");
    for (auto test_size : test_sizes) {
        auto raw = generate_test_data(test_size, rng);
        printf("  Testing decompression for size %u\n", test_size);
        run_config(Mode::TEST, raw);
        printf("  OK\n\n");
    }

    auto test_data_search_paths = std::vector<std::string>{"/kaggle/input/sample"};
    std::string test_data_path;
    for (auto test_data_search_path : test_data_search_paths) {
        auto candidate_path = test_data_search_path + "/rle_raw.bmp";
        if (std::filesystem::exists(candidate_path)) {
            test_data_path = candidate_path;
            break;
        }
    }
    if (test_data_path.empty()) {
        printf("Could not find test data file.\n");
        exit(1);
    }

    auto raw = std::vector<char>();
    {
        auto file = std::ifstream(test_data_path, std::ios::binary);
        if (!file) {
            printf("Could not open test data file '%s'.\n", test_data_path.c_str());
            exit(1);
        }
        file.seekg(0, std::ios::end);
        raw.resize(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(raw.data(), raw.size());
    }

    printf("Performance:\n\n");
    printf("  Testing decompression on file 'rle_raw.bmp' (size %zu)\n", raw.size());
    auto results = run_config(Mode::BENCHMARK, raw);
    printf("  Time: %.2f ms\n", results.time_ms);

    auto raw_sparse = generate_sparse(16 << 20, 1 << 10, rng);
    printf("\n  Testing decompression on sparse data (size %u)\n", 16 << 20);
    results = run_config(Mode::BENCHMARK, raw_sparse);
    printf("  Time: %.2f ms\n", results.time_ms);

    return 0;
}
