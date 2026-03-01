%%writefile matmul.cu
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

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
// CPU Reference Implementation (Too slow to actually run!)
//
// void matmul_cpu_naive(
//     int32_t size_i,
//     int32_t size_j,
//     int32_t size_k,
//     float const *a,
//     float const *b,
//     float *c) {
//     for (int32_t i = 0; i < size_i; ++i) {
//         for (int32_t j = 0; j < size_j; ++j) {
//             float sum = 0.0;
//             for (int32_t k = 0; k < size_k; ++k) {
//                 sum += a[i * size_k + k] * b[k * size_j + j];
//             }
//             c[i * size_j + j] = sum;
//         }
//     }
// }

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With L1 Cache using __ldg)

namespace matmul_l1_cache {

constexpr int TILE_SIZE = 32;

__global__ void matmul_l1_cache(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c) {

    // Output element coordinates
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Early exit for out-of-bounds threads
    if (row >= size_i || col >= size_j)
        return;

    // Accumulator (kept in register)
    float sum = 0.0f;

    // Loop over K dimension with tiling
    for (int k0 = 0; k0 < size_k; k0 += TILE_SIZE) {
        int kend = min(size_k, k0 + TILE_SIZE);

        // Compute directly without shared memory
        // Use __ldg() for read-only cache
        #pragma unroll
        for (int k = k0; k < kend; ++k) {
            sum += __ldg(&a[row * size_k + k]) * __ldg(&b[k * size_j + col]);
        }
    }

    // Write result
    c[row * size_j + col] = sum;
}

void launch_matmul_l1_cache(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim(
        (size_j + TILE_SIZE - 1) / TILE_SIZE,
        (size_i + TILE_SIZE - 1) / TILE_SIZE
    );

    matmul_l1_cache<<<grid_dim, block_dim>>>(size_i, size_j, size_k, a, b, c);
    CUDA_CHECK(cudaGetLastError());
}

}; // namespace matmul_l1_cache

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Reuse in L1/Shmem)

namespace matmul_l1 {

// Tile size for shared memory blocking
constexpr int TILE_SIZE = 32;

__global__ void matmul_l1(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    // Shared memory for A and B tiles
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    // Thread coordinates within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Output element coordinates in C
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Accumulator for C[row][col] (kept in register)
    float sum = 0.0f;

    // Loop over tiles of A and B required to compute C tile
    int num_tiles = (size_k + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        // Load tile of A into shared memory
        int a_row = row;
        int a_col = tile_idx * TILE_SIZE + tx;
        if (a_row < size_i && a_col < size_k) {
            tile_a[ty][tx] = a[a_row * size_k + a_col];
        } else {
            tile_a[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        int b_row = tile_idx * TILE_SIZE + ty;
        int b_col = col;
        if (b_row < size_k && b_col < size_j) {
            tile_b[ty][tx] = b[b_row * size_j + b_col];
        } else {
            tile_b[ty][tx] = 0.0f;
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_a[ty][k] * tile_b[k][tx];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result to C
    if (row < size_i && col < size_j) {
        c[row * size_j + col] = sum;
    }
}

void launch_matmul_l1(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    // Configure grid and block dimensions
    dim3 block_dim(TILE_SIZE, TILE_SIZE);
    dim3 grid_dim(
        (size_j + TILE_SIZE - 1) / TILE_SIZE,
        (size_i + TILE_SIZE - 1) / TILE_SIZE
    );

    // Launch kernel
    matmul_l1<<<grid_dim, block_dim>>>(size_i, size_j, size_k, a, b, c);

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());
}

}; // namespace matmul_l1

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation (With Reuse in L1/Shmem and Registers)

namespace matmul_l1_reg {

// Microtiling parameters
constexpr int BM = 64;   // Block tile size in M dimension
constexpr int BN = 64;   // Block tile size in N dimension
constexpr int BK = 8;    // Block tile size in K dimension
constexpr int TM = 8;    // Thread tile size in M dimension (per thread)
constexpr int TN = 8;    // Thread tile size in N dimension (per thread)

// Use __launch_bounds__ to control register usage
__global__ void __launch_bounds__(64) matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    // Shared memory for A and B tiles
    __shared__ float tile_a[BM][BK];
    __shared__ float tile_b[BK][BN];

    // Thread indices for microtile
    const int thread_row = threadIdx.x / (BN / TN);  // 0-7
    const int thread_col = threadIdx.x % (BN / TN);  // 0-7

    // Block indices
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    // Register arrays for accumulation (TM x TN per thread)
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    // Register arrays for temporary storage
    float reg_a[TM];
    float reg_b[TN];

    // Number of tiles in K dimension
    const int num_tiles = (size_k + BK - 1) / BK;

    // Loop over K dimension tiles
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        // Cooperatively load tile_a from global to shared memory
        // Each thread loads BM*BK/64 = 8 elements
        #pragma unroll
        for (int load_offset = 0; load_offset < (BM * BK) / 64; ++load_offset) {
            int elem_idx = threadIdx.x + load_offset * 64;
            int load_row = elem_idx / BK;
            int load_col = elem_idx % BK;

            int global_row = block_row * BM + load_row;
            int global_col = tile_idx * BK + load_col;

            if (global_row < size_i && global_col < size_k) {
                tile_a[load_row][load_col] = a[global_row * size_k + global_col];
            } else {
                tile_a[load_row][load_col] = 0.0f;
            }
        }

        // Cooperatively load tile_b from global to shared memory
        #pragma unroll
        for (int load_offset = 0; load_offset < (BK * BN) / 64; ++load_offset) {
            int elem_idx = threadIdx.x + load_offset * 64;
            int load_row = elem_idx / BN;
            int load_col = elem_idx % BN;

            int global_row = tile_idx * BK + load_row;
            int global_col = block_col * BN + load_col;

            if (global_row < size_k && global_col < size_j) {
                tile_b[load_row][load_col] = b[global_row * size_j + global_col];
            } else {
                tile_b[load_row][load_col] = 0.0f;
            }
        }

        __syncthreads();

        // Compute on this tile (register-level reuse happens here)
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Load TM elements from A into registers
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                reg_a[m] = tile_a[thread_row * TM + m][k];
            }

            // Load TN elements from B into registers
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                reg_b[n] = tile_b[k][thread_col * TN + n];
            }

            // Compute outer product: acc[TM][TN] += reg_a[TM] * reg_b[TN]
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    acc[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }

        __syncthreads();
    }

    // Write results to C
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            int global_row = block_row * BM + thread_row * TM + m;
            int global_col = block_col * BN + thread_col * TN + n;

            if (global_row < size_i && global_col < size_j) {
                c[global_row * size_j + global_col] = acc[m][n];
            }
        }
    }
}

void launch_matmul_l1_reg(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    // Thread block: 8x8 threads = 64 threads
    // Each thread computes 8x8 output elements
    // Each block computes 64x64 output tile
    dim3 block_dim(64);
    dim3 grid_dim(
        (size_j + BN - 1) / BN,
        (size_i + BM - 1) / BM
    );

    matmul_l1_reg<<<grid_dim, block_dim>>>(size_i, size_j, size_k, a, b, c);
    CUDA_CHECK(cudaGetLastError());
}

}; // namespace matmul_l1_reg

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

std::vector<float> read_data(std::string const &path, int32_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read " << path << std::endl;
        std::abort();
    }
    return data;
}

template <typename F>
double benchmark_ms(double target_time_ms, int32_t num_iters_inner, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        for (int32_t i = 0; i < num_iters_inner; ++i) {
            f();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms / num_iters_inner);
    }
    return best_time_ms;
}

struct BenchmarkResult {
    char const *name;
    double elapsed_ms;
};

struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
    int32_t size_k;
    bool save_result;
};

template <typename Impl>
void run_tests_for_size(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results,
    std::vector<BenchmarkConfig> const &configs) {
    for (auto config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;

        auto path_prefix = test_data_dir + "/test_" + std::to_string(size_i) + "x" +
            std::to_string(size_j) + "x" + std::to_string(size_k);
        auto a = read_data(path_prefix + "_a.bin", size_i * size_k);
        auto b = read_data(path_prefix + "_b.bin", size_k * size_j);
        auto c = read_data(path_prefix + "_c.bin", size_i * size_j);

        float *a_gpu;
        float *b_gpu;
        float *c_gpu;
        CUDA_CHECK(cudaMalloc(&a_gpu, size_i * size_k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b_gpu, size_k * size_j * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c_gpu, size_i * size_j * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(
            a_gpu,
            a.data(),
            size_i * size_k * sizeof(float),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            b_gpu,
            b.data(),
            size_k * size_j * sizeof(float),
            cudaMemcpyHostToDevice));

        Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu);

        std::vector<float> c_out_host(size_i * size_j);
        CUDA_CHECK(cudaMemcpy(
            c_out_host.data(),
            c_gpu,
            size_i * size_j * sizeof(float),
            cudaMemcpyDeviceToHost));

        double mse = 0.0;
        double ref_mean_square = 0.0;
        for (int32_t i = 0; i < size_i; ++i) {
            for (int32_t j = 0; j < size_j; ++j) {
                float diff = c_out_host[i * size_j + j] - c[i * size_j + j];
                mse += diff * diff;
                ref_mean_square += c[i * size_j + j] * c[i * size_j + j];
            }
        }
        mse /= size_i * size_j;
        ref_mean_square /= size_i * size_j;
        float rmse = std::sqrt(mse);
        float rel_rmse = rmse / std::sqrt(ref_mean_square);

        printf("  size %4d * %4d * %4d:\n", size_i, size_j, size_k);
        printf("    correctness: %.02e relative RMSE\n", rel_rmse);

        if (rel_rmse > 1e-5) {
            printf("    skipping benchmark (incorrect)\n");
        } else {
            double elapsed_ms = benchmark_ms(1000.0, 4, [&]() {
                Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu);
            });

            printf("    run time: %6.02f ms\n", elapsed_ms);

            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            printf("    throughput: %5.02f TFLOP/s\n", tflop / (elapsed_ms * 1e-3));

            if (config.save_result) {
                saved_results.push_back({Impl::name, elapsed_ms});
            }
        }

        printf("\n");
    }
}

template <typename Impl>
void run_all_tests(
    std::string const &test_data_dir,
    std::vector<BenchmarkResult> &saved_results)
{
    printf("%s:\n\n", Impl::name);
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{256, 256, 256, false}});
    run_tests_for_size<Impl>(test_data_dir, saved_results, {{3072, 3072, 3072, true}});
}

struct MatmulL1Cache {
    constexpr static char const *name = "matmul_l1_cache";
    static void run(
        int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
            matmul_l1_cache::launch_matmul_l1_cache(size_i, size_j, size_k, a, b, c);
        }
};

struct MatmulL1 {
    constexpr static char const *name = "matmul_l1 (shared memory)";
    static void run(
        int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
            matmul_l1::launch_matmul_l1(size_i, size_j, size_k, a, b, c);
        }
};

struct MatmulL1Reg {
    constexpr static char const *name = "matmul_l1_reg";
    static void run(
        int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c) {
            matmul_l1_reg::launch_matmul_l1_reg(size_i, size_j, size_k, a, b, c);
        }
};

int main(int argc, char **argv) {
    // Kaggle 的測試資料路徑
    std::string test_data_dir = "/kaggle/input/test-data";

    // 如果提供命令列參數，使用該路徑
    if (argc > 1) {
        test_data_dir = argv[1];
    }

    auto saved_results = std::vector<BenchmarkResult>();

    run_all_tests<MatmulL1Cache>(test_data_dir, saved_results);
    run_all_tests<MatmulL1>(test_data_dir, saved_results);
    run_all_tests<MatmulL1Reg>(test_data_dir, saved_results);

    if (saved_results.size() > 1) {
        printf("speedups on largest problem size:\n");
        for (int32_t j = 1; j < saved_results.size(); ++j) {
            printf("\n");
            for (int32_t i = j; i > 0;) {
                --i;
                auto const &first = saved_results.at(i);
                auto const &second = saved_results.at(j);
                printf(
                    "  speedup %s -> %s: %.02fx\n",
                    first.name,
                    second.name,
                    first.elapsed_ms / second.elapsed_ms);
            }
        }
    }

    return 0;
}
