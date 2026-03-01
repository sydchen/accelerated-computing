%%writefile matmul_2.cu
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#define HAS_LAB_4_BASELINE_IMPL

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

__device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" ::"r"(smem),
        "l"(glob_ptr),
        "n"(BYTES));
}

__device__ __forceinline__ void async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> __device__ __forceinline__ void async_wait_pending() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

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
// GPU Implementation with Reuse in L1/Shmem and Registers (Baseline from Lab 4)

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

} // namespace matmul_l1_reg

////////////////////////////////////////////////////////////////////////////////
// GPU Implementation with Vectorized Loads (Based on Lab 4 Baseline)

namespace matmul_l1_reg_vectorized {

// Same microtiling parameters as matmul_l1_reg
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

__global__ void __launch_bounds__(64) matmul_l1_reg_vectorized(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    // Shared memory with padding to avoid bank conflicts
    __shared__ float tile_a[BM][BK + 1];  // [64][9] - padding eliminates bank conflicts
    __shared__ float tile_b[BK][BN + 1];  // [8][65] - padding eliminates bank conflicts

    const int thread_row = threadIdx.x / (BN / TN);  // 0-7
    const int thread_col = threadIdx.x % (BN / TN);  // 0-7

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    // Register arrays for accumulation
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    float reg_a[TM];
    float reg_b[TN];

    const int num_tiles = (size_k + BK - 1) / BK;

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {

        // ====================================================================
        // Vectorized load tile_a: using float4 (16-byte aligned loads)
        // ====================================================================
        // tile_a: [64][8] = 512 floats, 64 threads → 8 floats/thread
        // Strategy: each thread loads 2 float4s (8 consecutive floats)

        #pragma unroll
        for (int load_offset = 0; load_offset < 2; ++load_offset) {
            // Each thread loads 2 float4s
            int elem_idx = threadIdx.x * 8 + load_offset * 4;
            int load_row = elem_idx / BK;
            int load_col = elem_idx % BK;

            int global_row = block_row * BM + load_row;
            int global_col = tile_idx * BK + load_col;

            // Check boundaries and ensure we can load 4 consecutive elements
            if (global_row < size_i && global_col + 3 < size_k && load_col + 3 < BK) {
                // Vectorized load using float4 (16-byte aligned)
                float4 vals = *reinterpret_cast<const float4*>(
                    &a[global_row * size_k + global_col]);

                tile_a[load_row][load_col]     = vals.x;
                tile_a[load_row][load_col + 1] = vals.y;
                tile_a[load_row][load_col + 2] = vals.z;
                tile_a[load_row][load_col + 3] = vals.w;
            } else {
                // Boundary case: scalar loads
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (load_col + i < BK && global_row < size_i && global_col + i < size_k) {
                        tile_a[load_row][load_col + i] = a[global_row * size_k + global_col + i];
                    } else {
                        tile_a[load_row][load_col + i] = 0.0f;
                    }
                }
            }
        }

        // ====================================================================
        // Vectorized load tile_b: using float4
        // ====================================================================
        // tile_b: [8][64] = 512 floats, 64 threads → 8 floats/thread

        #pragma unroll
        for (int load_offset = 0; load_offset < 2; ++load_offset) {
            int elem_idx = threadIdx.x * 8 + load_offset * 4;
            int load_row = elem_idx / BN;
            int load_col = elem_idx % BN;

            int global_row = tile_idx * BK + load_row;
            int global_col = block_col * BN + load_col;

            if (global_row < size_k && global_col + 3 < size_j && load_col + 3 < BN) {
                // Vectorized load using float4
                float4 vals = *reinterpret_cast<const float4*>(
                    &b[global_row * size_j + global_col]);

                tile_b[load_row][load_col]     = vals.x;
                tile_b[load_row][load_col + 1] = vals.y;
                tile_b[load_row][load_col + 2] = vals.z;
                tile_b[load_row][load_col + 3] = vals.w;
            } else {
                // Boundary case: scalar loads
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (load_col + i < BN && global_row < size_k && global_col + i < size_j) {
                        tile_b[load_row][load_col + i] = b[global_row * size_j + global_col + i];
                    } else {
                        tile_b[load_row][load_col + i] = 0.0f;
                    }
                }
            }
        }

        __syncthreads();

        // ====================================================================
        // Compute on this tile (register-level reuse)
        // Same logic as original, but reading from padded shared memory
        // ====================================================================
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

            // Compute outer product
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

void launch_matmul_l1_reg_vectorized(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    dim3 block_dim(64);
    dim3 grid_dim(
        (size_j + BN - 1) / BN,
        (size_i + BM - 1) / BM
    );

    matmul_l1_reg_vectorized<<<grid_dim, block_dim>>>(size_i, size_j, size_k, a, b, c);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace matmul_l1_reg_vectorized

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace matmul_improved {

// Microtiling parameters - optimized for Tesla P100
// Current best: 9.85 ms, 5.89 TFLOP/s
constexpr int BM = 128;  // Block tile size in M dimension
constexpr int BN = 128;  // Block tile size in N dimension
constexpr int BK = 32;   // K tile size for better compute intensity
constexpr int TM = 8;    // Thread tile size in M dimension
constexpr int TN = 8;    // Thread tile size in N dimension

// 128x128 output tile with 256 threads (8 warps)
// Shared memory: (128*32 + 32*128) * 4 = 32 KB
// Allows 2 blocks/SM on P100 (target ~50% occupancy for best performance)
__global__ void matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c /* pointer to GPU memory */) {

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
        // 256 threads load 128*32 = 4096 elements, each thread loads 16 elements
        #pragma unroll
        for (int load_offset = 0; load_offset < (BM * BK) / 256; ++load_offset) {
            int elem_idx = threadIdx.x + load_offset * 256;
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
        for (int load_offset = 0; load_offset < (BK * BN) / 256; ++load_offset) {
            int elem_idx = threadIdx.x + load_offset * 256;
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

        // Compute on this tile (register-level reuse)
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

void launch_matmul_improved(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a, /* pointer to GPU memory */
    float const *b, /* pointer to GPU memory */
    float *c /* pointer to GPU memory */) {

    // Use async version with double buffering
    dim3 block_dim(256);
    dim3 grid_dim(
        (size_j + BN - 1) / BN,
        (size_i + BM - 1) / BM
    );

    matmul_improved<<<grid_dim, block_dim>>>(size_i, size_j, size_k, a, b, c);
}

////////////////////////////////////////////////////////////////////////////////
// Async Copy Version with Double Buffering
////////////////////////////////////////////////////////////////////////////////

// Smaller BK for double buffering to fit in shared memory limit
constexpr int BK_ASYNC = 16;  // Use 16 instead of 32 to keep shmem under 48KB

__global__ void matmul_improved_async(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    // Double buffering: two sets of shared memory
    // Total: 2*(128*16 + 16*128)*4 = 32 KB (within 48 KB limit)
    __shared__ float tile_a[2][BM][BK_ASYNC];
    __shared__ float tile_b[2][BK_ASYNC][BN];

    // Thread indices for microtile
    const int thread_row = threadIdx.x / (BN / TN);
    const int thread_col = threadIdx.x % (BN / TN);

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
    const int num_tiles = (size_k + BK_ASYNC - 1) / BK_ASYNC;

    // Prefetch first tile (tile_idx=0, buffer=0)
    #pragma unroll
    for (int load_offset = 0; load_offset < (BM * BK_ASYNC) / 256; ++load_offset) {
        int elem_idx = threadIdx.x + load_offset * 256;
        int load_row = elem_idx / BK_ASYNC;
        int load_col = elem_idx % BK_ASYNC;

        int global_row = block_row * BM + load_row;
        int global_col = 0 * BK_ASYNC + load_col;

        if (global_row < size_i && global_col < size_k) {
            // Use cp_async for 16-byte aligned loads (4 floats)
            // For simplicity, fallback to regular load here
            tile_a[0][load_row][load_col] = a[global_row * size_k + global_col];
        } else {
            tile_a[0][load_row][load_col] = 0.0f;
        }
    }

    #pragma unroll
    for (int load_offset = 0; load_offset < (BK_ASYNC * BN) / 256; ++load_offset) {
        int elem_idx = threadIdx.x + load_offset * 256;
        int load_row = elem_idx / BN;
        int load_col = elem_idx % BN;

        int global_row = 0 * BK_ASYNC + load_row;
        int global_col = block_col * BN + load_col;

        if (global_row < size_k && global_col < size_j) {
            tile_b[0][load_row][load_col] = b[global_row * size_j + global_col];
        } else {
            tile_b[0][load_row][load_col] = 0.0f;
        }
    }

    __syncthreads();

    // Main loop with double buffering
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        int read_buffer = tile_idx % 2;
        int write_buffer = (tile_idx + 1) % 2;

        // Async load next tile while computing current tile
        if (tile_idx + 1 < num_tiles) {
            #pragma unroll
            for (int load_offset = 0; load_offset < (BM * BK_ASYNC) / 256; ++load_offset) {
                int elem_idx = threadIdx.x + load_offset * 256;
                int load_row = elem_idx / BK_ASYNC;
                int load_col = elem_idx % BK_ASYNC;

                int global_row = block_row * BM + load_row;
                int global_col = (tile_idx + 1) * BK_ASYNC + load_col;

                if (global_row < size_i && global_col < size_k) {
                    tile_a[write_buffer][load_row][load_col] = a[global_row * size_k + global_col];
                } else {
                    tile_a[write_buffer][load_row][load_col] = 0.0f;
                }
            }

            #pragma unroll
            for (int load_offset = 0; load_offset < (BK_ASYNC * BN) / 256; ++load_offset) {
                int elem_idx = threadIdx.x + load_offset * 256;
                int load_row = elem_idx / BN;
                int load_col = elem_idx % BN;

                int global_row = (tile_idx + 1) * BK_ASYNC + load_row;
                int global_col = block_col * BN + load_col;

                if (global_row < size_k && global_col < size_j) {
                    tile_b[write_buffer][load_row][load_col] = b[global_row * size_j + global_col];
                } else {
                    tile_b[write_buffer][load_row][load_col] = 0.0f;
                }
            }
        }

        // Compute on current tile
        #pragma unroll
        for (int k = 0; k < BK_ASYNC; ++k) {
            // Load TM elements from A into registers
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                reg_a[m] = tile_a[read_buffer][thread_row * TM + m][k];
            }

            // Load TN elements from B into registers
            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                reg_b[n] = tile_b[read_buffer][k][thread_col * TN + n];
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

}; // namespace matmul_improved

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation with Vectorized Loads

namespace matmul_improved_vectorized {

// Same microtiling parameters as matmul_improved
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int TM = 8;
constexpr int TN = 8;

// 128x128 output tile with 256 threads (8 warps)
// Shared memory with padding: (128*33 + 32*129) * 4 = 33.4 KB
// Allows 2 blocks/SM on P100 (target ~50% occupancy)
__global__ void matmul_improved_vectorized(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    // // Shared memory WITHOUT padding (to test if padding causes the slowdown)
    // __shared__ float tile_a[BM][BK];      // [128][32] - same as original
    // __shared__ float tile_b[BK][BN];      // [32][128] - same as original
    // Shared memory with padding to avoid bank conflicts
    __shared__ float tile_a[BM][BK + 1];  // [128][33] - padding eliminates bank conflicts
    __shared__ float tile_b[BK][BN + 1];  // [32][129] - padding eliminates bank conflicts    

    const int thread_row = threadIdx.x / (BN / TN);  // 0-15
    const int thread_col = threadIdx.x % (BN / TN);  // 0-15

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

    float reg_a[TM];
    float reg_b[TN];

    const int num_tiles = (size_k + BK - 1) / BK;

    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {

        // ====================================================================
        // Vectorized load tile_a: using float4 (16-byte aligned loads)
        // ====================================================================
        // tile_a: [128][32] = 4096 floats, 256 threads → 16 floats/thread
        // Strategy: each thread loads 4 float4s (16 consecutive floats)

        #pragma unroll
        for (int load_offset = 0; load_offset < 4; ++load_offset) {
            // Each thread loads 4 float4s
            int elem_idx = threadIdx.x * 16 + load_offset * 4;
            int load_row = elem_idx / BK;
            int load_col = elem_idx % BK;

            int global_row = block_row * BM + load_row;
            int global_col = tile_idx * BK + load_col;

            // Check boundaries and ensure we can load 4 consecutive elements
            if (global_row < size_i && global_col + 3 < size_k && load_col + 3 < BK) {
                // Vectorized load using float4 (16-byte aligned)
                float4 vals = *reinterpret_cast<const float4*>(
                    &a[global_row * size_k + global_col]);

                tile_a[load_row][load_col]     = vals.x;
                tile_a[load_row][load_col + 1] = vals.y;
                tile_a[load_row][load_col + 2] = vals.z;
                tile_a[load_row][load_col + 3] = vals.w;
            } else {
                // Boundary case: scalar loads
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (load_col + i < BK && global_row < size_i && global_col + i < size_k) {
                        tile_a[load_row][load_col + i] = a[global_row * size_k + global_col + i];
                    } else {
                        tile_a[load_row][load_col + i] = 0.0f;
                    }
                }
            }
        }

        // ====================================================================
        // Vectorized load tile_b: using float4
        // ====================================================================
        // tile_b: [32][128] = 4096 floats, 256 threads → 16 floats/thread

        #pragma unroll
        for (int load_offset = 0; load_offset < 4; ++load_offset) {
            int elem_idx = threadIdx.x * 16 + load_offset * 4;
            int load_row = elem_idx / BN;
            int load_col = elem_idx % BN;

            int global_row = tile_idx * BK + load_row;
            int global_col = block_col * BN + load_col;

            if (global_row < size_k && global_col + 3 < size_j && load_col + 3 < BN) {
                // Vectorized load using float4
                float4 vals = *reinterpret_cast<const float4*>(
                    &b[global_row * size_j + global_col]);

                tile_b[load_row][load_col]     = vals.x;
                tile_b[load_row][load_col + 1] = vals.y;
                tile_b[load_row][load_col + 2] = vals.z;
                tile_b[load_row][load_col + 3] = vals.w;
            } else {
                // Boundary case: scalar loads
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (load_col + i < BN && global_row < size_k && global_col + i < size_j) {
                        tile_b[load_row][load_col + i] = b[global_row * size_j + global_col + i];
                    } else {
                        tile_b[load_row][load_col + i] = 0.0f;
                    }
                }
            }
        }

        __syncthreads();

        // ====================================================================
        // Compute on this tile (register-level reuse)
        // Same logic as original, but reading from padded shared memory
        // ====================================================================
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

void launch_matmul_improved_vectorized(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    dim3 block_dim(256);
    dim3 grid_dim(
        (size_j + BN - 1) / BN,
        (size_i + BM - 1) / BM
    );

    matmul_improved_vectorized<<<grid_dim, block_dim>>>(size_i, size_j, size_k, a, b, c);
    CUDA_CHECK(cudaGetLastError());
}

}; // namespace matmul_improved_vectorized

////////////////////////////////////////////////////////////////////////////////
// Warp Specialization with Software Pipelining (Strategy 6B)

namespace matmul_warp_specialization {

// Microtiling parameters optimized for double buffering within 48KB limit
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 24;  // Maximizes use of 48KB shared memory limit on P100
constexpr int TM = 8;
constexpr int TN = 8;

// Software pipelining: overlap data loading and computation
// Using double buffering to hide memory latency
__global__ void matmul_warp_specialization(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    // Double buffered shared memory
    // Total: 2 * (128*24 + 24*128) * 4 = 49,152 bytes = 48 KB
    // Exactly fits P100's 48KB per-block limit
    __shared__ float tile_a[2][BM][BK];
    __shared__ float tile_b[2][BK][BN];

    const int thread_row = threadIdx.x / (BN / TN);
    const int thread_col = threadIdx.x % (BN / TN);

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    // Register arrays
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    float reg_a[TM];
    float reg_b[TN];

    const int num_tiles = (size_k + BK - 1) / BK;

    // ==================================================================
    // Software Pipeline: Prefetch first tile
    // ==================================================================
    int write_buffer = 0;

    // Load first tile into buffer 0
    #pragma unroll
    for (int load_offset = 0; load_offset < (BM * BK) / 256; ++load_offset) {
        int elem_idx = threadIdx.x + load_offset * 256;
        int load_row = elem_idx / BK;
        int load_col = elem_idx % BK;

        int global_row = block_row * BM + load_row;
        int global_col = 0 * BK + load_col;

        if (global_row < size_i && global_col < size_k) {
            tile_a[0][load_row][load_col] = a[global_row * size_k + global_col];
        } else {
            tile_a[0][load_row][load_col] = 0.0f;
        }
    }

    #pragma unroll
    for (int load_offset = 0; load_offset < (BK * BN) / 256; ++load_offset) {
        int elem_idx = threadIdx.x + load_offset * 256;
        int load_row = elem_idx / BN;
        int load_col = elem_idx % BN;

        int global_row = 0 * BK + load_row;
        int global_col = block_col * BN + load_col;

        if (global_row < size_k && global_col < size_j) {
            tile_b[0][load_row][load_col] = b[global_row * size_j + global_col];
        } else {
            tile_b[0][load_row][load_col] = 0.0f;
        }
    }

    __syncthreads();

    // ==================================================================
    // Main Pipeline Loop
    // ==================================================================
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        int read_buffer = tile_idx % 2;
        write_buffer = 1 - read_buffer;

        // ----------------------------------------------------------
        // Prefetch NEXT tile (if exists) - happens in BACKGROUND
        // ----------------------------------------------------------
        if (tile_idx + 1 < num_tiles) {
            #pragma unroll
            for (int load_offset = 0; load_offset < (BM * BK) / 256; ++load_offset) {
                int elem_idx = threadIdx.x + load_offset * 256;
                int load_row = elem_idx / BK;
                int load_col = elem_idx % BK;

                int global_row = block_row * BM + load_row;
                int global_col = (tile_idx + 1) * BK + load_col;

                if (global_row < size_i && global_col < size_k) {
                    tile_a[write_buffer][load_row][load_col] = a[global_row * size_k + global_col];
                } else {
                    tile_a[write_buffer][load_row][load_col] = 0.0f;
                }
            }

            #pragma unroll
            for (int load_offset = 0; load_offset < (BK * BN) / 256; ++load_offset) {
                int elem_idx = threadIdx.x + load_offset * 256;
                int load_row = elem_idx / BN;
                int load_col = elem_idx % BN;

                int global_row = (tile_idx + 1) * BK + load_row;
                int global_col = block_col * BN + load_col;

                if (global_row < size_k && global_col < size_j) {
                    tile_b[write_buffer][load_row][load_col] = b[global_row * size_j + global_col];
                } else {
                    tile_b[write_buffer][load_row][load_col] = 0.0f;
                }
            }
        }

        // ----------------------------------------------------------
        // Compute on CURRENT tile - happens in FOREGROUND
        // The GPU can overlap the memory loads above with computation below
        // ----------------------------------------------------------
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                reg_a[m] = tile_a[read_buffer][thread_row * TM + m][k];
            }

            #pragma unroll
            for (int n = 0; n < TN; ++n) {
                reg_b[n] = tile_b[read_buffer][k][thread_col * TN + n];
            }

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

    // Write results
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

void launch_matmul_warp_specialization(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c) {

    // Note: P100 (CC 6.0) has a hard limit of 48KB static shared memory per block
    // We use exactly 48KB with BK=24 double buffering
    // No configuration needed - static shared memory is automatically allocated

    dim3 block_dim(256);
    dim3 grid_dim(
        (size_j + BN - 1) / BN,
        (size_i + BM - 1) / BM
    );

    matmul_warp_specialization<<<grid_dim, block_dim>>>(size_i, size_j, size_k, a, b, c);
    CUDA_CHECK(cudaGetLastError());
}

}; // namespace matmul_warp_specialization

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation with Reduction along k

namespace matmul_improved_reduce {

// Use same tiling parameters as matmul_improved
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int TM = 8;
constexpr int TN = 8;

// Split-K main kernel: computes partial sums
__global__ void matmul_split_k(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    int32_t split_k,
    float const *a,
    float const *b,
    float *workspace) {

    // Shared memory for A and B tiles
    __shared__ float tile_a[BM][BK];
    __shared__ float tile_b[BK][BN];

    // Thread indices for microtile
    const int thread_row = threadIdx.x / (BN / TN);
    const int thread_col = threadIdx.x % (BN / TN);

    // Block indices - now includes split_k dimension
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int split_k_idx = blockIdx.z;

    // Calculate K range for this split
    const int k_per_split = (size_k + split_k - 1) / split_k;
    const int k_start = split_k_idx * k_per_split;
    const int k_end = min(k_start + k_per_split, size_k);

    // Register arrays for accumulation
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

    // Number of tiles in this K range
    const int num_tiles = (k_end - k_start + BK - 1) / BK;

    // Loop over K dimension tiles (only this split's range)
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const int k_tile_start = k_start + tile_idx * BK;

        // Cooperatively load tile_a from global to shared memory
        #pragma unroll
        for (int load_offset = 0; load_offset < (BM * BK) / 256; ++load_offset) {
            int elem_idx = threadIdx.x + load_offset * 256;
            int load_row = elem_idx / BK;
            int load_col = elem_idx % BK;

            int global_row = block_row * BM + load_row;
            int global_col = k_tile_start + load_col;

            if (global_row < size_i && global_col >= k_start && global_col < k_end) {
                tile_a[load_row][load_col] = a[global_row * size_k + global_col];
            } else {
                tile_a[load_row][load_col] = 0.0f;
            }
        }

        // Cooperatively load tile_b from global to shared memory
        #pragma unroll
        for (int load_offset = 0; load_offset < (BK * BN) / 256; ++load_offset) {
            int elem_idx = threadIdx.x + load_offset * 256;
            int load_row = elem_idx / BN;
            int load_col = elem_idx % BN;

            int global_row = k_tile_start + load_row;
            int global_col = block_col * BN + load_col;

            if (global_row >= k_start && global_row < k_end && global_col < size_j) {
                tile_b[load_row][load_col] = b[global_row * size_j + global_col];
            } else {
                tile_b[load_row][load_col] = 0.0f;
            }
        }

        __syncthreads();

        // Compute on this tile
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

            // Compute outer product
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

    // Write partial results to workspace
    // workspace layout: [split_k_idx][size_i][size_j]
    const int workspace_offset = split_k_idx * size_i * size_j;

    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            int global_row = block_row * BM + thread_row * TM + m;
            int global_col = block_col * BN + thread_col * TN + n;

            if (global_row < size_i && global_col < size_j) {
                workspace[workspace_offset + global_row * size_j + global_col] = acc[m][n];
            }
        }
    }
}

// Reduction kernel: combines partial sums
__global__ void reduce_split_k(
    int32_t size_i,
    int32_t size_j,
    int32_t split_k,
    float const *workspace,
    float *c) {

    // Each thread handles one output element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= size_i || col >= size_j)
        return;

    // Sum across all split_k partitions
    float sum = 0.0f;
    for (int k = 0; k < split_k; ++k) {
        sum += workspace[k * size_i * size_j + row * size_j + col];
    }

    c[row * size_j + col] = sum;
}

// Determine optimal split_k based on problem size
int get_split_k(int32_t size_i, int32_t size_j, int32_t size_k) {
    // Calculate number of output blocks without splitting
    int blocks_m = (size_i + BM - 1) / BM;
    int blocks_n = (size_j + BN - 1) / BN;
    int total_blocks = blocks_m * blocks_n;

    // P100 has 56 SMs, aim for 2-4 blocks per SM
    const int target_blocks = 56 * 3;  // 168 blocks

    if (total_blocks >= target_blocks) {
        // Enough parallelism, no splitting needed
        return 1;
    }

    // Calculate split_k to reach target
    int split_k = (target_blocks + total_blocks - 1) / total_blocks;

    // Cap at reasonable maximum (don't split too much)
    split_k = min(split_k, 32);

    // Also consider K dimension size
    int max_splits_by_k = (size_k + BK - 1) / BK;
    split_k = min(split_k, max_splits_by_k);

    return max(1, split_k);
}

size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
    int split_k = get_split_k(size_i, size_j, size_k);
    return static_cast<size_t>(size_i) * size_j * split_k * sizeof(float);
}

void launch_matmul_improved_reduce(
    int32_t size_i,
    int32_t size_j,
    int32_t size_k,
    float const *a,
    float const *b,
    float *c,
    void *workspace) {

    int split_k = get_split_k(size_i, size_j, size_k);

    // Launch split-k kernel
    dim3 block_dim(256);
    dim3 grid_dim(
        (size_j + BN - 1) / BN,
        (size_i + BM - 1) / BM,
        split_k
    );

    matmul_split_k<<<grid_dim, block_dim>>>(
        size_i, size_j, size_k, split_k, a, b, (float*)workspace);

    // Launch reduction kernel if needed
    if (split_k > 1) {
        dim3 reduce_block(16, 16);
        dim3 reduce_grid(
            (size_j + 15) / 16,
            (size_i + 15) / 16
        );

        reduce_split_k<<<reduce_grid, reduce_block>>>(
            size_i, size_j, split_k, (float const *)workspace, c);
    } else {
        // No reduction needed, copy workspace to output
        CUDA_CHECK(cudaMemcpy(c, workspace, size_i * size_j * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

}; // namespace matmul_improved_reduce

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

template <typename Reset, typename F>
double
benchmark_ms(double target_time_ms, int32_t num_iters_inner, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
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

struct BenchmarkConfig {
    int32_t size_i;
    int32_t size_j;
    int32_t size_k;
};

struct TestData {
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> a;
    std::map<std::tuple<int32_t, int32_t>, std::vector<float>> b;
    std::map<std::tuple<int32_t, int32_t, int32_t>, std::vector<float>> c;
};

TestData read_test_data(
    std::string const &test_data_dir,
    std::vector<BenchmarkConfig> const &configs) {
    auto data = TestData{};
    for (auto const &config : configs) {
        auto size_i = config.size_i;
        auto size_j = config.size_j;
        auto size_k = config.size_k;

        auto path_prefix = test_data_dir + "/test_";

        if (data.a.find({size_i, size_k}) == data.a.end()) {
            data.a[{size_i, size_k}] = read_data(
                path_prefix + "a_" + std::to_string(size_i) + "x" +
                    std::to_string(size_k) + ".bin",
                size_i * size_k);
        }

        if (data.b.find({size_k, size_j}) == data.b.end()) {
            data.b[{size_k, size_j}] = read_data(
                path_prefix + "b_" + std::to_string(size_k) + "x" +
                    std::to_string(size_j) + ".bin",
                size_k * size_j);
        }

        if (data.c.find({size_i, size_j, size_k}) == data.c.end()) {
            data.c[{size_i, size_j, size_k}] = read_data(
                path_prefix + "c_" + std::to_string(size_i) + "x" +
                    std::to_string(size_j) + "x" + std::to_string(size_k) + ".bin",
                size_i * size_j);
        }
    }
    return data;
}

struct BenchmarkResults {
    char const *name;
    std::map<std::tuple<int32_t, int32_t, int32_t>, double> elapsed_ms;
};

enum class Phase {
    WARMUP,
    BENCHMARK,
};

template <typename Impl>
void run_config(
    Phase phase,
    TestData const &data,
    BenchmarkConfig const &config,
    BenchmarkResults &results) {
    auto size_i = config.size_i;
    auto size_j = config.size_j;
    auto size_k = config.size_k;

    auto const &a = data.a.at({size_i, size_k});
    auto const &b = data.b.at({size_k, size_j});
    auto const &c = data.c.at({size_i, size_j, size_k});

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

    size_t workspace_size = Impl::get_workspace_size(size_i, size_j, size_k);
    void *workspace_gpu = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
    }

    void *flush_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&flush_gpu, 1024*1024*64));
    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));

    if (phase == Phase::BENCHMARK) {
        printf("  %6d  %6d  %6d", size_i, size_j, size_k);
    } else {
        printf("  warmup %6d  %6d  %6d", size_i, size_j, size_k);
    }

    Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);

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

    if (phase == Phase::BENCHMARK) {
        printf("  %8.02e", rel_rmse);
    }

    if (rel_rmse > 1e-5) {
        if (phase == Phase::BENCHMARK) {
            printf("  %9s  %7s", "-", "-");
        }
    } else {
        double target_time_ms = 40.0;
        double elapsed_ms = 0.0;
        if (phase == Phase::BENCHMARK) {
            elapsed_ms = benchmark_ms(
                target_time_ms,
                1,
                [&]() {
                    if (workspace_size > 0) {
                        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
                    }
                    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));
                },
                [&]() {
                    Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);
                });
        } else {
            elapsed_ms = benchmark_ms(
                target_time_ms,
                1,
                [&]() {
                    if (workspace_size > 0) {
                        CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
                    }
                    CUDA_CHECK(cudaMemset(flush_gpu, 1, 1024*1024*64));
                },
                [&]() {
                    Impl::run(size_i, size_j, size_k, a_gpu, b_gpu, c_gpu, workspace_gpu);
                }); 
        }

        if (phase == Phase::BENCHMARK) {
            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            printf("  %9.02f  %7.02f", elapsed_ms, tflop / (elapsed_ms * 1e-3));

            results.elapsed_ms[{size_i, size_j, size_k}] = elapsed_ms;
        }
    }

    printf("\n");

    CUDA_CHECK(cudaFree(a_gpu));
    CUDA_CHECK(cudaFree(b_gpu));
    CUDA_CHECK(cudaFree(c_gpu));
    CUDA_CHECK(cudaFree(flush_gpu));
    if (workspace_size > 0) {
        CUDA_CHECK(cudaFree(workspace_gpu));
    }
}

template <typename Impl>
BenchmarkResults run_all_configs(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = BenchmarkResults{Impl::name};
    if (phase == Phase::WARMUP) {
        printf("warmup %s:\n\n", Impl::name);
    } else {
        printf("%s:\n\n", Impl::name);
        printf(
            "  %-6s  %-6s  %-6s  %-8s  %-9s  %-7s\n",
            "size_i",
            "size_j",
            "size_k",
            "RRMSE",
            "time (ms)",
            "TFLOP/s");
        printf(
            "  %-6s  %-6s  %-6s  %-8s  %-9s  %-7s\n",
            "------",
            "------",
            "------",
            "--------",
            "---------",
            "-------");
    }
    for (auto const &config : configs) {
        run_config<Impl>(phase, data, config, results);
    }
    printf("\n");
    return results;
}

#ifdef HAS_LAB_4_BASELINE_IMPL

struct MatmulL1Reg {
    constexpr static char const *name = "matmul_l1_reg";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return 0;
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_l1_reg::launch_matmul_l1_reg(size_i, size_j, size_k, a, b, c);
    }
};

struct MatmulL1RegVectorized {
    constexpr static char const *name = "matmul_l1_reg_vectorized";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return 0;
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_l1_reg_vectorized::launch_matmul_l1_reg_vectorized(
            size_i, size_j, size_k, a, b, c);
    }
};

#endif

struct MatmulImproved {
    constexpr static char const *name = "matmul_improved";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return 0;
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_improved::launch_matmul_improved(size_i, size_j, size_k, a, b, c);
    }
};

struct MatmulImprovedVectorized {
    constexpr static char const *name = "matmul_improved_vectorized";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return 0;
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_improved_vectorized::launch_matmul_improved_vectorized(
            size_i, size_j, size_k, a, b, c);
    }
};

struct MatmulWarpSpecialization {
    constexpr static char const *name = "matmul_warp_specialization";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return 0;
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_warp_specialization::launch_matmul_warp_specialization(
            size_i, size_j, size_k, a, b, c);
    }
};

struct MatmulImprovedReduce {
    constexpr static char const *name = "matmul_improved_reduce";

    static size_t get_workspace_size(int32_t size_i, int32_t size_j, int32_t size_k) {
        return matmul_improved_reduce::get_workspace_size(size_i, size_j, size_k);
    }

    static void
    run(int32_t size_i,
        int32_t size_j,
        int32_t size_k,
        float const *a,
        float const *b,
        float *c,
        void *workspace) {
        matmul_improved_reduce::launch_matmul_improved_reduce(
            size_i,
            size_j,
            size_k,
            a,
            b,
            c,
            workspace);
    }
};

std::vector<BenchmarkResults> run_all_impls(
    Phase phase,
    TestData const &data,
    std::vector<BenchmarkConfig> const &configs) {
    auto results = std::vector<BenchmarkResults>{};
#ifdef HAS_LAB_4_BASELINE_IMPL
    results.push_back(run_all_configs<MatmulL1Reg>(phase, data, configs));
    results.push_back(run_all_configs<MatmulL1RegVectorized>(phase, data, configs));
#endif
    results.push_back(run_all_configs<MatmulImproved>(phase, data, configs));
    results.push_back(run_all_configs<MatmulWarpSpecialization>(phase, data, configs));
    // Commented out: vectorized version was slower due to warp divergence and instruction overhead
    // results.push_back(run_all_configs<MatmulImprovedVectorized>(phase, data, configs));
    results.push_back(run_all_configs<MatmulImprovedReduce>(phase, data, configs));
    return results;
}

void write_json_results(
    std::string const &path,
    std::vector<BenchmarkResults> const &results) {
    auto file = std::ofstream(path);
    file << "{\n";
    for (int32_t i = 0; i < results.size(); ++i) {
        auto const &result = results.at(i);
        file << "  \"" << result.name << "\": [\n";
        int32_t j = 0;
        for (auto const &[config, elapsed_ms] : result.elapsed_ms) {
            auto [size_i, size_j, size_k] = config;
            double tflop = 2.0 * size_i * size_k * size_j * 1e-12;
            double tflop_per_sec = tflop / (elapsed_ms * 1e-3);
            file << "    {\n";
            file << "      \"size_i\": " << size_i << ",\n";
            file << "      \"size_j\": " << size_j << ",\n";
            file << "      \"size_k\": " << size_k << ",\n";
            file << "      \"elapsed_ms\": " << elapsed_ms << ",\n";
            file << "      \"tflop_per_sec\": " << tflop_per_sec << "\n";
            file << "    }";
            if (j + 1 < result.elapsed_ms.size()) {
                file << ",";
            }
            file << "\n";
            ++j;
        }
        file << "  ]";
        if (i + 1 < results.size()) {
            file << ",";
        }
        file << "\n";
    }
    file << "}\n";
}

int main(int argc, char **argv) {
    // Kaggle 的測試資料路徑
    std::string test_data_dir = "/kaggle/working";

    // 如果提供命令列參數，使用該路徑
    if (argc > 1) {
        test_data_dir = argv[1];
    }

    auto configs = std::vector<BenchmarkConfig>{
        {3072, 3072, 3072},
        {512, 3072, 3072},
        {256, 3072, 3072},
        {128, 3072, 3072},
        {64, 3072, 3072},
        {32, 3072, 3072},
        {16, 3072, 3072},
        {1, 3072, 3072},
        {256, 256, 256},
        {256, 256, 1024},
        {256, 256, 8192},
        {128, 128, 32768},
    };
    auto data = read_test_data(test_data_dir, configs);
    run_all_impls(Phase::WARMUP, data, configs);
    auto results = run_all_impls(Phase::BENCHMARK, data, configs);

    for (int32_t j = 1; j < results.size(); ++j) {
        for (int32_t i = j; i > 0;) {
            --i;
            auto const &first = results.at(i);
            auto const &second = results.at(j);
            printf("\nspeedups %s -> %s:\n\n", first.name, second.name);
            printf("  %-6s  %-6s  %-6s  %-7s\n", "size_i", "size_j", "size_k", "speedup");
            printf("  %-6s  %-6s  %-6s  %-7s\n", "------", "------", "------", "-------");
            for (auto const &config : configs) {
                auto size_i = config.size_i;
                auto size_j = config.size_j;
                auto size_k = config.size_k;
                printf("  %6d  %6d  %6d", size_i, size_j, size_k);
                auto it_first = first.elapsed_ms.find({size_i, size_j, size_k});
                auto it_second = second.elapsed_ms.find({size_i, size_j, size_k});
                if (it_first != first.elapsed_ms.end() &&
                    it_second != second.elapsed_ms.end()) {
                    printf("  %6.02fx", it_first->second / it_second->second);
                } else {
                    printf("  %7s", "-");
                }
                printf("\n");
            }
        }
    }

    write_json_results("out/results.json", results);

    return 0;
}
