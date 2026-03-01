%%writefile mandelbrot_gpu_2.cu
// Optional arguments:
//  -r <img_size>
//  -b <max iterations>
//  -i <implementation: {
//      vector,
//      vector_ilp,
//      vector_multicore,
//      vector_multicore_multithread_single_sm,
//      vector_multicore_multithread_full,
//      vector_multicore_multithread_full_ilp,
//    }>

#include <cstdint>
#include <cuda_runtime.h>

constexpr float window_zoom = 1.0 / 10000.0f;
constexpr float window_x = -0.743643887 - 0.5 * window_zoom;
constexpr float window_y = 0.131825904 - 0.5 * window_zoom;
constexpr uint32_t default_max_iters = 2000;

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

/// <--- your code here --->

/*
    // OPTIONAL: Uncomment this block to include your GPU vector implementation
    // from Lab 1 for easy comparison.
    //
    // (If you do this, you'll need to update your code to use the new constants
    // 'window_zoom', 'window_x', and 'window_y'.)

    #define HAS_VECTOR_IMPL // <~~ keep this line if you want to benchmark the vector kernel!

    ////////////////////////////////////////////////////////////////////////////////
    // Vector

    __global__ void mandelbrot_gpu_vector(
        uint32_t img_size,
        uint32_t max_iters,
        uint32_t *out // pointer to GPU memory
    ) {
        // your (GPU) code here...
    }

    void launch_mandelbrot_gpu_vector(
        uint32_t img_size,
        uint32_t max_iters,
        uint32_t *out // pointer to GPU memory
    ) {
        // your (CPU) code here...
    }
*/

////////////////////////////////////////////////////////////////////////////////
// Vector + ILP

__global__ void mandelbrot_gpu_vector_ilp(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    // Strategy: Each CUDA thread processes multiple pixels (ILP chains)
    // to increase instruction-level parallelism within a single warp

    constexpr int NUM_ILP_PIXELS = 4;  // Each thread processes 4 pixels
    uint32_t tid = threadIdx.x;  // 0-31

    const float scale = window_zoom / float(img_size);

    for (uint32_t i = 0; i < img_size; i++) {
        float cy = float(i) * scale + window_y;

        // Each thread processes: j = tid, tid+32, tid+64, ...
        // But now each thread handles NUM_ILP_PIXELS at each j position
        for (uint32_t j_base = tid; j_base < img_size; j_base += 32 * NUM_ILP_PIXELS) {
            // Initialize ILP chains (4 independent pixels per thread)
            float cx[NUM_ILP_PIXELS];
            float x2[NUM_ILP_PIXELS];
            float y2[NUM_ILP_PIXELS];
            float w[NUM_ILP_PIXELS];
            uint32_t iters[NUM_ILP_PIXELS];
            bool active[NUM_ILP_PIXELS];

            // Initialize all chains
            #pragma unroll
            for (int chain = 0; chain < NUM_ILP_PIXELS; chain++) {
                uint32_t j = j_base + chain * 32;
                if (j < img_size) {
                    cx[chain] = float(j) * scale + window_x;
                    x2[chain] = 0.0f;
                    y2[chain] = 0.0f;
                    w[chain] = 0.0f;
                    iters[chain] = 0;
                    active[chain] = true;
                } else {
                    active[chain] = false;
                }
            }

            // Main iteration loop - process all chains together
            for (uint32_t iter = 0; iter < max_iters; iter++) {
                // Check if any chain is still active
                bool any_active = false;
                #pragma unroll
                for (int chain = 0; chain < NUM_ILP_PIXELS; chain++) {
                    any_active |= active[chain];
                }
                if (!any_active) break;

                // Process all chains (explicit interleaving for ILP)
                #pragma unroll
                for (int chain = 0; chain < NUM_ILP_PIXELS; chain++) {
                    if (active[chain]) {
                        float magnitude = x2[chain] + y2[chain];

                        // Check condition
                        if (magnitude <= 4.0f) {
                            // Increment iteration count
                            iters[chain]++;

                            // Compute next iteration
                            float x = x2[chain] - y2[chain] + cx[chain];
                            float y = w[chain] - magnitude + cy;
                            x2[chain] = x * x;
                            y2[chain] = y * y;
                            float z = x + y;
                            w[chain] = z * z;
                        } else {
                            active[chain] = false;
                        }
                    }
                }
            }

            // Store results from all chains
            #pragma unroll
            for (int chain = 0; chain < NUM_ILP_PIXELS; chain++) {
                uint32_t j = j_base + chain * 32;
                if (j < img_size) {
                    out[i * img_size + j] = iters[chain];
                }
            }
        }
    }
}

void launch_mandelbrot_gpu_vector_ilp(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    // Launch with single warp (32 threads) as per lab requirements
    mandelbrot_gpu_vector_ilp<<<1, 32>>>(img_size, max_iters, out);
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core (Full Machine)

__global__ void mandelbrot_gpu_vector_multicore(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    // Calculate global warp ID and lane ID
    uint32_t tid = threadIdx.x;  // Thread ID within block
    uint32_t warp_id_local = tid / 32;  // Warp ID within block
    uint32_t warp_id_global = blockIdx.x * (blockDim.x / 32) + warp_id_local;
    uint32_t lane_id = tid % 32;  // Lane ID within warp

    // Calculate total number of warps
    uint32_t total_warps = gridDim.x * (blockDim.x / 32);

    const float scale = window_zoom / float(img_size);

    // Work partitioning: horizontal (by rows)
    // Each warp processes a contiguous chunk of rows
    uint32_t rows_per_warp = (img_size + total_warps - 1) / total_warps;  // ceil division
    uint32_t start_row = warp_id_global * rows_per_warp;
    uint32_t end_row = start_row + rows_per_warp;
    if (end_row > img_size) end_row = img_size;

    // Each warp processes its assigned rows
    for (uint32_t i = start_row; i < end_row; ++i) {
        float cy = float(i) * scale + window_y;

        // Within each row, lanes process pixels in strided fashion
        // lane 0: j = 0, 32, 64, ...
        // lane 1: j = 1, 33, 65, ...
        for (uint32_t j = lane_id; j < img_size; j += 32) {
            float cx = float(j) * scale + window_x;

            // Mandelbrot computation for this pixel
            float x2 = 0.0f;
            float y2 = 0.0f;
            float w = 0.0f;
            uint32_t iters = 0;

            while (x2 + y2 <= 4.0f && iters < max_iters) {
                float magnitude = x2 + y2;
                float x = x2 - y2 + cx;
                float y = w - magnitude + cy;
                x2 = x * x;
                y2 = y * y;
                float z = x + y;
                w = z * z;
                ++iters;
            }

            // Store result
            out[i * img_size + j] = iters;
        }
    }
}

void launch_mandelbrot_gpu_vector_multicore(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    // Part 2: Full machine with one warp per scheduler
    // Tesla P100: 56 SMs × 2 schedulers/SM = 112 warp schedulers
    // Configuration: Can be set via compile-time defines or use defaults
    // Compile with: -DMULTICORE_NUM_BLOCKS=56 -DMULTICORE_THREADS_PER_BLOCK=64

    #ifndef MULTICORE_NUM_BLOCKS
    #define MULTICORE_NUM_BLOCKS 56  // Default: one block per SM (Tesla P100)
    #endif

    #ifndef MULTICORE_THREADS_PER_BLOCK
    #define MULTICORE_THREADS_PER_BLOCK 64  // Default: 2 warps per block (1 warp/scheduler)
    #endif

    int num_blocks = MULTICORE_NUM_BLOCKS;
    int threads_per_block = MULTICORE_THREADS_PER_BLOCK;

    // Launch configuration: <<<56, 64>>>
    // 56 blocks: one per SM
    // 64 threads per block: 2 warps per block (1 warp per scheduler)
    // Total: 56 * 2 = 112 warps (one per warp scheduler)
    mandelbrot_gpu_vector_multicore<<<num_blocks, threads_per_block>>>(img_size, max_iters, out);
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core + Multi-thread-per-core (Single SM)

__global__ void mandelbrot_gpu_vector_multicore_multithread_single_sm(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    // Part 3: Single SM with multiple warps per warp scheduler
    // This kernel runs on a single SM with multiple warps (> 4)
    // to demonstrate multi-threading within a single SM

    // Calculate warp ID and lane ID within this single block
    uint32_t tid = threadIdx.x;  // Thread ID within block
    uint32_t warp_id = tid / 32;  // Warp ID within block (0 to blockDim.x/32 - 1)
    uint32_t lane_id = tid % 32;  // Lane ID within warp (0-31)

    uint32_t total_warps = blockDim.x / 32;  // Total number of warps in this block

    const float scale = window_zoom / float(img_size);

    // Work partitioning: horizontal (by rows)
    // Each warp processes a contiguous chunk of rows
    uint32_t rows_per_warp = (img_size + total_warps - 1) / total_warps;  // ceil division
    uint32_t start_row = warp_id * rows_per_warp;
    uint32_t end_row = start_row + rows_per_warp;
    if (end_row > img_size) end_row = img_size;

    // Each warp processes its assigned rows
    for (uint32_t i = start_row; i < end_row; ++i) {
        float cy = float(i) * scale + window_y;

        // Within each row, lanes process pixels in strided fashion
        for (uint32_t j = lane_id; j < img_size; j += 32) {
            float cx = float(j) * scale + window_x;

            // Mandelbrot computation for this pixel
            float x2 = 0.0f;
            float y2 = 0.0f;
            float w = 0.0f;
            uint32_t iters = 0;

            while (x2 + y2 <= 4.0f && iters < max_iters) {
                float magnitude = x2 + y2;
                float x = x2 - y2 + cx;
                float y = w - magnitude + cy;
                x2 = x * x;
                y2 = y * y;
                float z = x + y;
                w = z * z;
                ++iters;
            }

            // Store result
            out[i * img_size + j] = iters;
        }
    }
}

void launch_mandelbrot_gpu_vector_multicore_multithread_single_sm(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    // Part 3: Launch with single block but multiple warps
    // Single SM has 4 warp schedulers, so we use more warps to test multi-threading
    // Each warp scheduler can handle up to 12 concurrent warps

    // To test different configurations, change SINGLE_SM_NUM_WARPS below:
    // Option 1:  4 warps (baseline: 1 warp per scheduler, no multi-threading)
    // Option 2:  8 warps (2 warps per scheduler)
    // Option 3: 12 warps (3 warps per scheduler)
    // Option 4: 32 warps (8 warps per scheduler, hardware limit)

    #ifndef SINGLE_SM_NUM_WARPS
    #define SINGLE_SM_NUM_WARPS 4  // Default: baseline configuration
    #endif

    constexpr int NUM_WARPS = SINGLE_SM_NUM_WARPS;
    constexpr int THREADS_PER_BLOCK = NUM_WARPS * 32;

    // Launch with 1 block, multiple warps per block
    // Configuration: NUM_WARPS warps = THREADS_PER_BLOCK threads on 1 SM
    mandelbrot_gpu_vector_multicore_multithread_single_sm<<<1, THREADS_PER_BLOCK>>>(
        img_size, max_iters, out);
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core + Multi-thread-per-core (Full Machine)

__global__ void mandelbrot_gpu_vector_multicore_multithread_full(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    // Part 3: Full machine with multi-threading
    // Use more warps than warp schedulers to exploit multi-threading

    // Calculate global warp ID and lane ID
    uint32_t tid = threadIdx.x;  // Thread ID within block
    uint32_t warp_id_local = tid / 32;  // Warp ID within block
    uint32_t warp_id_global = blockIdx.x * (blockDim.x / 32) + warp_id_local;
    uint32_t lane_id = tid % 32;  // Lane ID within warp

    // Calculate total number of warps
    uint32_t total_warps = gridDim.x * (blockDim.x / 32);

    const float scale = window_zoom / float(img_size);

    // Work partitioning: horizontal (by rows)
    // Each warp processes a contiguous chunk of rows
    uint32_t rows_per_warp = (img_size + total_warps - 1) / total_warps;  // ceil division
    uint32_t start_row = warp_id_global * rows_per_warp;
    uint32_t end_row = start_row + rows_per_warp;
    if (end_row > img_size) end_row = img_size;

    // Each warp processes its assigned rows
    for (uint32_t i = start_row; i < end_row; ++i) {
        float cy = float(i) * scale + window_y;

        // Within each row, lanes process pixels in strided fashion
        for (uint32_t j = lane_id; j < img_size; j += 32) {
            float cx = float(j) * scale + window_x;

            // Mandelbrot computation for this pixel
            float x2 = 0.0f;
            float y2 = 0.0f;
            float w = 0.0f;
            uint32_t iters = 0;

            while (x2 + y2 <= 4.0f && iters < max_iters) {
                float magnitude = x2 + y2;
                float x = x2 - y2 + cx;
                float y = w - magnitude + cy;
                x2 = x * x;
                y2 = y * y;
                float z = x + y;
                w = z * z;
                ++iters;
            }

            // Store result
            out[i * img_size + j] = iters;
        }
    }
}

void launch_mandelbrot_gpu_vector_multicore_multithread_full(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    // Part 3: Full machine with multi-threading
    // Tesla P100: 56 SMs × 2 schedulers/SM = 112 warp schedulers
    // Part 2 used: <<<56, 64>>> = 112 warps (1 warp per scheduler)
    // Part 3 goal: Use more warps to test multi-threading benefit

    // Configuration: Can be set via compile-time defines or use defaults
    // Compile with: -DFULL_GPU_NUM_BLOCKS=112 -DFULL_GPU_THREADS_PER_BLOCK=128

    #ifndef FULL_GPU_NUM_BLOCKS
    #define FULL_GPU_NUM_BLOCKS 112  // Default: 2 blocks per SM (Tesla P100)
    #endif

    #ifndef FULL_GPU_THREADS_PER_BLOCK
    #define FULL_GPU_THREADS_PER_BLOCK 128  // Default: 4 warps per block
    #endif

    int num_blocks = FULL_GPU_NUM_BLOCKS;
    int threads_per_block = FULL_GPU_THREADS_PER_BLOCK;

    // Launch configuration: <<<112, 128>>>
    // 112 blocks: 2 per SM
    // 128 threads per block: 4 warps per block
    // Total: 112 * 4 = 448 warps (4 warps per scheduler)
    mandelbrot_gpu_vector_multicore_multithread_full<<<num_blocks, threads_per_block>>>(
        img_size, max_iters, out);
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core + Multi-thread-per-core + ILP (Full Machine)

__global__ void mandelbrot_gpu_vector_multicore_multithread_full_ilp(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    // Part 4: Combine all techniques: Vector + Multi-core + Multi-threading + ILP
    // Strategy: Each warp processes rows, and each lane within the warp processes
    // multiple pixels using ILP chains

    constexpr int NUM_ILP_PIXELS = 4;  // Each thread processes 4 pixels with ILP

    // Calculate global warp ID and lane ID
    uint32_t tid = threadIdx.x;  // Thread ID within block
    uint32_t warp_id_local = tid / 32;  // Warp ID within block
    uint32_t warp_id_global = blockIdx.x * (blockDim.x / 32) + warp_id_local;
    uint32_t lane_id = tid % 32;  // Lane ID within warp

    // Calculate total number of warps
    uint32_t total_warps = gridDim.x * (blockDim.x / 32);

    const float scale = window_zoom / float(img_size);

    // Work partitioning: horizontal (by rows)
    // Each warp processes a contiguous chunk of rows
    uint32_t rows_per_warp = (img_size + total_warps - 1) / total_warps;  // ceil division
    uint32_t start_row = warp_id_global * rows_per_warp;
    uint32_t end_row = start_row + rows_per_warp;
    if (end_row > img_size) end_row = img_size;

    // Each warp processes its assigned rows
    for (uint32_t i = start_row; i < end_row; ++i) {
        float cy = float(i) * scale + window_y;

        // Within each row, lanes process pixels with ILP
        // Each lane handles: j_base, j_base+32, j_base+64, j_base+96 (4 chains)
        for (uint32_t j_base = lane_id; j_base < img_size; j_base += 32 * NUM_ILP_PIXELS) {
            // Initialize ILP chains (4 independent pixels per thread)
            float cx[NUM_ILP_PIXELS];
            float x2[NUM_ILP_PIXELS];
            float y2[NUM_ILP_PIXELS];
            float w[NUM_ILP_PIXELS];
            uint32_t iters[NUM_ILP_PIXELS];
            bool active[NUM_ILP_PIXELS];

            // Initialize all chains
            #pragma unroll
            for (int chain = 0; chain < NUM_ILP_PIXELS; chain++) {
                uint32_t j = j_base + chain * 32;
                if (j < img_size) {
                    cx[chain] = float(j) * scale + window_x;
                    x2[chain] = 0.0f;
                    y2[chain] = 0.0f;
                    w[chain] = 0.0f;
                    iters[chain] = 0;
                    active[chain] = true;
                } else {
                    active[chain] = false;
                }
            }

            // Main iteration loop - process all chains together
            for (uint32_t iter = 0; iter < max_iters; iter++) {
                // Check if any chain is still active
                bool any_active = false;
                #pragma unroll
                for (int chain = 0; chain < NUM_ILP_PIXELS; chain++) {
                    any_active |= active[chain];
                }
                if (!any_active) break;

                // Process all chains (explicit interleaving for ILP)
                #pragma unroll
                for (int chain = 0; chain < NUM_ILP_PIXELS; chain++) {
                    if (active[chain]) {
                        float magnitude = x2[chain] + y2[chain];

                        // Check condition
                        if (magnitude <= 4.0f) {
                            // Increment iteration count
                            iters[chain]++;

                            // Compute next iteration
                            float x = x2[chain] - y2[chain] + cx[chain];
                            float y = w[chain] - magnitude + cy;
                            x2[chain] = x * x;
                            y2[chain] = y * y;
                            float z = x + y;
                            w[chain] = z * z;
                        } else {
                            active[chain] = false;
                        }
                    }
                }
            }

            // Store results from all chains
            #pragma unroll
            for (int chain = 0; chain < NUM_ILP_PIXELS; chain++) {
                uint32_t j = j_base + chain * 32;
                if (j < img_size) {
                    out[i * img_size + j] = iters[chain];
                }
            }
        }
    }
}

void launch_mandelbrot_gpu_vector_multicore_multithread_full_ilp(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    // Part 4: Full machine with multi-threading + ILP
    // Combine the benefits of:
    // 1. Multi-core parallelism (all SMs)
    // 2. Multi-threading (multiple warps per scheduler)
    // 3. ILP (multiple pixels per thread to hide latency)

    // Use same configurable launch parameters as Part 3
    // Note: Part 3 and Part 4 share the same macros for consistency
    #ifndef FULL_GPU_NUM_BLOCKS
    #define FULL_GPU_NUM_BLOCKS 112  // Default: 2 blocks per SM (Tesla P100)
    #endif

    #ifndef FULL_GPU_THREADS_PER_BLOCK
    #define FULL_GPU_THREADS_PER_BLOCK 128  // Default: 4 warps per block
    #endif

    int num_blocks = FULL_GPU_NUM_BLOCKS;
    int threads_per_block = FULL_GPU_THREADS_PER_BLOCK;

    // Launch configuration: <<<112, 128>>> same as Part 3
    // 112 blocks: 2 per SM
    // 128 threads per block: 4 warps per block
    // Total: 112 * 4 = 448 warps (4 warps per scheduler)
    // Each thread now processes 4 pixels with ILP
    mandelbrot_gpu_vector_multicore_multithread_full_ilp<<<num_blocks, threads_per_block>>>(
        img_size, max_iters, out);
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

// Useful functions and structures.
enum MandelbrotImpl {
    VECTOR,
    VECTOR_ILP,
    VECTOR_MULTICORE,
    VECTOR_MULTICORE_MULTITHREAD_SINGLE_SM,
    VECTOR_MULTICORE_MULTITHREAD_FULL,
    VECTOR_MULTICORE_MULTITHREAD_FULL_ILP,
    ALL
};

// Command-line arguments parser.
int ParseArgsAndMakeSpec(
    int argc,
    char *argv[],
    uint32_t *img_size,
    uint32_t *max_iters,
    MandelbrotImpl *impl) {
    char *implementation_str = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-r") == 0) {
            if (i + 1 < argc) {
                *img_size = atoi(argv[++i]);
                if (*img_size % 32 != 0) {
                    std::cerr << "Error: Image width must be a multiple of 32"
                              << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: No value specified for -r" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                *max_iters = atoi(argv[++i]);
            } else {
                std::cerr << "Error: No value specified for -b" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-i") == 0) {
            if (i + 1 < argc) {
                implementation_str = argv[++i];
                if (strcmp(implementation_str, "vector") == 0) {
                    *impl = VECTOR;
                } else if (strcmp(implementation_str, "vector_ilp") == 0) {
                    *impl = VECTOR_ILP;
                } else if (strcmp(implementation_str, "vector_multicore") == 0) {
                    *impl = VECTOR_MULTICORE;
                } else if (
                    strcmp(
                        implementation_str,
                        "vector_multicore_multithread_single_sm") == 0) {
                    *impl = VECTOR_MULTICORE_MULTITHREAD_SINGLE_SM;
                } else if (
                    strcmp(implementation_str, "vector_multicore_multithread_full") ==
                    0) {
                    *impl = VECTOR_MULTICORE_MULTITHREAD_FULL;
                } else if (
                    strcmp(implementation_str, "vector_multicore_multithread_full_ilp") ==
                    0) {
                    *impl = VECTOR_MULTICORE_MULTITHREAD_FULL_ILP;
                } else if (strcmp(implementation_str, "all") == 0) {
                    *impl = ALL;
                } else {
                    std::cerr << "Error: unknown implementation" << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: No value specified for -i" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unknown flag: " << argv[i] << std::endl;
            return 1;
        }
    }
    std::cout << "Testing with image size " << *img_size << "x" << *img_size << " and "
              << *max_iters << " max iterations." << std::endl;

    return 0;
}

// Output image writers: BMP file header structure
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{0x4D42};   // File type, always "BM"
    uint32_t fileSize{0};        // Size of the file in bytes
    uint16_t reserved1{0};       // Always 0
    uint16_t reserved2{0};       // Always 0
    uint32_t dataOffset{54};     // Start position of pixel data
    uint32_t headerSize{40};     // Size of this header (40 bytes)
    int32_t width{0};            // Image width in pixels
    int32_t height{0};           // Image height in pixels
    uint16_t planes{1};          // Number of color planes
    uint16_t bitsPerPixel{24};   // Bits per pixel (24 for RGB)
    uint32_t compression{0};     // Compression method (0 for uncompressed)
    uint32_t imageSize{0};       // Size of raw bitmap data
    int32_t xPixelsPerMeter{0};  // Horizontal resolution
    int32_t yPixelsPerMeter{0};  // Vertical resolution
    uint32_t colorsUsed{0};      // Number of colors in the color palette
    uint32_t importantColors{0}; // Number of important colors
};
#pragma pack(pop)

void writeBMP(const char *fname, uint32_t img_size, const std::vector<uint8_t> &pixels) {
    uint32_t width = img_size;
    uint32_t height = img_size;

    BMPHeader header;
    header.width = width;
    header.height = height;
    header.imageSize = width * height * 3;
    header.fileSize = header.dataOffset + header.imageSize;

    std::ofstream file(fname, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    file.write(reinterpret_cast<const char *>(pixels.data()), pixels.size());
}

std::vector<uint8_t> iters_to_colors(
    uint32_t img_size,
    uint32_t max_iters,
    const std::vector<uint32_t> &iters) {
    uint32_t width = img_size;
    uint32_t height = img_size;
    uint32_t min_iters = max_iters;
    for (uint32_t i = 0; i < img_size; i++) {
        for (uint32_t j = 0; j < img_size; j++) {
            min_iters = std::min(min_iters, iters[i * img_size + j]);
        }
    }
    float log_iters_min = log2f(static_cast<float>(min_iters));
    float log_iters_range =
        log2f(static_cast<float>(max_iters) / static_cast<float>(min_iters));
    auto pixel_data = std::vector<uint8_t>(width * height * 3);
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            uint32_t iter = iters[i * width + j];

            uint8_t r = 0, g = 0, b = 0;
            if (iter < max_iters) {
                auto log_iter = log2f(static_cast<float>(iter)) - log_iters_min;
                auto intensity = static_cast<uint8_t>(log_iter * 222 / log_iters_range);
                r = 32;
                g = 32 + intensity;
                b = 32;
            }

            auto index = (i * width + j) * 3;
            pixel_data[index] = b;
            pixel_data[index + 1] = g;
            pixel_data[index + 2] = r;
        }
    }
    return pixel_data;
}

// Benchmarking macros and configuration.
#define BENCHPRESS(func, kNumOfOuterIterations, kNumOfInnerIterations, ...) \
    do { \
        std::cout << std::endl << "Running " << #func << " ...\n"; \
        std::vector<double> times(kNumOfOuterIterations); \
        for (size_t i = 0; i < kNumOfOuterIterations; ++i) { \
            auto start = std::chrono::high_resolution_clock::now(); \
            for (size_t j = 0; j < kNumOfInnerIterations; ++j) { \
                func(__VA_ARGS__); \
            } \
            CUDA_CHECK(cudaDeviceSynchronize()); \
            auto end = std::chrono::high_resolution_clock::now(); \
            times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start) \
                           .count() / \
                kNumOfInnerIterations; \
        } \
        std::sort(times.begin(), times.end()); \
        std::stringstream sstream; \
        sstream << std::fixed << std::setw(6) << std::setprecision(2) \
                << times[0] / 1'000'000; \
        std::cout << "  Runtime: " << sstream.str() << " ms" << std::endl; \
    } while (0)

// AUX CUDA check functions.
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

double difference(
    uint32_t img_size,
    uint32_t max_iters,
    std::vector<uint32_t> &result,
    std::vector<uint32_t> &ref_result) {
    int64_t diff = 0;
    for (uint32_t i = 0; i < img_size; i++) {
        for (uint32_t j = 0; j < img_size; j++) {
            diff +=
                abs(int(result[i * img_size + j]) - int(ref_result[i * img_size + j]));
        }
    }
    return diff / double(img_size * img_size * max_iters);
}

// CPU Scalar Mandelbrot set generation.
// Based on the "optimized escape time algorithm" in
// https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set
void mandelbrot_cpu_scalar(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    for (uint64_t i = 0; i < img_size; ++i) {
        for (uint64_t j = 0; j < img_size; ++j) {
            float cx = (float(j) / float(img_size)) * window_zoom + window_x;
            float cy = (float(i) / float(img_size)) * window_zoom + window_y;

            float x2 = 0.0f;
            float y2 = 0.0f;
            float w = 0.0f;
            uint32_t iters = 0;
            while (x2 + y2 <= 4.0f && iters < max_iters) {
                float x = x2 - y2 + cx;
                float y = w - (x2 + y2) + cy;
                x2 = x * x;
                y2 = y * y;
                float z = x + y;
                w = z * z;
                ++iters;
            }

            // Write result.
            out[i * img_size + j] = iters;
        }
    }
}

void dump_image(
    const char *fname,
    uint32_t img_size,
    uint32_t max_iters,
    const std::vector<uint32_t> &iters) {
    // Dump result as an image.
    auto pixel_data = iters_to_colors(img_size, max_iters, iters);
    writeBMP(fname, img_size, pixel_data);
}

// Main function.
// Compile with:
//  g++ -march=native -O3 -Wall -Wextra -o mandelbrot mandelbrot_cpu.cc
int main(int argc, char *argv[]) {
    // Get Mandelbrot spec.
    uint32_t img_size = 1024;
    uint32_t max_iters = default_max_iters;
    enum MandelbrotImpl impl = ALL;
    if (ParseArgsAndMakeSpec(argc, argv, &img_size, &max_iters, &impl))
        return -1;

    // Allocate memory.
    std::vector<uint32_t> ref_result(img_size * img_size);
    std::vector<uint32_t> result_host(img_size * img_size);

    // Compute the reference solution
    mandelbrot_cpu_scalar(img_size, max_iters, ref_result.data());

    // Allocate CUDA memory.
    uint32_t *result_device;
    CUDA_CHECK(cudaMalloc(&result_device, img_size * img_size * sizeof(uint32_t)));

    // Test the desired kernels.
#ifdef HAS_VECTOR_IMPL
    if (impl == VECTOR || impl == ALL) {
        CUDA_CHECK(cudaMemset(result_device, 0, img_size * img_size * sizeof(uint32_t)));
        BENCHPRESS(
            launch_mandelbrot_gpu_vector,
            1,
            4,
            img_size,
            max_iters,
            result_device);
        // Copy result back.
        CUDA_CHECK(cudaMemcpy(
            result_host.data(),
            result_device,
            img_size * img_size * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
        dump_image("out/mandelbrot_gpu_vector.bmp", img_size, max_iters, result_host);
        // Check for correctness.
        std::cout << "  Correctness: average output difference from reference "
                  << difference(img_size, max_iters, result_host, ref_result)
                  << std::endl;
    }
#endif

    if (impl == VECTOR_ILP || impl == ALL) {
        CUDA_CHECK(cudaMemset(result_device, 0, img_size * img_size * sizeof(uint32_t)));
        BENCHPRESS(
            launch_mandelbrot_gpu_vector_ilp,
            1,
            2,
            img_size,
            max_iters,
            result_device);
        // Copy result back.
        CUDA_CHECK(cudaMemcpy(
            result_host.data(),
            result_device,
            img_size * img_size * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
        dump_image("out/mandelbrot_gpu_vector_ilp.bmp", img_size, max_iters, result_host);
        // Check for correctness.
        std::cout << "  Correctness: average output difference from reference "
                  << difference(img_size, max_iters, result_host, ref_result)
                  << std::endl;
    }


    if (impl == VECTOR_MULTICORE || impl == ALL) {
        CUDA_CHECK(cudaMemset(result_device, 0, img_size * img_size * sizeof(uint32_t)));
        BENCHPRESS(
            launch_mandelbrot_gpu_vector_multicore,
            3,
            5,
            img_size,
            max_iters,
            result_device);
        // Copy result back.
        CUDA_CHECK(cudaMemcpy(
            result_host.data(),
            result_device,
            img_size * img_size * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
        dump_image(
            "out/mandelbrot_gpu_vector_multicore.bmp",
            img_size,
            max_iters,
            result_host);
        // Check for correctness.
        std::cout << "  Correctness: average output difference from reference "
                  << difference(img_size, max_iters, result_host, ref_result)
                  << std::endl;
    }

    if (impl == VECTOR_MULTICORE_MULTITHREAD_SINGLE_SM || impl == ALL) {
        CUDA_CHECK(cudaMemset(result_device, 0, img_size * img_size * sizeof(uint32_t)));
        BENCHPRESS(
            launch_mandelbrot_gpu_vector_multicore_multithread_single_sm,
            3,
            5,
            img_size,
            max_iters,
            result_device);
        // Copy result back.
        CUDA_CHECK(cudaMemcpy(
            result_host.data(),
            result_device,
            img_size * img_size * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
        dump_image(
            "out/mandelbrot_gpu_vector_multicore_multithread_single_sm.bmp",
            img_size,
            max_iters,
            result_host);
        // Check for correctness.
        std::cout << "  Correctness: average output difference from reference "
                  << difference(img_size, max_iters, result_host, ref_result)
                  << std::endl;
    }

    if (impl == VECTOR_MULTICORE_MULTITHREAD_FULL || impl == ALL) {
        CUDA_CHECK(cudaMemset(result_device, 0, img_size * img_size * sizeof(uint32_t)));
        BENCHPRESS(
            launch_mandelbrot_gpu_vector_multicore_multithread_full,
            3,
            5,
            img_size,
            max_iters,
            result_device);
        // Copy result back.
        CUDA_CHECK(cudaMemcpy(
            result_host.data(),
            result_device,
            img_size * img_size * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
        dump_image(
            "out/mandelbrot_gpu_vector_multicore_multithread_full.bmp",
            img_size,
            max_iters,
            result_host);
        // Check for correctness.
        std::cout << "  Correctness: average output difference from reference "
                  << difference(img_size, max_iters, result_host, ref_result)
                  << std::endl;
    }

    if (impl == VECTOR_MULTICORE_MULTITHREAD_FULL_ILP || impl == ALL) {
        CUDA_CHECK(cudaMemset(result_device, 0, img_size * img_size * sizeof(uint32_t)));
        BENCHPRESS(
            launch_mandelbrot_gpu_vector_multicore_multithread_full_ilp,
            3,
            5,
            img_size,
            max_iters,
            result_device);
        // Copy result back.
        CUDA_CHECK(cudaMemcpy(
            result_host.data(),
            result_device,
            img_size * img_size * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
        dump_image(
            "out/mandelbrot_gpu_vector_multicore_multithread_full_ilp.bmp",
            img_size,
            max_iters,
            result_host);
        // Check for correctness.
        std::cout << "  Correctness: average output difference from reference "
                  << difference(img_size, max_iters, result_host, ref_result)
                  << std::endl;
    }

    // Free CUDA memory.
    CUDA_CHECK(cudaFree(result_device));

    return 0;
}
