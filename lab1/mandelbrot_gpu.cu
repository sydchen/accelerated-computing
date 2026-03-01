#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

__global__ void mandelbrot_gpu_scalar(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    for (uint32_t i = 0; i < img_size; i++) {
        for (uint32_t j = 0; j < img_size; j++) {
            float cx = (float(j) / float(img_size)) * 2.5f - 2.0f;
            float cy = (float(i) / float(img_size)) * 2.5f - 1.25f;

            float x2 = 0.0f;
            float y2 = 0.0f;
            float w = 0.0f;
            uint32_t iters = 0;
            while (x2 + y2 <= 4.0f && iters < max_iters) {
                float x = x2 - y2 + cx;
                float y = w - x2 - y2 + cy;
                x2 = x * x;
                y2 = y * y;
                w = (x + y) * (x + y);
                ++iters;
            }
            out[i * img_size + j] = iters;
        }
    }
}

void launch_mandelbrot_gpu_scalar(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    mandelbrot_gpu_scalar<<<1, 1>>>(img_size, max_iters, out);
}

__global__ void mandelbrot_gpu_vector(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    // Each thread processes one pixel position in a row.
    // threadIdx.x is the thread index inside the block (0-31).
    uint32_t tid = threadIdx.x;

    // Iterate through all rows.
    for (uint32_t i = 0; i < img_size; i++) {
        float cy = (float(i) / float(img_size)) * 2.5f - 1.25f;
        // thread 0 handles j=0, 32, 64, ...
        // thread 1 handles j=1, 33, 65, ...
        // and so on.
        for (uint32_t j = tid; j < img_size; j += 32) {
            float cx = (float(j) / float(img_size)) * 2.5f - 2.0f;

            float x2 = 0.0f;
            float y2 = 0.0f;
            float w = 0.0f;
            uint32_t iters = 0;

            while (x2 + y2 <= 4.0f && iters < max_iters) {
                float x = x2 - y2 + cx;
                float y = w - x2 - y2 + cy;
                x2 = x * x;
                y2 = y * y;
                w = (x + y) * (x + y);
                ++iters;
            }

            out[i * img_size + j] = iters;
        }
    }
}

void launch_mandelbrot_gpu_vector(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out /* pointer to GPU memory */
) {
    // Launch 1 block with 32 threads.
    // The 32 threads execute in one warp (32-wide SIMD-style execution).
    mandelbrot_gpu_vector<<<1, 32>>>(img_size, max_iters, out);
}
