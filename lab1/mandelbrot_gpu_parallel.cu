#include <cstdint>

// GPU Parallel Scalar: each thread processes one pixel.
__global__ void mandelbrot_gpu_parallel_scalar(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out
) {
    // 2D grid mapping: each thread maps to one pixel.
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;  // column (x)
    uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;  // row (y)

    // Boundary check.
    if (i >= img_size || j >= img_size) return;

    // Compute complex coordinates for this pixel.
    float cx = (float(j) / float(img_size)) * 2.5f - 2.0f;
    float cy = (float(i) / float(img_size)) * 2.5f - 1.25f;

    // Mandelbrot iteration.
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

    // Write result.
    out[i * img_size + j] = iters;
}

void launch_mandelbrot_gpu_parallel_scalar(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out
) {
    // Use 16x16 blocks (256 threads per block).
    dim3 block(16, 16);

    // Compute required grid size.
    dim3 grid((img_size + block.x - 1) / block.x,
              (img_size + block.y - 1) / block.y);

    mandelbrot_gpu_parallel_scalar<<<grid, block>>>(img_size, max_iters, out);
}

// GPU Parallel Vector: use larger blocks to increase parallelism.
__global__ void mandelbrot_gpu_parallel_vector(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out
) {
    // 2D grid mapping.
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= img_size || j >= img_size) return;

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

void launch_mandelbrot_gpu_parallel_vector(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out
) {
    // Use 32x32 blocks (1024 threads per block, maximum).
    dim3 block(32, 32);

    // Compute required grid size.
    dim3 grid((img_size + block.x - 1) / block.x,
              (img_size + block.y - 1) / block.y);

    mandelbrot_gpu_parallel_vector<<<grid, block>>>(img_size, max_iters, out);
}
