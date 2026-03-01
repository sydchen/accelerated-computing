%%writefile circles.cu
// 158ms
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

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

class GpuMemoryPool {
  public:
    GpuMemoryPool() = default;

    ~GpuMemoryPool();

    GpuMemoryPool(GpuMemoryPool const &) = delete;
    GpuMemoryPool &operator=(GpuMemoryPool const &) = delete;
    GpuMemoryPool(GpuMemoryPool &&) = delete;
    GpuMemoryPool &operator=(GpuMemoryPool &&) = delete;

    void *alloc(size_t size);
    void reset();

  private:
    std::vector<void *> allocations_;
    std::vector<size_t> capacities_;
    size_t next_idx_ = 0;
};

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Already Written)

void render_cpu(
    int32_t width,
    int32_t height,
    int32_t n_circle,
    float const *circle_x,
    float const *circle_y,
    float const *circle_radius,
    float const *circle_red,
    float const *circle_green,
    float const *circle_blue,
    float const *circle_alpha,
    float *img_red,
    float *img_green,
    float *img_blue) {

    // Initialize background to white
    for (int32_t pixel_idx = 0; pixel_idx < width * height; pixel_idx++) {
        img_red[pixel_idx] = 1.0f;
        img_green[pixel_idx] = 1.0f;
        img_blue[pixel_idx] = 1.0f;
    }

    // Render circles
    for (int32_t i = 0; i < n_circle; i++) {
        float c_x = circle_x[i];
        float c_y = circle_y[i];
        float c_radius = circle_radius[i];
        for (int32_t y = int32_t(c_y - c_radius); y <= int32_t(c_y + c_radius + 1.0f);
             y++) {
            for (int32_t x = int32_t(c_x - c_radius); x <= int32_t(c_x + c_radius + 1.0f);
                 x++) {
                float dx = x - c_x;
                float dy = y - c_y;
                if (!(0 <= x && x < width && 0 <= y && y < height &&
                      dx * dx + dy * dy < c_radius * c_radius)) {
                    continue;
                }
                int32_t pixel_idx = y * width + x;
                float pixel_red = img_red[pixel_idx];
                float pixel_green = img_green[pixel_idx];
                float pixel_blue = img_blue[pixel_idx];
                float pixel_alpha = circle_alpha[i];
                pixel_red =
                    circle_red[i] * pixel_alpha + pixel_red * (1.0f - pixel_alpha);
                pixel_green =
                    circle_green[i] * pixel_alpha + pixel_green * (1.0f - pixel_alpha);
                pixel_blue =
                    circle_blue[i] * pixel_alpha + pixel_blue * (1.0f - pixel_alpha);
                img_red[pixel_idx] = pixel_red;
                img_green[pixel_idx] = pixel_green;
                img_blue[pixel_idx] = pixel_blue;
            }
        }
    }
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation
// Strategy: Pre-binning circles to tiles, then render only relevant circles

namespace circles_gpu {

#define TILE_SIZE 16
#define TILES_X 64   // 1024 / 16
#define TILES_Y 64
#define N_TILES (TILES_X * TILES_Y)  // 4096

// Kernel 1: Initialize canvas to white
__global__ void init_image_kernel(
    float *img_red,
    float *img_green,
    float *img_blue,
    int n_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels) return;

    img_red[idx] = 1.0f;
    img_green[idx] = 1.0f;
    img_blue[idx] = 1.0f;
}

// Kernel 2: Count circles per tile
__global__ void count_circles_per_tile(
    int32_t n_circle,
    const float *__restrict__ circle_x,
    const float *__restrict__ circle_y,
    const float *__restrict__ circle_radius,
    int *tile_counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_circle) return;

    float cx = circle_x[i];
    float cy = circle_y[i];
    float radius = circle_radius[i];

    // Calculate which tiles this circle overlaps
    int tile_x_min = max(0, (int)((cx - radius) / TILE_SIZE));
    int tile_x_max = min(TILES_X - 1, (int)((cx + radius) / TILE_SIZE));
    int tile_y_min = max(0, (int)((cy - radius) / TILE_SIZE));
    int tile_y_max = min(TILES_Y - 1, (int)((cy + radius) / TILE_SIZE));

    // Increment count for each overlapping tile
    for (int ty = tile_y_min; ty <= tile_y_max; ty++) {
        for (int tx = tile_x_min; tx <= tile_x_max; tx++) {
            int tile_idx = ty * TILES_X + tx;
            atomicAdd(&tile_counts[tile_idx], 1);
        }
    }
}

// Kernel 3: Assign circles to tiles
__global__ void assign_circles_to_tiles(
    int32_t n_circle,
    const float *__restrict__ circle_x,
    const float *__restrict__ circle_y,
    const float *__restrict__ circle_radius,
    const int *tile_offsets,
    int *tile_counts,  // Used as write positions, reset to 0 first
    int *circle_lists)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_circle) return;

    float cx = circle_x[i];
    float cy = circle_y[i];
    float radius = circle_radius[i];

    int tile_x_min = max(0, (int)((cx - radius) / TILE_SIZE));
    int tile_x_max = min(TILES_X - 1, (int)((cx + radius) / TILE_SIZE));
    int tile_y_min = max(0, (int)((cy - radius) / TILE_SIZE));
    int tile_y_max = min(TILES_Y - 1, (int)((cy + radius) / TILE_SIZE));

    for (int ty = tile_y_min; ty <= tile_y_max; ty++) {
        for (int tx = tile_x_min; tx <= tile_x_max; tx++) {
            int tile_idx = ty * TILES_X + tx;
            int pos = atomicAdd(&tile_counts[tile_idx], 1);
            circle_lists[tile_offsets[tile_idx] + pos] = i;
        }
    }
}

// Kernel 4: Sort circles within each tile (one thread per tile, simple insertion sort)
__global__ void sort_tile_circles(
    const int *tile_offsets,
    const int *tile_sizes,
    int *circle_lists)
{
    int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_idx >= N_TILES) return;

    int start = tile_offsets[tile_idx];
    int size = tile_sizes[tile_idx];

    // Simple insertion sort - fast for small arrays
    for (int i = 1; i < size; i++) {
        int key = circle_lists[start + i];
        int j = i - 1;
        while (j >= 0 && circle_lists[start + j] > key) {
            circle_lists[start + j + 1] = circle_lists[start + j];
            j--;
        }
        circle_lists[start + j + 1] = key;
    }
}

// Kernel 5: Render using pre-computed tile lists with shared memory
#define MAX_CIRCLES_PER_BATCH 128

__global__ void render_with_tile_lists(
    int32_t width, int32_t height,
    const float *__restrict__ circle_x,
    const float *__restrict__ circle_y,
    const float *__restrict__ circle_radius,
    const float *__restrict__ circle_red,
    const float *__restrict__ circle_green,
    const float *__restrict__ circle_blue,
    const float *__restrict__ circle_alpha,
    const int *tile_offsets,
    const int *tile_sizes,
    const int *circle_lists,
    float *img_red, float *img_green, float *img_blue)
{
    // Shared memory for circle data
    __shared__ float sh_cx[MAX_CIRCLES_PER_BATCH];
    __shared__ float sh_cy[MAX_CIRCLES_PER_BATCH];
    __shared__ float sh_radius[MAX_CIRCLES_PER_BATCH];
    __shared__ float sh_cr[MAX_CIRCLES_PER_BATCH];
    __shared__ float sh_cg[MAX_CIRCLES_PER_BATCH];
    __shared__ float sh_cb[MAX_CIRCLES_PER_BATCH];
    __shared__ float sh_alpha[MAX_CIRCLES_PER_BATCH];

    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int tile_idx = tile_y * TILES_X + tile_x;
    int tid = threadIdx.y * TILE_SIZE + threadIdx.x;
    int block_threads = TILE_SIZE * TILE_SIZE;

    int pixel_x = tile_x * TILE_SIZE + threadIdx.x;
    int pixel_y = tile_y * TILE_SIZE + threadIdx.y;

    bool valid = (pixel_x < width && pixel_y < height);
    int pixel_idx = pixel_y * width + pixel_x;
    float px = (float)pixel_x;
    float py = (float)pixel_y;

    float r = valid ? img_red[pixel_idx] : 0.0f;
    float g = valid ? img_green[pixel_idx] : 0.0f;
    float b = valid ? img_blue[pixel_idx] : 0.0f;

    int list_start = tile_offsets[tile_idx];
    int list_size = tile_sizes[tile_idx];

    // Process circles in batches using shared memory
    for (int batch_start = 0; batch_start < list_size; batch_start += MAX_CIRCLES_PER_BATCH) {
        int batch_end = min(batch_start + MAX_CIRCLES_PER_BATCH, list_size);
        int batch_size = batch_end - batch_start;

        // Cooperatively load circle data into shared memory
        for (int k = tid; k < batch_size; k += block_threads) {
            int i = circle_lists[list_start + batch_start + k];
            sh_cx[k] = circle_x[i];
            sh_cy[k] = circle_y[i];
            sh_radius[k] = circle_radius[i];
            sh_cr[k] = circle_red[i];
            sh_cg[k] = circle_green[i];
            sh_cb[k] = circle_blue[i];
            sh_alpha[k] = circle_alpha[i];
        }
        __syncthreads();

        // Process this batch
        for (int k = 0; k < batch_size; k++) {
            float cx = sh_cx[k];
            float cy = sh_cy[k];
            float radius = sh_radius[k];

            float dx = px - cx;
            float dy = py - cy;
            if (dx * dx + dy * dy >= radius * radius) continue;

            float alpha = sh_alpha[k];
            if (alpha < 0.001f) continue;

            float cr = sh_cr[k];
            float cg = sh_cg[k];
            float cb = sh_cb[k];
            float one_minus_alpha = 1.0f - alpha;

            r = cr * alpha + r * one_minus_alpha;
            g = cg * alpha + g * one_minus_alpha;
            b = cb * alpha + b * one_minus_alpha;
        }
        __syncthreads();
    }

    if (valid) {
        img_red[pixel_idx] = r;
        img_green[pixel_idx] = g;
        img_blue[pixel_idx] = b;
    }
}

void launch_render(
    int32_t width,
    int32_t height,
    int32_t n_circle,
    float const *circle_x,
    float const *circle_y,
    float const *circle_radius,
    float const *circle_red,
    float const *circle_green,
    float const *circle_blue,
    float const *circle_alpha,
    float *img_red,
    float *img_green,
    float *img_blue,
    GpuMemoryPool &memory_pool) {

    // 1. Initialize canvas to white
    int n_pixels = width * height;
    {
        dim3 block(256);
        dim3 grid((n_pixels + 255) / 256);
        init_image_kernel<<<grid, block>>>(img_red, img_green, img_blue, n_pixels);
        CUDA_CHECK(cudaGetLastError());
    }

    // 2. Allocate memory for binning
    int *tile_counts = (int*)memory_pool.alloc(N_TILES * sizeof(int));
    int *tile_offsets = (int*)memory_pool.alloc((N_TILES + 1) * sizeof(int));
    int *tile_sizes = (int*)memory_pool.alloc(N_TILES * sizeof(int));

    // Estimate max circles per tile (rough upper bound)
    // Most circles are small, but some fog circles cover everything
    size_t max_list_size = (size_t)n_circle * 4;  // Conservative estimate
    int *circle_lists = (int*)memory_pool.alloc(max_list_size * sizeof(int));

    // 3. Count circles per tile
    CUDA_CHECK(cudaMemset(tile_counts, 0, N_TILES * sizeof(int)));
    {
        dim3 block(256);
        dim3 grid((n_circle + 255) / 256);
        count_circles_per_tile<<<grid, block>>>(
            n_circle, circle_x, circle_y, circle_radius, tile_counts);
        CUDA_CHECK(cudaGetLastError());
    }

    // 4. Compute prefix sum (offsets) on CPU for simplicity
    std::vector<int> h_counts(N_TILES);
    std::vector<int> h_offsets(N_TILES + 1);
    CUDA_CHECK(cudaMemcpy(h_counts.data(), tile_counts, N_TILES * sizeof(int), cudaMemcpyDeviceToHost));

    h_offsets[0] = 0;
    for (int i = 0; i < N_TILES; i++) {
        h_offsets[i + 1] = h_offsets[i] + h_counts[i];
    }
    CUDA_CHECK(cudaMemcpy(tile_offsets, h_offsets.data(), (N_TILES + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tile_sizes, h_counts.data(), N_TILES * sizeof(int), cudaMemcpyHostToDevice));

    // 5. Assign circles to tiles
    CUDA_CHECK(cudaMemset(tile_counts, 0, N_TILES * sizeof(int)));  // Reset for write positions
    {
        dim3 block(256);
        dim3 grid((n_circle + 255) / 256);
        assign_circles_to_tiles<<<grid, block>>>(
            n_circle, circle_x, circle_y, circle_radius,
            tile_offsets, tile_counts, circle_lists);
        CUDA_CHECK(cudaGetLastError());
    }

    // 6. Sort circles within each tile (for correct alpha blending order)
    {
        dim3 block(256);
        dim3 grid((N_TILES + 255) / 256);
        sort_tile_circles<<<grid, block>>>(tile_offsets, tile_sizes, circle_lists);
        CUDA_CHECK(cudaGetLastError());
    }

    // 7. Render using tile lists
    {
        dim3 block(TILE_SIZE, TILE_SIZE);  // 16x16 = 256 threads
        dim3 grid(TILES_X, TILES_Y);       // 64x64 tiles
        render_with_tile_lists<<<grid, block>>>(
            width, height,
            circle_x, circle_y, circle_radius,
            circle_red, circle_green, circle_blue, circle_alpha,
            tile_offsets, tile_sizes, circle_lists,
            img_red, img_green, img_blue);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace circles_gpu

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

GpuMemoryPool::~GpuMemoryPool() {
    for (auto ptr : allocations_) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void *GpuMemoryPool::alloc(size_t size) {
    if (next_idx_ < allocations_.size()) {
        auto idx = next_idx_++;
        if (size > capacities_.at(idx)) {
            CUDA_CHECK(cudaFree(allocations_.at(idx)));
            CUDA_CHECK(cudaMalloc(&allocations_.at(idx), size));
            CUDA_CHECK(cudaMemset(allocations_.at(idx), 0, size));
            capacities_.at(idx) = size;
        }
        return allocations_.at(idx);
    } else {
        void *ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        CUDA_CHECK(cudaMemset(ptr, 0, size));
        allocations_.push_back(ptr);
        capacities_.push_back(size);
        next_idx_++;
        return ptr;
    }
}

void GpuMemoryPool::reset() {
    next_idx_ = 0;
    for (int32_t i = 0; i < allocations_.size(); i++) {
        CUDA_CHECK(cudaMemset(allocations_.at(i), 0, capacities_.at(i)));
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

struct Scene {
    int32_t width;
    int32_t height;
    std::vector<float> circle_x;
    std::vector<float> circle_y;
    std::vector<float> circle_radius;
    std::vector<float> circle_red;
    std::vector<float> circle_green;
    std::vector<float> circle_blue;
    std::vector<float> circle_alpha;

    int32_t n_circle() const { return circle_x.size(); }
};

struct Image {
    int32_t width;
    int32_t height;
    std::vector<float> red;
    std::vector<float> green;
    std::vector<float> blue;
};

float max_abs_diff(Image const &a, Image const &b) {
    float max_diff = 0.0f;
    for (int32_t idx = 0; idx < a.width * a.height; idx++) {
        float diff_red = std::abs(a.red.at(idx) - b.red.at(idx));
        float diff_green = std::abs(a.green.at(idx) - b.green.at(idx));
        float diff_blue = std::abs(a.blue.at(idx) - b.blue.at(idx));
        max_diff = std::max(max_diff, diff_red);
        max_diff = std::max(max_diff, diff_green);
        max_diff = std::max(max_diff, diff_blue);
    }
    return max_diff;
}

struct Results {
    bool correct;
    float max_abs_diff;
    Image image_expected;
    Image image_actual;
    double time_ms;
};

enum class Mode {
    TEST,
    BENCHMARK,
};

template <typename T> struct GpuBuf {
    T *data;

    explicit GpuBuf(size_t n) { CUDA_CHECK(cudaMalloc(&data, n * sizeof(T))); }

    explicit GpuBuf(std::vector<T> const &host_data) {
        CUDA_CHECK(cudaMalloc(&data, host_data.size() * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(
            data,
            host_data.data(),
            host_data.size() * sizeof(T),
            cudaMemcpyHostToDevice));
    }

    ~GpuBuf() { CUDA_CHECK(cudaFree(data)); }
};

Results run_config(Mode mode, Scene const &scene) {
    auto img_expected = Image{
        scene.width,
        scene.height,
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f)};

    render_cpu(
        scene.width,
        scene.height,
        scene.n_circle(),
        scene.circle_x.data(),
        scene.circle_y.data(),
        scene.circle_radius.data(),
        scene.circle_red.data(),
        scene.circle_green.data(),
        scene.circle_blue.data(),
        scene.circle_alpha.data(),
        img_expected.red.data(),
        img_expected.green.data(),
        img_expected.blue.data());

    auto circle_x_gpu = GpuBuf<float>(scene.circle_x);
    auto circle_y_gpu = GpuBuf<float>(scene.circle_y);
    auto circle_radius_gpu = GpuBuf<float>(scene.circle_radius);
    auto circle_red_gpu = GpuBuf<float>(scene.circle_red);
    auto circle_green_gpu = GpuBuf<float>(scene.circle_green);
    auto circle_blue_gpu = GpuBuf<float>(scene.circle_blue);
    auto circle_alpha_gpu = GpuBuf<float>(scene.circle_alpha);
    auto img_red_gpu = GpuBuf<float>(scene.height * scene.width);
    auto img_green_gpu = GpuBuf<float>(scene.height * scene.width);
    auto img_blue_gpu = GpuBuf<float>(scene.height * scene.width);

    auto memory_pool = GpuMemoryPool();

    auto reset = [&]() {
        CUDA_CHECK(
            cudaMemset(img_red_gpu.data, 0, scene.height * scene.width * sizeof(float)));
        CUDA_CHECK(cudaMemset(
            img_green_gpu.data,
            0,
            scene.height * scene.width * sizeof(float)));
        CUDA_CHECK(
            cudaMemset(img_blue_gpu.data, 0, scene.height * scene.width * sizeof(float)));
        memory_pool.reset();
    };

    auto f = [&]() {
        circles_gpu::launch_render(
            scene.width,
            scene.height,
            scene.n_circle(),
            circle_x_gpu.data,
            circle_y_gpu.data,
            circle_radius_gpu.data,
            circle_red_gpu.data,
            circle_green_gpu.data,
            circle_blue_gpu.data,
            circle_alpha_gpu.data,
            img_red_gpu.data,
            img_green_gpu.data,
            img_blue_gpu.data,
            memory_pool);
    };

    reset();
    f();

    auto img_actual = Image{
        scene.width,
        scene.height,
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f)};

    CUDA_CHECK(cudaMemcpy(
        img_actual.red.data(),
        img_red_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        img_actual.green.data(),
        img_green_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        img_actual.blue.data(),
        img_blue_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));

    float max_diff = max_abs_diff(img_expected, img_actual);

    if (max_diff > 5e-2) {
        return Results{
            false,
            max_diff,
            std::move(img_expected),
            std::move(img_actual),
            0.0,
        };
    }

    if (mode == Mode::TEST) {
        return Results{
            true,
            max_diff,
            std::move(img_expected),
            std::move(img_actual),
            0.0,
        };
    }

    double time_ms = benchmark_ms(1000.0, reset, f);

    return Results{
        true,
        max_diff,
        std::move(img_expected),
        std::move(img_actual),
        time_ms,
    };
}

template <typename Rng>
Scene gen_random(Rng &rng, int32_t width, int32_t height, int32_t n_circle) {
    auto unif_0_1 = std::uniform_real_distribution<float>(0.0f, 1.0f);
    auto z_values = std::vector<float>();
    for (int32_t i = 0; i < n_circle; i++) {
        float z;
        for (;;) {
            z = unif_0_1(rng);
            z = std::max(z, unif_0_1(rng));
            if (z > 0.01) {
                break;
            }
        }
        // float z = std::max(unif_0_1(rng), unif_0_1(rng));
        z_values.push_back(z);
    }
    std::sort(z_values.begin(), z_values.end(), std::greater<float>());

    auto colors = std::vector<uint32_t>{
        0xd32360,
        0xcc9f26,
        0x208020,
        0x2874aa,
    };
    auto color_idx_dist = std::uniform_int_distribution<int>(0, colors.size() - 1);
    auto alpha_dist = std::uniform_real_distribution<float>(0.0f, 0.3f);

    int32_t fog_interval = n_circle / 10;
    float fog_alpha = 0.2;

    auto scene = Scene{width, height};
    float base_radius_scale = 1.0f;
    int32_t i = 0;
    for (float z : z_values) {
        float max_radius = base_radius_scale / z;
        float radius = std::max(1.0f, unif_0_1(rng) * max_radius);
        float x = unif_0_1(rng) * (width + 2 * max_radius) - max_radius;
        float y = unif_0_1(rng) * (height + 2 * max_radius) - max_radius;
        int color_idx = color_idx_dist(rng);
        uint32_t color = colors[color_idx];
        scene.circle_x.push_back(x);
        scene.circle_y.push_back(y);
        scene.circle_radius.push_back(radius);
        scene.circle_red.push_back(float((color >> 16) & 0xff) / 255.0f);
        scene.circle_green.push_back(float((color >> 8) & 0xff) / 255.0f);
        scene.circle_blue.push_back(float(color & 0xff) / 255.0f);
        scene.circle_alpha.push_back(alpha_dist(rng));
        i++;
        if (i % fog_interval == 0 && i + 1 < n_circle) {
            scene.circle_x.push_back(float(width - 1) / 2.0f);
            scene.circle_y.push_back(float(height - 1) / 2.0f);
            scene.circle_radius.push_back(float(std::max(width, height)));
            scene.circle_red.push_back(1.0f);
            scene.circle_green.push_back(1.0f);
            scene.circle_blue.push_back(1.0f);
            scene.circle_alpha.push_back(fog_alpha);
        }
    }

    return scene;
}

constexpr float PI = 3.14159265359f;

Scene gen_overlapping_opaque() {
    int32_t width = 256;
    int32_t height = 256;

    auto scene = Scene{width, height};

    auto colors = std::vector<uint32_t>{
        0xd32360,
        0xcc9f26,
        0x208020,
        0x2874aa,
    };

    int32_t n_circle = 20;
    int32_t n_ring = 4;
    float angle_range = PI;
    for (int32_t ring = 0; ring < n_ring; ring++) {
        float dist = 20.0f * (ring + 1);
        float saturation = float(ring + 1) / n_ring;
        float hue_shift = float(ring) / (n_ring - 1);
        for (int32_t i = 0; i < n_circle; i++) {
            float theta = angle_range * i / (n_circle - 1);
            float x = width / 2.0f - dist * std::cos(theta);
            float y = height / 2.0f - dist * std::sin(theta);
            scene.circle_x.push_back(x);
            scene.circle_y.push_back(y);
            scene.circle_radius.push_back(16.0f);
            auto color = colors[(i + ring * 2) % colors.size()];
            scene.circle_red.push_back(float((color >> 16) & 0xff) / 255.0f);
            scene.circle_green.push_back(float((color >> 8) & 0xff) / 255.0f);
            scene.circle_blue.push_back(float(color & 0xff) / 255.0f);
            scene.circle_alpha.push_back(1.0f);
        }
    }

    return scene;
}

Scene gen_overlapping_transparent() {
    int32_t width = 256;
    int32_t height = 256;

    auto scene = Scene{width, height};

    float offset = 20.0f;
    float radius = 40.0f;
    scene.circle_x = std::vector<float>{
        (width - 1) / 2.0f - offset,
        (width - 1) / 2.0f + offset,
        (width - 1) / 2.0f + offset,
        (width - 1) / 2.0f - offset,
    };
    scene.circle_y = std::vector<float>{
        (height - 1) * 0.75f,
        (height - 1) * 0.75f,
        (height - 1) * 0.25f,
        (height - 1) * 0.25f,
    };
    scene.circle_radius = std::vector<float>{
        radius,
        radius,
        radius,
        radius,
    };
    // 0xd32360
    // 0x2874aa
    scene.circle_red = std::vector<float>{
        float(0xd3) / 255.0f,
        float(0x28) / 255.0f,
        float(0x28) / 255.0f,
        float(0xd3) / 255.0f,
    };
    scene.circle_green = std::vector<float>{
        float(0x23) / 255.0f,
        float(0x74) / 255.0f,
        float(0x74) / 255.0f,
        float(0x23) / 255.0f,
    };
    scene.circle_blue = std::vector<float>{
        float(0x60) / 255.0f,
        float(0xaa) / 255.0f,
        float(0xaa) / 255.0f,
        float(0x60) / 255.0f,
    };
    scene.circle_alpha = std::vector<float>{
        0.75f,
        0.75f,
        0.75f,
        0.75f,
    };
    return scene;
}

Scene gen_simple() {
    /*
        0xd32360,
        0xcc9f26,
        0x208020,
        0x2874aa,
    */
    int32_t width = 256;
    int32_t height = 256;
    auto scene = Scene{width, height};
    scene.circle_x = std::vector<float>{
        (width - 1) * 0.25f,
        (width - 1) * 0.75f,
        (width - 1) * 0.25f,
        (width - 1) * 0.75f,
    };
    scene.circle_y = std::vector<float>{
        (height - 1) * 0.25f,
        (height - 1) * 0.25f,
        (height - 1) * 0.75f,
        (height - 1) * 0.75f,
    };
    scene.circle_radius = std::vector<float>{
        40.0f,
        40.0f,
        40.0f,
        40.0f,
    };
    scene.circle_red = std::vector<float>{
        float(0xd3) / 255.0f,
        float(0xcc) / 255.0f,
        float(0x20) / 255.0f,
        float(0x28) / 255.0f,
    };
    scene.circle_green = std::vector<float>{
        float(0x23) / 255.0f,
        float(0x9f) / 255.0f,
        float(0x80) / 255.0f,
        float(0x74) / 255.0f,
    };
    scene.circle_blue = std::vector<float>{
        float(0x60) / 255.0f,
        float(0x26) / 255.0f,
        float(0x20) / 255.0f,
        float(0xaa) / 255.0f,
    };
    scene.circle_alpha = std::vector<float>{
        1.0f,
        1.0f,
        1.0f,
        1.0f,
    };
    return scene;
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

void write_bmp(
    std::string const &fname,
    uint32_t width,
    uint32_t height,
    const std::vector<uint8_t> &pixels) {
    BMPHeader header;
    header.width = width;
    header.height = height;

    uint32_t rowSize = (width * 3 + 3) & (~3); // Align to 4 bytes
    header.imageSize = rowSize * height;
    header.fileSize = header.dataOffset + header.imageSize;

    std::ofstream file(fname, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open file for writing: " << fname << std::endl;
        return;
    }

    file.write(reinterpret_cast<const char *>(&header), sizeof(header));

    // Write pixel data with padding
    std::vector<uint8_t> padding(rowSize - width * 3, 0);
    for (int32_t idx_y = height - 1; idx_y >= 0;
         --idx_y) { // BMP stores pixels from bottom to top
        const uint8_t *row = &pixels[idx_y * width * 3];
        file.write(reinterpret_cast<const char *>(row), width * 3);
        if (!padding.empty()) {
            file.write(reinterpret_cast<const char *>(padding.data()), padding.size());
        }
    }
    file.close();
}

uint8_t float_to_byte(float x) {
    if (x < 0) {
        return 0;
    } else if (x >= 1) {
        return 255;
    } else {
        return x * 255.0f;
    }
}

void write_image(std::string const &fname, Image const &img) {
    auto pixels = std::vector<uint8_t>(img.width * img.height * 3);
    for (int32_t idx = 0; idx < img.width * img.height; idx++) {
        float red = img.red.at(idx);
        float green = img.green.at(idx);
        float blue = img.blue.at(idx);
        // BMP stores pixels in BGR order
        pixels.at(idx * 3) = float_to_byte(blue);
        pixels.at(idx * 3 + 1) = float_to_byte(green);
        pixels.at(idx * 3 + 2) = float_to_byte(red);
    }
    write_bmp(fname, img.width, img.height, pixels);
}

Image compute_img_diff(Image const &a, Image const &b) {
    auto img_diff = Image{
        a.width,
        a.height,
        std::vector<float>(a.height * a.width, 0.0f),
        std::vector<float>(a.height * a.width, 0.0f),
        std::vector<float>(a.height * a.width, 0.0f),
    };
    for (int32_t idx = 0; idx < a.width * a.height; idx++) {
        img_diff.red.at(idx) = std::abs(a.red.at(idx) - b.red.at(idx));
        img_diff.green.at(idx) = std::abs(a.green.at(idx) - b.green.at(idx));
        img_diff.blue.at(idx) = std::abs(a.blue.at(idx) - b.blue.at(idx));
    }
    return img_diff;
}

struct SceneTest {
    std::string name;
    Mode mode;
    Scene scene;
};

int main(int argc, char const *const *argv) {
    // Create output directory if it doesn't exist
    mkdir("out", 0755);

    auto rng = std::mt19937(0xCA7CAFE);

    auto scenes = std::vector<SceneTest>();
    scenes.push_back({"simple", Mode::TEST, gen_simple()});
    scenes.push_back({"overlapping_opaque", Mode::TEST, gen_overlapping_opaque()});
    scenes.push_back(
        {"overlapping_transparent", Mode::TEST, gen_overlapping_transparent()});
    scenes.push_back(
        {"ten_million_circles", Mode::BENCHMARK, gen_random(rng, 1024, 1024, 10'000'000)});

    int32_t fail_count = 0;

    int32_t count = 0;
    for (auto const &scene_test : scenes) {
        auto i = count++;
        printf("\nTesting scene '%s'\n", scene_test.name.c_str());
        auto results = run_config(scene_test.mode, scene_test.scene);
        write_image(
            std::string("out/img") + std::to_string(i) + "_" + scene_test.name +
                "_cpu.bmp",
            results.image_expected);
        write_image(
            std::string("out/img") + std::to_string(i) + "_" + scene_test.name +
                "_gpu.bmp",
            results.image_actual);
        if (!results.correct) {
            printf("  Result did not match expected image\n");
            printf("  Max absolute difference: %.2e\n", results.max_abs_diff);
            auto diff = compute_img_diff(results.image_expected, results.image_actual);
            write_image(
                std::string("out/img") + std::to_string(i) + "_" + scene_test.name +
                    "_diff.bmp",
                diff);
            printf(
                "  (Wrote image diff to 'out/img%d_%s_diff.bmp')\n",
                i,
                scene_test.name.c_str());
            fail_count++;
            continue;
        } else {
            printf("  OK\n");
        }
        if (scene_test.mode == Mode::BENCHMARK) {
            printf("  Time: %f ms\n", results.time_ms);
        }
    }

    if (fail_count) {
        printf("\nCorrectness: %d tests failed\n", fail_count);
    } else {
        printf("\nCorrectness: All tests passed\n");
    }

    return 0;
}
