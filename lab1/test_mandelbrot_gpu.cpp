#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for GPU functions (from mandelbrot_gpu.cu)
void launch_mandelbrot_gpu_scalar(uint32_t img_size, uint32_t max_iters, uint32_t *out);
void launch_mandelbrot_gpu_vector(uint32_t img_size, uint32_t max_iters, uint32_t *out);

// Forward declarations for parallel GPU functions (from mandelbrot_gpu2.cu)
void launch_mandelbrot_gpu_parallel_scalar(uint32_t img_size, uint32_t max_iters, uint32_t *out);
void launch_mandelbrot_gpu_parallel_vector(uint32_t img_size, uint32_t max_iters, uint32_t *out);

// Helper function to save image as PPM
void save_ppm(const char* filename, uint32_t *data, uint32_t width, uint32_t height, uint32_t max_iters) {
    std::ofstream file(filename);
    file << "P3\n" << width << " " << height << "\n255\n";

    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            uint32_t val = data[i * width + j];
            // Simple color mapping
            if (val == max_iters) {
                file << "0 0 0 ";  // Black for points in the set
            } else {
                uint8_t r = (val * 9) % 256;
                uint8_t g = (val * 7) % 256;
                uint8_t b = (val * 5) % 256;
                file << (int)r << " " << (int)g << " " << (int)b << " ";
            }
        }
        file << "\n";
    }
    file.close();
}

// Verify two results match
bool verify_results(uint32_t *a, uint32_t *b, uint32_t size) {
    for (uint32_t i = 0; i < size * size; i++) {
        if (a[i] != b[i]) {
            std::cerr << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

int main(int argc, char *argv[]) {
    uint32_t max_iters = 256;

    // Test sizes
    std::vector<uint32_t> test_sizes = {256, 512, 1024};

    std::cout << "Mandelbrot GPU Benchmark\n";
    std::cout << "Testing sizes: 256x256, 512x512, 1024x1024\n";
    std::cout << "Max iterations: " << max_iters << "\n";
    std::cout << "==========================================\n\n";

    for (uint32_t img_size : test_sizes) {
        std::cout << "\n========================================\n";
        std::cout << "Image size: " << img_size << "x" << img_size << "\n";
        std::cout << "========================================\n";

        // Allocate output buffers
        std::vector<uint32_t> out_gpu_scalar(img_size * img_size);
        std::vector<uint32_t> out_gpu_vector(img_size * img_size);
        std::vector<uint32_t> out_gpu_parallel_scalar(img_size * img_size);
        std::vector<uint32_t> out_gpu_parallel_vector(img_size * img_size);

        // ========== GPU SCALAR ==========
        std::cout << "Running GPU scalar version...\n";

        // Allocate GPU memory
        uint32_t *d_out;
        size_t buffer_size = img_size * img_size * sizeof(uint32_t);
        CUDA_CHECK(cudaMalloc(&d_out, buffer_size));

        auto start = std::chrono::high_resolution_clock::now();
        launch_mandelbrot_gpu_scalar(img_size, max_iters, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        auto gpu_scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

        // Copy results back to CPU
        CUDA_CHECK(cudaMemcpy(out_gpu_scalar.data(), d_out, buffer_size, cudaMemcpyDeviceToHost));

        std::cout << "GPU Scalar time: " << gpu_scalar_time << " ms\n";

        // Save output (only for 512x512 to avoid too many files)
        if (img_size == 512) {
            save_ppm("mandelbrot_gpu_scalar.ppm", out_gpu_scalar.data(), img_size, img_size, max_iters);
            std::cout << "Saved output to mandelbrot_gpu_scalar.ppm\n";
        }

        // ========== GPU VECTOR ==========
        std::cout << "Running GPU vector version...\n";

        // Clear GPU memory
        CUDA_CHECK(cudaMemset(d_out, 0, buffer_size));

        start = std::chrono::high_resolution_clock::now();
        launch_mandelbrot_gpu_vector(img_size, max_iters, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        auto gpu_vector_time = std::chrono::duration<double, std::milli>(end - start).count();

        // Copy results back to CPU
        CUDA_CHECK(cudaMemcpy(out_gpu_vector.data(), d_out, buffer_size, cudaMemcpyDeviceToHost));

        std::cout << "GPU Vector time: " << gpu_vector_time << " ms\n";
        if (verify_results(out_gpu_scalar.data(), out_gpu_vector.data(), img_size)) {
            std::cout << "✓ GPU Vector results match GPU Scalar\n";
        } else {
            std::cout << "✗ GPU Vector results do NOT match GPU Scalar\n";
        }
        std::cout << "Speedup: " << (gpu_scalar_time / gpu_vector_time) << "x\n";

        // Save vector output (only for 512x512)
        if (img_size == 512) {
            save_ppm("mandelbrot_gpu_vector.ppm", out_gpu_vector.data(), img_size, img_size, max_iters);
            std::cout << "Saved output to mandelbrot_gpu_vector.ppm\n";
        }

        // ========== GPU PARALLEL SCALAR ==========
        std::cout << "Running GPU parallel scalar version (16x16 blocks)...\n";

        // Clear GPU memory
        CUDA_CHECK(cudaMemset(d_out, 0, buffer_size));

        start = std::chrono::high_resolution_clock::now();
        launch_mandelbrot_gpu_parallel_scalar(img_size, max_iters, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        auto gpu_parallel_scalar_time = std::chrono::duration<double, std::milli>(end - start).count();

        // Copy results back to CPU
        CUDA_CHECK(cudaMemcpy(out_gpu_parallel_scalar.data(), d_out, buffer_size, cudaMemcpyDeviceToHost));

        std::cout << "GPU Parallel Scalar time: " << gpu_parallel_scalar_time << " ms\n";
        if (verify_results(out_gpu_scalar.data(), out_gpu_parallel_scalar.data(), img_size)) {
            std::cout << "✓ GPU Parallel Scalar results match GPU Scalar\n";
        } else {
            std::cout << "✗ GPU Parallel Scalar results do NOT match GPU Scalar\n";
        }
        std::cout << "Speedup vs GPU Scalar: " << (gpu_scalar_time / gpu_parallel_scalar_time) << "x\n";

        // ========== GPU PARALLEL VECTOR ==========
        std::cout << "Running GPU parallel vector version (32x32 blocks)...\n";

        // Clear GPU memory
        CUDA_CHECK(cudaMemset(d_out, 0, buffer_size));

        start = std::chrono::high_resolution_clock::now();
        launch_mandelbrot_gpu_parallel_vector(img_size, max_iters, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        auto gpu_parallel_vector_time = std::chrono::duration<double, std::milli>(end - start).count();

        // Copy results back to CPU
        CUDA_CHECK(cudaMemcpy(out_gpu_parallel_vector.data(), d_out, buffer_size, cudaMemcpyDeviceToHost));

        std::cout << "GPU Parallel Vector time: " << gpu_parallel_vector_time << " ms\n";
        if (verify_results(out_gpu_scalar.data(), out_gpu_parallel_vector.data(), img_size)) {
            std::cout << "✓ GPU Parallel Vector results match GPU Scalar\n";
        } else {
            std::cout << "✗ GPU Parallel Vector results do NOT match GPU Scalar\n";
        }
        std::cout << "Speedup vs GPU Scalar: " << (gpu_scalar_time / gpu_parallel_vector_time) << "x\n";
        std::cout << "Speedup vs GPU Parallel Scalar: " << (gpu_parallel_scalar_time / gpu_parallel_vector_time) << "x\n";

        // Cleanup
        CUDA_CHECK(cudaFree(d_out));
    }

    std::cout << "\n==========================================\n";
    std::cout << "All tests completed!\n";
    std::cout << "==========================================\n";

    return 0;
}
