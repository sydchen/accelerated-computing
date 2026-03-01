#include <cstdint>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <immintrin.h>

// Aligned allocator for AVX-512 (64-byte alignment)
template<typename T>
class AlignedAllocator {
public:
    using value_type = T;

    AlignedAllocator() = default;
    template<typename U> AlignedAllocator(const AlignedAllocator<U>&) {}

    T* allocate(size_t n) {
        size_t size = n * sizeof(T);
        size = (size + 63) & ~63;  // Round up to 64-byte boundary
        void* ptr = std::aligned_alloc(64, size);
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, size_t) { std::free(p); }
};

template<typename T>
using aligned_vector = std::vector<T, AlignedAllocator<T>>;

#ifndef __AVX512F__
#error "This source file is AVX-512-only. Compile with AVX-512 enabled (e.g. -mavx512f)."
#endif

void mandelbrot_cpu_scalar(
    uint32_t img_size, /* usually greater than 100 */
    uint32_t max_iters, /* usually greater than 100 */
    uint32_t *out /* buffer of 'img_size * img_size' integers */
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

// x86 AVX-512 vector implementation (512-bit vectors = 16 floats)
void mandelbrot_cpu_vector(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out
) {
    const float inv_img = 1.0f / float(img_size);
    const __m512 v_step = _mm512_set1_ps(2.5f * inv_img);
    const __m512 v_offx = _mm512_set1_ps(-2.0f);
    const __m512 v_four = _mm512_set1_ps(4.0f);
    const __m512i v_one = _mm512_set1_epi32(1);

    // [0..15]
    const __m512 v_idx = _mm512_setr_ps(
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
        8.f, 9.f,10.f,11.f,12.f,13.f,14.f,15.f
    );

    for (uint32_t i = 0; i < img_size; ++i) {
        const float cy_s = (float(i) * inv_img) * 2.5f - 1.25f;
        const __m512 v_cy = _mm512_set1_ps(cy_s);

        for (uint32_t j = 0; j < img_size; j += 16) {
            const uint32_t remain = img_size - j;
            const __mmask16 lane_mask = (remain >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << remain) - 1u);

            // cx = ((j + [0..15]) * step) + offx
            const __m512 v_jbase = _mm512_set1_ps((float)j);
            const __m512 v_j = _mm512_add_ps(v_jbase, v_idx);
            const __m512 v_cx = _mm512_add_ps(_mm512_mul_ps(v_j, v_step), v_offx);

            __m512  v_x2 = _mm512_setzero_ps();
            __m512  v_y2 = _mm512_setzero_ps();
            __m512  v_w  = _mm512_setzero_ps();
            __m512i v_it = _mm512_setzero_epi32();

            for (uint32_t iter = 0; iter < max_iters; ++iter) {
                // Check condition: x2 + y2 <= 4.0
                const __m512 v_sum = _mm512_add_ps(v_x2, v_y2);
                const __mmask16 m_in = _mm512_cmp_ps_mask(v_sum, v_four, _CMP_LE_OQ);

                // Early exit if all lanes escaped
                if (!m_in) break;

                // Calculate x and y with EXACT same order as scalar
                // x = x2 - y2 + cx
                const __m512 v_x = _mm512_add_ps(_mm512_sub_ps(v_x2, v_y2), v_cx);
                // y = w - x2 - y2 + cy (NOT w - (x2+y2) + cy due to float precision!)
                const __m512 v_y = _mm512_add_ps(_mm512_sub_ps(_mm512_sub_ps(v_w, v_x2), v_y2), v_cy);

                // Calculate new values
                const __m512 v_x2n = _mm512_mul_ps(v_x, v_x);
                const __m512 v_y2n = _mm512_mul_ps(v_y, v_y);
                const __m512 v_xy = _mm512_add_ps(v_x, v_y);
                const __m512 v_wn = _mm512_mul_ps(v_xy, v_xy);

                // Update state with mask to prevent NaN
                v_x2 = _mm512_mask_mov_ps(v_x2, m_in, v_x2n);
                v_y2 = _mm512_mask_mov_ps(v_y2, m_in, v_y2n);
                v_w = _mm512_mask_mov_ps(v_w, m_in, v_wn);

                // Increment iteration count LAST (like scalar)
                v_it = _mm512_mask_add_epi32(v_it, m_in, v_it, v_one);
            }

            uint32_t* dst = out + i * img_size + j;
            // Use aligned store if possible (buffer is 64-byte aligned and j is multiple of 16)
            if ((img_size % 16 == 0) && (reinterpret_cast<uintptr_t>(out) % 64 == 0)) {
                _mm512_mask_store_epi32((void*)dst, lane_mask, v_it);
            } else {
                _mm512_mask_storeu_epi32((void*)dst, lane_mask, v_it);
            }
        }
    }
}

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

int main(int argc, char *argv[]) {
    uint32_t max_iters = 256;

    // Test sizes
    std::vector<uint32_t> test_sizes = {256, 512, 1024};

    std::cout << "Mandelbrot Benchmark\n";
    std::cout << "Testing sizes: 256x256, 512x512, 1024x1024\n";
    std::cout << "Max iterations: " << max_iters << "\n";
    std::cout << "==========================================\n\n";

    for (uint32_t img_size : test_sizes) {
        std::cout << "\n========================================\n";
        std::cout << "Image size: " << img_size << "x" << img_size << "\n";
        std::cout << "========================================\n";

        // Allocate output buffers (64-byte aligned for vector version)
        std::vector<uint32_t> out_scalar(img_size * img_size);
        aligned_vector<uint32_t> out_vector(img_size * img_size);

        // Test scalar version
        std::cout << "Running CPU scalar version...\n";
        auto start = std::chrono::high_resolution_clock::now();
        mandelbrot_cpu_scalar(img_size, max_iters, out_scalar.data());
        auto end = std::chrono::high_resolution_clock::now();
        auto scalar_time = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "CPU Scalar time: " << scalar_time << " ms\n";

        // Save scalar output (only for 512x512 to avoid too many files)
        if (img_size == 512) {
            save_ppm("mandelbrot_scalar.ppm", out_scalar.data(), img_size, img_size, max_iters);
            std::cout << "Saved output to mandelbrot_scalar.ppm\n";
        }

        // Test vector version
        std::cout << "Running CPU vector version...\n";
        start = std::chrono::high_resolution_clock::now();
        mandelbrot_cpu_vector(img_size, max_iters, out_vector.data());
        end = std::chrono::high_resolution_clock::now();
        auto vector_time = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "CPU Vector time: " << vector_time << " ms\n";

        // Verify results match
        if (verify_results(out_scalar.data(), out_vector.data(), img_size)) {
            std::cout << "✓ Vector results match scalar results\n";
        } else {
            std::cout << "✗ Vector results do NOT match scalar results\n";
        }

        std::cout << "Speedup: " << (scalar_time / vector_time) << "x\n";

        // Save vector output (only for 512x512)
        if (img_size == 512) {
            save_ppm("mandelbrot_vector.ppm", out_vector.data(), img_size, img_size, max_iters);
            std::cout << "Saved output to mandelbrot_vector.ppm\n";
        }
    }

    std::cout << "\n==========================================\n";
    std::cout << "All tests completed!\n";
    std::cout << "==========================================\n";

    return 0;
}
