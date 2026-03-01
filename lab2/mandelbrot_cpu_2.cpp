%%writefile mandelbrot_cpu_2.cpp
// Optional arguments:
//  -r <img_size>
//  -b <max iterations>
//  -i <implementation: {"scalar", "vector", "vector_ilp", "vector_multicore",
//  "vector_multicore_multithread", "vector_multicore_multithread_ilp", "all"}>

#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <pthread.h>

constexpr float window_zoom = 1.0 / 10000.0f;
constexpr float window_x = -0.743643887 - 0.5 * window_zoom;
constexpr float window_y = 0.131825904 - 0.5 * window_zoom;
constexpr uint32_t default_max_iters = 2000;

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

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// ILP (Instruction-Level Parallelism) configuration
// Number of independent computation chains to process simultaneously
// Can be overridden at compile-time with: -DNUM_ILP_CHAINS=N
// Default: 4 independent chains
#ifndef NUM_ILP_CHAINS
#define NUM_ILP_CHAINS 4
#endif

/// <--- your code here --->

/*
    // OPTIONAL: Uncomment this block to include your CPU vector implementation
    // from Lab 1 for easy comparison.
    //
    // (If you do this, you'll need to update your code to use the new constants
    // 'window_zoom', 'window_x', and 'window_y'.)

    #define HAS_VECTOR_IMPL // <~~ keep this line if you want to benchmark the vector kernel!

    ////////////////////////////////////////////////////////////////////////////////
    // Vector

    void mandelbrot_cpu_vector(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
        // your code here...
    }
*/

////////////////////////////////////////////////////////////////////////////////
// Vector + ILP

void mandelbrot_cpu_vector_ilp(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    // Strategy: Process NUM_ILP_CHAINS independent vectors (16 pixels each) simultaneously
    // to exploit Instruction-Level Parallelism (ILP)
    constexpr int VECTOR_WIDTH = 16;    // AVX-512 width
    constexpr int PIXELS_PER_ITER = NUM_ILP_CHAINS * VECTOR_WIDTH;  // NUM_ILP_CHAINS * 16 pixels

    const float inv_img = 1.0f / float(img_size);
    const __m512 v_step = _mm512_set1_ps(window_zoom * inv_img);
    const __m512 v_offx = _mm512_set1_ps(window_x);
    const __m512 v_four = _mm512_set1_ps(4.0f);
    const __m512i v_one = _mm512_set1_epi32(1);

    // [0..15]
    const __m512 v_idx = _mm512_setr_ps(
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
        8.f, 9.f,10.f,11.f,12.f,13.f,14.f,15.f
    );

    for (uint32_t i = 0; i < img_size; ++i) {
        const float cy_s = (float(i) * inv_img) * window_zoom + window_y;
        const __m512 v_cy = _mm512_set1_ps(cy_s);

        for (uint32_t j = 0; j < img_size; j += PIXELS_PER_ITER) {
            // Initialize 4 independent chains (explicit ILP)
            __m512  v_cx[NUM_ILP_CHAINS];
            __m512  v_x2[NUM_ILP_CHAINS];
            __m512  v_y2[NUM_ILP_CHAINS];
            __m512  v_w[NUM_ILP_CHAINS];
            __m512i v_it[NUM_ILP_CHAINS];
            __mmask16 lane_mask[NUM_ILP_CHAINS];

            #pragma unroll
            for (int chain = 0; chain < NUM_ILP_CHAINS; chain++) {
                uint32_t j_offset = j + chain * VECTOR_WIDTH;

                // Boundary mask for this chain
                const uint32_t remain = (j_offset < img_size) ? (img_size - j_offset) : 0;
                lane_mask[chain] = (remain >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << remain) - 1u);

                // cx = ((j_offset + [0..15]) * step) + offx
                const __m512 v_jbase = _mm512_set1_ps((float)j_offset);
                const __m512 v_j = _mm512_add_ps(v_jbase, v_idx);
                v_cx[chain] = _mm512_add_ps(_mm512_mul_ps(v_j, v_step), v_offx);

                // Initialize state
                v_x2[chain] = _mm512_setzero_ps();
                v_y2[chain] = _mm512_setzero_ps();
                v_w[chain]  = _mm512_setzero_ps();
                v_it[chain] = _mm512_setzero_epi32();
            }

            // Main Mandelbrot loop with explicit interleaving for ILP
            while (true) {
                // Check if any chain is still active
                __mmask16 any_active = 0;
                #pragma unroll
                for (int chain = 0; chain < NUM_ILP_CHAINS; chain++) {
                    const __m512 v_sum = _mm512_add_ps(v_x2[chain], v_y2[chain]);
                    const __mmask16 m_in = _mm512_cmp_ps_mask(v_sum, v_four, _CMP_LE_OQ);
                    const __mmask16 m_iter = _mm512_cmp_epi32_mask(v_it[chain], _mm512_set1_epi32((int)max_iters), _MM_CMPINT_LT);
                    const __mmask16 m_active = m_in & m_iter & lane_mask[chain];
                    any_active |= m_active;
                }
                if (!any_active) break;

                // Process all 4 chains (explicit interleaving for ILP)
                #pragma unroll
                for (int chain = 0; chain < NUM_ILP_CHAINS; chain++) {
                    const __m512 v_sum = _mm512_add_ps(v_x2[chain], v_y2[chain]);
                    const __mmask16 m_in = _mm512_cmp_ps_mask(v_sum, v_four, _CMP_LE_OQ);
                    const __mmask16 m_iter = _mm512_cmp_epi32_mask(v_it[chain], _mm512_set1_epi32((int)max_iters), _MM_CMPINT_LT);
                    const __mmask16 m_active = m_in & m_iter & lane_mask[chain];

                    // x = x2 - y2 + cx
                    // y = w - (x2 + y2) + cy
                    const __m512 v_x = _mm512_add_ps(_mm512_sub_ps(v_x2[chain], v_y2[chain]), v_cx[chain]);
                    const __m512 v_y = _mm512_add_ps(_mm512_sub_ps(v_w[chain], v_sum), v_cy);

                    const __m512 v_x2n = _mm512_mul_ps(v_x, v_x);
                    const __m512 v_y2n = _mm512_mul_ps(v_y, v_y);
                    const __m512 v_xy  = _mm512_add_ps(v_x, v_y);
                    const __m512 v_wn  = _mm512_mul_ps(v_xy, v_xy);

                    // Only update active lanes (hardware predication)
                    v_x2[chain] = _mm512_mask_mov_ps(v_x2[chain], m_active, v_x2n);
                    v_y2[chain] = _mm512_mask_mov_ps(v_y2[chain], m_active, v_y2n);
                    v_w[chain]  = _mm512_mask_mov_ps(v_w[chain],  m_active, v_wn);
                    v_it[chain] = _mm512_mask_add_epi32(v_it[chain], m_active, v_it[chain], v_one);
                }
            }

            // Store results for all 4 chains
            #pragma unroll
            for (int chain = 0; chain < NUM_ILP_CHAINS; chain++) {
                uint32_t j_offset = j + chain * VECTOR_WIDTH;
                if (j_offset < img_size) {
                    uint32_t* dst = out + i * img_size + j_offset;
                    _mm512_mask_storeu_epi32((void*)dst, lane_mask[chain], v_it[chain]);
                }
            }
        }
    }

}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core

// Thread data structure for passing parameters to worker threads
struct ThreadData {
    uint32_t img_size;       // Image size
    uint32_t max_iters;      // Maximum iterations
    uint32_t *out;           // Output buffer
    uint32_t row_start;      // Starting row for this thread
    uint32_t row_end;        // Ending row (exclusive) for this thread
};

// Worker function that each thread executes
void* mandelbrot_worker_vector(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    constexpr int VECTOR_WIDTH = 16;  // AVX-512 width
    const float inv_img = 1.0f / float(data->img_size);
    const __m512 v_step = _mm512_set1_ps(window_zoom * inv_img);
    const __m512 v_offx = _mm512_set1_ps(window_x);
    const __m512 v_four = _mm512_set1_ps(4.0f);
    const __m512i v_one = _mm512_set1_epi32(1);

    // [0..15] for j index calculation
    const __m512 v_idx = _mm512_setr_ps(
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
        8.f, 9.f,10.f,11.f,12.f,13.f,14.f,15.f
    );

    // Process rows assigned to this thread
    for (uint32_t i = data->row_start; i < data->row_end; ++i) {
        const float cy_s = (float(i) * inv_img) * window_zoom + window_y;
        const __m512 v_cy = _mm512_set1_ps(cy_s);

        // Process each row with vector parallelism
        for (uint32_t j = 0; j < data->img_size; j += VECTOR_WIDTH) {
            // Boundary mask for this vector
            const uint32_t remain = (j < data->img_size) ? (data->img_size - j) : 0;
            const __mmask16 lane_mask = (remain >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << remain) - 1u);

            // cx = ((j + [0..15]) * step) + offx
            const __m512 v_jbase = _mm512_set1_ps((float)j);
            const __m512 v_j = _mm512_add_ps(v_jbase, v_idx);
            const __m512 v_cx = _mm512_add_ps(_mm512_mul_ps(v_j, v_step), v_offx);

            // Initialize state vectors
            __m512 v_x2 = _mm512_setzero_ps();
            __m512 v_y2 = _mm512_setzero_ps();
            __m512 v_w = _mm512_setzero_ps();
            __m512i v_it = _mm512_setzero_epi32();

            // Mandelbrot iteration loop
            for (uint32_t iter = 0; iter < data->max_iters; ++iter) {
                const __m512 v_sum = _mm512_add_ps(v_x2, v_y2);
                const __mmask16 m_in = _mm512_cmp_ps_mask(v_sum, v_four, _CMP_LE_OQ);
                const __mmask16 m_iter = _mm512_cmp_epi32_mask(v_it, _mm512_set1_epi32((int)data->max_iters), _MM_CMPINT_LT);
                const __mmask16 m_active = m_in & m_iter & lane_mask;

                if (!m_active) break;  // Early exit if all lanes done

                // x = x2 - y2 + cx
                // y = w - (x2 + y2) + cy
                const __m512 v_x = _mm512_add_ps(_mm512_sub_ps(v_x2, v_y2), v_cx);
                const __m512 v_y = _mm512_add_ps(_mm512_sub_ps(v_w, v_sum), v_cy);

                const __m512 v_x2n = _mm512_mul_ps(v_x, v_x);
                const __m512 v_y2n = _mm512_mul_ps(v_y, v_y);
                const __m512 v_xy = _mm512_add_ps(v_x, v_y);
                const __m512 v_wn = _mm512_mul_ps(v_xy, v_xy);

                // Update active lanes only
                v_x2 = _mm512_mask_mov_ps(v_x2, m_active, v_x2n);
                v_y2 = _mm512_mask_mov_ps(v_y2, m_active, v_y2n);
                v_w = _mm512_mask_mov_ps(v_w, m_active, v_wn);
                v_it = _mm512_mask_add_epi32(v_it, m_active, v_it, v_one);
            }

            // Store results
            uint32_t* dst = data->out + i * data->img_size + j;
            _mm512_mask_storeu_epi32((void*)dst, lane_mask, v_it);
        }
    }

    return nullptr;
}

void mandelbrot_cpu_vector_multicore(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out) {
    constexpr uint32_t NUM_THREADS = 2;  // One thread per CPU core

    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    // Calculate rows per thread
    uint32_t rows_per_thread = img_size / NUM_THREADS;

    // Create worker threads
    for (uint32_t t = 0; t < NUM_THREADS; ++t) {
        thread_data[t].img_size = img_size;
        thread_data[t].max_iters = max_iters;
        thread_data[t].out = out;
        thread_data[t].row_start = t * rows_per_thread;
        thread_data[t].row_end = (t == NUM_THREADS - 1) ? img_size : (t + 1) * rows_per_thread;

        pthread_create(&threads[t], nullptr, mandelbrot_worker_vector, &thread_data[t]);
    }

    // Wait for all threads to complete
    for (uint32_t t = 0; t < NUM_THREADS; ++t) {
        pthread_join(threads[t], nullptr);
    }
}

/////////////////////////
// ChatGPT

typedef struct {
    uint32_t start_row;
    uint32_t end_row;
    uint32_t img_size;
    uint32_t max_iters;
    uint32_t *out;
} ThreadArgs;

// 每個 thread 負責一部分 rows 的計算
void* mandelbrot_thread_func(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    const uint32_t start_row = args->start_row;
    const uint32_t end_row   = args->end_row;
    const uint32_t img_size  = args->img_size;
    const uint32_t max_iters = args->max_iters;
    uint32_t* out            = args->out;

    const float inv_img = 1.0f / float(img_size);
    const __m512 v_step = _mm512_set1_ps(window_zoom * inv_img);
    const __m512 v_offx = _mm512_set1_ps(window_x);
    const __m512 v_four = _mm512_set1_ps(4.0f);
    const __m512i v_one = _mm512_set1_epi32(1);
    const __m512 v_idx = _mm512_setr_ps(
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
        8.f, 9.f,10.f,11.f,12.f,13.f,14.f,15.f
    );

    for (uint32_t i = start_row; i < end_row; ++i) {
        const float cy_s = (float(i) * inv_img) * window_zoom + window_y;
        const __m512 v_cy = _mm512_set1_ps(cy_s);

        for (uint32_t j = 0; j < img_size; j += 16) {
            const uint32_t remain = img_size - j;
            const __mmask16 lane_mask = (remain >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << remain) - 1u);

            const __m512 v_jbase = _mm512_set1_ps((float)j);
            const __m512 v_j = _mm512_add_ps(v_jbase, v_idx);
            const __m512 v_cx = _mm512_add_ps(_mm512_mul_ps(v_j, v_step), v_offx);

            __m512  v_x2 = _mm512_setzero_ps();
            __m512  v_y2 = _mm512_setzero_ps();
            __m512  v_w  = _mm512_setzero_ps();
            __m512i v_it = _mm512_setzero_epi32();

            while (true) {
                const __m512 v_sum = _mm512_add_ps(v_x2, v_y2);
                const __mmask16 m_in = _mm512_cmp_ps_mask(v_sum, v_four, _CMP_LE_OQ);
                const __mmask16 m_iter = _mm512_cmp_epi32_mask(v_it, _mm512_set1_epi32((int)max_iters), _MM_CMPINT_LT);
                const __mmask16 m_active = m_in & m_iter & lane_mask;
                if (!m_active) break;

                const __m512 v_x   = _mm512_add_ps(_mm512_sub_ps(v_x2, v_y2), v_cx);
                const __m512 v_y   = _mm512_add_ps(_mm512_sub_ps(v_w, _mm512_add_ps(v_x2, v_y2)), v_cy);

                const __m512 v_x2n = _mm512_mul_ps(v_x, v_x);
                const __m512 v_y2n = _mm512_mul_ps(v_y, v_y);
                const __m512 v_xy  = _mm512_add_ps(v_x, v_y);
                const __m512 v_wn  = _mm512_mul_ps(v_xy, v_xy);

                v_x2 = _mm512_mask_mov_ps(v_x2, m_active, v_x2n);
                v_y2 = _mm512_mask_mov_ps(v_y2, m_active, v_y2n);
                v_w  = _mm512_mask_mov_ps(v_w,  m_active, v_wn);
                v_it = _mm512_mask_add_epi32(v_it, m_active, v_it, v_one);
            }

            uint32_t* dst = out + i * img_size + j;
            _mm512_mask_storeu_epi32((void*)dst, lane_mask, v_it);
        }
    }

    return NULL;
}

// 主函數
void mandelbrot_cpu_vector_pthread(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out
) {
    const int num_threads = 2; // 可根據 CPU 核心數調整
    pthread_t threads[num_threads];
    ThreadArgs args[num_threads];

    const uint32_t rows_per_thread = img_size / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        args[t].start_row = t * rows_per_thread;
        args[t].end_row   = (t == num_threads - 1) ? img_size : args[t].start_row + rows_per_thread;
        args[t].img_size  = img_size;
        args[t].max_iters = max_iters;
        args[t].out       = out;

        pthread_create(&threads[t], NULL, mandelbrot_thread_func, &args[t]);
    }

    for (int t = 0; t < num_threads; ++t) {
        pthread_join(threads[t], NULL);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core + Multi-thread-per-core

void mandelbrot_cpu_vector_multicore_multithread(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out) {
    // Part 3: Use more threads than physical cores to utilize SMT/hyperthreading
    // Kaggle has 2 physical cores with hyperthreading (4 logical cores)
    // Using 4 threads allows 2 threads per physical core
    constexpr uint32_t NUM_THREADS = 4;

    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    // Calculate rows per thread
    uint32_t rows_per_thread = img_size / NUM_THREADS;

    // Create worker threads
    for (uint32_t t = 0; t < NUM_THREADS; ++t) {
        thread_data[t].img_size = img_size;
        thread_data[t].max_iters = max_iters;
        thread_data[t].out = out;
        thread_data[t].row_start = t * rows_per_thread;
        thread_data[t].row_end = (t == NUM_THREADS - 1) ? img_size : (t + 1) * rows_per_thread;

        pthread_create(&threads[t], nullptr, mandelbrot_worker_vector, &thread_data[t]);
    }

    // Wait for all threads to complete
    for (uint32_t t = 0; t < NUM_THREADS; ++t) {
        pthread_join(threads[t], nullptr);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core + Multi-thread-per-core + ILP

// Worker function with ILP: each thread processes multiple independent vector chains
void* mandelbrot_worker_vector_ilp(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    // ILP configuration: process NUM_ILP_CHAINS independent vectors simultaneously
    constexpr int VECTOR_WIDTH = 16;  // AVX-512 width
    constexpr int PIXELS_PER_ITER = NUM_ILP_CHAINS * VECTOR_WIDTH;  // NUM_ILP_CHAINS * 16 pixels

    const float inv_img = 1.0f / float(data->img_size);
    const __m512 v_step = _mm512_set1_ps(window_zoom * inv_img);
    const __m512 v_offx = _mm512_set1_ps(window_x);
    const __m512 v_four = _mm512_set1_ps(4.0f);
    const __m512i v_one = _mm512_set1_epi32(1);

    // [0..15] for j index calculation
    const __m512 v_idx = _mm512_setr_ps(
        0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
        8.f, 9.f,10.f,11.f,12.f,13.f,14.f,15.f
    );

    // Process rows assigned to this thread
    for (uint32_t i = data->row_start; i < data->row_end; ++i) {
        const float cy_s = (float(i) * inv_img) * window_zoom + window_y;
        const __m512 v_cy = _mm512_set1_ps(cy_s);

        // Process each row with ILP: 4 independent vector chains at a time
        for (uint32_t j = 0; j < data->img_size; j += PIXELS_PER_ITER) {
            // Initialize 4 independent chains (explicit ILP)
            __m512  v_cx[NUM_ILP_CHAINS];
            __m512  v_x2[NUM_ILP_CHAINS];
            __m512  v_y2[NUM_ILP_CHAINS];
            __m512  v_w[NUM_ILP_CHAINS];
            __m512i v_it[NUM_ILP_CHAINS];
            __mmask16 lane_mask[NUM_ILP_CHAINS];

            #pragma unroll
            for (int chain = 0; chain < NUM_ILP_CHAINS; chain++) {
                uint32_t j_offset = j + chain * VECTOR_WIDTH;

                // Boundary mask for this chain
                const uint32_t remain = (j_offset < data->img_size) ? (data->img_size - j_offset) : 0;
                lane_mask[chain] = (remain >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << remain) - 1u);

                // cx = ((j_offset + [0..15]) * step) + offx
                const __m512 v_jbase = _mm512_set1_ps((float)j_offset);
                const __m512 v_j = _mm512_add_ps(v_jbase, v_idx);
                v_cx[chain] = _mm512_add_ps(_mm512_mul_ps(v_j, v_step), v_offx);

                // Initialize state
                v_x2[chain] = _mm512_setzero_ps();
                v_y2[chain] = _mm512_setzero_ps();
                v_w[chain]  = _mm512_setzero_ps();
                v_it[chain] = _mm512_setzero_epi32();
            }

            // Main Mandelbrot loop with explicit interleaving for ILP
            while (true) {
                // Check if any chain is still active
                __mmask16 any_active = 0;
                #pragma unroll
                for (int chain = 0; chain < NUM_ILP_CHAINS; chain++) {
                    const __m512 v_sum = _mm512_add_ps(v_x2[chain], v_y2[chain]);
                    const __mmask16 m_in = _mm512_cmp_ps_mask(v_sum, v_four, _CMP_LE_OQ);
                    const __mmask16 m_iter = _mm512_cmp_epi32_mask(v_it[chain], _mm512_set1_epi32((int)data->max_iters), _MM_CMPINT_LT);
                    const __mmask16 m_active = m_in & m_iter & lane_mask[chain];
                    any_active |= m_active;
                }
                if (!any_active) break;

                // Process all 4 chains (explicit interleaving for ILP)
                #pragma unroll
                for (int chain = 0; chain < NUM_ILP_CHAINS; chain++) {
                    const __m512 v_sum = _mm512_add_ps(v_x2[chain], v_y2[chain]);
                    const __mmask16 m_in = _mm512_cmp_ps_mask(v_sum, v_four, _CMP_LE_OQ);
                    const __mmask16 m_iter = _mm512_cmp_epi32_mask(v_it[chain], _mm512_set1_epi32((int)data->max_iters), _MM_CMPINT_LT);
                    const __mmask16 m_active = m_in & m_iter & lane_mask[chain];

                    // x = x2 - y2 + cx
                    // y = w - (x2 + y2) + cy
                    const __m512 v_x = _mm512_add_ps(_mm512_sub_ps(v_x2[chain], v_y2[chain]), v_cx[chain]);
                    const __m512 v_y = _mm512_add_ps(_mm512_sub_ps(v_w[chain], v_sum), v_cy);

                    const __m512 v_x2n = _mm512_mul_ps(v_x, v_x);
                    const __m512 v_y2n = _mm512_mul_ps(v_y, v_y);
                    const __m512 v_xy  = _mm512_add_ps(v_x, v_y);
                    const __m512 v_wn  = _mm512_mul_ps(v_xy, v_xy);

                    // Only update active lanes (hardware predication)
                    v_x2[chain] = _mm512_mask_mov_ps(v_x2[chain], m_active, v_x2n);
                    v_y2[chain] = _mm512_mask_mov_ps(v_y2[chain], m_active, v_y2n);
                    v_w[chain]  = _mm512_mask_mov_ps(v_w[chain],  m_active, v_wn);
                    v_it[chain] = _mm512_mask_add_epi32(v_it[chain], m_active, v_it[chain], v_one);
                }
            }

            // Store results for all 4 chains
            #pragma unroll
            for (int chain = 0; chain < NUM_ILP_CHAINS; chain++) {
                uint32_t j_offset = j + chain * VECTOR_WIDTH;
                if (j_offset < data->img_size) {
                    uint32_t* dst = data->out + i * data->img_size + j_offset;
                    _mm512_mask_storeu_epi32((void*)dst, lane_mask[chain], v_it[chain]);
                }
            }
        }
    }

    return nullptr;
}

void mandelbrot_cpu_vector_multicore_multithread_ilp(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out) {
    // Part 4: Combine ALL techniques:
    // - Vector parallelism (AVX-512, 16 pixels per vector)
    // - ILP (4 independent vector chains)
    // - Multi-threading (4 threads for 2 physical cores with hyperthreading)

    constexpr uint32_t NUM_THREADS = 4;

    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    // Calculate rows per thread
    uint32_t rows_per_thread = img_size / NUM_THREADS;

    // Create worker threads with ILP
    for (uint32_t t = 0; t < NUM_THREADS; ++t) {
        thread_data[t].img_size = img_size;
        thread_data[t].max_iters = max_iters;
        thread_data[t].out = out;
        thread_data[t].row_start = t * rows_per_thread;
        thread_data[t].row_end = (t == NUM_THREADS - 1) ? img_size : (t + 1) * rows_per_thread;

        pthread_create(&threads[t], nullptr, mandelbrot_worker_vector_ilp, &thread_data[t]);
    }

    // Wait for all threads to complete
    for (uint32_t t = 0; t < NUM_THREADS; ++t) {
        pthread_join(threads[t], nullptr);
    }
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <vector>

// Useful functions and structures.
enum MandelbrotImpl {
    SCALAR,
    VECTOR,
    VECTOR_ILP,
    VECTOR_MULTICORE,
    VECTOR_MULTICORE_GPT,
    VECTOR_MULTICORE_MULTITHREAD,
    VECTOR_MULTICORE_MULTITHREAD_ILP,
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
                if (strcmp(implementation_str, "scalar") == 0) {
                    *impl = SCALAR;
                } else if (strcmp(implementation_str, "vector") == 0) {
                    *impl = VECTOR;
                } else if (strcmp(implementation_str, "vector_ilp") == 0) {
                    *impl = VECTOR_ILP;
                } else if (strcmp(implementation_str, "vector_multicore") == 0) {
                    *impl = VECTOR_MULTICORE;
                } else if (strcmp(implementation_str, "vector_multicore_gpt") == 0) {
                    *impl = VECTOR_MULTICORE_GPT;
                } else if (
                    strcmp(implementation_str, "vector_multicore_multithread") == 0) {
                    *impl = VECTOR_MULTICORE_MULTITHREAD;
                } else if (
                    strcmp(implementation_str, "vector_multicore_multithread_ilp") == 0) {
                    *impl = VECTOR_MULTICORE_MULTITHREAD_ILP;
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
static constexpr size_t kNumOfOuterIterations = 10;
static constexpr size_t kNumOfInnerIterations = 1;
#define BENCHPRESS(func, ...) \
    do { \
        std::cout << std::endl << "Running " << #func << " ...\n"; \
        std::vector<double> times(kNumOfOuterIterations); \
        for (size_t i = 0; i < kNumOfOuterIterations; ++i) { \
            auto start = std::chrono::high_resolution_clock::now(); \
            for (size_t j = 0; j < kNumOfInnerIterations; ++j) { \
                func(__VA_ARGS__); \
            } \
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
    std::vector<uint32_t> result(img_size * img_size);
    std::vector<uint32_t> ref_result(img_size * img_size);

    // Compute the reference solution
    mandelbrot_cpu_scalar(img_size, max_iters, ref_result.data());

    // Test the desired kernels.
    if (impl == SCALAR || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_scalar, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_scalar.bmp", img_size, max_iters, result);
    }

#ifdef HAS_VECTOR_IMPL
    if (impl == VECTOR || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_vector.bmp", img_size, max_iters, result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }
#endif

    if (impl == VECTOR_ILP || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector_ilp, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_vector_ilp.bmp", img_size, max_iters, result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    if (impl == VECTOR_MULTICORE || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector_multicore, img_size, max_iters, result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_multicore.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    // ChatGPT
    if (impl == VECTOR_MULTICORE_GPT || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector_pthread, img_size, max_iters, result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_parallel.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    if (impl == VECTOR_MULTICORE_MULTITHREAD || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(
            mandelbrot_cpu_vector_multicore_multithread,
            img_size,
            max_iters,
            result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_multicore_multithread.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    if (impl == VECTOR_MULTICORE_MULTITHREAD_ILP || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(
            mandelbrot_cpu_vector_multicore_multithread_ilp,
            img_size,
            max_iters,
            result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_multicore_multithread_ilp.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    return 0;
}
