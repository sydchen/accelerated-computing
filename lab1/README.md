# Lab 1 Notes: Mandelbrot (SIMD + CUDA)

Chinese version: `NOTES_ZH.md`

## Problem Setup

This lab computes the Mandelbrot set per pixel using the standard recurrence:

```text
z_{n+1} = z_n^2 + c,  z_0 = 0
```

A pixel is considered escaped when:

```text
|z|^2 = x^2 + y^2 > 4
```

In practice, each pixel stores the number of iterations before escape, or `max_iters` if it stays bounded.

From a performance-engineering perspective, this workload has four useful properties:

1. Pixel independence: each output pixel can be computed independently.
2. High parallel opportunity: typical image sizes contain many pixels.
3. Compute-dominated behavior: most time is spent in the inner iteration math.
4. Irregular per-pixel work: different pixels require different iteration counts.

## Output and Visualization

I use ASCII PPM (`P3`) because it is simple and easy to inspect during debugging:

```text
P3
<width> <height>
255
<R G B values...>
```

The color mapping in the code is intentionally simple: points that never escape are black, and escaped points use a periodic color transform such as `(val*9)%256`, `(val*7)%256`, `(val*5)%256`.

On macOS, converting output to PNG is straightforward:

```bash
sips -s format png mandelbrot_scalar.ppm --out mandelbrot_scalar.png
```

## Core SIMD Idea

The main performance challenge is divergence: nearby pixels can require very different iteration counts.  
The AVX-512 version handles this with per-lane masks. Each iteration checks `x2 + y2 <= 4.0`, updates an active mask, increments counters only for active lanes, and exits early once all lanes are inactive.

Numerically, the update is:

```text
x' = x^2 - y^2 + cx
y' = 2xy + cy
```

The implementation also uses:

```text
w = (x + y)^2, then 2xy = w - x^2 - y^2
```

## CPU vs GPU Mental Model

| Concept | CPU (AVX-512) | GPU (CUDA) |
|---|---|---|
| Programming style | Explicit SIMD intrinsics | Scalar-style SPMD kernel |
| Execution unit | Vector lanes | Warp (32 threads) |
| Divergence handling | Manual masks in code | Hardware predication/masking |
| Typical width | 16 lanes | 32 lanes per warp |

CUDA is written as SPMD code, but warp execution is still lock-step in practice, so many SIMD-style intuitions transfer.

## AVX-512 in This Codebase

In `mandelbrot_cpu.cpp`, AVX-512 is enabled when `__AVX512F__` is defined.  
The vector kernel works on 16 pixels at a time (`j += 16`) using:

- `__m512` for float vectors
- `__m512i` for iteration counters
- `__mmask16` for lane activity masks

All `_mm512_*` intrinsics currently used in this file:

- `_mm512_set1_ps(...)`: broadcast one float to all 16 lanes (constants like `4.0f`, offsets, scale factors).
- `_mm512_set1_epi32(...)`: broadcast one int32 value to all lanes (iteration increment value).
- `_mm512_setr_ps(...)`: build a lane index vector `[0..15]` for `cx` generation.
- `_mm512_setzero_ps()`: initialize float state vectors (`x2`, `y2`, `w`) to zero.
- `_mm512_setzero_epi32()`: initialize iteration counters to zero.
- `_mm512_add_ps(...)`: vector float add (state update and intermediate sums).
- `_mm512_sub_ps(...)`: vector float subtract (state update with scalar-order-preserving arithmetic).
- `_mm512_mul_ps(...)`: vector float multiply (`x*x`, `y*y`, `(x+y)*(x+y)`).
- `_mm512_cmp_ps_mask(...)`: compare per lane and return `__mmask16` active-lane mask.
- `_mm512_mask_mov_ps(...)`: masked state update, writing only active lanes.
- `_mm512_mask_add_epi32(...)`: masked iteration increment, active lanes only.
- `_mm512_mask_store_epi32(...)`: masked aligned store for vector results.
- `_mm512_mask_storeu_epi32(...)`: masked unaligned store fallback.

Tail processing uses `lane_mask`, and store path selection chooses aligned vs unaligned masked store based on pointer/alignment conditions.

In the official Lab 1 handout, CPU vectorization is framed as 16-wide AVX-512 row processing, and assumes image width is divisible by 16.  
This implementation keeps the same 16-wide strategy and additionally includes safe tail handling with `lane_mask`.

## What Changed in the GPU Version

`mandelbrot_gpu.cu` currently keeps both a scalar baseline and a warp-parallel version:

- Scalar baseline: `mandelbrot_gpu_scalar<<<1,1>>>`
- Warp version: `mandelbrot_gpu_vector<<<1,32>>>`

The core change is the work mapping. Instead of one thread walking every pixel, each thread uses `tid = threadIdx.x` and processes columns `j = tid, tid + 32, tid + 64, ...` across all rows.  
So the GPU version remains scalar at thread level, but parallel at warp level.

This matches the core Lab 1 GPU vector idea: switching from `<<<1,1>>>` to `<<<1,32>>>` and using `threadIdx.x` to map lanes to different pixels.

## AVX-512 and GPU: Practical Difference

Both implementations solve the same divergence problem, but in different ways.  
On CPU AVX-512, masks are explicit in source code. On CUDA, masking is mostly implicit through warp predication and scheduler behavior.

## Key Points in Lab 1

Lab 1 made one idea very concrete for me: writing code that is mathematically correct is only the starting point.  
To get performance, I need to think in terms of execution units, lane utilization, and divergence behavior.

### 1) SIMD and GPU are similar in hardware spirit, but very different in programming style

At a high level, both CPU SIMD and GPU warps execute multiple data elements together.  
But the way I express parallelism is different:

- On AVX-512 CPU, I explicitly manage vectors and masks with intrinsics.
- On CUDA GPU, I write per-thread scalar code, and the hardware groups threads into warps.

This difference changes how I debug and optimize:

- CPU SIMD optimization feels like "manual lane control."
- CUDA optimization feels like "thread mapping + warp behavior management."

### 2) Control-flow divergence is the core bottleneck pattern in Mandelbrot

The Mandelbrot kernel is a perfect divergence case because neighboring pixels often escape at very different iteration counts.

What I learned:

- Throughput is limited when many lanes/threads become inactive early.
- Correctness and performance both depend on how inactive work is handled.
- Early-exit and masked updates are not optional details; they are central design choices.

In AVX-512, this appears directly in code via `__mmask16` and masked ops.  
In CUDA, it appears indirectly through warp execution and hardware predication.

### 3) "Scalar-looking CUDA" can still be vector-like execution

Before this lab, it was easy to think scalar code means scalar execution.  
After implementing `<<<1,32>>>`, it became clear that a scalar thread function can still run as 32-lane lock-step warp execution.

So the real question is not "did I use vector intrinsics?" but:

- Did I map work to threads/warps well?
- Did I avoid wasting inactive lanes?
- Did I expose enough parallel work to hardware?

### 4) Data layout and alignment matter in practice

The AVX-512 path reinforced that alignment is a practical engineering concern, not just a theoretical one:

- aligned loads/stores are safer for predictable performance when data is prepared correctly
- unaligned paths are useful for robustness, especially at boundaries

I also learned to think about tail handling (`lane_mask`) as part of production-quality vector code, not as a corner case.

### 5) Performance work needs a repeatable workflow

The most useful workflow from this lab:

1. keep a scalar reference for correctness
2. introduce one optimization dimension at a time (vector width, mapping, masking)
3. verify output equivalence
4. then compare performance

This helped me avoid "fast but wrong" changes and made optimization decisions easier to explain.

## Short Answers to Lab Questions

1. GPU scalar vs CPU scalar: a GPU scalar kernel can be slower if it underutilizes parallel hardware and pays launch overhead without enough parallel work.
2. CPU vector implementation: speedup depends on lane utilization, divergence, and memory behavior, not just nominal lane count.
3. GPU vector vs GPU scalar: the warp version usually wins because it exposes parallelism directly through 32 concurrent threads.

These map directly to the official Lab 1 write-up prompts:
- Q1: scalar GPU vs scalar CPU runtime comparison and factors
- Q2: CPU vector design choices (`cx/cy` initialization, divergence handling, performance)
- Q3: GPU vector behavior and how SIMD-style execution handles divergent iteration counts

## Alignment Notes

For AVX-512 loads/stores:

- `_mm512_load_ps` expects 64-byte alignment (fast, unsafe if not aligned)
- `_mm512_loadu_ps` works for unaligned addresses (safe, often slightly slower)

Using aligned allocation (`alignas(64)`, `std::aligned_alloc`) keeps behavior predictable.

## CPU Benchmark Results

Benchmark configuration:

- Sizes: `256x256`, `512x512`, `1024x1024`
- `max_iters = 256`
- Scalar and AVX-512 vector outputs were verified to match

| Image Size | CPU Scalar (ms) | CPU Vector (ms) | Speedup |
|---|---:|---:|---:|
| `256x256` | 29.1412 | 2.9441 | 9.8982x |
| `512x512` | 112.1880 | 10.8191 | 10.3694x |
| `1024x1024` | 466.5860 | 41.2672 | 11.3065x |

Quick observations:

- The AVX-512 path consistently outperforms scalar by about `~10x` to `~11x`.
- Speedup increases slightly with larger image sizes in this test set.
- For `512x512`, both scalar and vector PPM outputs were saved and verified.

## GPU Benchmark Results

Benchmark configuration:

- Sizes: `256x256`, `512x512`, `1024x1024`
- `max_iters = 256`
- All GPU variants were verified against GPU scalar output

| Image Size | GPU Scalar (ms) | GPU Vector (ms) | GPU Parallel Scalar (ms) | GPU Parallel Vector (ms) |
|---|---:|---:|---:|---:|
| `256x256` | 251.626 | 10.729 | 30.7839 | 0.089322 |
| `512x512` | 912.735 | 36.235 | 0.126300 | 0.123392 |
| `1024x1024` | 3647.040 | 135.690 | 0.326564 | 0.346591 |

Derived speedups:

| Image Size | Vector vs Scalar | Parallel Scalar vs Scalar | Parallel Vector vs Scalar | Parallel Vector vs Parallel Scalar |
|---|---:|---:|---:|---:|
| `256x256` | 23.4529x | 8.1739x | 2817.06x | 344.64x |
| `512x512` | 25.1893x | 7226.72x | 7397.03x | 1.0236x |
| `1024x1024` | 26.8777x | 11167.9x | 10522.6x | 0.9422x |

Quick observations:

- `<<<1,32>>>` vector launch is consistently much faster than `<<<1,1>>>` scalar launch.
- Grid-parallel kernels dominate single-block kernels for medium/large sizes.
- At `512x512` and `1024x1024`, parallel scalar and parallel vector are very close.
- For `512x512`, scalar/vector PPM outputs were saved and verified.

## Takeaways

Mandelbrot is mathematically simple, but performance work quickly becomes about divergence control and memory behavior.  
A good workflow here is still: get correctness first, then profile, then optimize one variable at a time.
