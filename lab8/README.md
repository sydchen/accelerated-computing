# Lab 8: Ten Million Circles

**MIT 6.S894 - Accelerated Computing (Fall 2025)**

Official Course Resource: [Lab 8 - Ten Million Circles](https://accelerated-computing.academy/fall25/labs/lab8/)

## Overview

Implement a GPU circle renderer that correctly renders 10 million circles and meets performance targets.

**Core challenge:** Achieve highly parallel GPU rendering while preserving correct alpha blending order.

## Performance Results

| Goal | Requirement | Achieved |
|------|-------------|---------|
| **Full Credit** | < 200ms | 158ms |
| **Extra Credit** | < 100ms | **78ms** |
| **Correctness** | < 0.05 error | Pass |

**Test environment:** Tesla P100 (Kaggle)

## Key Constraints

1. **Order dependence:** Circles must be processed in order from 0 to n_circle-1 (required by alpha blending)
2. **Alpha blending formula:** `new_color = circle_color * alpha + old_color * (1 - alpha)`
3. **No external libraries:** Thrust and CUB are not allowed
4. **Memory management:** Use `GpuMemoryPool` to allocate temporary GPU memory

---

## Final Strategy: Pre-binning + Shared Memory

### Why This Approach?

**Approaches tried and failed:**

| Approach | Problem | Performance |
|----------|---------|-------------|
| Per-pixel (iterate all circles) | Checking 10M circles per pixel is too slow | > 2 minutes |
| Per-circle (one kernel per circle) | 10M kernel launches dominate runtime | 22.7 seconds |
| Batched + tile culling | Still checks all circles per tile (just skips non-overlapping) | 39–54 seconds |

**The approach that worked: Pre-binning**

Core idea: **precompute which circles each tile needs to process**, then at render time only process each tile's circle list.

### Algorithm

```
1. Initialize canvas to white

2. Pre-binning (preprocessing):
   a. count_circles_per_tile: count how many circles overlap each tile
   b. prefix_sum: compute list offset for each tile
   c. assign_circles_to_tiles: write circle IDs into each tile's list
   d. sort_tile_circles: sort each tile's list (to preserve alpha blending order)

3. Render:
   - Each block handles one tile
   - Only processes that tile's circle list (already sorted)
   - Uses shared memory to cache circle data
```

### Key Parameters

```cpp
#define TILE_SIZE 8                    // 8x8 pixels per tile (critical — smaller is better)
#define TILES_X   128                  // 1024 / 8
#define TILES_Y   128
#define N_TILES   16384                // 128 x 128
#define MAX_CIRCLES_PER_BATCH 256      // shared memory batch size
```

**Effect of tile size:**

| Tile Size | Tiles | Performance |
|-----------|-------|-------------|
| 32×32 | 1,024 | 749ms (too large, too many circles per tile) |
| 16×16 | 4,096 | 158ms |
| **8×8** | **16,384** | **78ms (best)** |

---

## Kernel Implementations

### Kernel 1: Initialize Canvas

```cuda
__global__ void init_image_kernel(
    float *img_red, float *img_green, float *img_blue, int n_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pixels) return;
    img_red[idx] = 1.0f;
    img_green[idx] = 1.0f;
    img_blue[idx] = 1.0f;
}
```

### Kernel 2: Count Circles per Tile

```cuda
__global__ void count_circles_per_tile(
    int32_t n_circle,
    const float *circle_x, const float *circle_y, const float *circle_radius,
    int *tile_counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_circle) return;

    float cx = circle_x[i], cy = circle_y[i], radius = circle_radius[i];

    int tile_x_min = max(0,         (int)((cx - radius) / TILE_SIZE));
    int tile_x_max = min(TILES_X-1, (int)((cx + radius) / TILE_SIZE));
    int tile_y_min = max(0,         (int)((cy - radius) / TILE_SIZE));
    int tile_y_max = min(TILES_Y-1, (int)((cy + radius) / TILE_SIZE));

    for (int ty = tile_y_min; ty <= tile_y_max; ty++)
        for (int tx = tile_x_min; tx <= tile_x_max; tx++)
            atomicAdd(&tile_counts[ty * TILES_X + tx], 1);
}
```

### Kernel 3: Assign Circles to Tiles

```cuda
__global__ void assign_circles_to_tiles(
    int32_t n_circle,
    const float *circle_x, const float *circle_y, const float *circle_radius,
    const int *tile_offsets, int *tile_counts, int *circle_lists)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_circle) return;

    // use atomicAdd to claim a write slot in each overlapping tile's list
    for (int ty = tile_y_min; ty <= tile_y_max; ty++) {
        for (int tx = tile_x_min; tx <= tile_x_max; tx++) {
            int tile_idx = ty * TILES_X + tx;
            int pos = atomicAdd(&tile_counts[tile_idx], 1);
            circle_lists[tile_offsets[tile_idx] + pos] = i;
        }
    }
}
```

### Kernel 4: Sort Each Tile's Circle List

```cuda
__global__ void sort_tile_circles(
    const int *tile_offsets, const int *tile_sizes, int *circle_lists)
{
    int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_idx >= N_TILES) return;

    int start = tile_offsets[tile_idx];
    int size  = tile_sizes[tile_idx];

    // insertion sort to restore ascending circle index order
    for (int i = 1; i < size; i++) {
        int key = circle_lists[start + i], j = i - 1;
        while (j >= 0 && circle_lists[start + j] > key) {
            circle_lists[start + j + 1] = circle_lists[start + j];
            j--;
        }
        circle_lists[start + j + 1] = key;
    }
}
```

### Kernel 5: Render with Shared Memory

```cuda
__global__ void render_with_tile_lists(
    int32_t width, int32_t height,
    const float *circle_x, const float *circle_y, const float *circle_radius,
    const float *circle_red, const float *circle_green,
    const float *circle_blue, const float *circle_alpha,
    const int *tile_offsets, const int *tile_sizes, const int *circle_lists,
    float *img_red, float *img_green, float *img_blue)
{
    __shared__ float sh_cx[MAX_CIRCLES_PER_BATCH];
    __shared__ float sh_cy[MAX_CIRCLES_PER_BATCH];
    __shared__ float sh_radius[MAX_CIRCLES_PER_BATCH];
    __shared__ float sh_cr[MAX_CIRCLES_PER_BATCH];
    __shared__ float sh_cg[MAX_CIRCLES_PER_BATCH];
    __shared__ float sh_cb[MAX_CIRCLES_PER_BATCH];
    __shared__ float sh_alpha[MAX_CIRCLES_PER_BATCH];

    int tile_idx  = blockIdx.y * TILES_X + blockIdx.x;
    int tid       = threadIdx.y * TILE_SIZE + threadIdx.x;
    int pixel_x   = blockIdx.x * TILE_SIZE + threadIdx.x;
    int pixel_y   = blockIdx.y * TILE_SIZE + threadIdx.y;
    int pixel_idx = pixel_y * width + pixel_x;

    float r = img_red[pixel_idx];
    float g = img_green[pixel_idx];
    float b = img_blue[pixel_idx];

    int list_start = tile_offsets[tile_idx];
    int list_size  = tile_sizes[tile_idx];

    for (int batch_start = 0; batch_start < list_size; batch_start += MAX_CIRCLES_PER_BATCH) {
        int batch_size = min(MAX_CIRCLES_PER_BATCH, list_size - batch_start);

        // collaboratively load this batch of circle data into shared memory
        for (int k = tid; k < batch_size; k += TILE_SIZE * TILE_SIZE) {
            int ci = circle_lists[list_start + batch_start + k];
            sh_cx[k] = circle_x[ci];  sh_cy[k] = circle_y[ci];
            sh_radius[k] = circle_radius[ci];
            sh_cr[k] = circle_red[ci];  sh_cg[k] = circle_green[ci];
            sh_cb[k] = circle_blue[ci]; sh_alpha[k] = circle_alpha[ci];
        }
        __syncthreads();

        for (int k = 0; k < batch_size; k++) {
            float dx = pixel_x - sh_cx[k];
            float dy = pixel_y - sh_cy[k];
            if (dx*dx + dy*dy < sh_radius[k] * sh_radius[k]) {
                float a = sh_alpha[k];
                r = sh_cr[k] * a + r * (1.0f - a);
                g = sh_cg[k] * a + g * (1.0f - a);
                b = sh_cb[k] * a + b * (1.0f - a);
            }
        }
        __syncthreads();
    }

    img_red[pixel_idx] = r;
    img_green[pixel_idx] = g;
    img_blue[pixel_idx] = b;
}
```

---

## Optimization Journey

| Version | Strategy | Performance | Speedup |
|---------|----------|-------------|---------|
| v0 | Per-pixel, iterate all circles | > 54s | — |
| v1 | Per-circle kernel | 22.7s | 2.4× |
| v2 | Batched + tile culling | 39s | regression |
| v3 | Pre-binning (16×16 tiles) | 205ms | **190×** |
| v4 | + Shared memory render | 158ms | 1.3× |
| **v5** | **8×8 tiles** | **78ms** | **2×** |

**Key takeaways:**
1. **Pre-binning is the breakthrough**—avoids checking all circles per pixel
2. **Smaller tiles are better**—8×8 is 2× faster than 16×16
3. **Shared memory pays off**—reduces global memory traffic for circle data

---

## Memory Usage

```
Temporary buffers (allocated via memory_pool):
  tile_counts[16,384]     ~  64 KB
  tile_offsets[16,385]    ~  64 KB
  tile_sizes[16,384]      ~  64 KB
  circle_lists[~40M]      ~ 160 MB (depends on circle distribution)
```

---

## Interesting Bugs

1. **Alpha blending order corruption:** The initial implementation didn't sort circles within each tile, producing visually incorrect output. `atomicAdd` in `assign_circles_to_tiles` doesn't guarantee insertion order, so a sort step is required.

2. **Shared memory overflow:** Using batch size 512 exceeded shared memory capacity, causing silent numerical errors (not a crash). Reducing to 256 fixed it.

3. **Counterintuitive tile size result:** 16×16 tiles gave 158ms, but switching to 8×8 jumped to 78ms—2× faster. The initial assumption was that larger tiles would be more efficient (fewer shared memory loads). Experiments proved the opposite: shorter per-tile circle lists matter more.

4. **Kaggle directory issue:** The `out/` directory doesn't exist by default, causing image writes to silently fail. Fixed by adding `mkdir("out", 0755)` at startup.

---

## Design Reflection

The implementation uses spatial pre-binning combined with per-pixel parallelism and shared memory optimization, organized as five kernels.

**Core insight:** Avoid 10 trillion distance checks. If every pixel checked all 10M circles, that's 10T redundant computations. By pre-binning circles into 8×8 spatial tiles, each pixel only checks ~20–30 circles instead of 10M.

**Parallelization strategy:**
- Circle-parallel in the counting and assignment kernels (each circle decides which tiles it affects)
- Pixel-parallel in the render kernel (each thread independently processes its own pixel)
- This hybrid exploits both: circles run in parallel during binning, pixels run in parallel during rendering

**Potential further optimizations:**
- **Early termination:** Track accumulated alpha per pixel; once it approaches 1.0, skip remaining circles. Estimated 20–30% savings.
- **Better sorting:** Replace insertion sort with GPU-friendly bitonic sort for tiles with many circles
- **Warp-level early exit:** Use `__ballot_sync` to detect when an entire warp misses a circle and skip it as a group
- **Adaptive tile size:** Adjust tile granularity based on local circle density
