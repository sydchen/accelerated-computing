# Lab 8: 一千萬個圓形

**MIT 6.S894 - Accelerated Computing (Fall 2025)**

官方課程資源：[Lab 8 - Ten Million Circles](https://accelerated-computing.academy/fall25/labs/lab8/)

## 概述

實現 GPU 圓形渲染器，能夠正確渲染 1000 萬個圓形，並達到性能目標。

**核心挑戰：** 在保持正確 alpha blending 順序的前提下，實現高度並行的 GPU 渲染。

## 性能結果

| 目標 | 要求 | 實際達成 |
|------|------|---------|
| **Full Credit** | < 200ms |  158ms |
| **Extra Credit** | < 100ms |  **78ms** |
| **正確性** | < 0.05 誤差 |  通過 |

**測試環境：** Tesla P100 (Kaggle)

## 關鍵約束

1. **順序依賴：** 圓形必須按 0 到 n_circle-1 順序處理（alpha blending 要求）
2. **Alpha Blending 公式：** `new_color = circle_color * alpha + old_color * (1 - alpha)`
3. **禁用外部庫：** 不能使用 Thrust、CUB
4. **記憶體管理：** 使用 `GpuMemoryPool` 分配臨時 GPU 記憶體

---

## 最終實現策略：Pre-binning + Shared Memory

### 為何選擇這個策略？

**嘗試過但失敗的方法：**

| 方法 | 問題 | 性能 |
|------|------|------|
| Per-pixel（遍歷所有圓形） | 每像素檢查 10M 圓形太慢 | > 2 分鐘 |
| Per-circle（每圓形一個 kernel） | 10M kernel launches 開銷太大 | 22.7 秒 |
| 分批 + Tile culling | 仍然要檢查所有圓形（只是跳過不重疊的） | 39–54 秒 |

**成功的方法：Pre-binning**

核心思想：**預先計算每個 tile 需要處理哪些圓形**，渲染時只處理該 tile 的圓形列表。

### 算法流程

```
1. 初始化畫布為白色

2. Pre-binning（預處理）：
   a. count_circles_per_tile：計算每個 tile 的圓形數量
   b. prefix_sum：計算每個 tile 的列表偏移量
   c. assign_circles_to_tiles：將圓形 ID 寫入各 tile 的列表
   d. sort_tile_circles：排序每個 tile 的列表（保持 alpha blending 順序）

3. 渲染：
   - 每個 block 處理一個 tile
   - 只處理該 tile 的圓形列表（已排序）
   - 使用 shared memory 緩存圓形數據
```

### 關鍵參數

```cpp
#define TILE_SIZE 8                    // 8x8 像素 per tile（關鍵！小 tile 效果更好）
#define TILES_X   128                  // 1024 / 8
#define TILES_Y   128
#define N_TILES   16384                // 128 x 128
#define MAX_CIRCLES_PER_BATCH 256      // shared memory batch size
```

**Tile 大小的影響：**

| Tile Size | Tiles | 性能 |
|-----------|-------|------|
| 32×32 | 1,024 | 749ms（太大，每 tile 圓形太多） |
| 16×16 | 4,096 | 158ms |
| **8×8** | **16,384** | **78ms（最佳）** |

---

## Kernel 實現

### Kernel 1：初始化畫布

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

### Kernel 2：計算每個 Tile 的圓形數量

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

### Kernel 3：將圓形分配到 Tiles

```cuda
__global__ void assign_circles_to_tiles(
    int32_t n_circle,
    const float *circle_x, const float *circle_y, const float *circle_radius,
    const int *tile_offsets, int *tile_counts, int *circle_lists)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_circle) return;

    // 計算覆蓋的 tiles，用 atomicAdd 獲取寫入位置
    for (int ty = tile_y_min; ty <= tile_y_max; ty++) {
        for (int tx = tile_x_min; tx <= tile_x_max; tx++) {
            int tile_idx = ty * TILES_X + tx;
            int pos = atomicAdd(&tile_counts[tile_idx], 1);
            circle_lists[tile_offsets[tile_idx] + pos] = i;
        }
    }
}
```

### Kernel 4：排序每個 Tile 的圓形列表

```cuda
__global__ void sort_tile_circles(
    const int *tile_offsets, const int *tile_sizes, int *circle_lists)
{
    int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_idx >= N_TILES) return;

    int start = tile_offsets[tile_idx];
    int size  = tile_sizes[tile_idx];

    // Insertion sort（保持 alpha blending 順序）
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

### Kernel 5：渲染（使用 Shared Memory）

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

    int tile_idx = blockIdx.y * TILES_X + blockIdx.x;
    int tid      = threadIdx.y * TILE_SIZE + threadIdx.x;
    int pixel_x  = blockIdx.x * TILE_SIZE + threadIdx.x;
    int pixel_y  = blockIdx.y * TILE_SIZE + threadIdx.y;
    int pixel_idx = pixel_y * width + pixel_x;

    float r = img_red[pixel_idx];
    float g = img_green[pixel_idx];
    float b = img_blue[pixel_idx];

    int list_start = tile_offsets[tile_idx];
    int list_size  = tile_sizes[tile_idx];

    for (int batch_start = 0; batch_start < list_size; batch_start += MAX_CIRCLES_PER_BATCH) {
        int batch_size = min(MAX_CIRCLES_PER_BATCH, list_size - batch_start);

        // 協作載入這批圓形數據到 shared memory
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

## 性能優化歷程

| 階段 | 策略 | 性能 | 改進 |
|------|------|------|------|
| v0 | Per-pixel，遍歷所有圓形 | > 54s | — |
| v1 | Per-circle kernel | 22.7s | 2.4× |
| v2 | 分批 + tile culling | 39s | 退步 |
| v3 | Pre-binning（16×16 tiles） | 205ms | **190×** |
| v4 | + Shared memory render | 158ms | 1.3× |
| **v5** | **8×8 tiles** | **78ms** | **2×** |

**關鍵洞察：**
1. **Pre-binning 是關鍵**——避免每像素檢查所有圓形
2. **小 tile 更好**——8×8 比 16×16 快 2 倍
3. **Shared memory 有效**——減少全域記憶體存取

---

## 記憶體使用

```
臨時 Buffer（透過 memory_pool 分配）：
  tile_counts[16,384]     ~  64 KB
  tile_offsets[16,385]    ~  64 KB
  tile_sizes[16,384]      ~  64 KB
  circle_lists[~40M]      ~ 160 MB（取決於圓形分布）
```

---

## 遇到的有趣 Bug

1. **Alpha blending 順序損壞**：初始實現沒有排序 tile 內的圓形，產生視覺上不正確的結果。`assign_circles_to_tiles` 中的 `atomicAdd` 不保證順序，所以需要排序步驟。

2. **Shared memory 容量問題**：使用 batch size 512 超過了 shared memory 容量，導致隱藏的數值錯誤（不是 crash）。縮回 256 後修好。

3. **Tile 大小的反直覺結果**：16×16 達到 158ms，但換成 8×8 直接跳到 78ms——快了整整 2 倍。一開始以為 tile 越大效率越高（shared memory 載入次數少），實驗推翻了這個直覺。

4. **Kaggle 目錄問題**：`out/` 目錄預設不存在，圖片寫入靜默失敗，誤以為渲染結果是空的。加上 `mkdir("out", 0755)` 才解決。

---

## 設計反思

我的實現採用空間預分箱策略，結合按像素並行化與 shared memory 優化，總共五個 kernel。

**核心洞察**：避免 10 兆次的距離計算。如果每個像素都檢查所有 10M 個圓形，就會產生 10 兆次多餘計算。透過將圓形預分箱到 8×8 的空間 tile，每個像素只需檢查約 20–30 個圓形而不是 10M 個。

**並行化策略**：
- 按圓形並行化發生在計數和分配 kernel 中（每個圓形決定它影響哪些 tile）
- 按像素並行化發生在渲染 kernel 中（每個 thread 獨立處理自己的像素）
- 這種混合方法利用了兩者的優勢：分箱時圓形並行，渲染時像素並行

**可能的進一步優化**：
- **提前終止**：追蹤累積 alpha，一旦接近 1.0 就停止處理後續圓形，估計省 20–30%
- **更好的排序**：用 GPU 友好的 bitonic sort 替換 insertion sort，應對圓形很多的 tile
- **Warp 級早退**：用 `__ballot_sync` 偵測整個 warp 都在圓形外，整批跳過
- **動態 tile 大小**：根據圓形密度自適應調整 tile 大小
