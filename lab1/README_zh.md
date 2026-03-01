# Lab 1 筆記：Mandelbrot（SIMD + CUDA）

## 問題定義

Lab 1 的核心是對每個像素做 Mandelbrot 迭代，計算該點在多少次迭代後發散（或在 `max_iters` 內不發散）。

```text
z_{n+1} = z_n^2 + c, z_0 = 0
發散條件：|z|^2 = x^2 + y^2 > 4
```

從效能工程角度，這個 workload 有四個重點特性：

1. 像素彼此獨立：每個像素可獨立計算。
2. 平行度高：影像尺寸大時可同時處理大量像素。
3. 計算主導（compute-dominated）：瓶頸主要在迭代運算。
4. 工作量不規則：不同像素需要的迭代次數差異很大。

## 本 Lab 的主要重點

### 1) SIMD 與 GPU 架構差異

同樣是「一次處理多筆資料」，但寫法與控制方式差很多：

- CPU AVX-512：程式碼中要明確處理向量與 mask（顯式 SIMD）
- GPU CUDA：以 thread 為單位寫 scalar 程式，硬體在 warp 層做 lock-step 執行（SPMD）

### 2) 控制流發散是 Mandelbrot 的核心難點

相鄰像素會在不同迭代次數發散，導致同一批 lane/thread 活躍程度不同。  
這讓「如何管理 active/inactive lanes」變成效能關鍵。

- CPU 端：以 mask 控制每個 lane 是否更新
- GPU 端：由 warp predication 機制處理分歧

### 3) 「看起來是 scalar」不代表硬體是 scalar

在 CUDA 版本中，即使 kernel 程式看起來是 scalar，`<<<1,32>>>` 仍然會以 1 個 warp（32 threads）執行。  
因此實務上要看的不是語法像不像向量，而是工作是否正確映射到 warp。

## CPU 與 GPU 對照

| 面向 | CPU (AVX-512) | GPU (CUDA) |
|---|---|---|
| 程式模型 | 顯式 SIMD intrinsics | Scalar 風格 SPMD |
| 執行單位 | 16-lane 向量 | 32-thread warp |
| 分歧處理 | 程式手動管理 mask | 硬體 predication |

## 與原始 baseline 相比，我做的重點調整

### CPU (`mandelbrot_cpu.cpp`)

- 保留 scalar reference 版本做 correctness 對照
- 向量版本以 AVX-512 為主，重點在：
  - 每次處理 16 個像素
  - 使用 `__mmask16` 處理 lane 活躍狀態
  - 以 masked update 避免無效 lane 汙染狀態
  - row 尾端用 `lane_mask` 安全處理

#### `mandelbrot_cpu.cpp` 實際用到的 `_mm512_*` 指令

- `_mm512_set1_ps(...)`：把單一 float 廣播到 16 個 lanes（常數、比例、偏移）。
- `_mm512_set1_epi32(...)`：把單一 int32 廣播到 16 個 lanes（迭代增量）。
- `_mm512_setr_ps(...)`：建立 `[0..15]` 的 lane 索引向量，用來計算 `cx`。
- `_mm512_setzero_ps()`：初始化浮點狀態向量（`x2`、`y2`、`w`）為 0。
- `_mm512_setzero_epi32()`：初始化迭代計數向量為 0。
- `_mm512_add_ps(...)`：向量浮點加法（狀態更新與中間和）。
- `_mm512_sub_ps(...)`：向量浮點減法（按 scalar 順序更新狀態）。
- `_mm512_mul_ps(...)`：向量浮點乘法（`x*x`、`y*y`、`(x+y)*(x+y)`）。
- `_mm512_cmp_ps_mask(...)`：逐 lane 比較，輸出 `__mmask16` 活躍遮罩。
- `_mm512_mask_mov_ps(...)`：masked 寫回，只更新活躍 lanes。
- `_mm512_mask_add_epi32(...)`：masked 迭代累加，只累加活躍 lanes。
- `_mm512_mask_store_epi32(...)`：masked 對齊寫回結果。
- `_mm512_mask_storeu_epi32(...)`：masked 非對齊寫回（fallback）。

對照官方 Lab 1 文件，CPU vectorization 的核心是 16-wide AVX-512 的 row-wise 處理，且常見假設是寬度可被 16 整除。  
目前這份實作保持 16-wide 策略，另外用 `lane_mask` 補上尾端保護。

### GPU (`mandelbrot_gpu.cu`)

- 保留 `<<<1,1>>>` 的 scalar baseline（參考版本）
- 新增/保留 `<<<1,32>>>` 的 warp 版本
- 工作分配方式：
  - `tid = threadIdx.x`
  - 每個 thread 負責 `j = tid, tid+32, tid+64, ...`
  - row 仍逐列走訪

這和官方 Lab 1 的 GPU vector 重點一致：從 `<<<1,1>>>` 切到 `<<<1,32>>>`，並透過 `threadIdx.x` 映射不同像素。

## Lab 問題的實作觀察（簡述）

1. GPU scalar vs CPU scalar  
GPU scalar 不一定快，若平行硬體沒有被打開，還會承擔 kernel launch 的固定成本。

2. CPU vector 實作  
理論加速比不等於實際加速比，會受 divergence、記憶體行為與對齊影響。

3. GPU vector vs scalar  
warp 版本通常更快，因為把平行度直接映射到 32 threads。

這三點可直接對應官方 write-up 問題：

- Q1：GPU scalar vs CPU scalar 的比較與原因
- Q2：CPU vector 設計（`cx/cy` 初始化、控制流、效能）
- Q3：GPU vector 的執行行為與 divergence 處理

## 輸出與視覺化

目前使用 ASCII PPM（`P3`）做輸出，方便除錯與驗證；必要時再轉 PNG：

```bash
sips -s format png mandelbrot_scalar.ppm --out mandelbrot_scalar.png
```

## CPU 測試數據整理

測試設定：

- 尺寸：`256x256`、`512x512`、`1024x1024`
- `max_iters = 256`
- AVX-512 向量結果已與 scalar 結果比對一致

| 影像尺寸 | CPU Scalar (ms) | CPU Vector (ms) | Speedup |
|---|---:|---:|---:|
| `256x256` | 29.1412 | 2.9441 | 9.8982x |
| `512x512` | 112.1880 | 10.8191 | 10.3694x |
| `1024x1024` | 466.5860 | 41.2672 | 11.3065x |

重點觀察：

- AVX-512 相對 scalar 約有 `~10x` 到 `~11x` 加速。
- 影像尺寸變大時，speedup 有小幅提升。
- `512x512` 的 scalar/vector PPM 輸出已保存並驗證。

## GPU 測試數據整理

測試設定：

- 尺寸：`256x256`、`512x512`、`1024x1024`
- `max_iters = 256`
- GPU 各版本均與 GPU scalar 結果比對一致

| 影像尺寸 | GPU Scalar (ms) | GPU Vector (ms) | GPU Parallel Scalar (ms) | GPU Parallel Vector (ms) |
|---|---:|---:|---:|---:|
| `256x256` | 251.626 | 10.729 | 30.7839 | 0.089322 |
| `512x512` | 912.735 | 36.235 | 0.126300 | 0.123392 |
| `1024x1024` | 3647.040 | 135.690 | 0.326564 | 0.346591 |

速度提升（推導）：

| 影像尺寸 | Vector vs Scalar | Parallel Scalar vs Scalar | Parallel Vector vs Scalar | Parallel Vector vs Parallel Scalar |
|---|---:|---:|---:|---:|
| `256x256` | 23.4529x | 8.1739x | 2817.06x | 344.64x |
| `512x512` | 25.1893x | 7226.72x | 7397.03x | 1.0236x |
| `1024x1024` | 26.8777x | 11167.9x | 10522.6x | 0.9422x |

重點觀察：

- `<<<1,32>>>` 的 vector launch 相對 `<<<1,1>>>` scalar launch 有穩定優勢。
- 中大尺寸下，grid-parallel kernel 明顯優於單 block 版本。
- `512x512` 與 `1024x1024` 下，parallel scalar 與 parallel vector 的差距很小。

## 實務結論

這個 Lab 最大的收穫不是「把 Mandelbrot 算出來」，而是建立一個更穩定的效能優化流程：

1. 先有可比對的 scalar baseline  
2. 再加入向量化/平行化  
3. 每一步先驗證 correctness  
4. 最後再談 speedup 與瓶頸
