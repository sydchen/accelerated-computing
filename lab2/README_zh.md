# Lab 2 筆記：大規模平行 Mandelbrot

來源：https://accelerated-computing.academy/fall25/labs/lab2/

## Lab 目標

Lab 2 在 Lab 1 的 Mandelbrot 基礎上，新增更多層次的平行化：

1. **指令級平行（ILP）**：在單一指令流中利用彼此獨立的指令
2. **多核心平行（Multi-Core）**：把工作分配到多個 CPU cores 或 GPU SM
3. **多執行緒平行（Multi-Threaded）**：每個執行單元同時管理多個 threads/warps

核心目標是對齊 CPU 與 GPU 的執行模型，並透過平行層次疊加來提升整體效能。

## 與 Lab 1 的重要差異

起始碼將這一行改成：

```cpp
// old
float y = w - x2 - y2 + cy;
// new
float y = w - (x2 + y2) + cy;
```

由於浮點數的非結合律（non-associativity），這會同時影響效能與輸出結果。

另外，影像設定也更激進（更高解析/品質與不同縮放視窗），例如 `window_zoom`、`window_x`、`window_y`。

## 術語對照（CPU vs GPU）

| CUDA 文件中的 GPU 術語 | 對應的 CPU 類比（本課脈絡） |
|---|---|
| CUDA thread | Vector lane |
| Warp | Thread（指令流） |
| Warp scheduler | 類 core 的執行單元 |

在這個 Lab 裡，重點是以 warp-scheduler/core 粒度思考 occupancy 與 throughput。

## 硬體背景（課程頁）

- CPU：AMD Ryzen 7 7700，8 cores，SMT 最多 2 threads/core。
- GPU：NVIDIA RTX 4000 Ada，48 個 SM，每個 SM 4 個 warp schedulers（總計 192）。

## Part 0：FFMA 延遲 + ILP Prelab

### 交付項目

- 在 `fma_latency.cu` 實作 `measure_fma_latency(..)`。
- 用 SASS 驗證是否生成 `FFMA`（telerun 流程中的 `-s`）。
- 完成 interleaved / non-interleaved 的 ILP 版本。

### 需要回答的問題

- Q0.1：FFMA latency 是多少？
- Q0.2：顯式 interleaved ILP 觀察到的 latency 是什麼？
- Q0.3：非顯式 interleave 的結果是否相同？為什麼？
- Q0.4：throughput 在多少 warp 數達到峰值？如何用 FFMA latency 解釋？

### 筆記重點

- 用 dependent chains 測量 latency 才穩定。
- latency 與 throughput 的分析要分開看。
- 搭配 `warp_scheduler.cu` 與 `plot.py` 觀察縮放行為。

## Part 1：Mandelbrot 的 ILP

### CPU

- 檔案：`mandelbrot_cpu_2.cpp`
- 實作：`mandelbrot_cpu_vector_ilp`

### GPU

- 檔案：`mandelbrot_gpu_2.cu`
- 實作：`mandelbrot_gpu_vector_ilp`、`launch_mandelbrot_gpu_vector_ilp`
- launch 維持 `<<<1, 32>>>`（單 warp 基準類比）

### 設計檢查清單

- 多個獨立 vectors 的 state 要怎麼組織
- divergence / control flow 如何處理
- 同時處理幾個 vectors（ILP factor）
- row-wise 或 tile-wise 的工作分配
- `#pragma unroll` 是否在可控範圍帶來收益

### 最終報告問題對應

- Q1：CPU/GPU 上的 ILP 加速比、分工策略、控制流策略、選擇的 ILP factor、限制因素。

### CPU vs GPU 實作差異

**CPU（AVX-512）**
- State：向量暫存器陣列 `__m512 v_x2[4]`
- Divergence：用 `__mmask16` 做硬體 predication
- Memory pattern：水平連續（row-major）
- Control flow：用 mask 降低 branch 成本

**GPU（CUDA）**
- State：純量暫存器陣列 `float x2[4]`
- Divergence：boolean flag + branch（thread 內，成本低）
- Memory pattern：stride 存取（warp 內 coalesced）
- Control flow：可接受每 thread 分歧

### 預期效能提升

- **CPU**：1.2x - 2.0x（受 register 壓力與記憶體頻寬限制）
- **GPU**：1.5x - 3.0x（GPU 對 ILP 延遲隱藏更敏感）

## Part 2：Multi-Core Parallelism

### CPU Multi-Core

- 在 `mandelbrot_cpu_2.cpp` 實作：
  - `mandelbrot_cpu_vector_multicore`
- 使用 pthread（`pthread_create`、`pthread_join`）做 8-core 擴展。

Q2 重點：
- 相對單 core vector baseline 的加速
- 工作切分策略對效能的影響

### GPU Multi-Core

- 在 `mandelbrot_gpu_2.cu` 實作：
  - `mandelbrot_gpu_vector_multicore`
  - `launch_mandelbrot_gpu_vector_multicore`

課程目標概念：
- 讓 192 個 warp schedulers 都有對應 warp
- baseline 參考配置：`<<<48, 4 * 32>>>`

Q3 重點：
- 相對單 warp vector baseline 的加速
- 與 CPU multicore 的絕對時間比較
- block/warp 的分工策略

Q4 重點：
- 比較 launch：
  - `<<<48, 4*32>>>`
  - `<<<96, 2*32>>>`
  - `<<<24, 8*32>>>`
- 解釋工作分配行為與測得時間差異

### 效能分析：Launch Configuration 影響

關鍵發現：**Block 粒度比 total warps 更重要**

| 配置 | Blocks | Threads/Block | Total Warps | Warps/Scheduler | Blocks/SM | Runtime | 備註 |
|---|---|---|---|---|---|---|---|
| <<<48, 128>>> | 48 | 128 | 192 | 1.0 | 1.0 | baseline | 完整 1:1 SM 對應 |
| <<<96, 64>>> | 96 | 64 | 192 | 1.0 | 2.0 | ~2.5% faster | 動態排程更好 |
| <<<24, 256>>> | 24 | 256 | 192 | 1.0 | 0.5 | ~17% slower | 只用到 48 個 SM 中的 24 個 |

**關鍵洞察**：即使 total warps 相同，較多 blocks（每 SM 2 個）通常會有更好的負載平衡與排程彈性。

### 預期加速

- **CPU Multi-Core**：6-7.5x（理論 8x，受負載平衡與 cache contention 限制）
- **GPU Multi-Core**：相對單 warp baseline 約 150-180x（192 個 warp schedulers 同時運作）

## Part 3：Multi-Threaded Parallelism

### CPU Multi-Threaded

- 實作：
  - `mandelbrot_cpu_vector_multicore_multithread`
- 每個 core 使用超過 1 個 thread。

Q5 重點：
- 每 core 加入 multithreading 後的加速
- 最佳 thread 數量與原因

### GPU Multi-Threaded

- 單一 SM 研究：
  - `mandelbrot_cpu_vector_multicore_multithread_single_sm`
  - `launch_cpu_vector_multicore_multithread_single_sm`
- 全機多執行緒 GPU：
  - `mandelbrot_gpu_vector_multicore_multithread_full`
  - `launch_mandelbrot_gpu_vector_multicore_multithread_full`

Q6 重點：
- warps/block 超過 4 之後的時間趨勢
- 是否一路提升到 32 warps/block

### 單一 SM 多執行緒結果

實驗顯示到硬體上限前都持續改善：

| Config | Warps | Warps/Scheduler | Runtime (ms) | Relative Speedup | Improvement % |
|---|---|---|---|---|---|
| Baseline | 4 | 1 | 132.10 | 1.00x | - |
| 8 warps | 8 | 2 | 84.53 | 1.56x | +56.0% |
| 12 warps | 12 | 3 | 70.82 | 1.87x | +46.4% |
| 32 warps | 32 | 8 | 58.31 | 2.27x | +55.9% |

**關鍵發現**：效能持續上升、未見衰退，表示此配置下資源（register/shared memory）足以支撐每 scheduler 8 warps。

**為何有效**：當某 warp 在等待 FMA latency（約 4-11 cycles）時，scheduler 會切到其他 warp，維持硬體忙碌。

Q7 重點：
- 全機多執行緒加速比
- 最佳 warp 數與主要影響因素

### 全機多執行緒（P100 範例）

| Configuration | Warps | Warps/Scheduler | Blocks/SM | Runtime | Speedup vs Baseline |
|---|---|---|---|---|---|
| <<<56, 64>>> | 112 | 1.0 | 1.0 | 7.21 ms | 1.00x (baseline) |
| <<<56, 128>>> | 224 | 2.0 | 1.0 | 3.84 ms | 1.88x |
| <<<56, 256>>> | 448 | 4.0 | 1.0 | 2.83 ms | 2.55x |
| <<<112, 128>>> | 448 | 4.0 | 2.0 | **2.37 ms** | **3.04x**（最佳） |
| <<<56, 512>>> | 896 | 8.0 | 1.0 | 3.12 ms | 2.31x |
| <<<112, 256>>> | 896 | 8.0 | 2.0 | 2.39 ms | 3.02x |

**關鍵發現**：**block 粒度（2 blocks/SM）比單純加 warp 數更重要**

- <<<56, 256>>>（1 block/SM，4 W/S）：2.83 ms
- <<<112, 128>>>（2 blocks/SM，4 W/S）：2.37 ms（快 19%）

較多小 blocks -> 更高排程彈性 -> 更好的負載平衡

## Part 4：整合所有技術（ILP + Multi-Core + Multi-Threaded）

- CPU：
  - `mandelbrot_cpu_vector_multicore_multithread_ilp`
- GPU：
  - `mandelbrot_gpu_vector_multicore_multithread_full_ilp`
  - `launch_mandelbrot_gpu_vector_multicore_multithread_full_ilp`

Q8 重點：
- 在 multicore + multithread 之上再加入 ILP 的額外收益
- 與單 core / 單 thread 場景下 ILP 收益比較
- 最佳 thread/warp 數與最佳內層 ILP factor

### CPU 效能數據（Kaggle 2-core）

| Implementation | Runtime (ms) | Speedup vs Scalar | Improvement vs Previous |
|---|---|---|---|
| Scalar (baseline) | 751.78 | 1.00x | - |
| Vector + ILP (single-thread) | 65.29 | 11.52x | - |
| Vector + Multi-core | 51.08 | 14.72x | -21.8% |
| Vector + Multi-thread (4 threads) | 34.92 | 21.53x | -31.6% |
| Vector + Multi-thread + ILP | 32.78 | 22.94x | -6.1% |

**關鍵發現**：在多執行緒環境中，ILP 僅帶來 **6.1%** 進步，遠低於單執行緒 ILP 的收益。

**為何收益有限**：
1. **資源已飽和**：4 threads 幾乎吃滿 2 個實體核心
2. **Register 壓力**：ILP 需要約 4 倍暫存器，容易 spill
3. **Cache Thrashing**：16 條活躍計算流（4 threads × 4 chains）競爭 L1/L2
4. **指令流已交錯**：Hyperthreading 本身已提供每 core 的交錯執行

**結論**：在這個平台上，從成本效益來看，Part 3（34.92 ms）其實已很接近最佳；Part 4 雖更快但邊際收益小（32.78 ms）。

### GPU 效能數據（Tesla P100）

| Implementation | Runtime (ms) | Speedup vs Single-Warp |
|---|---|---|
| Part 3 (no ILP) | 2.36 ms | 365x |
| Part 4 (with ILP 4 chains) | 5.51 ms | 157x |

**負面影響**：加入 ILP 後效能下降 **133%**（慢 2.33x）。

**主因**：4 pixel chains 帶來的 register 壓力造成：
- occupancy 下降（每 SM 活躍 warps 變少）
- spill 到 local memory（單次存取高延遲）
- 原本靠 multithreading 的 latency-hiding 被破壞

**核心洞察**：在 GPU 上，硬體 multithreading 常比軟體 ILP 更有效；額外 ILP 會競爭有限資源。

## 建議實驗表格格式

建議全階段用同一張表，方便報告一致性：

| Variant | CPU/GPU | Launch / Thread Config | Runtime (ms) | Speedup vs Baseline | Notes |
|---|---|---|---:|---:|---|
| vector baseline | CPU | single-core |  |  |  |
| vector + ILP | CPU | single-core |  |  |  |
| vector multicore | CPU | 8 cores |  |  |  |
| vector multicore+mt | CPU | N threads |  |  |  |
| vector multicore+mt+ILP | CPU | N threads |  |  |  |
| vector baseline | GPU | `<<<1,32>>>` |  |  |  |
| vector + ILP | GPU | `<<<1,32>>>` |  |  |  |
| vector multicore | GPU | `<<<48,4*32>>>` |  |  |  |
| vector multicore+mt full | GPU | tuned |  |  |  |
| vector multicore+mt+ILP | GPU | tuned |  |  |  |

## 目前本地檔案

- `fma_latency.cu`
- `fma_latency.md`
- `warp_scheduler.cu`
- `mandelbrot_cpu_2.cpp`
- `mandelbrot_gpu_2.cu`
- `plot.py`

此文件可作為 Lab 2 的工作紀錄與最終報告骨架。

---

## 附錄：整合自 docs/README.md

# Lab 2 重點摘要

Lab 2 的 window 是 10000x 縮放 到一個非常小的區域：
- 更多像素會快速逃逸（迭代次數少）
- Early exit 更頻繁

Lab 1 的 window 是完整 Mandelbrot set：
- 很多像素在 set 內部或邊界附近
- 需要跑完全部 2000 次迭代
- 計算量大很多

---
在 Lab 1 的 vector parallelism 基礎上，新增三種並行：

1. Instruction-Level Parallelism (ILP) - 單一指令流內的並行
2. Multi-Core Parallelism - 跨多個物理核心
3. Multi-Threaded Parallelism - 單一核心內多執行緒

### 硬體規格

CPU: AMD Ryzen 7 7700
- 8 cores @ 3.8 GHz
- 每個 core 支援 2 個同時執行的 threads (SMT)

GPU: NVIDIA RTX 4000 Ada
- 48 SMs @ 2.175 GHz
- 每個 SM 有 4 個 warp schedulers
- 總共 192 個 warp schedulers (類比 CPU cores)
- 每個 warp scheduler 最多支援 12 個並行 warps

---
實際跑在Kaggle 上面, GPU是Tesla P100
- 56 SMs
- 每個 SM 有 2 個 warp schedulers
- 總共 112 個 warp schedulers (類比 CPU cores)

---
### 術語釐清

| GPU 概念    | CPU 概念    |
|-------------|-------------|
| CUDA Thread | Vector Lane |
| Warp        | Thread      |

## Part 0 (Prelab): FMA 指令延遲測量

Deliverable: 測量 FFMA 指令延遲
- 使用 fma_latency.cu
- 用 -s flag 驗證 SASS 輸出

Prelab Questions:
1. FFMA 指令延遲是多少？
2. 用 interleaved FMA 測試 ILP，觀察到什麼延遲？
3. 不明確 interleave 的版本效能如何？為什麼？
4. 何時達到最大 throughput？用 FMA 延遲解釋

## Part 1: ILP in Mandelbrot

核心概念: 同時處理多個獨立向量的像素

Deliverables:
- mandelbrot_cpu_vector_ilp
- mandelbrot_gpu_vector_ilp + launch function
- GPU 保持 <<<1, 32>>> 配置

考慮因素:
1. 狀態變數如何處理？
2. 控制流如何處理？
3. CPU vs GPU 的控制流策略差異？
4. 同時處理幾個向量？
5. 從影像的哪裡取向量？(行? 2D tile?)

工具: #pragma unroll - 展開迴圈

### Question 1: ILP 加速比？策略？控制流處理？處理幾個向量？限制因素？

## Part 2: Multi-Core Parallelism

### CPU Multi-Core

Deliverable: mandelbrot_cpu_vector_multicore
- 使用 pthread API
- 產生 8 個 threads (每個 core 一個)
- 同步完成

工具: pthread_create(), pthread_join()

### Question 2: 8 cores 的加速比？work partitioning 策略影響？

### GPU Multi-Core

Deliverable: mandelbrot_gpu_vector_multicore + launch
- 目標: 192 個 warp schedulers 各跑一個 warp
- Launch config: <<<48, 4*32>>>
  - 48 blocks (每個 SM 一個)
  - 128 CUDA threads/block (4 warps)

關鍵變數:
- threadIdx.x - 在 block 內的索引
- blockIdx.x - block 索引
- gridDim.x - 總 blocks 數
- blockDim.x - 每 block 的 threads 數

Questions:
3. 192 warp schedulers 的加速比？與 CPU 比較？partitioning 策略？
4. 試試 <<<96, 2*32>>> 和 <<<24, 8*32>>>，效能差異？

## Part 3: Multi-Threaded Parallelism

### CPU Multi-Threaded

Deliverable: mandelbrot_cpu_vector_multicore_multithread
- 每個 core 產生 > 1 個 thread

Question 5: 加速比？最佳 thread 數？決定因素？

### GPU Multi-Threaded

Deliverable 1: mandelbrot_cpu_vector_multicore_multithread_single_sm
- 單一 block，多個 warps (最多 32)

### Question 6: 超過 4 warps 後效能如何變化？到 32 warps 持續改善嗎？

Deliverable 2: mandelbrot_gpu_vector_multicore_multithread_full
- 全規模多 warps、多 blocks

### Question 7: 加速比？最佳 warp 數？決定因素？

## Part 4: 組合所有技術

Deliverables:
- mandelbrot_cpu_vector_multicore_multithread_ilp
- mandelbrot_gpu_vector_multicore_multithread_full_ilp

### Question 8: ILP + multi-core + multi-thread 的加速比？與單 thread ILP 比較？最佳參數？

---
## FMA 延遲和 Dependent Chains 詳解

### 1. FMA (Fused Multiply-Add) 是什麼？

單一指令完成兩個運算

```cpp
// 傳統方式（兩個指令）
temp = a * b;      // 乘法
result = temp + c; // 加法

// FMA 方式（一個指令）
result = a * b + c;  // 一次完成！
```

優勢：
- 更快（一個指令 vs 兩個）
- 更精確（中間結果不會 round）
- 省能源

---
### 2. FMA 延遲 (Latency) 是什麼？

延遲 = 從指令開始到結果可用的時間

```text
Cycle 0: 開始執行 FMA
Cycle 1: 內部計算中...
Cycle 2: 內部計算中...
Cycle 3: 內部計算中...
Cycle 4: 結果可用！ ✓
```

FMA Latency = 4 cycles

#### 重要區別：Latency vs Throughput

| 概念       | 意義             | 例子                |
|------------|------------------|---------------------|
| Latency    | 單一指令完成時間 | 4 cycles            |
| Throughput | 每 cycle 可以發射多少指令 | 1 instruction/cycle |

類比：
- Latency = 洗衣機洗一次衣服需要 30 分鐘
- Throughput = 如果有 4 台洗衣機，每 7.5 分鐘可以開始洗一批

---
### 3. Dependent Chain 是什麼？

定義：每個指令都依賴前一個的結果

```cpp
x = 1.0f;
x = x * x + x;  // 指令 1
x = x * x + x;  // 指令 2（等待指令 1）
x = x * x + x;  // 指令 3（等待指令 2）
x = x * x + x;  // 指令 4（等待指令 3）
```

時間線圖

```text
指令 1:  [████]........    (cycles 0-3)
            ↓ (must wait)
指令 2:  ....[████].....   (cycles 4-7)
               ↓ (must wait)
指令 3:  ........[████]..  (cycles 8-11)
                  ↓ (must wait)
指令 4:  ............[████] (cycles 12-15)
```

總時間 = 16 cycles（sequential）

關鍵問題：指令 2 無法在 cycle 1 開始，因為它需要指令 1 的結果！

---
### 4. Independent Chains

定義：指令之間沒有依賴關係

```cpp
x = 1.0f;
y = 2.0f;  // 獨立的變數！

x = x * x + x;  // Chain 1
y = y * y + y;  // Chain 2（不依賴 x！）

x = x * x + x;  // Chain 1
y = y * y + y;  // Chain 2
```

時間線圖（並行執行）

```text
Chain 1:
指令 1: [████]........    (cycles 0-3)
           ↓
指令 2: ....[████].....   (cycles 4-7)

Chain 2:（同時進行！）
指令 1: [████]........    (cycles 0-3) ← 同時！
           ↓
指令 2: ....[████].....   (cycles 4-7) ← 同時！
```

總時間 = 8 cycles（但完成了 2x 工作！）

---
### 5. 為什麼 Dependent Chain 慢？

問題：Pipeline Stall（管線停頓）

硬體視角：

```text
Cycle 0: 發射 FMA #1
Cycle 1: FMA #1 計算中... FMA #2 想發射但不行！（需要 #1 的結果）
Cycle 2: FMA #1 計算中... FMA #2 還在等...
Cycle 3: FMA #1 計算中... FMA #2 還在等...
Cycle 4: FMA #1 完成 ✓    FMA #2 現在可以發射了
Cycle 5: FMA #2 計算中...
...
```

浪費的 cycles：Cycle 1-3，硬體閒置！

---
### 6. ILP 如何解決？

Instruction-Level Parallelism

雖然有4個chains, 但實際上真的有平行處理嗎?

答案：沒有真正的平行處理！

在 `mandelbrot_cpu_vector_ilp` 中：
- ✅ 有 4 個獨立的 chains
- ❌ 並沒有 4 個 CPU cores 同時執行
- ❌ 並沒有 4 個執行單元真正「平行」運作

那為什麼還能加速？關鍵：Pipeline Parallelism

是 Instruction Pipelining，不是 Multi-threading

單一 CPU core 內部有 pipeline：

```text
Clock 0: [Fetch chain0]
Clock 1: [Decode chain0] [Fetch chain1]
Clock 2: [Execute chain0] [Decode chain1] [Fetch chain2]
Clock 3: [WriteBack chain0] [Execute chain1] [Decode chain2] [Fetch chain3]
Clock 4: [WriteBack chain1] [Execute chain2] [Decode chain3] [Fetch chain0_next]
         ↑                  ↑                ↑                ↑
         4 個指令同時在 pipeline 的不同階段！
```

關鍵：
- 只有 1 個 CPU core 在執行
- 但因為指令是獨立的，CPU 可以「重疊」執行它們，每個 cycle 都在做有用的工作，沒有浪費

---
### 7. 實際例子：Mandelbrot

原始版本（Dependent Chain）

```cpp
float x = 0, y = 0;
for (int iter = 0; iter < max_iters; iter++) {
    float x_new = x*x - y*y + cx;
    float y_new = 2*x*y + cy;
    x = x_new;
    y = y_new;

    if (x*x + y*y > 4.0f) break;
}
```

問題：每次迭代都依賴前一次的 x 和 y

ILP 優化版本（4 個獨立 pixels）

```cpp
// 同時計算 4 個 pixels
float x[4] = {0, 0, 0, 0};
float y[4] = {0, 0, 0, 0};
float cx[4] = {...};  // 4 個不同的 pixels

for (int iter = 0; iter < max_iters; iter++) {
    #pragma unroll
    for (int p = 0; p < 4; p++) {
        float x_new = x[p]*x[p] - y[p]*y[p] + cx[p];
        float y_new = 2*x[p]*y[p] + cy[p];
        x[p] = x_new;
        y[p] = y_new;
    }
}
```

優勢：4 個 pixels 的計算可以並行！

---
### 8. 視覺化對比

Dependent Chain（慢）- 必須等待：

```text
時間 →
Cycle: 0   4   8   12  16  20  24  28  32

Pixel 1: [████]
              └──等待──┐
                       [████]
                            └──等待──┐
                                     [████]
                                          └──等待──┐
                                                   [████]
                                                        ↑
                                            總共 16 cycles 完成 4 次迭代
```

問題：每次計算都要等前一次完成，CPU 大部分時間在閒置。

---
ILP with 4 Chains（快）- Pipeline 重疊：

```text
時間 →
Cycle: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19

Pixel 1: [████]    [████]    [████]    [████]
            └───────┴─────────┴─────────┴── 4 次迭代完成

Pixel 2:  [████]    [████]    [████]    [████]
Pixel 3:   [████]    [████]    [████]    [████]
Pixel 4:    [████]    [████]    [████]    [████]
            ↑
     16 cycles 完成 16 次迭代（4 pixels × 4 迭代）！
```

實際上是「交錯執行」，不是「同時執行」

注意：
- Pixel 1, 2, 3, 4 的指令交錯在 pipeline 中
- 不是 4 個 cores 同時執行 4 個 pixels
- 單一 core 快速切換，讓 pipeline 保持滿載

---
更精確的 Pipeline 階段圖

假設 FMA 指令有 4 個 pipeline 階段：

```text
時間 →
Cycle:  0   1   2   3   4   5   6   7   8

階段:
Fetch   [P1][P2][P3][P4][P1][P2][P3][P4][P1]...
Decode      [P1][P2][P3][P4][P1][P2][P3][P4]...
Execute         [P1][P2][P3][P4][P1][P2][P3]...
Write               [P1][P2][P3][P4][P1][P2]...
                        ↑
            Cycle 4: P1 的第 1 次迭代完成
                     同時 P4 剛進入 Fetch
```

每個 cycle 有 4 條指令在 pipeline 的不同階段

這才是真相：
- 單一 core 一次只 Fetch 一條指令
- 但因為 4 個 pixels 是獨立的，可以填滿 pipeline
- 看起來像「並行」，實際是「重疊」

---
對比：Multi-Core 真正的並行

Part 2: Multi-Core（8 cores 真正同時）

```text
時間 →
Cycle: 0   4   8   12  16

Core 0 (Pixel 0-15):     [████][████][████][████]
Core 1 (Pixel 16-31):    [████][████][████][████]  } 這些是
Core 2 (Pixel 32-47):    [████][████][████][████]  } 物理上
Core 3 (Pixel 48-63):    [████][████][████][████]  } 同時
Core 4 (Pixel 64-79):    [████][████][████][████]  } 執行的
Core 5 (Pixel 80-95):    [████][████][████][████]  }
Core 6 (Pixel 96-111):   [████][████][████][████]  }
Core 7 (Pixel 112-127):  [████][████][████][████]  }
                          ↑
            所有 cores 在 Cycle 0 同時開始
            這才是真正的「並行」！
```

---
用高速公路比喻

Dependent Chain = 單車道，車子排隊：

```text
→ [🚗]──等待──[🚗]──等待──[🚗]──等待──[🚗]

每輛車必須等前車完全通過才能開始

ILP = 單車道，但有多個收費亭階段：

收費亭 1    收費亭 2    收費亭 3    收費亭 4
   ↓           ↓           ↓           ↓
  [🚗P1]   →  [🚗P2]   →  [🚗P3]   →  [🚗P4]

下一秒：
  [🚗P2]   →  [🚗P3]   →  [🚗P4]   →  [🚗P1-next]

每個收費亭都有車（pipeline 滿載），但仍是單車道！

Multi-Core = 8 條真正的平行車道：

車道 1: [🚗][🚗][🚗][🚗]
車道 2: [🚗][🚗][🚗][🚗]  } 真正同時
車道 3: [🚗][🚗][🚗][🚗]  } 在不同
車道 4: [🚗][🚗][🚗][🚗]  } 車道上
車道 5: [🚗][🚗][🚗][🚗]
車道 6: [🚗][🚗][🚗][🚗]
車道 7: [🚗][🚗][🚗][🚗]
車道 8: [🚗][🚗][🚗][🚗]

這才是真正的並行！
```

---
### CPU 執行時實際發生的事

#### Part 1 (ILP) - 單一 Thread：

偽代碼展示 CPU 實際執行順序

```
while True:
    # CPU 快速循環處理 4 個 chains（但仍是序列）
    issue_instruction(pixel1, "v_x2[0] = ...")  # Cycle 0
    issue_instruction(pixel2, "v_x2[1] = ...")  # Cycle 1
    issue_instruction(pixel3, "v_x2[2] = ...")  # Cycle 2
    issue_instruction(pixel4, "v_x2[3] = ...")  # Cycle 3
    # Cycle 4: pixel1 的指令完成，可以繼續
    issue_instruction(pixel1, "v_y2[0] = ...")  # Cycle 4
    issue_instruction(pixel2, "v_y2[1] = ...")  # Cycle 5
    ...
```

關鍵：
- CPU 一次只發射 1 條指令（或少數幾條）
- 但因為獨立，不用等前一條完成
- 透過快速切換保持 pipeline 忙碌

#### Part 2 (Multi-Core) - 8 Threads：

```cpp
# 8 個 threads 真正同時執行在不同 cores
Core0.execute(thread0)  # 處理 rows 0-127
Core1.execute(thread1)  # 處理 rows 128-255  } 同一時刻
Core2.execute(thread2)  # 處理 rows 256-383  } 所有 cores
Core3.execute(thread3)  # 處理 rows 384-511  } 都在工作
Core4.execute(thread4)  # 處理 rows 512-639
Core5.execute(thread5)  # 處理 rows 640-767
Core6.execute(thread6)  # 處理 rows 768-895
Core7.execute(thread7)  # 處理 rows 896-1023
```

---
"4 個 chains 交錯處理"
"4 個 pixels 的指令在 pipeline 中重疊執行"
"透過 instruction-level parallelism 隱藏延遲"

---
為什麼還叫 "Parallelism"？

雖然不是真正的 multi-core 並行，但從指令層級看確實是「平行的」：

時間點 T：
```text
Pipeline Stage 1: 正在 fetch pixel1 的指令
Pipeline Stage 2: 正在 decode pixel2 的指令  } 同時發生
Pipeline Stage 3: 正在 execute pixel3 的指令 } 在不同
Pipeline Stage 4: 正在 writeback pixel4 的指令 } 階段
```

所以叫 Instruction-Level Parallelism：
- "Instruction-Level"：指令層級的（不是 thread 層級）
- "Parallelism"：pipeline 中有多條指令同時在不同階段

---
總結：三種不同的「並行」

| 類型                | 真正並行？ | 實際機制               | 比喻       |
|---------------------|-------|-----------------------------|------------|
| ILP (Part 1)        | ❌ 否   | 單一 core 的 pipeline 重疊       | 單車道的收費站流水線 |
| SIMD                | ✅ 是   | Vector unit 一次算 16 個 floats | 16 車道同時收費  |
| Multi-Core (Part 2) | ✅ 是   | 8 個物理 cores 同時執行            | 8 條獨立高速公路  |

Part 1 的加速來源：
- 不是因為「並行」
- 是因為「不浪費」
- 讓單一 core 的 pipeline 保持滿載

Part 2/3 的加速來源：
- 真正的並行
- 8 個 cores 同時工作
- 理論上可以接近 8x 加速

---
### 9. 關鍵公式

Dependent Chain

```text
時間 = N × Latency
例：100 FMAs × 4 cycles = 400 cycles
```

Independent Chains (ILP)

```text
時間 = N × Latency / min(ILP_width, Hardware_width)
例：400 FMAs × 4 cycles / 4 chains = 400 cycles
    相同時間完成 4x 工作量！（4x throughput）
```

---
### 10. 總結

| 概念    | Dependent Chain | Independent Chains (ILP) |
|-------|-----------------|--------------------------|
| 定義    | 每個指令依賴前一個       | 多個獨立指令流                  |
| 執行    | Sequential（序列）  | Parallel（並行）             |
| 延遲    | 暴露 full latency | 隱藏 latency               |
| 吞吐量   | 低               | 高（N×）                    |
| 硬體利用率 | 低（閒置）           | 高（忙碌）                    |

核心洞察：
- Latency is unavoidable（延遲無法避免）
- But can be hidden with ILP（但可以用 ILP 隱藏）
- Key: Find independent work（關鍵：找到獨立的工作）

這就是為什麼 Part 1 要同時處理多個 vectors/pixels——不是為了降低延遲，而是為了提高吞吐量！

---
## Q & A

### 2 FLOPs 是只要花費2個指令操作時間?

> 不完全是。

- 2 FLOPs 是「運算量」(一次乘 + 一次加)，不是「兩個指令時段」或「兩個 cycles」。
- FMA/FFMA 用「一條指令」完成 2 FLOPs。這條指令有自己的延遲（L 個 cycles 才能被下一條相依指令使用），也有吞吐率（每個 cycle 可發出多少條 FFMA）。
- 在你的程式中，每次迭代就是一條 FFMA；因此 (end-start)/100 ≈ 單條 FFMA 的延遲 L。雖然每迭代做了 2 FLOPs，但不代表用了 2 個 cycles。
- 若未融合成 FMA，而是分開的乘法+加法，才會變成兩條指令（通常更慢、延遲也更長）。
- 性能觀念拆分：
    - 延遲：單條相依鏈耗時 ≈ 次數 × FFMA 延遲。
    - 吞吐：在足夠 ILP/多 warp 下，每 cycle 可發出多條 FFMA；峰值 FLOPs/s ≈ 發出率 × 2（因為每條 FFMA = 2 FLOPs）。

---
### FMA可以算是執行一個指令的時間包含延遲, 是嗎?

> 不完全正確。更精確地說：

- FMA 是「一條指令」（FFMA），而「延遲」是這條指令的屬性之一：從指令被發射到其結果可被下一條相依指令使用，需要 L 個 cycles（指令延遲）。
- 指令的「發射/佔用 issue slot」通常只佔 1 個 cycle；但結果要在後面的 L 個 cycles 才會可用。期間可以繼續發射其他獨立的指令（管線重疊）。
- 因此在管線化處理器上，不能把「一條指令的執行時間」簡化為「延遲」。延遲是結果可用的等待時間；而吞吐描述的是每個 cycle 能發出多少條指令，兩者可同時存在：高吞吐但仍有固定延遲。
- 在你的程式中，單一相依鏈測到的 (end-start)/100 近似 FFMA 的延遲 L；交錯兩條獨立鏈則展示高吞吐可重疊延遲，不代表延遲消失。

---
| 版本                        | Loop 次數 | 每次迭代的 FFMA | 總 FFMA 數 | 測量時間        |
|---------------------------|---------|------------|----------|-------------|
| fma_latency               | 100     | 1          | 100      | 1135 cycles |
| fma_latency_interleaved   | 100     | 2          | 200      | 962 cycles  |
| fma_latency_no_interleave | 100+100 | 1+1        | 200      | 916 cycles  |

關鍵：
- Interleaved 版本執行了 200 個 FFMA
- 但因為並行執行，時間只比 100 個 FFMA 多一點點
- 這證明了 ILP (Instruction-Level Parallelism) 的效果！

---
## ILP 重新理解

---
### ILP vs Multi-Core Parallelism

Part 1 (ILP) - 單一 Thread：

```cpp
// mandelbrot_cpu_vector_ilp
void mandelbrot_cpu_vector_ilp(...) {
    // 只有 1 個 thread 執行
    for (...) {
        for (int chain = 0; chain < 4; chain++) {
            // 這 4 個 chains 在同一個 CPU core 上
            // 透過 pipeline 重疊執行
            v_x2[chain] = ...;  // ← 獨立的指令
        }
    }
}
```

執行模型：
```text
CPU Core 0: 執行所有 4 個 chains 的指令
	            但因為獨立，可以 pipeline 重疊

CPU Core 1-7: 閒置 😴
```

Part 2 (Multi-Core) - 真正平行：

```cpp
// mandelbrot_cpu_vector_multicore
void mandelbrot_cpu_vector_multicore(...) {
    // 產生 8 個 threads
    pthread_create(&thread0, ...);  // Core 0
    pthread_create(&thread1, ...);  // Core 1
    pthread_create(&thread2, ...);  // Core 2
    // ...
    pthread_create(&thread7, ...);  // Core 7

    // 8 個 cores 真正同時執行！
}
```

執行模型：
```text
CPU Core 0: Thread 0 處理 rows 0-127
CPU Core 1: Thread 1 處理 rows 128-255  } 同時執行！
CPU Core 2: Thread 2 處理 rows 256-383
...
CPU Core 7: Thread 7 處理 rows 896-1023
```

---
### 視覺化對比

Part 1 (ILP) - Pipeline 重疊：

```text
時間 →

Single Core:
[C0_inst1][C1_inst1][C2_inst1][C3_inst1][C0_inst2][C1_inst2]...
    ↑        ↑        ↑        ↑
    同一個 core，但指令重疊在 pipeline 中
```

實際上是「序列」執行，但因為 pipeline 重疊，看起來像「並行」

Part 2 (Multi-Core) - 真正平行：

```text
時間 →

Core 0: [Thread0_work.........................]
Core 1: [Thread1_work.........................]  } 真正同時
Core 2: [Thread2_work.........................]
...
Core 7: [Thread7_work.........................]
```

8 個 cores 物理上同時執行不同的工作

---
為什麼 ILP 能加速？

CPU Pipeline 如何處理？

現代 CPU 的 Out-of-Order Execution：

```cpp
// 程式碼
for (int chain = 0; chain < 4; chain++) {
    v_x2[chain] = _mm512_mul_ps(v_x[chain], v_x[chain]);
}

// CPU 看到的（簡化）：
VMULPS zmm0, zmm0, zmm0  // chain 0: v_x2[0] = v_x[0] * v_x[0]
VMULPS zmm1, zmm1, zmm1  // chain 1: v_x2[1] = v_x[1] * v_x[1]
VMULPS zmm2, zmm2, zmm2  // chain 2: v_x2[2] = v_x[2] * v_x[2]
VMULPS zmm3, zmm3, zmm3  // chain 3: v_x2[3] = v_x[3] * v_x[3]
```

CPU 分析：
- 這 4 個指令沒有數據依賴關係
- 可以亂序執行（Out-of-Order）
- 可以同時在不同的 execution units 執行

CPU 內部結構（簡化）：

```text
                 ┌─────────────┐
                 │   Decoder   │
                 └──────┬──────┘
                        │
         ┌──────────────┼──────────────┐
         ↓              ↓              ↓
    [FP Unit 0]    [FP Unit 1]    [FP Unit 2]  ← 多個執行單元
         ↓              ↓              ↓
    zmm0 * zmm0    zmm1 * zmm1    zmm2 * zmm2  ← 同時執行！
```

關鍵：
- 現代 CPU 有多個 Floating-Point Units
- 但這些 units 屬於同一個 core
- 不是 multi-core parallelism！

---
Part 1 (ILP)：

目的：讓單一 thread 充分利用 CPU pipeline 和多個 execution units

限制：
- ❌ 只用 1 個 CPU core
- ❌ 其他 7 個 cores 閒置
- ✅ 但讓這 1 個 core 達到最高效率

加速比：通常 1.5x - 2.5x（相對於單 chain）

---
Part 2 (Multi-Core)：

目的：使用所有 8 個 CPU cores 同時工作

優勢：
- ✅ 8 個 cores 真正並行
- ✅ 每個 core 可以同時用 ILP
- ✅ 組合效果：8 cores × ILP

加速比：理想接近 8x（相對於單 core）

---
Part 3 (Multi-Threading)：

目的：每個 core 跑多個 threads（利用 SMT/Hyper-Threading）

AMD Ryzen 7 7700：
- 8 physical cores
- 每個 core 支援 2 個 hardware threads
- 總共 16 個 logical threads

組合：Multi-core + Multi-threading + ILP

---
回答你的問題

雖然有4個chains, 但實際上真的有平行處理嗎？

答案：
- ❌ 不是真正的「平行執行」（不是 multi-core）
- ✅ 是「pipeline 重疊」（instruction-level parallelism）
- ✅ 仍然能加速，因為隱藏了指令延遲

Part 2 和 Part 3 才是真正的平行：

- Part 1 (ILP)：單一 core 的效率優化
- Part 2 (Multi-Core)：多個 cores 真正並行
- Part 3 (Multi-Threading)：每個 core 多個 threads
- Part 4 (組合)：ILP + Multi-Core + Multi-Threading

---
總結：不同層級的並行

| 層級                | 技術              | 硬體單位                   | 真正平行？ | 加速原理             |
|-------------------|-----------------|------------------------|-------|------------------|
| Instruction-Level | ILP             | 單一 core 的 pipeline     | ❌ 否   | 隱藏延遲             |
| Data-Level        | SIMD            | 單一 core 的 vector units | ✅ 是   | 16 個 floats 同時計算 |
| Thread-Level      | Multi-Core      | 多個 physical cores      | ✅ 是   | 8 個 cores 同時工作   |
| Thread-Level      | Multi-Threading | SMT/Hyper-Threading    | ✅ 部分  | 共享 core 資源       |

Part 1 的 ILP 是「偽平行」：
- 看起來像平行（4 個 chains）
- 實際是序列執行在單一 core
- 但透過 pipeline 重疊獲得加速

Part 2/3 才是「真平行」：
- 多個 cores 物理上同時執行
- 這才是下面 parts 的重點！

---
## Instruction-Level Parallelism 這樣聽起來是平行的, 但實際上也許叫做 Instruction-Level Pipeline 比較貼切

為什麼叫 "Parallelism" 而不是 "Pipelining"？

歷史和學術慣例

在計算機架構領域，這些術語有特定的含義：

1. Pipelining：指令在 pipeline 階段中流動的機制
  - Fetch → Decode → Execute → Write
  - 是一種實現技術
2. Instruction-Level Parallelism (ILP)：多個指令可以同時處於不同執行階段的能力
  - 是一種特性或效果
  - 測量的是「單位時間內完成的指令數」

---
兩種視角

視角 1：從指令的角度（Sequential）

```text
時間 →
Inst 1: [Fetch][Decode][Execute][Write]
Inst 2:        [Fetch] [Decode][Execute][Write]
Inst 3:               [Fetch] [Decode][Execute][Write]
Inst 4:                      [Fetch] [Decode][Execute][Write]
```

這看起來是 Sequential + Pipelining

視角 2：從硬體單元的角度（Parallel）

```text
時間點 T：
Fetch Unit:    正在處理 Inst 4
Decode Unit:   正在處理 Inst 3  } 4 個單元
Execute Unit:  正在處理 Inst 2  } 同時工作
Write Unit:    正在處理 Inst 1  } 這是並行！
```

4 個硬體單元真的在同時工作！

---
ILP 的完整定義

Instruction-Level Parallelism 實際上指的是：

"在單一 thread 內，可以被並行執行的獨立指令數量"

它測量的是：
- 有多少獨立指令可以被發現（compiler/hardware）
- 有多少指令可以同時在執行（Out-of-Order）

Pipelining 是實現 ILP 的一種方式

```text
ILP (概念) ─┬─> Pipelining (基礎技術)
            ├─> Superscalar (多條 pipeline)
            ├─> Out-of-Order Execution (亂序執行)
            ├─> Speculative Execution (推測執行)
            └─> Multiple Issue (同時發射多條指令)
```

---
在 Mandelbrot 例子中：

```cpp
for (int chain = 0; chain < 4; chain++) {
    v_x2[chain] = _mm512_mul_ps(v_x[chain], v_x[chain]);
}
```

這 4 條指令：
- 確實透過 Pipelining 重疊執行
- 確實不是「4 個 threads 同時執行」
- 確實是在單一 core 上序列發射

所以你說的 "Instruction-Level Pipeline" 確實更直觀！

---
但為什麼學術界堅持叫 "Parallelism"？

原因 1：多個執行單元真的在並行

現代 CPU 不是只有一條 pipeline，而是：

```text
                 ┌─────────────┐
                 │   Frontend  │
                 └──────┬──────┘
                        │
         ┌──────────────┼──────────────┬──────────────┐
         ↓              ↓              ↓              ↓
    [FP Unit 0]    [FP Unit 1]    [Int Unit 0]  [Load/Store]
         ↓              ↓              ↓              ↓
    zmm0 * zmm0    zmm1 * zmm1    i++            read mem
         ↑              ↑              ↑              ↑
         同一個 cycle，4 個單元真的同時執行不同指令！
```

這確實是並行！ 只是不是 multi-core 那種並行。

原因 2：Out-of-Order Execution

```cpp
// 程式碼順序
v_x2[0] = v_x[0] * v_x[0];  // 1
v_x2[1] = v_x[1] * v_x[1];  // 2
v_x2[2] = v_x[2] * v_x[2];  // 3
v_x2[3] = v_x[3] * v_x[3];  // 4

// CPU 實際執行順序可能是
Cycle 0: 同時發射 1, 2, 3, 4 到不同 execution units
         ↓
這 4 條指令真的在「並行」執行！
```

關鍵：CPU 不是等 1 執行完才執行 2，而是同時執行 1, 2, 3, 4！

---
更準確的術語對比

| 術語           | 描述的是什麼         | 層級   |
|--------------|----------------|------|
| Pipelining   | 指令流水線機制        | 實現技術 |
| Superscalar  | 每 cycle 發射多條指令 | 架構特性 |
| Out-of-Order | 亂序執行機制         | 實現技術 |
| ILP          | 程式中可被並行執行的指令數  | 程式特性 |

實際上是這樣：

```text
你的程式有 ILP 潛力（4 個獨立 chains）
        ↓
CPU 用 Pipelining 來利用這個 ILP
        ↓
CPU 用 Superscalar 來加強 ILP（多個 execution units）
        ↓
CPU 用 Out-of-Order 來最大化 ILP
```

---
學術界的視角：

```text
1. ILP 測量的是程式特性，不是硬體技術
這段程式的 ILP = 4  （有 4 個獨立指令）
這段程式的 ILP = 1  （全是 dependent chain）
2. 從結果看確實是並行
時間 T：4 條指令在不同 execution units 同時執行
時間 T+1：另外 4 條指令在不同 units 同時執行

這不就是並行嗎？
3. 區分不同層級的並行
Instruction-Level Parallelism  (單 thread 內)
Thread-Level Parallelism       (多 threads)
Data-Level Parallelism         (SIMD)
```

---
## Part 2 GPU Multi-Core 詳細解說

目標: 在 GPU 上運行 one warp on each of the 192 warp schedulers

類比 CPU：
- CPU: 8 cores → 8 threads
- GPU: 192 warp schedulers → 192 warps

CUDA 的兩層架構

硬體架構

```text
GPU (NVIDIA RTX 4000 Ada)
├── 48 SMs (Streaming Multiprocessors)
│   ├── SM 0: 4 warp schedulers
│   ├── SM 1: 4 warp schedulers
│   └── ...
│   └── SM 47: 4 warp schedulers
Total: 48 × 4 = 192 warp schedulers
```

軟體架構（CUDA）

Kernel Launch: <<<# Blocks, # Threads per Block>>>

關鍵規則

1. CUDA Threads 組織成 Blocks
  - Launch 時指定：block 數量 + 每個 block 的 thread 數量
  - <<<48, 128>>> 表示：48 blocks，每 block 128 threads
2. Block 與 SM 的對應
  - 一個 block 的所有 warps 保證在同一個 SM 上執行
  - 不能跨 SM
  - 一個 SM 可以同時運行多個 blocks（但這不是本 lab 的重點）
3. Warp 的定義
  - 32 個連續的 CUDA threads = 1 warp
  - 例如：128 threads = 128/32 = 4 warps

為什麼不能用 <<<1, 6144>>>？

```text
<<<1, 6144>>>
→ 1 block with 6144 threads
→ 6144/32 = 192 warps
```

問題：所有 192 warps 會被分配到同一個 SM！
- 只用了 1 個 SM（out of 48）
- 其他 47 個 SMs 閒置
- 無法利用全部的 192 warp schedulers

正確的Launch Configuration

<<<48, 4 * 32>>>  // 即 <<<48, 128>>>

分析：
- 48 blocks → 使用全部 48 SMs（每個 SM 1 block）
- 128 threads/block → 128/32 = 4 warps/block
- Total: 48 × 4 = 192 warps（每個 warp scheduler 1 warp）

視覺化

```text
SM 0: Block 0 → 4 warps → 分配給 SM 0 的 4 個 warp schedulers
SM 1: Block 1 → 4 warps → 分配給 SM 1 的 4 個 warp schedulers
...
SM 47: Block 47 → 4 warps → 分配給 SM 47 的 4 個 warp schedulers
```

關鍵 CUDA 變數

1. threadIdx.x

- 範圍：0-127（在 block 內）
- 每個 block 重新計數
- Warp 0: threadIdx.x = 0-31
- Warp 1: threadIdx.x = 32-63
- Warp 2: threadIdx.x = 64-95
- Warp 3: threadIdx.x = 96-127

2. blockIdx.x

- 範圍：0-47
- 當前 block 的編號

3. gridDim.x

- 值：48（total blocks）

4. blockDim.x

- 值：128（threads per block）

計算全局 Warp ID

```cpp
__global__ void mandelbrot_gpu_vector_multicore(...) {
    // 計算全局 warp ID (0-191)
    uint32_t warp_id_in_block = threadIdx.x / 32;  // 0-3
    uint32_t global_warp_id = blockIdx.x * 4 + warp_id_in_block;  // 0-191

    // 計算 lane ID (0-31)
    uint32_t lane_id = threadIdx.x % 32;

    // 每個 warp 處理圖像的一部分
    // 例如：horizontal partitioning
    // warp 0 處理 rows [0, 5)
    // warp 1 處理 rows [5, 10)
    // ...
}
```

### Work Partitioning 策略

選項 1: Horizontal (按行分割)

```cpp
uint32_t rows_per_warp = img_size / 192;
uint32_t start_row = global_warp_id * rows_per_warp;
uint32_t end_row = (global_warp_id == 191) ? img_size : start_row + rows_per_warp;
```

選項 2: Interleaved

```cpp
for (uint32_t i = global_warp_id; i < img_size; i += 192) {
    // 處理第 i 行
}
```

實作範例骨架

```cpp
__global__ void mandelbrot_gpu_vector_multicore(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out
) {
    // 計算 warp 和 lane 的 ID
    uint32_t tid = threadIdx.x;  // 0-127
    uint32_t warp_id_local = tid / 32;  // 0-3
    uint32_t warp_id_global = blockIdx.x * (blockDim.x / 32) + warp_id_local;  // 0-191
    uint32_t lane_id = tid % 32;  // 0-31

    // Work partitioning
    uint32_t rows_per_warp = (img_size + 191) / 192;
    uint32_t start_row = warp_id_global * rows_per_warp;
    uint32_t end_row = min(start_row + rows_per_warp, img_size);

    // 類似 Lab 1 的向量化邏輯，但只處理分配的行
    for (uint32_t i = start_row; i < end_row; ++i) {
        for (uint32_t j = lane_id; j < img_size; j += 32) {
            // Mandelbrot 計算...
        }
    }
}

void launch_mandelbrot_gpu_vector_multicore(...) {
    mandelbrot_gpu_vector_multicore<<<48, 128>>>(...);
}
```

總結

- <<<48, 128>>> = 使用全部 192 warp schedulers
- 每個 warp = Lab 1 的向量化邏輯，但只處理圖像的一部分
- Work partitioning = 決定每個 warp 處理哪些像素

這就是 GPU Multi-Core 實作的核心！
