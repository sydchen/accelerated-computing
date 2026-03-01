# Lab 3：波動模擬 - 記憶體階層最佳化

來源: https://accelerated-computing.academy/fall25/labs/lab3/

## 📋 快速導航

- [核心目標](#核心目標)
- [實驗結構](#實驗結構)
- [詳細學習筆記](#詳細學習筆記)
- [實現參考](#實現參考)
- [學習資源](#學習資源)

---

## 核心目標

### 從計算瓶頸到記憶體瓶頸

本實驗重點在於理解 GPU 記憶體階層並透過共享記憶體技術最佳化記憶體密集型工作負載。

**Lab 1-2 (Mandelbrot)：Compute-Dominated**
- 主要時間花在 ALU 運算
- 記憶體存取時間可忽略

**Lab 3 (Wave Simulation)：Memory-Dominated**
- 每個像素讀取鄰居值（像素間通訊）
- 記憶體存取成為主要瓶頸
- 需要利用 locality 優化記憶體使用

### 學習重點

#### 1️⃣ GPU 記憶體階層理解

| 記憶體層級 | 容量 | 頻寬 | 說明 |
|-----------|------|------|------|
| Global Memory (DRAM) | 20 GB | 360 GB/s | 主存儲 |
| L2 Cache | 48 MB | ~2.5 TB/s | 共享 |
| L1 Cache / Shared Memory | 128 KB/SM | 13.4 TB/s | 每 SM |

#### 2️⃣ 兩種實作策略

- **Naive 方法**：每個 timestep 都從 global memory 讀寫
- **Optimized 方法**：使用 Shared Memory 減少 global memory 存取

#### 3️⃣ 實際應用

- Wave Equation（波動方程）模擬
- 模擬聲波、水波、光波等物理現象
- 經典的 Stencil Computation 問題

---

## 實驗結構

### Part 0：記憶體子系統實證研究

**重點：** 透過 PTX 層級分析瞭解快取行為。

#### 任務 1：測量記憶體延遲 (Prelab Question 1)

**三種 PTX Load 指令：**

| 指令 | Cache 行為 | 測試目標 |
|------|----------|---------|
| `ld.global.ca.f32` | L1 + L2 | 測量 L1 cache 延遲 |
| `ld.global.cg.f32` | 只 L2 | 測量 L2 cache 延遲 |
| `ld.global.cv.f32` | 無快取 | 測量 DRAM 延遲 |

**預期結果：**
- L1 Cache: ~30-40 cycles
- L2 Cache: ~200-300 cycles
- Global Memory (DRAM): ~400-600 cycles

#### 任務 2：Memory Coalescing 效果 (Prelab Question 2)

**Non-Coalesced 存取模式：**
```
每個 lane 負責連續的 x 個元素
Lane 0: [0, 1, 2, ..., x-1]
Lane 1: [x, x+1, ..., 2x-1]
Lane 2: [2x, 2x+1, ..., 3x-1]

→ 記憶體存取跳躍，每個 warp 需要 32 個獨立 transaction
```

**Coalesced 存取模式：**
```
每個 lane 負責間隔 blockDim.x 的元素
Lane 0: [0, blockDim.x, 2*blockDim.x, ...]
Lane 1: [1, blockDim.x+1, 2*blockDim.x+1, ...]

→ 記憶體存取連續，整個 warp 只需 1 個 transaction
```

**預期加速：** Coalesced 通常比 Non-Coalesced 快 **5-10 倍**

---

### Part 1：基礎 GPU 實作

**工作負載：** 2D 域上的波動方程數值模擬。

核心演算法根據以下各項更新每個像素：
- 當前和前一時間步的值
- 正交相鄰像素的值（4 點模板）
- 位置相關的阻尼、源注入和牆壁反射

**實作細節：**
- 每個時間步啟動一個核心，確保正確同步
- 多維執行緒/區塊索引簡化 2D 域映射
- Y-主序陣列配置影響記憶體合併模式
- 小域進行正確性測試；3201×3201 網格執行 12800 時間步進行性能測試

**性能分析（問題 1）：**

計算以下各項：
- 緩衝區大小 vs L2 快取容量（48 MB）
- 每次核心啟動載入的位元組數
- L2 快取未中率和 DRAM 流量
- DRAM/L2 頻寬的理論執行時間下限
- 與實際執行時間的比較

---

### Part 2：共享記憶體最佳化

**策略：** 利用局部性將模擬磁貼載入 SM 本地 SRAM（每個 SM 128 KB），在將結果寫回全域記憶體前計算多個時間步。

**關鍵約束：** 資訊每時間步傳播一個像素，隨著邊緣在多步後變為無效，磁貼中心的「有效區域」逐漸縮小。

**技術實作要求：**

動態共享記憶體分配啟用靈活的緩衝區大小調整：

```cuda
extern __shared__ float shmem[];
// 啟動時帶大小參數：kernel<<<blocks, threads, bytes>>>
```

超過 48 KB 的共享記憶體需透過 `cudaFuncSetAttribute` 明確選擇。

**實作設計決策：**
- 每次核心啟動的時間步數（可調參數）
- 處理非倍數時間步計數
- 面對有效區域縮小時的區塊-磁貼工作分割
- 執行緒-像素映射（初始一對一，可推廣）
- 暫存器 vs 共享記憶體資料儲存
- 邊界外存取防止

**性能測量（問題 2）：**
對比 Part 1 的加速倍數，記錄遭遇的權衡，並討論實作挑戰。

---

## 詳細學習筆記

### PTX 與記憶體延遲測量

#### 什麼是 PTX？

**PTX = Parallel Thread Execution**

- NVIDIA GPU 的虛擬指令集 (Virtual ISA)
- 類似 x86 組合語言，但：
  - 不是真正機器碼，需再編譯成 SASS
  - 跨 GPU 架構可執行
  - 無固定暫存器編號（編譯器分配）

**編譯流程：**
```
CUDA C++ → PTX (虛擬ISA) → SASS (真實機器碼) → GPU 執行
```

#### 在 CUDA 中嵌入 PTX

**基本語法：**

```cpp
asm volatile(
    "ptx指令1;\n\t"
    "ptx指令2;\n\t"
    : 輸出運算元列表
    : 輸入運算元列表
    : 破壞列表
);
```

**運算元約束 (Operand Constraints)：**

| 約束 | 類型 | 說明 | PTX 類型 |
|------|------|------|---------|
| `"r"` | int/unsigned | 32-bit 整數暫存器 | `.u32 / .s32` |
| `"l"` | long | 64-bit 整數暫存器 | `.u64 / .s64` |
| `"f"` | float | 32-bit 浮點暫存器 | `.f32` |
| `"d"` | double | 64-bit 浮點暫存器 | `.f64` |

**修飾符：**
- `"="` - 只寫 (write-only)
- `"+"` - 讀寫 (read-write)
- 無修飾符 - 只讀 (read-only)

#### 簡單範例：加法

```cpp
__global__ void add_kernel() {
    int a = 10, b = 20, c;

    asm volatile(
        "add.s32 %0, %1, %2;\n\t"  // c = a + b
        : "=r"(c)      // %0：輸出
        : "r"(a),      // %1：輸入
          "r"(b)       // %2：輸入
    );

    printf("c = %d\n", c);  // c = 30
}
```

#### PTX 常用指令

**算術指令：**
```ptx
add.s32 %0, %1, %2;        // 有符號 32-bit 加法
sub.f32 %0, %1, %2;        // 浮點減法
mul.lo.s32 %0, %1, %2;     // 乘法 (低 32 位)
div.f32 %0, %1, %2;        // 浮點除法
```

**記憶體指令：**
```ptx
ld.global.ca.f32 %0, [%1];    // 快取所有層級
ld.global.cg.f32 %0, [%1];    // 快取於 L2 only
ld.global.cv.f32 %0, [%1];    // 不快取 (volatile)
ld.shared.f32 %0, [%1];       // 從 shared memory 載入

st.global.f32 [%0], %1;       // 寫入 global memory
st.shared.f32 [%0], %1;       // 寫入 shared memory
```

**資料移動：**
```ptx
mov.u32 %0, %1;                // 32-bit 移動
mov.u64 %0, %%clock64;         // 讀取 clock counter
```

**同步與屏障：**
```ptx
membar.gl;                     // Global memory barrier
membar.cta;                    // Block memory barrier
bar.sync 0;                    // Thread barrier (__syncthreads)
```

#### memory_latency.cu 詳解

**測量流程：**
```
1. Warm-up    : 預熱 cache
2. 計時開始   : mov.u64 %1, %%clock64;
3. Load 指令  : ld.global.<qualifier>.f32 %2, [%0];
4. 計時結束   : mov.u64 %4, %%clock64;
5. 計算延遲   : latency = end_time - start_time - 2
```

**三種 Kernel 實現：**

L1 Cache Latency：
```ptx
"ld.global.ca.f32 %2, [%0];\n\t"
```
- 使用 `.ca` (cache all) qualifier，快取在 L1 和 L2
- 測量 L1 cache hit 延遲

L2 Cache Latency：
```ptx
"ld.global.cg.f32 %2, [%0];\n\t"
```
- 使用 `.cg` (cache global) qualifier，只快取在 L2，bypass L1
- 測量 L2 cache hit 延遲

Global Memory Latency：
```ptx
"ld.global.cv.f32 %1, [%3];\n\t"
```
- 使用 `.cv` (cache volatile) qualifier，不快取
- 測量 DRAM 延遲

**編譯與運行：**
```bash
nvcc -O2 -arch=sm_89 mem-latency.cu -o mem-latency
./mem-latency
```

**預期輸出範例：**
```
global_mem_latency latency =  450 cycles
l2_mem_latency latency =          280 cycles
l1_mem_latency latency =           32 cycles
```

#### Pointer Chasing 技術

**核心概念：** 防止 GPU 並行執行多個 load 操作，通過創建依賴的記憶體訪問鏈：

```cpp
// Pointer chasing - 依賴鏈
addr1 = load(addr0);  // 必須等待完成
addr2 = load(addr1);  // 依賴 addr1，無法並行
addr3 = load(addr2);  // 依賴 addr2
```

**為什麼需要 Pointer Chasing？**

沒有 pointer chasing（錯誤）：
```cpp
// 獨立的 loads - GPU 可並行執行
result1 = load(addr);
result2 = load(addr);
result3 = load(addr);
result4 = load(addr);
```
→ GPU 同時發出所有 load，測量到的只是 1 個 load 的延遲

有 pointer chasing（正確）：
```cpp
// 依賴鏈 - 必須順序執行
ptr = load(ptr);  // 讀取 ptr 指向的值
ptr = load(ptr);  // 必須等待上一個完成
ptr = load(ptr);
```
→ 測量到的是所有 load 的總延遲

**實現方式對比：**

| 方法 | 優點 | 缺點 | 用途 |
|------|------|------|------|
| 真正指針鏈 | 測量真實隨機訪問 | 複雜 | cache miss latency |
| Self-pointing + add | 簡單、同一地址 | 非真正 pointer chase | cache hit latency |
| 獨立 loads | 最簡單 | 無法測量延遲 | ❌ 不適用 |

---

### Memory Coalescing 分析

#### Non-Coalesced 存取模式

```cpp
int base_idx = tid * x;  // tid=0: 0, tid=1: 1024, tid=2: 2048...
for (int i = 0; i < x; i++) {
    dst[base_idx + i] = src[base_idx + i];
}
```

**存取示意：**
```
Thread 0: [0] ← 相距 1024 個元素
Thread 1: [1024]
Thread 2: [2048] ← 不連續
...
Thread 31: [31744]

→ 每個 warp 需要 32 個獨立 transactions → 慢
```

#### Coalesced 存取模式

```cpp
int idx = tid + i * total_threads;  // stride = blockDim.x × gridDim.x
for (int i = 0; i < x; i++) {
    dst[idx] = src[idx];
}
```

**存取示意：**
```
Thread 0: [0] ← 相鄰！
Thread 1: [1]
Thread 2: [2] ← 連續！
...
Thread 31: [31]

→ 整個 warp 只需 1 個 coalesced transaction → 快
```

#### 視覺化對比

**Non-Coalesced（每個 thread 讀連續塊）：**
```
Thread 0: ████████████... (1024 elements)
Thread 1:                 ████████████... (1024 elements)
Thread 2:                                 ████████████...
        ↑ 同一時刻，warp 內的 threads 訪問不連續的地址
```

**Coalesced（每個 thread 讀 strided 元素）：**
```
Thread 0: █   █   █   █   ...
Thread 1:  █   █   █   █   ...
Thread 2:   █   █   █   █   ...
        ↑ 同一時刻，warp 內的 threads 訪問連續的地址
```

#### 預期性能差異

- **理論加速**：coalesced 應比 non-coalesced 快 **5-10 倍**
- **原因**：
  - Non-coalesced: 32 個獨立 transactions/iteration
  - Coalesced: 1 個 coalesced transaction/iteration

#### 關鍵洞察：Cache 效應很重要

> **在 GPU 上，cache 局部性往往比 memory coalescing 更重要**

**Coalescing 優勢場景：**
- ✅ 隨機訪問（沒有空間局部性）
- ✅ 單次訪問（沒有時間重用）
- ✅ Bandwidth-bound（記憶體帶寬是瓶頸）

**測試場景（有 cache 重用）：**
- ✓ 有空間局部性（non-coalesced 是順序的）
- ✓ 有時間重用（cache 可緩存）
- ✗ Compute-bound（每個 thread 做很多工作）

---

### 記憶體訪問優化

#### 優化技巧

**1. 減少暫存器壓力**

原始版本（需額外暫存器）：
```cpp
int idx = tid + i * total_threads;
dst[idx] = src[idx];  // 兩次使用 idx
```

優化版本（內聯計算）：
```cpp
dst[i * total_threads + tid] = src[i * total_threads + tid];
```

→ 編譯器直接內聯，減少暫存器使用，提高 occupancy

**2. `__restrict__` 關鍵字**

```cpp
void kernel(float * __restrict__ dst, const float * __restrict__ src)
```

**作用：**
- 告訴編譯器 `dst` 和 `src` 不會 alias
- 允許編譯器大膽重排序 load/store
- 啟用更激進的優化

**3. `const` 關鍵字**

```cpp
const float * __restrict__ src
```

**作用：**
- 告訴編譯器 `src` 只讀
- 可使用 texture cache 或 read-only cache
- 允許更多 prefetch 優化

#### L1/L2/DRAM 頻寬測量

**Tesla P100 規格：**
- 56 SMs
- L1 總容量：~3.5 MB (每 SM 128 KB)
- L2 容量：4 MB

**測試配置：**

| 層級 | 工作集大小 | 迭代數 | 測量方法 |
|------|----------|--------|---------|
| **L1** | 4 KB | 80,000 | 確保完全在 L1 |
| **L2** | 4 MB | 5,000 | 接近 L2 容量 |
| **DRAM** | 256 MB | 500 | 遠大於 L2 |

**典型結果：**

```
L1 Cache 頻寬：2.08 TB/s  (約 70-100% 理論效率)
L2 Cache 頻寬：1.21 TB/s  (約 40-60% 理論效率)
DRAM 頻寬：     0.49 TB/s  (約 68% 理論效率) ✓
```

#### 優化要點

✅ **有效的改進：**

1. **極小工作集** - 確保資料完全在 L1
2. **多個獨立累加器** - 增加指令級並行度 (ILP)
3. **利用 Dual-Issue** - 地址計算與讀取重疊
4. **每次循環多個讀取** - 最大化並行性

---

## 實現參考

### 記憶體階層詳細資訊

RTX 4000 Ada 包含四層階層：

| 層級 | 容量 | 頻寬 | 說明 |
|------|------|------|------|
| **DRAM** | 20 GB | 360 GB/sec | 5 個 GDDR6 晶片聚合頻寬 |
| **L2 快取** | 48 MB | ~2.5 TB/sec | 共享 |
| **L1 快取 + Scratchpad** | 128 KB/SM | 13.4 TB/sec | 每個 SM 聚合 |
| **暫存器檔案** | 64 KB/warp 排程器 | — | 最快 |

**注意：** L1 快取缺乏跨 SM 一致性，大多數載入略過 L1，透過 L2 路由。

### 關鍵檔案

- `mem-latency.cu`：Part 0 記憶體延遲測量
- `coalesced-loads.cu`：Part 0 合併測試
- `wave-naive.cu`：Part 1 基礎實作
- `wave-shared.cu`：Part 2 共享記憶體最佳化
- `docs/PART1.md`：Part 1 詳細分析筆記
- `docs/PART2.md`：Part 2 詳細分析筆記
- `docs/NOTES.md`：完整學習筆記

---

## 學習資源

### 官方文件與工具

1. **官方 PTX 文件**：https://docs.nvidia.com/cuda/parallel-thread-execution/

2. **查看生成的 PTX：**
```bash
nvcc -ptx your_code.cu -o your_code.ptx
cat your_code.ptx
```

3. **線上 PTX 編譯器**：https://godbolt.org/

4. **記憶體頻寬測試參考**：https://www.evolvebenchmark.com/blog-posts/learning-about-gpus-through-measuring-memory-bandwidth

### 編譯與測試

```bash
# 編譯
nvcc -O2 -arch=sm_60 coalesced-loads.cu -o coalesced-loads

# 運行
./coalesced-loads
```

---

## 🎯 關鍵要點回顧

### PTX Load 指令的三種差異

```ptx
ld.global.ca.f32 %2, [%0];    // L1 測試：快取所有層級
ld.global.cg.f32 %2, [%0];    // L2 測試：只快取 L2
ld.global.cv.f32 %1, [%3];    // DRAM 測試：不快取
```

### Memory Coalescing 核心原則

- **不是**總是更快（考慮 cache 效應）
- **是**在隨機訪問、單次使用時更快
- **需要** warp 內 threads 訪問連續地址

