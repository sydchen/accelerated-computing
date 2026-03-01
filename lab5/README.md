# Lab 5: 矩陣乘法 – 改進的排程

## 課程背景

**MIT 6.S894 - Accelerated Computing (Fall 2025)**

官方課程資源：[Lab 5 - Matrix Multiply – Improved Scheduling](https://accelerated-computing.academy/fall25/labs/lab5/)

Lab 5 專注於透過改進排程和並行化管理來最佳化 GPU 矩陣乘法核心。本實驗強調佔用率、記憶體延遲和硬體利用率之間的關係，並透過 Little's Law 和 Roofline Model 進行性能分析。

## 實驗目標

### 效能目標

對於 3072 × 3072 × 3072 矩陣乘法：

| 實現 | 目標時間 | 實測時間 | 達成率 |
|------|--------|--------|--------|
| **基準**（無最佳化） | - | ~100 ms | - |
| **Lab 4 風格**（Shared Memory） | ~45 ms | 10.10 ms | ✓ |
| **Lab 5 目標**（改進排程） | **< 5 ms** | 9.86 ms | ✓✓ |

---

## Part 0: Little's Law 和 Occupancy 分析

### 核心概念：Little's Law

**Little's Law** 是排隊理論中的基本定律：

$$\text{所需並行度} = \text{延遲} \times \text{吞吐量}$$

在 GPU 記憶體系統的背景下：

**所需資料量 = 記憶體延遲 × 記憶體頻寬**

#### 直覺理解

- 如果記憶體請求需要 800 cycles 才能完成
- 而要達到 360 GB/s 的吞吐量
- 必須同時有足夠的資料在傳輸中來「填滿這條管道」

#### 計算範例（P100）

```
延遲: ~800 cycles
GPU 時脈: 1.328 GHz
DRAM 頻寬: 732 GB/s

所需資料量 = (800 / 1.328×10⁹) × 732×10⁹ bytes
           = 441 KB
```

**結論**：需要約 **441 KB** 的資料同時在傳輸中才能飽和 P100 的 DRAM 頻寬。

### Occupancy 的意義

**Occupancy** = (active warps) / (max warps per SM)

- P100：每個 SM 最多 64 warps
- Occupancy 100% = 64 warps active
- Occupancy 50% = 32 warps active

#### 為什麼 Occupancy 重要？

1. **記憶體延遲隱藏**：當一個 warp 等待記憶體時，其他 warps 可以執行
2. **計算資源利用**：更多 active warps = 更多機會保持 ALUs 忙碌
3. **並行記憶體請求**：更多 warps = 更多同時進行的記憶體傳輸

#### Occupancy 受限於：

1. **Shared Memory**：每個 SM 有固定容量（P100: 64 KB）
2. **Registers**：每個 SM 有固定數量的暫存器
3. **Block 數量**：硬體限制每個 SM 的最大 blocks

### Occupancy 配置實驗（P100）

#### 實驗設計

透過控制 shared memory 使用量，達到不同的 occupancy 水準：

**配置參數**：`{0, 2520, 4096, 9362}` bytes per block

#### 配置對應分析

| Shared Memory | Blocks/SM | Active Warps | Occupancy | 目標 |
|--------------|-----------|--------------|-----------|------|
| 0 bytes | 32 | 64 | 100% | 基準線 |
| 2,520 bytes | 26 | 52 | ~80% | 高 occupancy |
| 4,096 bytes | 16 | 32 | 50% | 中 occupancy |
| 9,362 bytes | 7 | 14 | ~20% | 低 occupancy |

**計算公式**（P100 規格）：
- Shared memory per SM: 65,536 bytes
- Threads per block: 64 = 2 warps
- Max warps per SM: 64

```
Blocks/SM = floor(65,536 / shared_memory_per_block)
Active warps = Blocks/SM × 2
Occupancy = Active warps / 64
```

#### 實驗目的

透過這 4 種配置，觀察：
1. Occupancy 對記憶體頻寬的影響
2. 達到峰值頻寬所需的最小 occupancy
3. Shared memory 使用量如何限制並行度
4. 理論最小值（Little's Law 441 KB）在實務中的表現

### 實驗發現：Occupancy 的倒 U 型效應 ⭐

根據 P100 實測數據，傳統「Occupancy 越高越好」的認知是**錯誤的**！

#### 實測效能曲線

| Shared Memory | Time(ms) | BW(GB/s) | 效率(%) | Occ(%) | Blocks/SM | Bytes in-flight | 備註 |
|--------------|----------|----------|---------|--------|-----------|-----------------|------|
| 0 bytes | 2.528 | 331.8 | 45.3% | 100.0% | 32 | 2.1 MB | **最差** ❌ |
| 2,520 bytes | 1.554 | 539.8 | 73.7% | 78.1% | 26 | 1.6 MB | 優秀 |
| 4,096 bytes | 1.552 | **540.7** | **73.8%** | 50.0% | 16 | **1.0 MB** | **最佳** ⭐ |
| 9,362 bytes | 1.586 | 529.0 | 72.3% | 18.8% | 7 | 384 KB | 仍佳 |

**關鍵發現**：
- ✅ **50-78% occupancy** 達最佳效能（540 GB/s，73.8% 效率）
- ❌ **100% occupancy 表現最差**（331 GB/s，45.3% 效率）
- 差距：50% occupancy 比 100% occupancy **快 63%**！
- **Bytes in-flight** 最佳範圍：1-1.6 MB
  - 384 KB（18.8% occ）：仍達 72.3% 效率 ✓ 驗證 Little's Law
  - 1.0 MB（50% occ）：達最佳 73.8% 效率 ⭐
  - 2.1 MB（100% occ）：反而降至 45.3% 效率 ❌

#### 為什麼 100% Occupancy 表現差？

原因分析：
1. **Cache thrashing**：32 blocks/SM 同時競爭 L1/L2 cache，導致 miss 率高
2. **Memory controller 過度飽和**：過多並行請求導致排隊延遲
3. **Bank conflicts**：更多 warps 增加 shared memory bank conflicts 機率
4. **TLB thrashing**：過多並行存取導致 TLB miss 率上升

#### Little's Law 實驗驗證

理論預測需要 441 KB in-flight，實測對比：

```
18.8% occupancy: 384 KB in-flight → 72.3% 效率 ✓ (接近理論值)
50% occupancy:  1.0 MB in-flight  → 73.8% 效率 ✓ (超過最小值)
78% occupancy:  1.6 MB in-flight  → 73.7% 效率 ✓ (遠超最小值)
100% occupancy: 2.1 MB in-flight  → 45.3% 效率 ❌ (過猶不及)
```

**結論**：Little's Law 預測的 441 KB 是達到峰值的**最小值**，但實務上 1-1.6 MB 是**最佳效能區間**。

---

## Part 1: Lab 4 vs Lab 5 優化對比

### Lab 4 (Register 重用)

- **Block tile**: 64×64
- **K tile**: 8
- **Threads**: 64 (2 warps)
- **Shared memory**: 4 KB
- **__launch_bounds__**: 有 (64)
- **效能**: 10.10 ms (5.74 TFLOP/s)

### Lab 5 改進策略（矩陣乘法 – 改進的排程）

Lab 5 是 Lab 4 的 **4× scale-up 版本**，採用系統性優化：

#### 1. 更大的 Block Tile Size (BM × BN)

- Lab 4: 64×64 → Lab 5: 128×128 ✅ (4倍大)

**影響**：
- 減少 grid-level overhead（blocks 數量減少 4 倍）
- 每個 block 處理更多工作，攤銷 kernel launch 和 scheduling overhead
- 更好的 L2 cache 利用率

#### 2. 更大的 K Tile Size (BK)

- Lab 4: 8 → Lab 5: 32 ✅ (4倍大)

**影響**：
- 更高的 arithmetic intensity：每次載入 tile 後執行更多計算
  - Lab 4：載入 tile → 8 次 outer product
  - Lab 5：載入 tile → 32 次 outer product
- 減少 `__syncthreads()` 次數：384 次 → 96 次（**減少 75%**）
- 更少的 global memory 訪問

#### 3. 更多 Threads per Block

- Lab 4: 64 threads (2 warps) → Lab 5: 256 threads ✅ (8 warps, 4倍大)

**影響**：
- 更好的記憶體 coalescing
- 隱藏延遲能力更強
- 每個 thread 載入量：8 elements → 16 elements

#### 4. 更大的 Shared Memory 使用量

- Lab 4: 4 KB → Lab 5: 32 KB ✅ (8倍大)

**影響**：
- 更激進地使用 on-chip memory
- **針對 P100 優化**：目標 ~50% occupancy 而非 100%（基於 Part 0 發現）
- 32 KB shared memory → 最多 2 blocks/SM (65536/32768=2)

#### 5. 移除 __launch_bounds__ 限制

- Lab 4: 有限制 → Lab 5: 無限制 ✅

**影響**：
- 讓編譯器自由優化 register 分配
- 實驗發現不限制時效能更好（9.85 ms vs 13.03 ms with bounds）

### 對比總結表

| 參數 | Lab 4 | Lab 5 | 改進 |
|------|--------|--------|---------|
| Block tile | 64×64 | 128×128 | 4× 更大 |
| K tile | 8 | 32 | 4× 更大 |
| Threads | 64 (2 warps) | 256 (8 warps) | 4× 更多 |
| Shared memory | 4 KB | 32 KB | 8× 更大 |
| Sync 次數 | 384 次 | 96 次 | 減少 75% |
| Occupancy | 高 (~100%) | 中 (~50%) | 針對 P100 優化 |
| 效能 (3072³) | 10.10 ms | 9.86 ms | 快 2.4% |
| TFLOP/s | 5.74 | 5.88 | +2.4% |

### 核心優化思想

Lab 5 的改進遵循「更大 tile + 更高計算強度」的策略：

1. **擴大工作粒度**：減少 overhead，提高效率
2. **增加 arithmetic intensity**：每次記憶體訪問後做更多計算
3. **針對硬體調優**：根據 P100 特性選擇最佳 occupancy 而非盲目追求 100%

---

## Part 2: 性能分析與 Roofline Model

### 分析方法論

對每個問題規模計算以下指標：

1. **Total FLOPs**: `2 × size_i × size_j × size_k`
2. **Compute-bound 最小時間**: `FLOPs / (9.3 × 10¹²)` seconds
3. **Unique bytes**: `size_i×size_k + size_k×size_j + size_i×size_j`
4. **Bandwidth-bound 最小時間**: `bytes / (732 × 10⁹)` seconds
5. **Combined lower bound**: `max(compute_bound, bandwidth_bound)`
6. **分類**: Compute-bound 或 Bandwidth-bound

### 主要測試案例（P100）

#### 案例 1：3072 × 3072 × 3072

```
Total FLOPs: 5.80 × 10¹⁰ (58.0 GFLOP)
Compute-bound time: 6.24 ms
Unique bytes: 113.2 MB
Bandwidth-bound time: 0.155 ms

下限: 6.24 ms (Compute-bound)
分類: Compute-bound
最大 TFLOP/s: 9.3 (理論峰值)
Threadblocks: 24×24 = 576 (10.3 blocks/SM) ✓

實測效能:
- matmul_improved: 9.86 ms → 5.88 TFLOP/s (63% of peak)
- matmul_improved_reduce: 9.88 ms → 5.87 TFLOP/s (63% of peak)
```

#### 案例 2：1×3072 (極端小矩陣)

```
Total FLOPs: 1.89 × 10⁷ (0.0189 GFLOP)
Compute-bound time: 0.002 ms
Unique bytes: 24.6 MB
Bandwidth-bound time: 0.034 ms

下限: 0.034 ms (Bandwidth-bound)
分類: Bandwidth-bound
最大 TFLOP/s: 0.56
Threadblocks: 1×24 = 24 (0.43 blocks/SM) ← 嚴重不足

優化方案: Split-K
- split_k=7 → 168 blocks (3.0 blocks/SM) ✓
- 實測加速: 2.6× (27.08 ms → 10.38 ms)
```

### Split-K 策略

對於輸出 tile 不足以充分利用 SM 的問題大小（如小矩陣）：

#### 實現方式

```
Kernel 1: 沿 K 維度分割，計算部分和
  - 輸出: workspace[split_k_idx * size_i * size_j + ...]

Kernel 2: 規約，合併所有 split_k 分片的結果
  - 輸出: C[i,j] = sum(workspace[k][i,j] for k in split_k)
```

#### Split-K 策略決策

```
標準方法:     split_k=1, grid: 4×4 blocks    (16 blocks)
Split-K 方法: split_k=7, grid: 4×4×7 blocks (112 blocks)
```

### 性能分析總結

#### 大矩陣（充分並行度）

| 規模 | Blocks/SM | Occupancy | 效能(TFLOP/s) | % Peak |
|------|----------|-----------|--------------|---------|
| 3072³ | 10.3 | 中 (~50%) | 5.88 | 63% |
| 512×3072² | 1.8 | 低 | 4.89 | 53% |

#### 小矩陣（Split-K 優化）

| 規模 | split_k | Blocks/SM | 加速 | 新效能 |
|------|---------|----------|------|--------|
| 1×3072 | 7 | 3.0 | 2.6× | 5.89 TFLOP/s |
| 16×3072 | 11 | 4.4 | 8.6× | 5.64 TFLOP/s |
| 256×256 | 32 | 2.3 | 27× | 0.48 TFLOP/s |

### 關鍵洞察

1. **Compute-bound vs Bandwidth-bound**
   - 大矩陣：計算密集，受計算能力限制
   - 小矩陣：記憶體密集，需要 Split-K 來增加並行度

2. **Occupancy 的實務應用**
   - 50-78% 是黃金區間，不要盲目追求 100%
   - Little's Law 提供下限，但實務中需要留有 buffer

3. **Split-K 的價值**
   - 將平行度不足的小矩陣問題轉化為充分利用 GPU 資源的計算
   - 在 128²×32768 等極端案例中，將 SM 使用率從 1.8% 提升至 57%

