# Lab 5: Matrix Multiply – Improved Scheduling

## Course Background

**MIT 6.S894 - Accelerated Computing (Fall 2025)**

Official Course Resource: [Lab 5 - Matrix Multiply – Improved Scheduling](https://accelerated-computing.academy/fall25/labs/lab5/)

Lab 5 focuses on optimizing GPU matrix multiplication kernels through improved scheduling and parallelism management. This lab emphasizes the relationship between occupancy, memory latency, and hardware utilization, analyzing performance through Little's Law and the Roofline Model.

## Lab Objectives

### Performance Goals

For 3072 × 3072 × 3072 matrix multiplication:

| Implementation | Target Time | Measured Time | Achievement |
|------|--------|--------|--------|
| **Baseline** (No optimization) | - | ~100 ms | - |
| **Lab 4 Style** (Shared Memory) | ~45 ms | 10.10 ms | ✓ |
| **Lab 5 Target** (Improved Scheduling) | **< 5 ms** | 9.86 ms | ✓✓ |

---

## Part 0: Little's Law and Occupancy Analysis

### Core Concept: Little's Law

**Little's Law** is a fundamental principle from queuing theory:

$$\text{Required Parallelism} = \text{Latency} \times \text{Throughput}$$

In the context of GPU memory systems:

**Required Data In-Flight = Memory Latency × Memory Bandwidth**

#### Intuitive Explanation

- If a memory request takes ~800 cycles to complete
- And we want to achieve 360 GB/s throughput
- We must have enough data in-flight to "fill the pipe"

#### Calculation Example (P100)

```
Latency: ~800 cycles
GPU Clock: 1.328 GHz
DRAM Bandwidth: 732 GB/s

Required Data = (800 / 1.328×10⁹) × 732×10⁹ bytes
              = 441 KB
```

**Conclusion**: Approximately **441 KB** of data must be in-flight simultaneously to saturate P100's DRAM bandwidth.

### Occupancy Definition

**Occupancy** = (active warps) / (max warps per SM)

- P100: Maximum 64 warps per SM
- 100% occupancy = 64 warps active
- 50% occupancy = 32 warps active

#### Why Occupancy Matters

1. **Memory latency hiding**: When one warp waits for memory, others can execute
2. **Compute resource utilization**: More active warps = more chance to keep ALUs busy
3. **Parallel memory requests**: More warps = more simultaneous memory transfers

#### Occupancy Constraints

1. **Shared Memory**: Each SM has fixed capacity (P100: 64 KB)
2. **Registers**: Each SM has a fixed number of registers
3. **Block count**: Hardware limits blocks per SM

### Occupancy Configuration Experiments (P100)

#### Experimental Design

By controlling shared memory usage, we achieve different occupancy levels:

**Configuration Parameters**: `{0, 2520, 4096, 9362}` bytes per block

#### Configuration Analysis

| Shared Memory | Blocks/SM | Active Warps | Occupancy | Target |
|--------------|-----------|--------------|-----------|------|
| 0 bytes | 32 | 64 | 100% | Baseline |
| 2,520 bytes | 26 | 52 | ~80% | High occupancy |
| 4,096 bytes | 16 | 32 | 50% | Medium occupancy |
| 9,362 bytes | 7 | 14 | ~20% | Low occupancy |

**Calculation Formula** (P100 Specs):
- Shared memory per SM: 65,536 bytes
- Threads per block: 64 = 2 warps
- Max warps per SM: 64

```
Blocks/SM = floor(65,536 / shared_memory_per_block)
Active warps = Blocks/SM × 2
Occupancy = Active warps / 64
```

#### Experimental Purpose

Through these 4 configurations, we observe:
1. Impact of occupancy on memory bandwidth
2. Minimum occupancy needed to reach peak bandwidth
3. How shared memory usage limits parallelism
4. Practical validation of Little's Law's 441 KB minimum

### Measured Performance Curve

| Shared Memory | Time(ms) | BW(GB/s) | Efficiency(%) | Occ(%) | Blocks/SM | Bytes In-Flight | Remark |
|--------------|----------|----------|---------|--------|-----------|-----------------|------|
| 0 bytes | 2.528 | 331.8 | 45.3% | 100.0% | 32 | 2.1 MB | **Worst** |
| 2,520 bytes | 1.554 | 539.8 | 73.7% | 78.1% | 26 | 1.6 MB | Excellent |
| 4,096 bytes | 1.552 | **540.7** | **73.8%** | 50.0% | 16 | **1.0 MB** | **Best** ⭐ |
| 9,362 bytes | 1.586 | 529.0 | 72.3% | 18.8% | 7 | 384 KB | Still Good |

**Key Findings**:
-  **50-78% occupancy** achieves best performance (540 GB/s, 73.8% efficiency)
-  **100% occupancy performs worst** (331 GB/s, 45.3% efficiency)
- Gap: 50% occupancy is **63% faster than 100%**!
- **Bytes in-flight** optimal range: 1-1.6 MB
  - 384 KB (18.8% occ): Achieves 72.3% efficiency ✓ Validates Little's Law
  - 1.0 MB (50% occ): Achieves best 73.8% efficiency ⭐
  - 2.1 MB (100% occ): Drops to 45.3% efficiency

#### Why 100% Occupancy Performs Poorly

Root cause analysis:
1. **Cache thrashing**: 32 blocks/SM competing for L1/L2 cache, high miss rate
2. **Memory controller saturation**: Too many requests cause queueing delays
3. **Bank conflicts**: More warps increase shared memory bank conflict probability
4. **TLB thrashing**: Excessive concurrent accesses increase TLB misses

#### Little's Law Experimental Validation

Theory predicts 441 KB minimum in-flight data, practical comparison:

```
18.8% occupancy: 384 KB in-flight  → 72.3% efficiency (near theory)
50% occupancy:   1.0 MB in-flight  → 73.8% efficiency (exceeds minimum)
78% occupancy:   1.6 MB in-flight  → 73.7% efficiency (well above minimum)
100% occupancy:  2.1 MB in-flight  → 45.3% efficiency (overloaded)
```

**Conclusion**: Little's Law's predicted 441 KB is the **minimum** to reach peak performance, but practically 1-1.6 MB is the **optimal efficiency range**.

---

## Part 1: Lab 4 vs Lab 5 Optimization Comparison

### Lab 4 (Register Reuse)

- **Block tile**: 64×64
- **K tile**: 8
- **Threads**: 64 (2 warps)
- **Shared memory**: 4 KB
- **__launch_bounds__**: Present (64)
- **Performance**: 10.10 ms (5.74 TFLOP/s)

### Lab 5 Improvement Strategy (4× Scale-up)

Lab 5 is a **4× scale-up version** of Lab 4 with systematic optimization:

#### 1. Larger Block Tile Size (BM × BN)

- Lab 4: 64×64 → Lab 5: 128×128  (4× larger)

**Impact**:
- Reduce grid-level overhead (4× fewer blocks)
- Amortize kernel launch and scheduling overhead across more work
- Better L2 cache utilization

#### 2. Larger K Tile Size (BK)

- Lab 4: 8 → Lab 5: 32  (4× larger)

**Impact**:
- Higher arithmetic intensity: more computation per tile load
  - Lab 4: Load tile → 8 outer products
  - Lab 5: Load tile → 32 outer products
- Reduce `__syncthreads()` calls: 384 → 96 (**75% reduction**)
- Fewer global memory accesses

#### 3. More Threads per Block

- Lab 4: 64 threads (2 warps) → Lab 5: 256 threads  (8 warps, 4× more)

**Impact**:
- Better memory coalescing
- Stronger latency hiding capability
- Per-thread load: 8 elements → 16 elements

#### 4. Larger Shared Memory Usage

- Lab 4: 4 KB → Lab 5: 32 KB  (8× larger)

**Impact**:
- More aggressive on-chip memory usage
- **Optimized for P100**: Target ~50% occupancy instead of 100% (from Part 0 findings)
- 32 KB shared memory → maximum 2 blocks/SM (65536/32768=2)

#### 5. Removing __launch_bounds__ Restriction

- Lab 4: With restriction → Lab 5: No restriction 

**Impact**:
- Allow compiler free optimization of register allocation
- Experiments show better performance without bounds (9.85 ms vs 13.03 ms with bounds)

### Comparison Summary Table

| Parameter | Lab 4 | Lab 5 | Improvement |
|------|--------|--------|---------|
| Block tile | 64×64 | 128×128 | 4× larger |
| K tile | 8 | 32 | 4× larger |
| Threads | 64 (2 warps) | 256 (8 warps) | 4× more |
| Shared memory | 4 KB | 32 KB | 8× larger |
| Sync calls | 384 | 96 | 75% reduction |
| Occupancy | High (~100%) | Medium (~50%) | Optimized for P100 |
| Performance (3072³) | 10.10 ms | 9.86 ms | 2.4% faster |
| TFLOP/s | 5.74 | 5.88 | +2.4% |

### Core Optimization Strategy

Lab 5 optimization follows the "larger tiles + higher compute intensity" strategy:

1. **Expand work granularity**: Reduce overhead, improve efficiency
2. **Increase arithmetic intensity**: More computation per memory access
3. **Hardware-specific tuning**: Choose optimal occupancy for P100 rather than blindly pursuing 100%

---

## Part 2: Performance Analysis and Roofline Model

### Analysis Methodology

For each problem size, compute the following metrics:

1. **Total FLOPs**: `2 × size_i × size_j × size_k`
2. **Compute-bound minimum time**: `FLOPs / (9.3 × 10¹²)` seconds
3. **Unique bytes**: `size_i×size_k + size_k×size_j + size_i×size_j`
4. **Bandwidth-bound minimum time**: `bytes / (732 × 10⁹)` seconds
5. **Combined lower bound**: `max(compute_bound, bandwidth_bound)`
6. **Classification**: Compute-bound or Bandwidth-bound

### Major Test Cases (P100)

#### Case 1: 3072 × 3072 × 3072

```
Total FLOPs: 5.80 × 10¹⁰ (58.0 GFLOP)
Compute-bound time: 6.24 ms
Unique bytes: 113.2 MB
Bandwidth-bound time: 0.155 ms

Lower bound: 6.24 ms (Compute-bound)
Classification: Compute-bound
Max TFLOP/s: 9.3 (theoretical peak)
Threadblocks: 24×24 = 576 (10.3 blocks/SM) ✓

Measured Performance:
- matmul_improved: 9.86 ms → 5.88 TFLOP/s (63% of peak)
- matmul_improved_reduce: 9.88 ms → 5.87 TFLOP/s (63% of peak)
```

#### Case 2: 1×3072 (Extreme Small Matrix)

```
Total FLOPs: 1.89 × 10⁷ (0.0189 GFLOP)
Compute-bound time: 0.002 ms
Unique bytes: 24.6 MB
Bandwidth-bound time: 0.034 ms

Lower bound: 0.034 ms (Bandwidth-bound)
Classification: Bandwidth-bound
Max TFLOP/s: 0.56
Threadblocks: 1×24 = 24 (0.43 blocks/SM) ← Severely underutilized

Optimization: Split-K
- split_k=7 → 168 blocks (3.0 blocks/SM) ✓
- Measured speedup: 2.6× (27.08 ms → 10.38 ms)
```

### Split-K Strategy

For problem sizes with insufficient output tiles to fully saturate SMs (e.g., small matrices):

#### Implementation

```
Kernel 1: Split along K dimension, compute partial sums
  - Output: workspace[split_k_idx * size_i * size_j + ...]

Kernel 2: Reduction, aggregate results from all split_k partitions
  - Output: C[i,j] = sum(workspace[k][i,j] for k in split_k)
```

#### Split-K Strategy Decision

```
Standard Approach:  split_k=1, grid: 4×4 blocks    (16 blocks)
Split-K Approach:   split_k=7, grid: 4×4×7 blocks (112 blocks)
```

### Performance Analysis Summary

#### Large Matrices (Sufficient Parallelism)

| Size | Blocks/SM | Occupancy | Performance(TFLOP/s) | % Peak |
|------|----------|-----------|--------------|---------|
| 3072³ | 10.3 | Medium (~50%) | 5.88 | 63% |
| 512×3072² | 1.8 | Low | 4.89 | 53% |

#### Small Matrices (Split-K Optimized)

| Size | split_k | Blocks/SM | Speedup | New Performance |
|------|---------|----------|------|--------|
| 1×3072 | 7 | 3.0 | 2.6× | 5.89 TFLOP/s |
| 16×3072 | 11 | 4.4 | 8.6× | 5.64 TFLOP/s |
| 256×256 | 32 | 2.3 | 27× | 0.48 TFLOP/s |

### Key Insights

1. **Compute-bound vs Bandwidth-bound**
   - Large matrices: Compute-intensive, limited by compute capability
   - Small matrices: Memory-intensive, need Split-K for increased parallelism

2. **Practical Occupancy Application**
   - 50-78% is the golden range, don't blindly pursue 100%
   - Little's Law provides lower bound, but practical systems need headroom

3. **Split-K Value**
   - Transform parallelism-starved small matrix problems into GPU-saturating computations
   - In extreme cases (128²×32768), increase SM utilization from 1.8% to 57%

