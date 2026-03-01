# Lab 5: Matrix Multiply – Improved Scheduling

## Course Background

**MIT 6.S894 - Accelerated Computing (Fall 2025)**

Official Course Resource: [Lab 5 - Matrix Multiply – Improved Scheduling](https://accelerated-computing.academy/fall25/labs/lab5/)

Lab 5 focuses on optimizing GPU matrix multiplication kernels through improved scheduling and parallelism management. This lab emphasizes the relationship between occupancy, memory latency, and hardware utilization.

## Overview

### Core Concepts

#### 1. Occupancy and Little's Law

**Little's Law**: "Required Parallelism = Latency × Throughput"

This establishes the theoretical foundation for understanding how much parallel work is needed to saturate memory bandwidth.

**Occupancy Definition**: The percentage of an SM's capacity that is actively used.

**Example**:
- Global memory request completion time: ~700-800 cycles
- L1 cache request completion time: ~35 cycles
- Difference: 23× latency

#### 2. Memory Latency Considerations

**Serial Dependency Problem**:
```cpp
// ❌ Bad: Each memory request waits for the previous one
float a = A[i];  // Latency ~700 cycles
float b = B[a];  // Wait above, then latency ~700 cycles
sum += a * b;
```

**Solution: Batch Operations**:
```cpp
// ✅ Good: Issue multiple memory requests simultaneously
float a0 = A[i];
float a1 = A[i+1];
float a2 = A[i+2];  // 3 requests in flight simultaneously
// ...do other computations...
sum += a0 * B[a0];  // a0 is ready now
```

#### 3. Overlapping Strategies

Three approaches for overlapping data movement with computation:

##### Strategy 1: Co-resident Blocks

Multiple blocks run simultaneously on the same SM.

**Advantages**:
- Simple to implement
- Good SM utilization

**Disadvantages**:
- Register and shared memory contention
- May reduce occupancy

##### Strategy 2: Warp Specialization

Different warps perform different tasks:

```cpp
// Warp 0-1: Load data
if (warpIdx < 2) {
    // Load global → shared memory
    tile_a[...] = a[...];
    tile_b[...] = b[...];
}
__syncthreads();

// Warp 2-7: Compute
if (warpIdx >= 2) {
    // Compute using loaded tiles
    acc += tile_a[...] * tile_b[...];
}
```

**Advantages**:
- Hide memory latency effectively
- Increase compute-memory overlap

**Disadvantages**:
- Careful synchronization required
- Code complexity increases

##### Strategy 3: Asynchronous Copy

Use `memcpy_async` for direct global-to-shared memory transfer.

```cpp
__shared__ float tile_a[2][32][32];  // Double buffer

// Initiate async copy
__memcpy_async(&tile_a[0][ty][tx], &A[...], sizeof(float), 32);
__syncwarp();

for (int k = 0; k < num_tiles; ++k) {
    // Compute current tile
    for (int i = 0; i < 32; ++i) {
        sum += tile_a[k & 1][ty][i] * tile_b[i][tx];
    }

    // Simultaneously prefetch next tile
    if (k + 1 < num_tiles) {
        __memcpy_async(&tile_a[(k+1) & 1][ty][tx],
                       &A[...], sizeof(float), 32);
    }
    __syncwarp();
}
```

**Advantages**:
- Completely hide load latency
- Hardware-accelerated memory transfer

**Disadvantages**:
- Requires newer SM support
- Double-buffering increases memory usage

#### 4. Vectorized Loads

Load 2-4 words in a single instruction instead of scalar loads.

```cpp
// ❌ Scalar loads (4 instructions)
float a0 = A[i];
float a1 = A[i+32];
float a2 = A[i+64];
float a3 = A[i+96];

// ✅ Vectorized load (1-2 instructions)
float4 v = *((float4*)&A[i]);  // a0, a1, a2, a3
```

**Advantages**:
- Reduce instruction count
- Increase memory bandwidth utilization
- No bank conflicts (contiguous address access)

**Key Insight**: "Vectorized loads do not incur bank conflicts...where all lanes access contiguous addresses."

#### 5. Split-K Partitioning

For problem sizes with insufficient output tiles to fully saturate SMs.

**Problem Scenario**:
- Matrix size: 1024 × 1024
- Per-block tile size: 256 × 256
- Result: Only 16 blocks
- GPU SMs: 100 → 84 SMs idle

**Solution**: Split work along K dimension.

```
Standard Approach:      Split-K Approach:
grid: 4×4 blocks        grid: 4×4×8 blocks
      (16 blocks)             (128 blocks)
```

**Implementation**:
```cpp
// Kernel 1: Compute partial sums
__global__ void matmul_partial(float *C_partial, ...) {
    // Compute subset of C, write to C_partial
    // No inter-block synchronization needed
}

// Kernel 2: Reduce
__global__ void reduce_partial(float *C, float *C_partial, ...) {
    // Aggregate partial sums: C[i,j] = sum(C_partial[k][i,j])
}
```

**Advantages**:
- Increase number of schedulable blocks
- Better SM utilization

**Disadvantages**:
- Requires two kernels
- Extra global memory I/O

---

## Lab Objectives

### Performance Goals

For 3072 × 3072 × 3072 matrix multiplication:

| Implementation | Target Time |
|------|--------|
| **Baseline** (No optimization) | ~100 ms |
| **Lab 4 Style** (Shared Memory) | ~45 ms |
| **Lab 5 Target** (Improved Scheduling) | **< 5 ms** |

### Key Metrics

- **Throughput**: > 50 TFLOP/s (RTX 4000 Ada)
- **Occupancy**: > 70%
- **Memory Bandwidth Utilization**: Effectively hide latency

---

## Implementation Strategy Comparison

| Strategy | Complexity | Performance | Use Case |
|------|--------|------|---------|
| Co-resident Blocks | Low | Medium | Simple applications |
| Warp Specialization | Medium | High | General optimization |
| Async Copy | High | Very High | Need extreme performance |
| Split-K | High | High | Small matrices or special sizes |

---

## Lab Progression

### Part 0: Prelab
- Answer Little's Law questions
- Analyze memory latency impact

### Part 1: Baseline Experiments
- Establish performance baseline
- Measure occupancy

### Part 2: Implementation
- Implement warp specialization or async copy
- Measure performance improvement

### Part 3: Optimization Analysis
- Write answers to three analysis questions
- Explain why target is/isn't achieved

---

## Key Insights

### How Lab 5 Differs from Lab 4

**Lab 4** (Data Reuse):
- Focus: Reduce memory access frequency
- Tool: Shared Memory Tiling
- Performance Improvement: 10-20×

**Lab 5** (Scheduling Optimization):
- Focus: Hide memory latency
- Tool: Parallel work batching, Async Copy
- Performance Improvement: 5-10× (additional)

### Occupancy vs Performance

**Misconception**: "Higher occupancy = better performance"

**Reality**:
- Occupancy is just one metric
- What truly matters: **Hide enough latency**
- Little's Law guidance: `Parallelism ≥ Latency × Throughput`
- Example: On P100, `Parallelism ≥ 800 cycles × (732 GB/s ÷ 4) ≈ 146K`

### Value of Vectorization

Benchmark Example:
```
Scalar loads (4 instructions):
  - Instruction overhead: High
  - Bandwidth utilization: Low (1 word per load)

Vectorized loads (1-2 instructions):
  - Instruction overhead: Low
  - Bandwidth utilization: High (4 words per load)
  - Performance improvement: 2-3×
```

---

## Expected Outcomes

After successfully completing Lab 5, you should be able to:

1. ✅ Implement < 5ms matrix multiplication kernel
2. ✅ Achieve > 50 TFLOP/s throughput
3. ✅ Understand relationship between parallelism and latency hiding
4. ✅ Design and implement warp specialization or async copy
5. ✅ Use Little's Law to reason about performance decisions

---

## Further Resources

- Official Lab 5 Documentation: https://accelerated-computing.academy/fall25/labs/lab5/
- NVIDIA Async Copy Documentation: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- Warp Specialization Paper: Scott Gray et al., 2022
