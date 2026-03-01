# Lab 3: Wave Simulation - Memory Hierarchy Optimization

Source: https://accelerated-computing.academy/fall25/labs/lab3/

## 📋 Quick Navigation

- [Core Objectives](#core-objectives)
- [Lab Structure](#lab-structure)
- [Detailed Learning Notes](#detailed-learning-notes)
- [Implementation Reference](#implementation-reference)
- [Learning Resources](#learning-resources)

---

## Core Objectives

### From Compute Bottleneck to Memory Bottleneck

This lab focuses on understanding GPU memory hierarchy and optimizing memory-intensive workloads through shared memory techniques.

**Lab 1-2 (Mandelbrot): Compute-Dominated**
- Most time spent on ALU operations
- Memory access time is negligible

**Lab 3 (Wave Simulation): Memory-Dominated**
- Each pixel reads neighbor values (pixel-to-pixel communication)
- Memory access becomes the main bottleneck
- Need to leverage locality to optimize memory usage

### Learning Objectives

#### 1️⃣ Understanding GPU Memory Hierarchy

| Memory Level | Capacity | Bandwidth | Description |
|--------------|----------|-----------|-------------|
| Global Memory (DRAM) | 20 GB | 360 GB/s | Main storage |
| L2 Cache | 48 MB | ~2.5 TB/s | Shared |
| L1 Cache / Shared Memory | 128 KB/SM | 13.4 TB/s | Per SM |

#### 2️⃣ Two Implementation Strategies

- **Naive Method**: Read/write from global memory every timestep
- **Optimized Method**: Use Shared Memory to reduce global memory access

#### 3️⃣ Real-World Application

- Wave Equation simulation
- Simulating acoustic, water, and light waves
- Classic Stencil Computation problem

---

## Lab Structure

### Part 0: Memory Subsystem Empirical Study

**Focus:** Understanding cache behavior through PTX-level analysis.

#### Task 1: Measure Memory Latency (Prelab Question 1)

**Three PTX Load Instructions:**

| Instruction | Cache Behavior | Test Target |
|------------|----------------|-------------|
| `ld.global.ca.f32` | L1 + L2 | Measure L1 cache latency |
| `ld.global.cg.f32` | L2 only | Measure L2 cache latency |
| `ld.global.cv.f32` | No cache | Measure DRAM latency |

**Expected Results:**
- L1 Cache: ~30-40 cycles
- L2 Cache: ~200-300 cycles
- Global Memory (DRAM): ~400-600 cycles

#### Task 2: Memory Coalescing Effect (Prelab Question 2)

**Non-Coalesced Access Pattern:**
```
Each lane loads x consecutive elements
Lane 0: [0, 1, 2, ..., x-1]
Lane 1: [x, x+1, ..., 2x-1]
Lane 2: [2x, 2x+1, ..., 3x-1]

→ Memory accesses are scattered, each warp needs 32 independent transactions
```

**Coalesced Access Pattern:**
```
Each lane loads elements with stride = blockDim.x
Lane 0: [0, blockDim.x, 2*blockDim.x, ...]
Lane 1: [1, blockDim.x+1, 2*blockDim.x+1, ...]

→ Memory accesses are consecutive, entire warp needs 1 transaction
```

**Expected Speedup:** Coalesced typically **5-10x faster** than Non-Coalesced

---

### Part 1: Basic GPU Implementation

**Workload:** Numerical simulation of wave equation on 2D domain.

Core algorithm updates each pixel based on:
- Current and previous timestep values
- Values from orthogonally-adjacent neighbors (4-point stencil)
- Position-dependent damping, source injection, and wall reflection

**Implementation Details:**
- One kernel launch per timestep ensures correct synchronization
- Multi-dimensional thread/block indices simplify 2D domain mapping
- Y-major array layout affects memory coalescing patterns
- Correctness testing on small domains; performance testing on 3201×3201 grids for 12800 timesteps

**Performance Analysis (Question 1):**

Calculate the following:
- Buffer size vs L2 cache capacity (48 MB)
- Bytes loaded per kernel launch
- L2 cache miss rates and DRAM traffic
- Theoretical execution time bounds from DRAM/L2 bandwidth
- Comparison with actual execution time

---

### Part 2: Shared Memory Optimization

**Strategy:** Exploit locality by loading simulation tiles into SM-local SRAM (128 KB per SM) and computing multiple timesteps before writing results back to global memory.

**Key Constraint:** Information propagates one pixel per timestep, creating a shrinking "valid region" at tile centers as edges become invalid after multiple steps.

**Technical Implementation Requirements:**

Dynamic shared memory allocation enables flexible buffer sizing:

```cuda
extern __shared__ float shmem[];
// Launch with size parameter: kernel<<<blocks, threads, bytes>>>
```

Shared memory sizes exceeding 48 KB require explicit opt-in via `cudaFuncSetAttribute`.

**Implementation Design Decisions:**
- Timesteps per kernel launch (tunable parameter)
- Handling non-multiple timestep counts
- Block-to-tile work partitioning amid valid region shrinkage
- Thread-to-pixel mapping (initially one-to-one, generalizable)
- Register vs shared memory data storage
- Out-of-bounds access prevention

**Performance Measurement (Question 2):**
Compare speedup against Part 1, document trade-offs encountered, and discuss implementation challenges.

---

## Detailed Learning Notes

### PTX and Memory Latency Measurement

#### What is PTX?

**PTX = Parallel Thread Execution**

- NVIDIA GPU's virtual instruction set (Virtual ISA)
- Similar to x86 assembly language, but:
  - Not true machine code, needs further compilation to SASS
  - Cross-GPU architecture compatible
  - No fixed register numbering (compiler allocates)

**Compilation Flow:**
```
CUDA C++ → PTX (Virtual ISA) → SASS (True Machine Code) → GPU Execution
```

#### Embedding PTX in CUDA

**Basic Syntax:**

```cpp
asm volatile(
    "ptx_instruction1;\n\t"
    "ptx_instruction2;\n\t"
    : output operand list
    : input operand list
    : clobber list
);
```

**Operand Constraints:**

| Constraint | Type | Description | PTX Type |
|-----------|------|-------------|----------|
| `"r"` | int/unsigned | 32-bit integer register | `.u32 / .s32` |
| `"l"` | long | 64-bit integer register | `.u64 / .s64` |
| `"f"` | float | 32-bit float register | `.f32` |
| `"d"` | double | 64-bit float register | `.f64` |

**Modifiers:**
- `"="` - Write-only
- `"+"` - Read-write
- No modifier - Read-only

#### Simple Example: Addition

```cpp
__global__ void add_kernel() {
    int a = 10, b = 20, c;

    asm volatile(
        "add.s32 %0, %1, %2;\n\t"  // c = a + b
        : "=r"(c)      // %0: output
        : "r"(a),      // %1: input
          "r"(b)       // %2: input
    );

    printf("c = %d\n", c);  // c = 30
}
```

#### Common PTX Instructions

**Arithmetic Instructions:**
```ptx
add.s32 %0, %1, %2;        // signed 32-bit addition
sub.f32 %0, %1, %2;        // float subtraction
mul.lo.s32 %0, %1, %2;     // multiplication (low 32 bits)
div.f32 %0, %1, %2;        // float division
```

**Memory Instructions:**
```ptx
ld.global.ca.f32 %0, [%1];    // Load, cache all levels
ld.global.cg.f32 %0, [%1];    // Load, cache in L2 only
ld.global.cv.f32 %0, [%1];    // Load, no caching (volatile)
ld.shared.f32 %0, [%1];       // Load from shared memory

st.global.f32 [%0], %1;       // Store to global memory
st.shared.f32 [%0], %1;       // Store to shared memory
```

**Data Movement:**
```ptx
mov.u32 %0, %1;                // 32-bit move
mov.u64 %0, %%clock64;         // Read clock counter
```

**Synchronization and Barriers:**
```ptx
membar.gl;                     // Global memory barrier
membar.cta;                    // Block memory barrier
bar.sync 0;                    // Thread barrier (__syncthreads)
```

#### memory_latency.cu Explanation

**Measurement Flow:**
```
1. Warm-up    : Preheat cache
2. Start timing: mov.u64 %1, %%clock64;
3. Load instruction: ld.global.<qualifier>.f32 %2, [%0];
4. End timing: mov.u64 %4, %%clock64;
5. Calculate latency: latency = end_time - start_time - 2
```

**Three Kernel Implementations:**

L1 Cache Latency:
```ptx
"ld.global.ca.f32 %2, [%0];\n\t"
```
- Uses `.ca` (cache all) qualifier, caches in L1 and L2
- Measures L1 cache hit latency

L2 Cache Latency:
```ptx
"ld.global.cg.f32 %2, [%0];\n\t"
```
- Uses `.cg` (cache global) qualifier, caches only in L2, bypasses L1
- Measures L2 cache hit latency

Global Memory Latency:
```ptx
"ld.global.cv.f32 %1, [%3];\n\t"
```
- Uses `.cv` (cache volatile) qualifier, no caching
- Measures DRAM latency

**Compilation and Execution:**
```bash
nvcc -O2 -arch=sm_89 mem-latency.cu -o mem-latency
./mem-latency
```

**Expected Output Example:**
```
global_mem_latency latency =  450 cycles
l2_mem_latency latency =          280 cycles
l1_mem_latency latency =           32 cycles
```

#### Pointer Chasing Technique

**Core Concept:** Prevent GPU from executing multiple loads in parallel by creating dependent memory access chains:

```cpp
// Pointer chasing - dependency chain
addr1 = load(addr0);  // Must wait to complete
addr2 = load(addr1);  // Depends on addr1, cannot parallelize
addr3 = load(addr2);  // Depends on addr2
```

**Why Pointer Chasing?**

Without pointer chasing (wrong):
```cpp
// Independent loads - GPU can execute in parallel
result1 = load(addr);
result2 = load(addr);
result3 = load(addr);
result4 = load(addr);
```
→ GPU issues all loads simultaneously, measures only 1 load's latency

With pointer chasing (correct):
```cpp
// Dependency chain - must execute sequentially
ptr = load(ptr);  // Read value pointed to by ptr
ptr = load(ptr);  // Must wait for previous load to complete
ptr = load(ptr);
```
→ Measures total latency of all loads

**Implementation Method Comparison:**

| Method | Advantages | Disadvantages | Purpose |
|--------|-----------|----------------|---------|
| True pointer chain | Measures real random access | Complex | Cache miss latency |
| Self-pointing + add | Simple, same address | Not true pointer chase | Cache hit latency |
| Independent loads | Simplest | Cannot measure latency | ❌ Not applicable |

---

### Memory Coalescing Analysis

#### Non-Coalesced Access Pattern

```cpp
int base_idx = tid * x;  // tid=0: 0, tid=1: 1024, tid=2: 2048...
for (int i = 0; i < x; i++) {
    dst[base_idx + i] = src[base_idx + i];
}
```

**Access Pattern:**
```
Thread 0: [0] ← 1024 elements apart
Thread 1: [1024]
Thread 2: [2048] ← Non-consecutive
...
Thread 31: [31744]

→ Each warp needs 32 independent transactions → Slow
```

#### Coalesced Access Pattern

```cpp
int idx = tid + i * total_threads;  // stride = blockDim.x × gridDim.x
for (int i = 0; i < x; i++) {
    dst[idx] = src[idx];
}
```

**Access Pattern:**
```
Thread 0: [0] ← Adjacent!
Thread 1: [1]
Thread 2: [2] ← Consecutive!
...
Thread 31: [31]

→ Entire warp needs 1 coalesced transaction → Fast
```

#### Visual Comparison

**Non-Coalesced (each thread reads consecutive block):**
```
Thread 0: ████████████... (1024 elements)
Thread 1:                 ████████████... (1024 elements)
Thread 2:                                 ████████████...
        ↑ At same time, warp threads access non-consecutive addresses
```

**Coalesced (each thread reads strided elements):**
```
Thread 0: █   █   █   █   ...
Thread 1:  █   █   █   █   ...
Thread 2:   █   █   █   █   ...
        ↑ At same time, warp threads access consecutive addresses
```

#### Expected Performance Difference

- **Theoretical Speedup**: Coalesced should be **5-10x faster** than Non-Coalesced
- **Reason**:
  - Non-coalesced: 32 independent transactions/iteration
  - Coalesced: 1 coalesced transaction/iteration

#### Key Insight: Cache Effects Matter

> **On GPU, cache locality often matters more than memory coalescing**

**Coalescing Advantages in:**
- ✅ Random access (no spatial locality)
- ✅ Single access (no time reuse)
- ✅ Bandwidth-bound (memory bandwidth is bottleneck)

**Test Scenario (with cache reuse):**
- ✓ Has spatial locality (non-coalesced is sequential)
- ✓ Has time reuse (cache can buffer)
- ✗ Compute-bound (each thread does lots of work)

---

### Memory Access Optimization

#### Optimization Techniques

**1. Reduce Register Pressure**

Original version (needs extra registers):
```cpp
int idx = tid + i * total_threads;
dst[idx] = src[idx];  // Use idx twice
```

Optimized version (inline calculation):
```cpp
dst[i * total_threads + tid] = src[i * total_threads + tid];
```

→ Compiler inlines directly, reduces register usage, improves occupancy

**2. `__restrict__` Keyword**

```cpp
void kernel(float * __restrict__ dst, const float * __restrict__ src)
```

**Effects:**
- Tells compiler `dst` and `src` don't alias
- Allows compiler to aggressively reorder load/store
- Enables more aggressive optimizations

**3. `const` Keyword**

```cpp
const float * __restrict__ src
```

**Effects:**
- Tells compiler `src` is read-only
- Can use texture cache or read-only cache
- Allows more prefetch optimizations

#### L1/L2/DRAM Bandwidth Measurement

**Tesla P100 Specifications:**
- 56 SMs
- L1 total capacity: ~3.5 MB (128 KB per SM)
- L2 capacity: 4 MB

**Test Configuration:**

| Level | Working Set | Iterations | Method |
|-------|------------|------------|--------|
| **L1** | 4 KB | 80,000 | Ensure fully in L1 |
| **L2** | 4 MB | 5,000 | Near L2 capacity |
| **DRAM** | 256 MB | 500 | Much larger than L2 |

**Typical Results:**

```
L1 Cache Bandwidth:  2.08 TB/s  (70-100% theoretical efficiency)
L2 Cache Bandwidth:  1.21 TB/s  (40-60% theoretical efficiency)
DRAM Bandwidth:      0.49 TB/s  (68% theoretical efficiency) ✓
```

#### Optimization Key Points

✅ **Effective Improvements:**

1. **Minimal Working Set** - Ensure data fits entirely in L1
2. **Multiple Independent Accumulators** - Increase instruction-level parallelism (ILP)
3. **Leverage Dual-Issue** - Overlap address calculation with reads
4. **Multiple Reads per Loop** - Maximize parallelism

---

## Implementation Reference

### Memory Hierarchy Details

RTX 4000 Ada contains four hierarchy levels:

| Level | Capacity | Bandwidth | Description |
|-------|----------|-----------|-------------|
| **DRAM** | 20 GB | 360 GB/sec | 5 GDDR6 chips aggregate bandwidth |
| **L2 Cache** | 48 MB | ~2.5 TB/sec | Shared |
| **L1 Cache + Scratchpad** | 128 KB/SM | 13.4 TB/sec | Per SM aggregate |
| **Register File** | 64 KB/warp scheduler | — | Fastest |

**Note:** L1 cache lacks cross-SM coherence, most loads bypass L1 and route through L2.

### Key Files

- `mem-latency.cu`: Part 0 memory latency measurement
- `coalesced-loads.cu`: Part 0 coalescing test
- `wave-naive.cu`: Part 1 basic implementation
- `wave-shared.cu`: Part 2 shared memory optimization
- `docs/PART1.md`: Part 1 detailed analysis notes
- `docs/PART2.md`: Part 2 detailed analysis notes
- `docs/NOTES.md`: Complete learning notes

---

## Learning Resources

### Official Documentation and Tools

1. **Official PTX Documentation**: https://docs.nvidia.com/cuda/parallel-thread-execution/

2. **View Generated PTX:**
```bash
nvcc -ptx your_code.cu -o your_code.ptx
cat your_code.ptx
```

3. **Online PTX Compiler**: https://godbolt.org/

4. **Memory Bandwidth Testing Reference**: https://www.evolvebenchmark.com/blog-posts/learning-about-gpus-through-measuring-memory-bandwidth

### Compilation and Testing

```bash
# Compile
nvcc -O2 -arch=sm_60 coalesced-loads.cu -o coalesced-loads

# Run
./coalesced-loads
```

---

## 🎯 Key Points Summary

### Three PTX Load Instruction Differences

```ptx
ld.global.ca.f32 %2, [%0];    // L1 test: cache all levels
ld.global.cg.f32 %2, [%0];    // L2 test: cache L2 only
ld.global.cv.f32 %1, [%3];    // DRAM test: no caching
```

### Memory Coalescing Core Principles

- **Not** always faster (consider cache effects)
- **Is** faster for random access, single use
- **Requires** warp threads access consecutive addresses

