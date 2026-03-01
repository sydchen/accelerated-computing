# Lab 2 Notes: Massively-Parallel Mandelbrot

Source: https://accelerated-computing.academy/fall25/labs/lab2/

## Lab Objective

Lab 2 extends Lab 1 Mandelbrot by adding more levels of parallelism:

1. **Instruction-Level Parallelism (ILP)** - exploiting instruction-level parallelism within a single instruction stream
2. **Multi-Core Parallelism** - distributing work across multiple CPU cores or GPU streaming multiprocessors
3. **Multi-Threaded Parallelism** - managing multiple concurrent threads/warps per execution unit

The central goal is to map CPU and GPU execution models to each other while scaling performance through hardware parallelism stacking.

## Important Changes vs Lab 1

The starter code changes this line:

```cpp
// old
float y = w - x2 - y2 + cy;
// new
float y = w - (x2 + y2) + cy;
```

This can change both performance and output due to floating-point non-associativity.

The image configuration is also more ambitious (higher resolution/quality and different zoom window), with parameters such as `window_zoom`, `window_x`, and `window_y`.

## Terminology Mapping (CPU vs GPU)

| GPU term in CUDA docs | Closest CPU analogy (course framing) |
|---|---|
| CUDA thread | Vector lane |
| Warp | Thread (instruction stream) |
| Warp scheduler | Core-like execution unit |

For this lab, the practical takeaway is to reason about occupancy and throughput at warp-scheduler/core granularity.

## Hardware Context (from course page)

- CPU: AMD Ryzen 7 7700, 8 cores, SMT up to 2 threads/core.
- GPU: NVIDIA RTX 4000 Ada, 48 SMs, 4 warp schedulers/SM (192 total warp schedulers).

## Part 0: FFMA Latency + ILP Prelab

### Deliverables

- Implement `measure_fma_latency(..)` in `fma_latency.cu`.
- Verify generated `FFMA` in SASS (`-s` in telerun flow).
- Complete interleaved/non-interleaved ILP variants.

### Questions to answer

- Q0.1: What is FFMA latency?
- Q0.2: What latency is observed with explicitly interleaved ILP?
- Q0.3: Does non-explicit interleaving behave the same, and why?
- Q0.4: At what warp count does throughput peak, and how does FFMA latency explain it?

### Notes

- Use dependent chains for stable latency measurement.
- Separate latency reasoning from throughput reasoning.
- Use `warp_scheduler.cu` + `plot.py` to inspect scaling behavior.

## Part 1: ILP in Mandelbrot

### CPU

- File: `mandelbrot_cpu_2.cpp`
- Implement: `mandelbrot_cpu_vector_ilp`

### GPU

- File: `mandelbrot_gpu_2.cu`
- Implement: `mandelbrot_gpu_vector_ilp`, `launch_mandelbrot_gpu_vector_ilp`
- Keep launch at `<<<1, 32>>>` (single warp baseline analogy)

### Design checklist

- How state is organized for multiple independent vectors at once
- How divergence/control flow is handled
- How many vectors are processed in parallel (ILP factor)
- Row-wise vs tile-wise work grouping
- Whether `#pragma unroll` helps in controlled places

### Final write-up question mapping

- Q1: ILP speedup on CPU/GPU, partitioning strategy, control-flow strategy, chosen ILP factor, limiting factors.

### CPU vs GPU Implementation Differences

**CPU (AVX-512)**
- State: vector registers with arrays `__m512 v_x2[4]`
- Divergence handling: hardware predication using masks `__mmask16`
- Memory pattern: horizontal (row-major, consecutive pixels)
- Control flow: use masks to avoid branches

**GPU (CUDA)**
- State: scalar registers with arrays `float x2[4]`
- Divergence handling: boolean flags + branch (thread-internal, low cost)
- Memory pattern: strided access (coalesced within warps)
- Control flow: per-thread divergence is acceptable

### Expected Performance Improvements

- **CPU**: 1.2x - 2.0x speedup (limited by register pressure and memory bandwidth)
- **GPU**: 1.5x - 3.0x speedup (GPUs benefit more from ILP due to higher latency)

## Part 2: Multi-Core Parallelism

### CPU Multi-Core

- Implement in `mandelbrot_cpu_2.cpp`:
  - `mandelbrot_cpu_vector_multicore`
- Use pthread model (`pthread_create`, `pthread_join`) for 8-core scaling.

Q2 focus:
- Speedup over single-core vector baseline
- Impact of work partitioning strategy

### GPU Multi-Core

- Implement in `mandelbrot_gpu_2.cu`:
  - `mandelbrot_gpu_vector_multicore`
  - `launch_mandelbrot_gpu_vector_multicore`

Target idea from course:
- one warp per warp scheduler across all 192 warp schedulers
- baseline configuration to reason about: `<<<48, 4 * 32>>>`

Q3 focus:
- Speedup vs single-warp vector baseline
- Absolute runtime vs CPU multicore
- Work partitioning strategy across blocks/warps

Q4 focus:
- Compare launch alternatives:
  - `<<<48, 4*32>>>`
  - `<<<96, 2*32>>>`
  - `<<<24, 8*32>>>`
- Explain assignment behavior and measured runtime differences

### Performance Analysis: Launch Configuration Impact

Key Finding: **Block granularity matters more than total warps**

| Configuration | Blocks | Threads/Block | Total Warps | Warps/Scheduler | Blocks/SM | Runtime | Notes |
|---|---|---|---|---|---|---|---|
| <<<48, 128>>> | 48 | 128 | 192 | 1.0 | 1.0 | baseline | Perfect 1:1 SM mapping |
| <<<96, 64>>> | 96 | 64 | 192 | 1.0 | 2.0 | ~2.5% faster | Better dynamic scheduling |
| <<<24, 256>>> | 24 | 256 | 192 | 1.0 | 0.5 | ~17% slower | Only uses 24 of 48 SMs |

**Critical Insight**: More blocks (2 per SM) provide better load balancing and scheduling flexibility compared to fewer, larger blocks, even when total warps remain constant.

### Expected Speedups

- **CPU Multi-Core**: 6-7.5x (theoretical 8x, limited by load balancing and cache contention)
- **GPU Multi-Core**: 150-180x relative to single warp baseline (192 warp schedulers in parallel)

## Part 3: Multi-Threaded Parallelism

### CPU Multi-Threaded

- Implement:
  - `mandelbrot_cpu_vector_multicore_multithread`
- Use more than one thread per core.

Q5 focus:
- Speedup from adding per-core multithreading
- Best thread count and why

### GPU Multi-Threaded

- Single-SM study:
  - `mandelbrot_cpu_vector_multicore_multithread_single_sm`
  - `launch_cpu_vector_multicore_multithread_single_sm`
- Full-scale multi-threaded GPU:
  - `mandelbrot_gpu_vector_multicore_multithread_full`
  - `launch_mandelbrot_gpu_vector_multicore_multithread_full`

Q6 focus:
- Runtime trend as warps/block increase beyond 4 warps/SM
- Whether gains continue up to 32 warps/block

### Single SM Multi-Threading Results

Experimental data shows **continuous improvement** up to hardware limits:

| Config | Warps | Warps/Scheduler | Runtime (ms) | Relative Speedup | Improvement % |
|---|---|---|---|---|---|
| Baseline | 4 | 1 | 132.10 | 1.00x | - |
| 8 warps | 8 | 2 | 84.53 | 1.56x | +56.0% |
| 12 warps | 12 | 3 | 70.82 | 1.87x | +46.4% |
| 32 warps | 32 | 8 | 58.31 | 2.27x | +55.9% |

**Key Finding**: Performance consistently improves without degradation, indicating sufficient resources (registers, shared memory) to support 8 warps per scheduler.

**Why Multi-Threading Works**: When one warp waits for FMA latency (~4-11 cycles), the scheduler switches to another warp, keeping hardware units busy.

Q7 focus:
- Full-scale multithread speedup
- Optimal warp count and contributing factors

### Full Machine Multi-Threading (P100 Example)

| Configuration | Warps | Warps/Scheduler | Blocks/SM | Runtime | Speedup vs Baseline |
|---|---|---|---|---|---|
| <<<56, 64>>> | 112 | 1.0 | 1.0 | 7.21 ms | 1.00x (baseline) |
| <<<56, 128>>> | 224 | 2.0 | 1.0 | 3.84 ms | 1.88x |
| <<<56, 256>>> | 448 | 4.0 | 1.0 | 2.83 ms | 2.55x |
| <<<112, 128>>> | 448 | 4.0 | 2.0 | **2.37 ms** | **3.04x** ✨ (BEST) |
| <<<56, 512>>> | 896 | 8.0 | 1.0 | 3.12 ms | 2.31x |
| <<<112, 256>>> | 896 | 8.0 | 2.0 | 2.39 ms | 3.02x |

**Critical Finding**: **Block granularity (2 blocks/SM) beats raw warp count**

- <<<56, 256>>> (1 block/SM, 4 W/S): 2.83 ms
- <<<112, 128>>> (2 blocks/SM, 4 W/S): 2.37 ms (19% faster)

More small blocks → better scheduler flexibility → superior load balancing

## Part 4: Put It All Together (ILP + Multi-Core + Multi-Threaded)

- CPU:
  - `mandelbrot_cpu_vector_multicore_multithread_ilp`
- GPU:
  - `mandelbrot_gpu_vector_multicore_multithread_full_ilp`
  - `launch_mandelbrot_gpu_vector_multicore_multithread_full_ilp`

Q8 focus:
- Additional speedup from reintroducing ILP on top of multicore+multithread
- Comparison to ILP gains in single-core/single-thread setting
- Optimal thread/warp counts and optimal inner-loop ILP factor

### Performance Data: CPU (Kaggle 2-core platform)

| Implementation | Runtime (ms) | Speedup vs Scalar | Improvement vs Previous |
|---|---|---|---|
| Scalar (baseline) | 751.78 | 1.00x | - |
| Vector + ILP (single-thread) | 65.29 | 11.52x | - |
| Vector + Multi-core | 51.08 | 14.72x | -21.8% |
| Vector + Multi-thread (4 threads) | 34.92 | 21.53x | -31.6% |
| Vector + Multi-thread + ILP | 32.78 | 22.94x | -6.1% |

**Critical Finding**: ILP provides **only 6.1% improvement** in multi-threaded environment, far below single-thread ILP gains (11.52x).

**Why Limited Improvement?**
1. **Resource Saturation**: 4 threads already fully utilize 2 physical cores
2. **Register Pressure**: ILP requires 4x more registers, causing spilling
3. **Cache Thrashing**: 16 active compute streams (4 threads × 4 chains) compete for L1/L2 cache
4. **Instruction Flow Already Interleaved**: Hyperthreading already provides per-core thread interleaving

**Conclusion**: Multi-threading alone (Part 3: 34.92 ms) is superior to combined approach (Part 4: 32.78 ms) in terms of **cost-benefit ratio** on this platform.

### Performance Data: GPU (Tesla P100)

| Implementation | Runtime (ms) | Speedup vs Single-Warp |
|---|---|---|
| Part 3 (no ILP) | 2.36 ms | 365x |
| Part 4 (with ILP 4 chains) | 5.51 ms | 157x |

**Negative Impact**: ILP degrades performance by **133%** (makes it 2.33x slower).

**Root Cause**: Register pressure from 4 pixel chains causes:
- Occupancy reduction (active warps per SM decrease)
- Register spilling to local memory (117-cycle latency per access)
- Elimination of latency-hiding benefits

**Key Insight**: GPU multi-threading is already more effective than software ILP for latency hiding. Additional ILP competes for limited resources.

## Suggested Experiment Table

Use a single table format for all stages to keep write-up consistent:

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

## Current Local Files

- `fma_latency.cu`
- `fma_latency.md`
- `warp_scheduler.cu`
- `mandelbrot_cpu_2.cpp`
- `mandelbrot_gpu_2.cu`
- `plot.py`

This note is intended as the working log and final write-up scaffold for Lab 2.

---

## Appendix: Integrated from docs/README.md

# Lab 2 Key Notes

Lab 2 uses a **10000x zoomed-in window** over a tiny region:
- More pixels escape quickly (fewer iterations)
- Early exit happens more frequently

Lab 1 uses the full Mandelbrot set window:
- Many pixels are inside the set or near boundaries
- Often needs all 2000 iterations
- Much heavier compute load

---
On top of Lab 1 vector parallelism, Lab 2 adds three parallelism levels:

1. Instruction-Level Parallelism (ILP): parallelism inside a single instruction stream
2. Multi-Core Parallelism: parallelism across physical cores
3. Multi-Threaded Parallelism: multiple threads per core

### Hardware Specs

CPU: AMD Ryzen 7 7700
- 8 cores @ 3.8 GHz
- 2 simultaneous hardware threads per core (SMT)

GPU: NVIDIA RTX 4000 Ada
- 48 SMs @ 2.175 GHz
- 4 warp schedulers per SM
- 192 warp schedulers total (CPU-core analogy)
- Up to 12 active warps per scheduler

---
Actual runs were done on Kaggle, with Tesla P100:
- 56 SMs
- 2 warp schedulers per SM
- 112 warp schedulers total (CPU-core analogy)

---
### Terminology Clarification

| GPU Concept | CPU Analogy |
|-------------|-------------|
| CUDA Thread | Vector Lane |
| Warp        | Thread      |

## Part 0 (Prelab): FMA Instruction Latency Measurement

Deliverable: measure FFMA instruction latency
- Use `fma_latency.cu`
- Use `-s` flag to inspect SASS

Prelab Questions:
1. What is FFMA latency?
2. What do you observe when using interleaved FMA to test ILP?
3. How does the non-explicitly interleaved version perform, and why?
4. When does throughput peak, and how does FMA latency explain it?

## Part 1: ILP in Mandelbrot

Core idea: process multiple independent pixel vectors at the same time.

Deliverables:
- `mandelbrot_cpu_vector_ilp`
- `mandelbrot_gpu_vector_ilp` + launch function
- Keep GPU launch as `<<<1, 32>>>`

Design considerations:
1. How should state variables be managed?
2. How should control flow be handled?
3. CPU vs GPU control-flow strategy differences?
4. How many vectors should be processed simultaneously?
5. How should vectors be mapped from the image (rows? 2D tiles?)

Tool: `#pragma unroll`

### Question 1
What is the ILP speedup? Which strategy was used? How was control flow handled? How many vectors were processed? What are the limiting factors?

## Part 2: Multi-Core Parallelism

### CPU Multi-Core

Deliverable: `mandelbrot_cpu_vector_multicore`
- Use pthread API
- Spawn 8 threads (one per core)
- Join/synchronize all threads

Tools: `pthread_create()`, `pthread_join()`

### Question 2
What is the speedup with 8 cores? How does work partitioning affect it?

### GPU Multi-Core

Deliverable: `mandelbrot_gpu_vector_multicore` + launch
- Goal: run one warp on each of 192 warp schedulers
- Launch config: `<<<48, 4*32>>>`
  - 48 blocks (one per SM)
  - 128 CUDA threads per block (4 warps)

Key variables:
- `threadIdx.x`: index inside block
- `blockIdx.x`: block index
- `gridDim.x`: total number of blocks
- `blockDim.x`: threads per block

Questions:
3. What is the speedup for 192 warp schedulers? How does it compare to CPU? How does partitioning strategy matter?
4. Try `<<<96, 2*32>>>` and `<<<24, 8*32>>>`; what performance differences do you see?

## Part 3: Multi-Threaded Parallelism

### CPU Multi-Threaded

Deliverable: `mandelbrot_cpu_vector_multicore_multithread`
- Use more than one thread per core

Question 5: speedup, best thread count, and deciding factors?

### GPU Multi-Threaded

Deliverable 1: `mandelbrot_cpu_vector_multicore_multithread_single_sm`
- Single block, multiple warps (up to 32)

### Question 6
How does performance change after 4 warps? Does it keep improving up to 32 warps?

Deliverable 2: `mandelbrot_gpu_vector_multicore_multithread_full`
- Full-scale multi-warp, multi-block implementation

### Question 7
What is the speedup? Best warp count? Key deciding factors?

## Part 4: Combine All Techniques

Deliverables:
- `mandelbrot_cpu_vector_multicore_multithread_ilp`
- `mandelbrot_gpu_vector_multicore_multithread_full_ilp`

### Question 8
What is the speedup from ILP + multi-core + multi-threading? How does it compare with single-thread ILP? What are the best parameters?

---
## FMA Latency and Dependent Chains

### 1. What is FMA (Fused Multiply-Add)?

A single instruction performs two operations:

```cpp
// Traditional (two instructions)
temp = a * b;
result = temp + c;

// FMA (one instruction)
result = a * b + c;
```

Benefits:
- Faster (one instruction vs two)
- More precise (no intermediate rounding)
- Better energy efficiency

---
### 2. What is FMA Latency?

Latency = time from instruction issue until result is usable.

```text
Cycle 0: Issue FMA
Cycle 1: Internal compute...
Cycle 2: Internal compute...
Cycle 3: Internal compute...
Cycle 4: Result ready
```

FMA latency = 4 cycles.

#### Latency vs Throughput

| Concept    | Meaning                              | Example            |
|------------|--------------------------------------|--------------------|
| Latency    | Time for one instruction to complete | 4 cycles           |
| Throughput | Instructions issued per cycle        | 1 instruction/cycle |

Analogy:
- Latency: one washer takes 30 minutes to finish one load
- Throughput: with 4 washers, you can start a load every 7.5 minutes

---
### 3. What is a Dependent Chain?

Definition: each instruction depends on the previous result.

```cpp
x = 1.0f;
x = x * x + x;  // inst 1
x = x * x + x;  // inst 2 (wait for inst 1)
x = x * x + x;  // inst 3 (wait for inst 2)
x = x * x + x;  // inst 4 (wait for inst 3)
```

```text
Inst 1: [####]........
Inst 2: ....[####].....
Inst 3: ........[####]..
Inst 4: ............[####]
```

Total: 16 cycles (sequential).

---
### 4. Independent Chains

Definition: no dependency between chains.

```cpp
x = 1.0f;
y = 2.0f;

x = x * x + x;  // chain 1
y = y * y + y;  // chain 2

x = x * x + x;
y = y * y + y;
```

```text
Chain 1 and Chain 2 overlap in time.
Total is lower for the same amount of useful work.
```

---
### 5. Why Dependent Chains Are Slow

Root cause: pipeline stalls.

```text
Cycle 0: issue FMA #1
Cycle 1-3: #2 waits for #1 result
Cycle 4: #1 result ready, #2 can issue
```

Idle cycles are wasted cycles.

---
### 6. How ILP Helps

Even with 4 chains, this is **not** multi-core parallel execution.

In `mandelbrot_cpu_vector_ilp`:
- 4 independent chains exist
- Not 4 CPU cores in parallel
- Not 4 OS threads in parallel

Speedup source: **pipeline overlap** on a single core.

```text
Clock 0: Fetch chain0
Clock 1: Decode chain0, Fetch chain1
Clock 2: Execute chain0, Decode chain1, Fetch chain2
Clock 3: Write chain0, Execute chain1, Decode chain2, Fetch chain3
```

Key point:
- One core
- Multiple independent instructions in different pipeline stages
- Better hardware utilization

---
### 7. Mandelbrot Example

Dependent version:

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

ILP version (4 independent pixels):

```cpp
float x[4] = {0,0,0,0};
float y[4] = {0,0,0,0};
float cx[4] = {...};

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

---
### 8. Visual Comparison

Dependent chain (slow):

```text
Must wait each iteration before the next can proceed.
```

ILP with 4 chains (faster):

```text
Instructions are interleaved in one core pipeline.
Looks parallel, but is overlap, not multi-core execution.
```

Multi-core (Part 2) is true physical parallelism:

```text
8 cores run 8 thread partitions concurrently.
```

---
### 9. Key Formulas

Dependent chain:

```text
Time = N * Latency
```

Independent chains (ILP):

```text
Time = N * Latency / min(ILP_width, hardware_width)
```

---
### 10. Summary

| Concept         | Dependent Chain        | Independent Chains (ILP) |
|----------------|------------------------|---------------------------|
| Definition     | each instruction depends on previous | multiple independent streams |
| Execution      | sequential             | overlapped/parallel at instruction level |
| Latency effect | fully exposed          | partially hidden          |
| Throughput     | low                    | higher                    |
| Utilization    | low                    | high                      |

Core insight:
- Latency is unavoidable
- But latency can be hidden with ILP
- Key is finding independent work

---
## Q&A

### Does 2 FLOPs mean 2 instruction-time units?

Not exactly.
- 2 FLOPs means one multiply + one add in terms of arithmetic work.
- FFMA executes both in one instruction.
- One instruction has latency and throughput properties; FLOP count does not directly map to cycles.

### Is FMA execution time equal to latency?

Not exactly.
- Latency: when result becomes usable by dependent instruction
- Throughput: how many instructions can be issued each cycle
- You can have fixed latency and still sustain high throughput with enough independent work

---
## ILP Reframed

### ILP vs Multi-Core Parallelism

Part 1 (ILP):
- Single thread and one core
- Multiple independent instructions interleaved and overlapped

Part 2 (Multi-Core):
- Multiple threads on multiple physical cores
- True simultaneous execution

### Why ILP speeds up

Modern CPUs can issue/execute independent instructions out of order.
Multiple execution units can run independent ops in the same cycle.

```cpp
for (int chain = 0; chain < 4; chain++) {
    v_x2[chain] = _mm512_mul_ps(v_x[chain], v_x[chain]);
}
```

This gives the CPU scheduling flexibility and better unit utilization.

---
## Is “Instruction-Level Pipeline” a better name than ILP?

It is intuitive, but architecture terminology separates concepts:
- **Pipelining**: mechanism
- **ILP**: amount of independent instruction work available

ILP is realized by multiple mechanisms:
- Pipelining
- Superscalar issue
- Out-of-order execution
- Speculation

So “ILP” stays as the standard term.

---
## Part 2 GPU Multi-Core Detailed Notes

Goal: run one warp on each of 192 warp schedulers.

### Hardware view

```text
48 SMs * 4 warp schedulers = 192 warp schedulers
```

### CUDA launch mapping

Use:

```cpp
<<<48, 128>>>
```

Reason:
- 48 blocks -> maps across 48 SMs
- 128 threads/block -> 4 warps/block
- Total warps = 48 * 4 = 192

Why not `<<<1, 6144>>>`?
- One giant block cannot distribute across all SMs the same way
- It fails the intended one-SM-per-block mapping objective in this lab

### Useful CUDA indices

- `threadIdx.x`: local thread id in block
- `blockIdx.x`: block id
- `blockDim.x`: threads per block
- `gridDim.x`: number of blocks

Global warp id example:

```cpp
uint32_t warp_id_in_block = threadIdx.x / 32;
uint32_t global_warp_id = blockIdx.x * 4 + warp_id_in_block;
uint32_t lane_id = threadIdx.x % 32;
```

### Work partitioning options

Horizontal split:

```cpp
uint32_t rows_per_warp = img_size / 192;
uint32_t start_row = global_warp_id * rows_per_warp;
uint32_t end_row = (global_warp_id == 191) ? img_size : start_row + rows_per_warp;
```

Interleaved split:

```cpp
for (uint32_t i = global_warp_id; i < img_size; i += 192) {
    // process row i
}
```

Skeleton:

```cpp
__global__ void mandelbrot_gpu_vector_multicore(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    uint32_t tid = threadIdx.x;
    uint32_t warp_id_local = tid / 32;
    uint32_t warp_id_global = blockIdx.x * (blockDim.x / 32) + warp_id_local;
    uint32_t lane_id = tid % 32;

    uint32_t rows_per_warp = (img_size + 191) / 192;
    uint32_t start_row = warp_id_global * rows_per_warp;
    uint32_t end_row = min(start_row + rows_per_warp, img_size);

    for (uint32_t i = start_row; i < end_row; ++i) {
        for (uint32_t j = lane_id; j < img_size; j += 32) {
            // Mandelbrot work
        }
    }
}
```

Final takeaway:
- `<<<48, 128>>>` targets full warp-scheduler coverage
- Per-warp partitioning strategy strongly impacts balance and performance

---

## Design Principles & Best Practices

### Priority of Parallelism Levels

When optimizing with limited resources, apply in this order:

1. **Thread-Level Parallelism** (multi-core/SM utilization)
   - True parallel execution on different hardware
   - Provides largest speedups

2. **Multi-Threading** (per-scheduler/core management)
   - Latency hiding via rapid context switching
   - Very effective cost-benefit ratio

3. **ILP** (per-thread instruction chains)
   - Pipeline overlap and interleaving
   - Only effective when thread count insufficient
   - Diminishing returns in well-threaded environments

Rationale: Hardware threads are more powerful than software-level ILP. Use ILP only when hardware limits prevent sufficient threading.

### Resource Management Guidelines

**Register Pressure**
- CPU: 32 vector registers available; limit to 4 ILP chains max
- GPU: 255 registers/thread typical; 4-8 pixel chains before spilling

**Cache Strategy**
- CPU: Horizontal (row-major) access for spatial locality
- GPU: Strided patterns ensure coalesced memory transactions
- Both: Avoid cache thrashing from too many simultaneous threads

**GPU Occupancy**
- Always use 2+ blocks per SM for scheduler flexibility
- Avoid pure register-pressure walls by profiling occupancy

### GPU Launch Configuration Formula

```
Total Warps = (Blocks × Threads/Block) / 32

Best Practice:
- Blocks ≥ SMs (ensure all SMs have work)
- Blocks = 1.5-2.0x SMs (optimal for dynamic load balancing)
- Threads/Block = multiple of 32 (no wasted lanes)
- Threads/Block ≤ 1024 (hardware limit)
```

### Why ILP Effectiveness Drops with Multi-Threading

**Single-Thread Scenario (Part 1)**
- Only mechanism to hide latency
- Significant speedups (1.2-3.0x typical)
- Register cost acceptable

**Multi-Thread Scenario (Parts 3-4)**
- Hardware already switches between threads/warps
- ILP adds:
  - 4x more register pressure
  - Cache contention
  - Complex control flow
- Returns marginal gains (5-15% at best)
- Often degrades performance due to resource competition

**Empirical Evidence from Lab**
- CPU Part 1 (ILP only): 11.5x speedup
- CPU Part 3 (4 threads): 21.5x speedup
- CPU Part 4 (4 threads + ILP): 22.9x speedup (+6.1% over Part 3)
- GPU Part 3 (no ILP): 365x speedup
- GPU Part 4 (with ILP): 157x speedup (-57% regression)

### Warp Scheduling Math

For complete latency hiding:

```
Warps Needed = FMA_Latency / FMA_Throughput

Example (Pascal P100):
- FMA Latency = 11 cycles
- FMA Throughput = 0.5 FMA/cycle
- Warps Needed = 11 / 0.5 = 22... but practical sweet spot is 6 warps/scheduler
- Total Optimal Warps = 112 schedulers × 6 = 672 warps
```

Performance beyond optimal point degrades due to:
- Register spilling to local memory (117-cycle latency per access)
- Occupancy reduction
- Cache thrashing

### Key Metrics for Each Optimization Level

| Metric | Part 1 (ILP) | Part 2 (Multi-Core) | Part 3 (Multi-Thread) | Part 4 (All) |
|---|---|---|---|---|
| Parallelism Type | Software | Hardware | Hardware | Mixed |
| Cost | Register pressure | Memory overhead | Context switching | Register + thread overhead |
| Typical Speedup | 1.2-3.0x | 6-8x | 1.1-1.3x per core | Varies widely |
| Best Use Case | Single-threaded | Utilize all cores/SMs | Hide latency with limited threads | Rarely beneficial |
| Risk Factor | Register spilling | Load imbalance | Diminishing returns | Negative returns |
