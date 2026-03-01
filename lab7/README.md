# Lab 7: Run-Length Compression

**MIT 6.S894 - Accelerated Computing (Fall 2025)**

Official Course Resource: [Lab 7 - Run-Length Compression](https://accelerated-computing.academy/fall25/labs/lab7/)

## Core Theme

This lab shifts from "regular memory access patterns" to **"irregular memory access patterns"**—a characteristic of many real-world applications. Data dependencies are unpredictable, memory accesses are incoherent, which is where parallel computing becomes challenging.

## Performance Targets

| Part | Task | Target | Test Scale |
|------|------|--------|-----------|
| **0** | Warp Shuffle vs Shared Memory | Performance Comparison | 32 elements |
| **1** | Parallel Scan | ≥144 GB/s (40% of peak) | 2²⁶ (~67M) elements |
| **2** | RLE Compression | ≤1.6 ms | ~16MB file |
| **3** | RLE Decompression | ≤0.8 ms | Dynamic allocation |

## What This Lab Does

We built a GPU-based Run-Length Encoding compression system. Simply put, it compresses consecutive repeated values into a "value + count" format, then accelerates it using warp shuffle and parallel scan.

Core techniques: Warp Shuffle (zero-latency intra-warp communication) + Parallel Scan (hierarchical scanning)

---

## Part 0: Warp Shuffle Experiment

### Problem

Implement cumulative sum for 32 elements, comparing two approaches:
1. Using shared memory
2. Using warp shuffle

### How Much Faster

| Method | Clock Cycles | Speedup |
|--------|------------|---------|
| Shared Memory | 1659 | Baseline |
| **Warp Shuffle** | **264** | **6.28× faster** |

This result was a bit surprising. Expected 3-5×, but got 6× directly.

### Why the Huge Difference?

**Shared memory version's problems**:
- Each iteration requires: read memory → sync → write memory → sync again
- Loop runs 5 times (log₂32), each with double synchronization overhead
- L1/L2 cache contention and bank conflicts

**Warp shuffle's elegance**:
- The 32 threads in a warp are naturally synchronized (lockstep execution)
- Use `__shfl_up_sync()` to transfer values directly between registers
- No shared memory reads/writes, no synchronization waits
- Hardware-native support with ultra-low latency

### Real Experience

When using shuffle, I realized how elegant GPU design is. A warp is the GPU's fundamental execution unit, naturally "all doing the same thing simultaneously." Since 32 threads are already in lockstep, why bother using shared memory and explicit synchronization to simulate? Just let the hardware transfer values directly—that's it.

---

## Part 1: Parallel Scan (Inclusive Scan)

### Core Idea

To compute prefix sum for N elements, we can't solve it at the block level alone because N can be huge (millions). So we use three levels:

```
Level 1: Intra-block scan (256 threads, shared memory + warp shuffle)
  └─ Each block outputs one aggregate value (cumulative sum of last element)

Level 2: Scan block aggregate values
  └─ Get each block's "starting offset" to add

Level 3: Fix-up (add offset back)
  └─ Each block's elements add their offset, completing global scan
```

### Intra-block Scan Strategy

Each block has 8 warps (256 threads). Strategy:

1. **Intra-warp scan** (using warp shuffle)
   - Each warp's 32 elements scanned once with shuffle
   - Fastest, no synchronization overhead

2. **Aggregate 8 warp results**
   - Extract each warp's final cumulative value
   - Scan these 8 numbers
   - Get each warp's offset to add

3. **Distribute offsets**
   - Broadcast each warp's offset to its 32 threads
   - Each thread adds its value plus the offset

The benefit is fully leveraging warp shuffle's speed while using shared memory for block-level synchronization.

### Performance Target

Achieve ≥144 GB/s (40% of RTX 4000 Ada's 360 GB/s peak)

I think the 40% target is very realistic. Memory-bound algorithms rarely exceed 60-70%, and scanning involves extensive global memory access with synchronization overhead. Achieving 40% is quite good implementation.

### Common Pitfalls

**Warp-level vs Block-level**: Easy to confuse warp-internal and block-internal synchronization. Warp needs no sync (lockstep), block needs `__syncthreads()`.

**Cross-block communication**: Threads in different blocks can't directly synchronize, so must split into multiple kernels or use global barrier. This is a design constraint.

**Boundary cases**: When data size isn't a multiple of 256, the last block needs special handling.

### Implementation Details

We discovered several gotchas when doing the scan:

**Boundary marking**: Use auxiliary arrays to record run start positions. Each thread compares itself with the previous element, producing a boolean value. The implicit structure of this boolean array is important—it determines subsequent scan steps.

**Dynamic shared memory**: Since the assignment restricts CUB library usage, we manually allocate shared memory. Be careful: in C++, dynamic shared memory requires casting through char* pointers. `__shared__ char sm[];` then manually calculate offsets. Got stuck initially with wrong offset calculations causing memory corruption.

**Test scale**: Experiments use 2²⁶ (~67 million) elements. At this scale, scan overhead becomes significant. Smaller scales don't show the effect.

---

## Part 2: RLE Compression

### The Actual Problem

Given a pixel array with many consecutive repeated values, compress them into "value + count" format.

Example:
```
Input:  [255, 255, 255, 0, 0, 128, 128, 128, 128]
Output: [255, 3, 0, 2, 128, 4]
```

### How to Parallelize

The trick here is that a single thread can't solve this independently—you don't know "how many times does this value repeat?" You must look ahead. That's why we need scan.

**Step 1: Find boundaries**
```
Compare adjacent elements. If A[i] ≠ A[i-1], it's a new run start
Example: [255, 255, 255, 0, 0, 128, ...]
Boundaries: [1, 0, 0, 1, 0, 1, ...]  ← Mark run start positions
```

**Step 2: Scan boundaries**
```
Prefix sum on boundary markers to get "run ID"
Boundaries: [1, 0, 0, 1, 0, 1, ...]
Run ID:     [1, 1, 1, 2, 2, 3, ...]
```

**Step 3: Compaction**
```
Only threads at boundaries execute output:
- Output current pixel value
- Calculate run length (next run start position - current position)
- Write to output buffer
```

### Why This Design?

Intuitively it seems like "each thread calculates its own," but actually it can't. Because:
- You don't know your value's repetition count (depends on later data)
- Output position is also fixed (depends on earlier runs)

So we must use scan—a "global perspective" operation. This is typical of parallel algorithms: when local operations can't solve it, use global scan to transform into parallel form.

### Performance Considerations

**Reading**: Sequential input reads, very fast
**Writing**: Sparse output writes, depends on run density. If most pixels differ (high entropy), output ≈ input, poor compression. If many repeats, output is small.

Memory bandwidth utilization depends entirely on data's run-length distribution.

### Implementation Details

This part's optimization space lies in "boundary marking" and "output strategy":

**Boundary marking parallelization**: Each thread independently compares itself with the previous element. First thread always marks 1 (run start). Be careful at boundaries: the next element after the last doesn't exist, can't read past boundary.

**Sparse output position calculation**: After scan gives run IDs, only run boundary threads write output. Output positions calculated directly from scan results. Easiest mistake: when calculating run length, can't assume next run's position is known since some threads don't execute output at all.

**Test scale**: ~16MB file. At this size, I/O cost isn't the bottleneck; focus is on algorithm correctness and memory access efficiency.

---

## Part 3: RLE Decompression (Optional)

### Reverse Operation

Given a compressed array, expand it back to original pixels.

```
Input:  [255, 3, 0, 2, 128, 4]
Output: [255, 255, 255, 0, 0, 128, 128, 128, 128]
```

### Strategy

1. Each (value, count) pair calculates its starting position in output
2. Multiple threads write this run's pixels in parallel

For example, (255, 3) corresponds to output positions [0, 1, 2], which 3 threads can write in parallel.

Long runs can be distributed to multiple threads, fully utilizing parallelism.

### Implementation Details

Decompression is more interesting than compression since output size is unknown beforehand:

**Dynamic memory allocation**: Output size depends on sum of all run counts. The assignment requires `GpuAllocCache` to manage temporary memory. First scan the input to get total size, then allocate output array dynamically.

**Parallel scatter writes**: Each (value, count) pair can be handled by multiple threads in parallel. If count=1000, use 1000 threads to write 1000 pixels in parallel. This requires prefix scan to get each pair's starting position.

**Load balancing**: If some runs are long (count=10000) while others are short (count=1), execution becomes uneven. Simple strategy: each thread writes its assigned position range. More complex approaches can use warp-level synchronization for further optimization.

**Target**: Complete decompression in ≤0.8 ms. This is quite challenging for million-scale outputs.

---

## Lessons from Actual Implementation

### How Warp Shuffle Changed Things

Before, I had "shared memory + sync" mentality. Then realized warp level has another way. Once understanding lockstep execution, you can fully leverage hardware characteristics. The 6× speedup demonstrates what "aligning with hardware design philosophy" means.

### Parallel Scan is a Great Example

Scan looks simple, but implementation involves many layers of consideration:
- Warp level (no sync)
- Block level (`__syncthreads()`)
- Global (multiple kernels or global barrier)

Each level has different synchronization mechanisms and performance characteristics. This taught me that parallel algorithms aren't just "write loops"—details matter enormously.

### RLE Application is Quite Interesting

Compression unlike matrix multiplication is "computation-intensive"—it's "data-dependency intensive." Must use scan techniques to break dependencies for parallelism. This helped me understand why scan is fundamental in parallel computing—it solves "how to derive global state from local dependencies."

---

## What This Lab Taught Me

**Warp is the minimal execution unit**
- 32 threads naturally lockstep
- Can transfer values via shuffle at zero cost
- Much faster than shared memory

**Parallel scan is more than "one algorithm"**
- It's hierarchical
- Each level has different sync mechanisms
- Done well, achieves 40% peak efficiency

**Algorithm and hardware design must align**
- Go against hardware design (unnecessary sync), performance suffers
- Align with hardware (warp shuffle), get several times speedup

**Parallelization isn't just adding parallel markers**
- Some problems need scan, gather operations—global operations—to break
- Must understand data dependencies, choose right algorithm

---

## Further Exploration

Directions worth trying:
- **Larger scans**: Millions of elements, how does performance scale
- **Custom reduction**: Beyond addition, try other operators
- **Multi-GPU**: Global scan across GPUs
- **Other applications**: Sort, prefix-sum related problems
- **Stream processing**: Real-time compress while reading

---

## Summary

Lab 7 is the process of upgrading from "knowing these operations exist" to "understanding why they're so fast." Warp shuffle and parallel scan seem like just two techniques, but actually open a new perspective—how to leverage hardware's own design to implement efficient parallel algorithms.

