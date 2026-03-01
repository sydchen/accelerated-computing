# fma_latency: FFMA Latency and ILP Notes

## What is FMA

- FMA (Fused Multiply-Add) is a single instruction that computes `a*b + c`; in NVIDIA SASS, the float variant is commonly `FFMA`.
- It performs only one rounding step, so it is more accurate than separate multiply then add. In throughput terms, it is often counted as 2 FLOPs (1 multiply + 1 add), while still being a single instruction.
- GPU floating-point units are built around FMA as a core primitive. Two key properties are latency (cycles before a result can be consumed by a dependent instruction) and throughput (how many instructions can be issued per cycle).

### Relationship with cycles

- `clock64` returns the current SM clock counter. `end - start` is the number of cycles consumed by that code section on that SM.
- With a single dependency chain (`fma_latency`), each step must wait for the previous step. Total cycles are approximately: iteration count × single FFMA latency (plus small measurement overhead).
- With two independent interleaved chains (`interleaved`), ILP can overlap latency by issuing an FFMA from one chain while the other is waiting, approaching the throughput limit of roughly one FFMA issued per cycle. Total cycles become noticeably lower than executing both chains serially.
- With non-interleaved two chains (`no_interleave`), total cycles are close to the sum of two single chains, so overlap is weakest.
- To estimate per-FFMA latency, use approximately `(end-start)/100` from `fma_latency`. To reason about throughput behavior, compare `interleaved` vs `no_interleave`. Cycles can be converted to seconds by dividing by GPU core clock frequency.

---
## fma_latency.cu

- The file `fma_latency.cu` measures floating-point FMA latency and the effect of instruction-level parallelism (ILP) using three CUDA kernels.
- It uses `clock64` to read cycle counters before and after pure-compute loops, then reports total cycles.
- The three kernels differ as follows:
  - `fma_latency`: single dependency chain (`x = x*x + x` repeated 100 times), measuring pure latency behavior.
  - `fma_latency_interleaved`: two independent chains interleaved (`x` and `y` updated alternately), creating ILP so hardware/scheduler can overlap latency.
  - `fma_latency_no_interleave`: two independent chains executed sequentially (all `x`, then all `y`), with weaker overlap.
- The host code allocates small buffers, initializes values, runs the kernels in order, and prints `end-start` in cycles.

---
## Extending fma_latency.cu

Additional latency measurements were added, especially for memory-subsystem-related operations:

1. Integer arithmetic: IADD (integer add), IMUL (integer multiply)
2. Memory operations:
   - Global memory load
   - Shared memory load
   - L1 cache latency (via texture memory)
3. Other floating-point operations: FADD, FMUL, FDIV, FSQRT

---
Added test items:

1. Integer arithmetic latency:

- `iadd_latency`: integer add (IADD)
- `imul_latency`: integer multiply + add (IMUL + IADD)

2. Floating-point arithmetic latency:

- `fadd_latency`: floating-point add (FADD)
- `fmul_latency`: floating-point multiply (FMUL)
- `fdiv_latency`: floating-point divide (FDIV)
- `fsqrt_latency`: floating-point square root (FSQRT)

3. Memory subsystem latency ⭐ (brownie points):

- `global_mem_latency`: global memory load latency
  - Uses pointer-chasing pattern; each load depends on the previous result
  - Measures true memory latency (harder for cache/prefetch to hide)
- `shared_mem_latency`: shared memory load latency
  - Also uses pointer-chasing pattern
  - Compares shared-memory vs global-memory latency

Measurement method:

All tests use a dependency chain:
- Each operation depends on the previous result
- Reduces compiler/hardware parallelization opportunities
- Better exposes true instruction latency

Pointer-chasing pattern for memory tests:

```cpp
idx = indices[idx];  // Next location depends on current value
```

Expected results (Tesla P100):

Based on NVIDIA documentation, rough expected latencies are:
- FFMA: ~6 cycles
- FADD/FMUL: ~6 cycles
- FDIV: ~20-30 cycles
- FSQRT: ~20-30 cycles
- IADD: ~6 cycles
- IMUL: ~10 cycles
- Shared memory: ~28 cycles
- Global memory (L2 cache hit): ~200 cycles
- Global memory (DRAM): ~400-800 cycles

---
### Results

1. Floating-point fused multiply-add (FFMA)

- Single dependency chain: 1027 cycles (100 iterations) -> ~10.27 cycles/op
- Interleaved dual-chain (ILP): 971 cycles -> ~9.71 cycles/op
- Non-interleaved dual-chain: 1090 cycles -> ~10.90 cycles/op

Analysis:
- Measured FFMA behavior is around the 10 cycles/op range.
- Interleaving reduces total cycles (1027 -> 971).
- `no_interleave` is slowest, matching the expectation of weaker latency overlap.

2. Integer arithmetic

- IADD: 58 cycles -> ~0.58 cycles/op ⭐
- IMUL+IADD: 515 cycles -> ~5.15 cycles/op

Analysis:
- IADD is very fast, possibly due to multiple integer execution paths or shallow pipeline cost.
- IMUL latency is around 5 cycles, which is within expected range.

3. Floating-point arithmetic

- FADD: 1011 cycles -> ~10.11 cycles/op
- FMUL: 957 cycles -> ~9.57 cycles/op
- FDIV: 12548 cycles -> ~125.48 cycles/op ⚠️
- FSQRT: 11850 cycles -> ~118.5 cycles/op ⚠️

Analysis:
- FADD/FMUL are close to FFMA scale (~10 cycles/op).
- FDIV and FSQRT are very expensive (about 10x+ vs basic FP ops).
- This is why division and square root should be avoided in hot GPU loops when possible.

4. Memory subsystem ⭐ (brownie points)

- Global memory: 11431 cycles -> ~114.31 cycles/op
- Shared memory: 3770 cycles -> ~37.7 cycles/op

Analysis:
- Shared memory is about 3x faster than global memory in this test.
- Global memory latency is around 114 cycles (possibly mostly L2 behavior).
- Shared memory latency around 38 cycles is consistent with on-chip SRAM expectations.

Key finding table:

| Instruction type | Latency (cycles/op) | Relative to FFMA |
|---|---:|---:|
| IADD | 0.58 | 0.05x |
| IMUL+IADD | 5.15 | 0.50x |
| FADD | 10.11 | 0.98x |
| FFMA (ILP) | 9.71 | 0.95x |
| FMUL | 9.57 | 0.93x |
| FFMA (single chain) | 10.27 | 1.00x (baseline) |
| FFMA (no-interleave) | 10.90 | 1.06x |
| Shared memory | 37.7 | 3.67x |
| Global memory | 114.31 | 11.13x |
| FSQRT | 118.5 | 11.54x |
| FDIV | 125.48 | 12.22x |

Conclusion:

1. Integer arithmetic is very fast: IADD is close to one operation per cycle.
2. Basic floating-point ops (FADD/FMUL/FFMA) are around 10 cycles/op.
3. Avoid DIV/SQRT in hot paths: latency is 10x+ compared to basic FP ops.
4. Shared memory is important: about 3x faster than global memory in this setup.
5. ILP benefit: in this measurement, interleaved is about 5.5% faster than single-chain.

These results explain:
- Why Mandelbrot Part 4 (ILP) may show limited gains in heavily threaded scenarios.
- Why shared memory is often preferred.
- Why division should be minimized in performance-critical kernels.
