# MIT 6.S894 Accelerated Computing

Course page: https://accelerated-computing.academy/fall25/

## Note

This repository is my personal learning notes and implementation record for MIT 6.S894.
Most experiments and runs were executed on Kaggle and Google Colab GPU environments.

**Acknowledgment:** The implementation and documentation in this repository have been developed with the assistance of Claude Code, an AI coding assistant by Anthropic. Claude Code has helped with code development, optimization strategies, and documentation throughout the course.

## What This Portfolio Demonstrates

### CUDA Engineering Fundamentals

- Parallel decomposition: grid/block/warp mapping
- Memory hierarchy optimization: global/L2/L1/shared/register reuse
- Control-flow behavior: divergence, predication, warp-level programming
- Scheduling and occupancy tradeoffs (Little's Law, latency hiding)
- Tensor Core programming (WMMA / MMA / WGMMA)

### Performance Workflow

- Baseline -> hypothesis -> kernel change -> measurement -> conclusion
- Runtime benchmarking and correctness checks
- Profiling-driven tuning (through lab-specific notes and scripts)

## Lab Map (Implementation + Notes)

| Lab | Topic | Learning Objectives |
|---|---|---|
| `lab1` | Mandelbrot, SIMD to GPU parallelism | Understand parallel decomposition and GPU memory layout; Profile and optimize single-kernel implementations |
| `lab2` | Roofline, scheduler intuition, Mandelbrot scaling | Analyze performance with Roofline Model; Understand compute-bound vs memory-bound workloads |
| `lab3` | Memory-dominated workloads, cache/coalescing | Optimize memory access patterns; Explore cache hierarchy and memory coalescing strategies |
| `lab4` | Matmul tiling and reuse | Implement data reuse across memory levels; Optimize L1/L2 cache and shared memory usage |
| `lab5` | Improved matmul scheduling and occupancy | Balance occupancy vs register pressure; Apply scheduling and blocking optimizations |
| `lab6` | Tensor Core matmul (WMMA/MMA) | Leverage hardware accelerators (WMMA/MMA); Achieve significant speedup with specialized instructions |
| `lab7` | Warp shuffle + parallel scan + RLE | Implement intra-warp communication; Design efficient scan and compression algorithms |
| `lab8` | GPU renderer (10M circles) | Build complex rendering workloads; Manage large-scale parallel processing and load balancing |
| `lab9` | H100 TMA and warp scheduler | Understand advanced H100 features (TMA); Optimize data movement with tensor memory acceleration |
| `lab10` | H100 WGMMA / swizzle matmul | Master WGMMA programming; Implement swizzle-based memory layout optimizations |
| `lab11` | TPU collectives and tensor parallelism (JAX/Pallas) | Explore distributed computing; Apply collective communication and tensor parallelism patterns |

