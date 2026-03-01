# MIT 6.S894 Accelerated Computing - CUDA Learning Portfolio

Course page: https://accelerated-computing.academy/fall25/

This repository is my hands-on implementation portfolio for MIT 6.S894.  
The focus is not only "finishing labs", but also showing:

- I can implement CUDA kernels end-to-end.
- I can reason about performance with profiling and hardware constraints.
- I can document optimization decisions with reproducible evidence.

## Note

This repository is my personal learning notes and implementation record for MIT 6.S894.
Most experiments and runs were executed on Kaggle and Google Colab GPU environments.

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

| Lab | Topic | Key Files |
|---|---|---|
| `lab1` | Mandelbrot, SIMD to GPU parallelism | `mandelbrot_gpu.cu`, `mandelbrot_gpu_parallel.cu`, `PROFILING_GUIDE.md`, `Lab1_SUMMARY.md` |
| `lab2` | Roofline, scheduler intuition, Mandelbrot scaling | `roofline.cu`, `warp_scheduler.cu`, `PART*_IMPLEMENTATION_NOTES.md` |
| `lab3` | Memory-dominated workloads, cache/coalescing | `mem-latency.cu`, `coalesced-loads.cu`, `PART1.md`, `PART2.md` |
| `lab4` | Matmul tiling and reuse | `matmul.cu`, `matmul_cute.cu`, `LAB4_SUMMARY.md` |
| `lab5` | Improved matmul scheduling and occupancy | `matmul_2.cu`, `OCCUPANCY_CONFIG.md`, `PART2_PERFORMANCE_ANALYSIS.md` |
| `lab6` | Tensor Core matmul (WMMA/MMA) | `exercise_mma.cu`, `matmul_3.cu`, `PART0.md` |
| `lab7` | Warp shuffle + parallel scan + RLE | `shuffle.cu`, `scan.cu`, `rle_compress.cu`, `LAB7_SUMMARY.md` |
| `lab8` | GPU renderer (10M circles) | `circles.cu`, `LAB8_SUMMARY.md` |
| `lab9` | H100 TMA and warp scheduler | `0-tma-single-load.cu` ... `5-tma-swizzle.cu`, `LAB9_SUMMARY.md` |
| `lab10` | H100 WGMMA / swizzle matmul | `0-m64n8k16-wgmma.cu`, `1-swizzle-m64n8k32-wgmma.cu`, `h100-matmul.cu` |
| `lab11` | TPU collectives and tensor parallelism (JAX/Pallas) | `collectives.py`, `collective_matmul.py`, `LAB11_SUMMARY.md` |

## Reproducibility

### Environment

- NVIDIA GPU (labs vary by architecture; some parts target P100/T4/H100)
- CUDA toolkit + `nvcc`
- Python 3 (for helper scripts/plots in several labs)

### Typical Build/Run Pattern

Some labs are hardware-specific (for example H100 TMA/WGMMA in `lab9`/`lab10`), so results should be interpreted relative to the target GPU.

