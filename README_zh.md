# MIT 6.S894 加速運算

課程頁面：https://accelerated-computing.academy/fall25/

本儲存庫是我在 MIT 6.S894 課程中的實戰實作作品集。
重點不僅僅是「完成實驗」，還要展示：

- 我能夠從頭到尾實作 CUDA 核心。
- 我能夠透過性能分析和硬體約束進行性能推理。
- 我能夠用可重現的證據記錄最佳化決策。

## 說明

本儲存庫是我個人在 MIT 6.S894 課程中的學習筆記和實作記錄。
大多數實驗和執行都是在 Kaggle 和 Google Colab GPU 環境上進行的。

**致謝：** 本儲存庫中的實作和文件是在 Anthropic 的人工智慧編程助手 Claude Code 的協助下開發完成的。在整個課程中，Claude Code 在程式碼開發、最佳化策略和文件編寫方面提供了幫助。

## 本作品集展示的內容

### CUDA 工程基礎

- 並行分解：grid/block/warp 映射
- 記憶體階層最佳化：全域記憶體/L2/L1/共享記憶體/暫存器複用
- 控制流行為：分歧、謂詞化、warp 層級編程
- 排程和佔用率權衡（Little's Law、延遲隱藏）
- Tensor Core 編程（WMMA / MMA / WGMMA）

### 性能工作流程

- 基線 -> 假設 -> 核心修改 -> 測量 -> 結論
- 執行時間基準測試和正確性檢查
- 性能分析驅動的調優（透過實驗室特定的筆記和指令碼）

## 實驗映射（實作 + 筆記）

| 實驗 | 主題 | 關鍵檔案 |
|---|---|---|
| `lab1` | Mandelbrot、從 SIMD 到 GPU 並行化 | `mandelbrot_gpu.cu`, `mandelbrot_gpu_parallel.cu`, `PROFILING_GUIDE.md`, `Lab1_SUMMARY.md` |
| `lab2` | Roofline、排程器直覺、Mandelbrot 擴展 | `roofline.cu`, `warp_scheduler.cu`, `PART*_IMPLEMENTATION_NOTES.md` |
| `lab3` | 記憶體密集型工作負載、快取/合併 | `mem-latency.cu`, `coalesced-loads.cu`, `PART1.md`, `PART2.md` |
| `lab4` | 矩陣乘法分塊和複用 | `matmul.cu`, `matmul_cute.cu`, `LAB4_SUMMARY.md` |
| `lab5` | 改進的矩陣乘法排程和佔用率 | `matmul_2.cu`, `OCCUPANCY_CONFIG.md`, `PART2_PERFORMANCE_ANALYSIS.md` |
| `lab6` | Tensor Core 矩陣乘法（WMMA/MMA） | `exercise_mma.cu`, `matmul_3.cu`, `PART0.md` |
| `lab7` | Warp shuffle + 並行掃描 + RLE | `shuffle.cu`, `scan.cu`, `rle_compress.cu`, `LAB7_SUMMARY.md` |
| `lab8` | GPU 轉譯器（1000萬個圓形） | `circles.cu`, `LAB8_SUMMARY.md` |
| `lab9` | H100 TMA 和 warp 排程器 | `0-tma-single-load.cu` ... `5-tma-swizzle.cu`, `LAB9_SUMMARY.md` |
| `lab10` | H100 WGMMA / swizzle 矩陣乘法 | `0-m64n8k16-wgmma.cu`, `1-swizzle-m64n8k32-wgmma.cu`, `h100-matmul.cu` |
| `lab11` | TPU 集合操作和張量並行（JAX/Pallas） | `collectives.py`, `collective_matmul.py`, `LAB11_SUMMARY.md` |
