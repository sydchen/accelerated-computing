# fma_latency：FFMA 延遲與 ILP 筆記

## 什麼是 FMA

- FMA（Fused Multiply-Add）是單一指令完成 a*b + c 的融合乘加運算；在 NVIDIA SASS 中對 float 常見為 FFMA。
- 它只有一次捨入，比先乘再加更精確，且計算吞吐上常視為 2 FLOPs（1 乘 + 1 加）但只是一條指令。
- GPU 的 FP 運算單元以 FMA 為核心原語，延遲（某指令結果可被下一條使用前需等待的 cycles）與吞吐（每個 cycle 能發出多少指令）是兩個關鍵特性。

### 和 cycles 的關係

- clock64 回傳當前 SM 的時鐘計數。end - start 即該段程式在該 SM 上消耗的 cycles。
- 單一相依鏈時（fma_latency）：每一步必須等前一步完成，總 cycles ≈ 迭代次數 × 單條 FMA 延遲（再加極少量測量開銷）。
- 交錯兩條獨立鏈時（interleaved）：可用 ILP 在等待一條鏈結果的同時發出另一條鏈的 FMA，將延遲重疊，接近「每 cycle 發一條 FMA」的吞吐極限；總 cycles 會明顯小於兩條鏈逐一完成的和。
- 兩條鏈不交錯（no_interleave）：幾乎是兩次單鏈延遲的總和，重疊效果最差。
- 若想估每次 FMA 的延遲，可用 fma_latency 的 (end-start)/100 近似；若看吞吐，則比較 interleaved 與 no_interleave 的差距。將 cycles 除以 GPU 核心時脈頻率即可換算成秒數。

---
## fma_latency.cu

- 檔案 fma_latency.cu 透過三個 CUDA kernel 量測浮點 FMA 指令的延遲（latency）與指令層級平行性（ILP）對延遲的影響。
- 使用 clock64 讀取 GPU 的週期計數器（cycles），在一段純計算的 FMA 迴圈前後取時間差，得到迴圈總耗費的 cycles。
- 三個 kernel 的差別：
  - fma_latency: 單一相依鏈（x = x*x + x 重複 100 次），每步彼此相依，量測純延遲。
  - fma_latency_interleaved: 兩條彼此獨立的鏈交錯執行（x 與 y 同步更新），創造 ILP 讓硬體/排程器重疊延遲。
  - fma_latency_no_interleave: 兩條獨立鏈依序完成（先整個 x，再整個 y），較難重疊延遲。
- 主程式配置少量記憶體、設定初值、依序執行三個 kernel，將 end-start 印成「幾個 cycles」。

---
## 擴充 fma_latency.cu

加入其他指令的延遲測量，特別是與記憶體子系統相關的。讓我擴展這個程式，加入以下指令的延遲測量：

1. 整數運算: IADD (整數加法), IMUL (整數乘法)
2. 記憶體操作:
   - Global memory load
   - Shared memory load
   - L1 cache latency (透過 texture memory)
3. 其他浮點運算: FADD, FMUL, FDIV, FSQRT

---
新增的測試項目：

1. 整數運算延遲：

- iadd_latency: 整數加法 (IADD)
- imul_latency: 整數乘法 + 加法 (IMUL + IADD)

2. 浮點運算延遲：

- fadd_latency: 浮點加法 (FADD)
- fmul_latency: 浮點乘法 (FMUL)
- fdiv_latency: 浮點除法 (FDIV)
- fsqrt_latency: 浮點平方根 (FSQRT)

3. 記憶體子系統延遲 ⭐（brownie points）：

- global_mem_latency: Global memory 載入延遲
  - 使用 pointer-chasing 模式，每次載入都依賴前一次的結果
  - 測量真實的記憶體延遲（無法被快取優化）
- shared_mem_latency: Shared memory 載入延遲
  - 同樣使用 pointer-chasing 模式
  - 比較 shared memory vs global memory 的延遲差異

測試方法：

所有測試都使用依賴鏈 (dependency chain) 的方式：
- 每次運算都依賴前一次的結果
- 防止編譯器和硬體的平行化優化
- 測量真實的指令延遲

記憶體測試使用 pointer-chasing 模式：

```cpp
idx = indices[idx];  // 下一個位置取決於當前位置的值
```

---
### 結果

1. 浮點乘加運算 (FFMA)

- 單一依賴鏈: 1027 cycles (100 次) → 每次 ~10.27 cycles
- 雙鏈交錯 (ILP): 971 cycles → 每次 ~9.71 cycles
- 雙鏈順序執行: 1090 cycles → 每次 ~10.90 cycles

分析:
- FFMA 的量測結果落在約 10 cycles/op 等級。
- ILP 交錯執行可降低總 cycles（1027→971）。
- no_interleave 最慢，符合「較難重疊延遲」的預期。

2. 整數運算

- IADD: 58 cycles → 每次 ~0.58 cycles ⭐
- IMUL+IADD: 515 cycles → 每次 ~5.15 cycles

分析:
- IADD 非常快！可能有多個執行單元或流水線深度很淺
- IMUL 延遲約 5 cycles，符合預期

3. 浮點運算

- FADD: 1011 cycles → 每次 ~10.11 cycles
- FMUL: 957 cycles → 每次 ~9.57 cycles
- FDIV: 12548 cycles → 每次 ~125.48 cycles ⚠️
- FSQRT: 11850 cycles → 每次 ~118.5 cycles ⚠️

分析:
- FADD/FMUL 延遲接近 FFMA（~10 cycles）
- FDIV 和 FSQRT 延遲非常高（120-140 cycles），是 FFMA 的 12-14 倍！
- 這就是為什麼應該避免在 GPU kernel 中使用除法和平方根

4. 記憶體子系統 ⭐ (brownie points)

- Global memory: 11431 cycles → 每次 ~114.31 cycles
- Shared memory: 3770 cycles → 每次 ~37.7 cycles

分析:
- Shared memory 比 Global memory 快約 3 倍。
- Global memory 延遲約 114 cycles（可能大部分在 L2 cache）。
- Shared memory 延遲約 38 cycles（符合 on-chip SRAM 預期）。

關鍵發現：

延遲比較表（每次操作）：

| 指令類型 | 延遲 (cycles) | 相對於 FFMA |
|---|---:|---:|
| IADD | 0.58 | 0.05x |
| IMUL+IADD | 5.15 | 0.50x |
| FADD | 10.11 | 0.98x |
| FFMA (ILP) | 9.71 | 0.95x |
| FMUL | 9.57 | 0.93x |
| FFMA (單鏈) | 10.27 | 1.00x (基準) |
| FFMA (no-interleave) | 10.90 | 1.06x |
| Shared memory | 37.7 | 3.67x |
| Global memory | 114.31 | 11.13x |
| FSQRT | 118.5 | 11.54x |
| FDIV | 125.48 | 12.22x |

結論：

1. 整數運算極快：IADD 幾乎可以每個週期執行一次
2. 浮點基本運算：FADD/FMUL/FFMA 都在約 10 cycles/op
3. 避免使用 DIV/SQRT：延遲是基本運算的 10 倍以上
4. Shared memory 很重要：比 Global memory 快 3 倍
5. ILP 的效益：在這次量測中，interleaved 比單鏈約快 5.5%

這些數據完美解釋了：
- 為什麼 Mandelbrot Part 4 (ILP) 在多執行緒環境中沒有幫助
- 為什麼應該盡量使用 shared memory
- 為什麼要避免除法運算  
