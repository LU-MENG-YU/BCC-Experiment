# BCC Experiment

## 目錄
- [背景簡介](#背景簡介)  
- [環境要求](#環境要求)  
- [安裝與執行](#安裝與執行)  
- [程式架構與檔案說明](#程式架構與檔案說明) 
- [函式與論文對應](#函式與論文對應)  
- [整體執行流程](#整體執行流程)  

---

## 背景簡介
本程式實作 J. Li & P. Zhang (2023) Single Depot Budgeted Cycle Cover（BCC）問題的 **Algorithm S**，並在各層中嵌入 Arkin et al. (2006) 的 **Path_Min_k** 3-approximation 來解子問題。  
使用者可一次 sweep 多組參數 \((n, \beta, k, \text{seed})\)，自動輸出結果並繪製折線圖，用於分析不同 β 與預算倍率 k 下 cycle 數的變化。

---

## 環境要求
- Ubuntu 24.04.1 LTS (GNU/Linux 6.6.87.2-microsoft-standard-WSL2 x86_64)
- Python 3.8+  
- 建議使用虛擬環境（venv）  
- 需要套件  
  ```bash
  pip install numpy pandas matplotlib
  ```

---

## 安裝與執行
1. **下載程式**  
   把 `bcc_experiment.py` 拉到你的 WSL 專案資料夾，例如 `~/BCCproj`。

2. **建立並啟動虛擬環境**  
   ```bash
   cd ~/BCCproj
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **安裝套件**  
   ```bash
   pip install numpy pandas matplotlib
   ```

4. **（可選）調整參數**  
   編輯檔案最上方的設定區段：
   ```python
   n_list     = [10, 50]
   beta_list  = [0.5, 0.55, …, 0.95]
   k_list     = [1.5, 2.0, …, 10.0]
   seeds      = [0, 1, 2]
   ```

5. **執行程式**  
   ```bash
   python3 bcc_experiment.py
   ```

6. **檔案輸出**  
   - `results.csv`：實驗結果  
   - `plot_n10_vs_beta.png`、`plot_n10_vs_k.png`  ：統計圖表
   - `plot_n50_vs_beta.png`、`plot_n50_vs_k.png`  ：統計圖表

---

## 程式架構與檔案說明

- **`bcc_experiment.py`**  
  - `gen_graph(n, beta, seed)`  
    - 生成 \(n\) 個隨機點、計算歐氏距離，並線性映射到 [1,(1+β)/(1-β)] 

  - `bpc_approx(graph, lam)`  
    - 實作 Path_Min_k 最小生成樹 + 路徑拆分  
    - **對應論文：** Fig. 2 Algorithm Path_Min_k（3-approximation）  

   - `bcc_algorithm_s(graph, B)`  
    - 實作 Algorithm 2.1 (Step 1–4)：  
      1. **Step 1** 移除距離 ≥ B/2 的「遠節點」  
      2. **Step 2** 計算 μ、ε = 2μ / B，t = ⌈log₂(1 / ε)⌉，並依 Eq.(4) 分層  
      3. **Step 3** 對每層呼叫 `bpc_approx(subgraph, 2⁻ʲ ε B)`  
      4. **Step 4** 將 path 加上 root 後轉成 cycle  
    - 回傳所有 cycles 數量  

  - `run_experiments()`  
    - 遍歷所有 \((n, β, k, seed)\) 組合，呼叫 `gen_graph` → `bcc_algorithm_s`，寫入 `results.csv`  

  - `plot_results(df)`  
    - 讀取 DataFrame，依 β、k 計算平均 cycle 數，並繪製折線圖  

---

## 函式與論文對應

| 論文/公式                  | 程式位置 & 函式                                 |
|---------------------------|-----------------------------------------------|
| **Algorithm 2.1 Step 1**  | `bcc_algorithm_s()` 中 `far_nodes = [...]`      |
| **Eq.(4) 層劃分**         | `bcc_algorithm_s()` 中 `layers[j-1].append(v)` |
| **Algorithm 2.1 Step 2** | 計算 `μ`, `ε`, `t`                          |
| **Algorithm 2.1 Step 3** | `paths = bpc_approx(subgraph, lam)`            |
| **Fig. 2 Path_Min_k**     | `bpc_approx()`                                 |
| **Algorithm 2.1 Step 4** | 在 `bcc_algorithm_s()` 中組裝 cycle             |
| **實驗流程**              | `run_experiments()`                            |
| **視覺化**                | `plot_results()`                               |

---

## 整體執行流程

1. **Graph Generation**  
   `gen_graph()` → 取得距離矩陣  

2. **Main Algorithm S**  
   `bcc_algorithm_s()` →  
   - 處理「遠節點」  
   - 計算層級  
   - Layer j → 呼叫 `bpc_approx()`  
   - 合併為 cycles  

3. **Path_Min_k 子例程**  
   `bpc_approx()` → MST + 路徑拆分  

4. **批次實驗**  
   `run_experiments()` → CSV 輸出  

5. **繪圖**  
   `plot_results()` → PNG 圖檔  
