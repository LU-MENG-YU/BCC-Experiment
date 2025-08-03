# BCC Experiment

## 目錄
- [背景簡介](#背景簡介)  
- [環境要求](#環境要求)  
- [安裝與執行](#安裝與執行)  
- [程式架構與檔案說明](#程式架構與檔案說明)  
  - [β-metric 矩陣生成說明](#β-metric-矩陣生成說明)  
- [函式與論文對應](#函式與論文對應)  
- [整體執行流程](#整體執行流程)  

---

## 背景簡介
本程式實作 J. Li & P. Zhang (2023) 單 Depot Budgeted Cycle Cover（BCC）問題的 **Algorithm S**，並在各層中嵌入 Arkin et al. (2006) 的 **Path_Min_k** 3-approximation 來解子問題。  
使用者可一次 sweep 多組參數 \((n, \beta, k, \text{seed})\)，自動輸出結果並繪製折線圖，用於分析不同 β 與預算倍率 k 下 cycle 數的變化。

---

## 環境要求
- Python 3.8+  
- 建議使用虛擬環境（venv）  
- 需要套件  
  ```bash
  pip install numpy pandas matplotlib
  ```

---

## 安裝與執行
1. **下載程式**  
   把 `bcc_experiment.py` 拉到你的 WSL 專案資料夾。

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
   - `plot_n10_vs_beta.png`、`plot_n10_vs_k.png`  
   - `plot_n50_vs_beta.png`、`plot_n50_vs_k.png`  

---

## 程式架構與檔案說明

- **`bcc_experiment.py`**  

  ### β-metric 矩陣生成說明
  - **函式**：`gen_beta_metric(n, beta, seed)`  
  - **目的**：在「度量空間」直接生成一個符合 β‑metric 條件的距離矩陣，不受限於任意坐標系。  
  - **步驟**：  
    1. 隨機產生各節點到 depot（root=0）的距離  
       ```python
       d_min, d_max = 1.0, (1+beta)/(1-beta)
       dr = np.random.uniform(d_min, d_max, size=n)
       ```  
    2. 對每對節點 (i,j) 產生距離 w  
       ```python
       low  = abs(dr[i] - dr[j])            # 度量性下界
       high = beta * (dr[i] + dr[j])        # β-鬆弛三角上界
       w    = random.uniform(low, high)     # 均勻抽樣
       W[i,j] = W[j,i] = w
       ```  
    3. 回傳 `(dr, W)`，其中 dr[i] = d(root, i)，W = n×n 對稱矩陣  

  - `bpc_approx(graph, lam)`  
    - 實作 Path_Min_k（MST + 路徑拆分）  
    - 對應文獻：Fig. 2 Algorithm Path_Min_k  

  - `bcc_algorithm_s(dr, W, B)`  
    - 實作 Algorithm 2.1 (Step 1–4)：  
      1. **Step 1** 移除「遠節點」 dr[v] ≥ B/2  
      2. **Step 2** 計算 μ, ε=2μ/B, t=⌈log₂(1/ε)⌉，依 Eq.(4) 分層  
      3. **Step 3** 每層呼叫 `bpc_approx(subW, lam)` 得到 paths  
      4. **Step 4** 把 paths 組成 cycles  

- `run_experiments()`  
  - Sweep (n, β, k, seed)、寫入 `results.csv`

- `plot_results(df)`  
  - 依 β、k 計算平均 cycle，繪製折線圖

---

## 函式與論文對應

| 論文/公式                  | 程式位置 & 函式                   |
|---------------------------|---------------------------------|
| 距離矩陣生成               | `gen_beta_metric()`             |
| Algorithm 2.1 Step 1      | `bcc_algorithm_s()` 中 `far = […]` |
| Eq.(4) 層劃分             | `layers[j-1].append(v)`         |
| Algorithm 2.1 Step 2      | 計算 `mu`, `eps`, `t`            |
| Algorithm 2.1 Step 3      | `paths = bpc_approx(subW, lam)` |
| Algorithm 2.1 Step 4      | 組裝 cycles                      |
| Path_Min_k                | `bpc_approx()`                  |
| 批次實驗                  | `run_experiments()`             |
| 視覺化                    | `plot_results()`                |

---

## 整體執行流程

1. **生成 β‑metric 矩陣**  
   `gen_beta_metric()` → `(dr, W)`  

2. **Algorithm S 主流程**  
   `bcc_algorithm_s(dr, W, B)` → cycles  

3. **Path_Min_k 子例程**  
   `bpc_approx()` → 3‑approx paths  

4. **批次實驗 & 輸出**  
   `run_experiments()` → `results.csv`  

5. **繪圖**  
   `plot_results()` → PNG 圖檔  

---
