import numpy as np
import random
import csv
import math
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product

# ====== 實驗參數（可自訂） ======
n_list    = [50]
beta_list = [0.5,0.6,0.7,0.8,0.9,0.99]  # 0.5, 0.55, …, 0.95
k_list    = [50]   # 1.5, 2.0, …, 10.0
seeds     = list(range(10001))

# ====== 生成任意維度 β-metric 矩陣 ======
def gen_beta_metric(n, beta, seed):
    seed = int(seed)               # 確保 seed 為 Python int
    random.seed(seed)
    np.random.seed(seed)
    # 定義到 root(r=0) 的距離 dr[i]
    d_min, d_max = 1.0, (1 + beta)/(1 - beta)
    dr = np.random.uniform(d_min, d_max, size=n)
    # 生成完整距離矩陣 W
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            low  = abs(dr[i] - dr[j])
            high = beta * (dr[i] + dr[j])
            w = random.uniform(low, high)
            W[i, j] = W[j, i] = w
    return dr, W

# ====== Path_Min_k 子例程 (3-approximation) ======
def bpc_approx(graph, lam):
    n = len(graph)
    # Kruskal MST
    edges = [(i, j, graph[i][j]) for i in range(n) for j in range(i+1, n)]
    edges.sort(key=lambda x: x[2])
    parent = list(range(n))
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    def union(u, v):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            return True
        return False
    F = []
    for u, v, w in edges:
        if union(u, v):
            F.append((u, v, w))
            if len(F) == n - 1:
                break
    # DFS 拆成路徑
    visited = [False]*n
    def dfs(u, path):
        visited[u] = True
        path.append(u)
        for a, b, _ in F:
            v = b if a == u else a if b == u else None
            if v is not None and not visited[v]:
                dfs(v, path)
    paths = []
    for i in range(n):
        if not visited[i]:
            p = []
            dfs(i, p)
            if p:
                paths.append(p)
    # 按 lam 切分
    final = []
    for p in paths:
        cur = [p[0]]
        length = 0
        for u, v in zip(p, p[1:]):
            length += graph[u][v]
            cur.append(v)
            if length > lam:
                final.append(cur[:-1])
                cur = [u, v]
                length = graph[u][v]
        final.append(cur)
    return final

# ====== Algorithm S 主流程 ======
def bcc_algorithm_s(dr, W, B):
    n = len(dr)
    root = 0
    # Step 1: 遠節點
    far = [v for v in range(n) if v != root and dr[v] >= B/2]
    cycles = [[root, v, root] for v in far]
    # 其餘節點
    rest = [v for v in range(n) if v != root and v not in far]
    if not rest:
        return len(cycles)
    # Step 2: 分層參數
    dists = sorted({dr[v] for v in rest})
    mu = min(abs(dists[i] - dists[i-1]) for i in range(1, len(dists))) if len(dists) > 1 else 1e-6
    eps = 2 * mu / B
    t   = math.ceil(math.log(1/eps, 2))
    # 分層 (Eq.4)
    layers = [[] for _ in range(t)]
    for v in rest:
        for j in range(1, t+1):
            low  = (1 - 2**j * eps) * B / 2
            high = (1 - 2**(j-1) * eps) * B / 2
            if low < dr[v] <= high:
                layers[j-1].append(v)
                break
    # Step 3 & 4: 各層 BPC + cycle
    for j, layer in enumerate(layers):
        if not layer:
            continue
        subW = W[np.ix_(layer, layer)]
        lam  = (2**j) * eps * B
        pats = bpc_approx(subW, lam)
        for p in pats:
            if p:
                cycle = [root] + [ layer[i] for i in p ] + [root]
                cycles.append(cycle)
    return len(cycles)

# ====== 批次實驗 & 輸出 ======
def run_experiments():
    rows = []
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "beta", "k", "seed", "cycles"])
        for n, beta, k, seed in product(n_list, beta_list, k_list, seeds):
            dr, W = gen_beta_metric(n, beta, seed)
            B      = k * dr.max()
            cnt    = bcc_algorithm_s(dr, W, B)
            rows.append((n, beta, k, seed, cnt))
            writer.writerow([n, beta, k, seed, cnt])
    return pd.DataFrame(rows, columns=["n", "beta", "k", "seed", "cycles"])

# ====== 繪圖 ======
def plot_results(df):
    for n in df["n"].unique():
        sub = df[df["n"] == n]
        avg_b = sub.groupby("beta")["cycles"].mean()
        avg_k = sub.groupby("k")["cycles"].mean()

        plt.figure()
        avg_b.plot(marker="o")
        plt.title(f"Avg Cycles vs Beta (n={n})")
        plt.xlabel("Beta")
        plt.ylabel("Avg Cycles")
        plt.grid(True)
        plt.savefig(f"plot_n{n}_vs_beta.png")

        plt.figure()
        avg_k.plot(marker="o")
        plt.title(f"Avg Cycles vs k (n={n})")
        plt.xlabel("k")
        plt.ylabel("Avg Cycles")
        plt.grid(True)
        plt.savefig(f"plot_n{n}_vs_k.png")

if __name__ == "__main__":
    df = run_experiments()
    plot_results(df)
    print("Done: results.csv + plots generated.")
