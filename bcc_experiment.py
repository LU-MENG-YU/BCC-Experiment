import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import random
from itertools import product
import math
import pandas as pd
from typing import List

# Set default experiment parameters
n_list =    [50]
beta_list = [0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9]
k_list =    [3,4,5,6]
seeds =     list(range(1001)) #[]
results = []

# Graph generation with beta-metric adjustment
def gen_graph(n: int, beta: float, seed: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    coords = np.random.rand(n, 2)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                euclidean = np.linalg.norm(coords[i] - coords[j])
                dist_matrix[i][j] = euclidean
    d_min = dist_matrix[dist_matrix > 0].min()
    d_max = ((1 + beta) / (1 - beta)) * d_min
    dist_matrix = np.interp(dist_matrix, (d_min, dist_matrix.max()), (1, d_max))
    return dist_matrix

# Approximate BPC solver (Path_Min_k-style) with 3-approximation
def bpc_approx(graph: np.ndarray, lam: float) -> List[List[int]]:
    n = len(graph)
    edges = [(i, j, graph[i][j]) for i in range(n) for j in range(i + 1, n)]
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

    paths = []
    visited = [False] * n

    def dfs(u, path):
        visited[u] = True
        path.append(u)
        for v in range(n):
            if not visited[v] and any(((a == u and b == v) or (a == v and b == u)) for a, b, _ in F):
                dfs(v, path)

    for i in range(n):
        if not visited[i]:
            path = []
            dfs(i, path)
            if path:
                paths.append(path)

    final_paths = []
    for path in paths:
        if len(path) < 2:
            final_paths.append(path)
            continue
        temp = []
        cur_len = 0
        temp_path = [path[0]]
        for i in range(1, len(path)):
            cur_len += graph[path[i - 1]][path[i]]
            temp_path.append(path[i])
            if cur_len > lam:
                final_paths.append(temp_path[:-1])
                temp_path = [path[i - 1], path[i]]
                cur_len = graph[path[i - 1]][path[i]]
        final_paths.append(temp_path)
    return final_paths

# Algorithm S main procedure
def bcc_algorithm_s(graph: np.ndarray, B: float) -> int:
    n = len(graph)
    root = 0
    far_nodes = [v for v in range(n) if graph[root][v] >= B / 2 and v != root]
    cycles = [[root, v, root] for v in far_nodes]

    V_rest = [v for v in range(n) if v not in far_nodes and v != root]
    if not V_rest:
        return len(cycles)

    dists = sorted(set([graph[root][v] for v in V_rest]))
    mu = min([abs(dists[i] - dists[i - 1]) for i in range(1, len(dists))]) if len(dists) > 1 else 1e-6
    eps = 2 * mu / B
    t = int(math.ceil(math.log(1 / eps, 2)))

    layers = [[] for _ in range(t)]
    for v in V_rest:
        for j in range(1, t + 1):
            low = (1 - 2 ** j * eps) * B / 2
            high = (1 - 2 ** (j - 1) * eps) * B / 2
            if low < graph[root][v] <= high:
                layers[j - 1].append(v)
                break

    for j, layer in enumerate(layers):
        if not layer:
            continue
        subgraph = graph[np.ix_(layer, layer)]
        lam = (2 ** j) * eps * B
        paths = bpc_approx(subgraph, lam)
        for p in paths:
            if len(p) >= 1:
                cycle = [root] + [layer[i] for i in p] + [root]
                cycles.append(cycle)
    return len(cycles)

# Run full experiment
def run_experiments():
    csv_file = "results.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["n", "beta", "k", "seed", "cycles"])

        for n, beta, k, seed in product(n_list, beta_list, k_list, seeds):
            graph = gen_graph(n, beta, seed)
            d_max = graph.max()
            B = k * d_max
            cycle_count = bcc_algorithm_s(graph, B)
            results.append((n, beta, k, seed, cycle_count))
            writer.writerow([n, beta, k, seed, cycle_count])

    return pd.DataFrame(results, columns=["n", "beta", "k", "seed", "cycles"])

# Plotting results
def plot_results(df: pd.DataFrame):
    for n in df["n"].unique():
        subset = df[df["n"] == n]
        avg_by_beta = subset.groupby("beta")["cycles"].mean()
        avg_by_k = subset.groupby("k")["cycles"].mean()

        plt.figure()
        avg_by_beta.plot(marker='o')
        plt.title(f"Avg Cycles vs Beta (n={n})")
        plt.xlabel("Beta")
        plt.ylabel("Average Cycles")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plot_n{n}_vs_beta.png")

        plt.figure()
        avg_by_k.plot(marker='o')
        plt.title(f"Avg Cycles vs k (n={n})")
        plt.xlabel("Budget Multiplier k")
        plt.ylabel("Average Cycles")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plot_n{n}_vs_k.png")

# Execute all
df = run_experiments()
plot_results(df)
