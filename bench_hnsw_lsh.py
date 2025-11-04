import faiss
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# ---------- 辅助函数 ----------
def read_fvecs(fname):
    fv = np.fromfile(fname, dtype=np.int32)
    d = fv[0]
    fv = fv.reshape(-1, d + 1)
    return fv[:, 1:].view("float32")

def read_ivecs(fname):
    iv = np.fromfile(fname, dtype=np.int32)
    d = iv[0]
    iv = iv.reshape(-1, d + 1)
    return iv[:, 1:]

def recall_at_1(I_pred, I_true):
    return np.mean(I_pred[:, 0] == I_true[:, 0])

def qps(num_queries, elapsed_sec):
    return num_queries / elapsed_sec if elapsed_sec > 0 else float("inf")


# ---------- 主程序 ----------
def main():
    # 1️⃣ 加载数据
    data_dir = "sift"
    xb = read_fvecs(os.path.join(data_dir, "sift_base.fvecs"))
    xq = read_fvecs(os.path.join(data_dir, "sift_query.fvecs"))
    gt = read_ivecs(os.path.join(data_dir, "sift_groundtruth.ivecs"))
    print("✅ 数据加载完成:", xb.shape, xq.shape, gt.shape)
    d = xb.shape[1]

    results = {"HNSW": [], "LSH": []}

    # 2️⃣ HNSW 实验
    print("\n⚙️  开始 HNSW 实验 (M=32)...")
    index_hnsw = faiss.IndexHNSWFlat(d, 32)
    index_hnsw.metric_type = faiss.METRIC_L2
    index_hnsw.hnsw.efConstruction = 200
    index_hnsw.add(xb)

    for efSearch in [10, 50, 100, 200]:
        index_hnsw.hnsw.efSearch = efSearch
        start = time.time()
        D, I = index_hnsw.search(xq, 1)
        elapsed = time.time() - start
        r1 = recall_at_1(I, gt)
        q = qps(xq.shape[0], elapsed)
        results["HNSW"].append((efSearch, r1, q))
        print(f"efSearch={efSearch:<4}  Recall@1={r1:.4f}  QPS={q:.2f}")

    # 3️⃣ LSH 实验
    print("\n⚙️  开始 LSH 实验...")
    for nbits in [32, 64, 512, 768]:
        index_lsh = faiss.IndexLSH(d, nbits)
        index_lsh.add(xb)
        start = time.time()
        D, I = index_lsh.search(xq, 1)
        elapsed = time.time() - start
        r1 = recall_at_1(I, gt)
        q = qps(xq.shape[0], elapsed)
        results["LSH"].append((nbits, r1, q))
        print(f"nbits={nbits:<4}  Recall@1={r1:.4f}  QPS={q:.2f}")

    # 4️⃣ 绘图
    plt.figure(figsize=(8, 6))
    for label, data in results.items():
        recalls = [r for _, r, _ in data]
        qps_vals = [q for _, _, q in data]
        annos = [str(p[0]) for p in data]
        plt.plot(recalls, qps_vals, "-o", label=label)
        for x, y, txt in zip(recalls, qps_vals, annos):
            plt.annotate(txt, (x, y))
    plt.xlabel("1-Recall@1")
    plt.ylabel("QPS (queries per second)")
    plt.title("HNSW vs LSH on SIFT1M")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part1_qps_vs_recall.png", dpi=200)
    plt.show()

    print("\n✅ 图已保存为 part1_qps_vs_recall.png")


if __name__ == "__main__":
    main()
