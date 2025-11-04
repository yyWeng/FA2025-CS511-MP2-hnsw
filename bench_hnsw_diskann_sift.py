import os, time
import numpy as np
import faiss
import diskannpy as dap
import matplotlib.pyplot as plt


# ----------------- helpers -----------------
def read_fvecs(fname):
    a = np.fromfile(fname, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].astype('float32')

def read_ivecs(fname):
    a = np.fromfile(fname, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]


def recall_at_1(I_pred, I_true):
    return float(np.mean(I_pred[:, 0] == I_true[:, 0]))


# ----------------- HNSW -----------------
def bench_hnsw(xb, xq, gt):
    d = xb.shape[1]
    results = []
    M_vals = [8, 16, 32]
    ef_vals = [50, 100, 200]
    for M in M_vals:
        for ef in ef_vals:
            index = faiss.IndexHNSWFlat(d, M)
            index.hnsw.efConstruction = 200
            index.add(xb)
            index.hnsw.efSearch = ef

            t0 = time.time()
            D, I = index.search(xq, 1)
            latency = (time.time() - t0) / len(xq) * 1000.0
            recall = recall_at_1(I, gt)
            results.append(("HNSW", M, ef, recall, latency))
            print(f"HNSW M={M} ef={ef} → Recall={recall:.3f}, Latency={latency:.4f} ms")
    return results


# ----------------- DiskANN -----------------
def bench_diskann(xb, xq, gt):
    results = []
    tmpdir = "diskann_sift"
    os.makedirs(tmpdir, exist_ok=True)
    # 参数选择
    R_vals = [16, 32]
    L_vals = [10, 50, 100]

    for R in R_vals:
        for L in L_vals:
            index_dir = os.path.join(tmpdir, f"R{R}_L{L}")
            dap.build_disk_index(
                data=xb,
                distance_metric="l2",
                index_directory=index_dir,
                graph_degree=R,
                complexity=L,
                num_threads=os.cpu_count(),
            )
            index = dap.StaticDiskIndex(index_directory=index_dir, distance_metric="l2")
            index.load(num_threads=os.cpu_count())

            t0 = time.time()
            idxs, dists = index.batch_search(xq, k=1, complexity=L, num_threads=os.cpu_count())
            latency = (time.time() - t0) / len(xq) * 1000.0
            recall = recall_at_1(idxs, gt)
            results.append(("DiskANN", R, L, recall, latency))
            print(f"DiskANN R={R} L={L} → Recall={recall:.3f}, Latency={latency:.4f} ms")
    return results


# ----------------- Plot -----------------
def plot_latency_vs_recall(results, outpng):
    plt.figure(figsize=(8, 6))
    for algo in ["HNSW", "DiskANN"]:
        pts = [r for r in results if r[0] == algo]
        xs = [1 - r[3] for r in pts]
        ys = [r[4] for r in pts]
        plt.plot(xs, ys, "-o", label=algo)
        for r in pts:
            if algo == "HNSW":
                label = f"M={r[1]},ef={r[2]}"
            else:
                label = f"R={r[1]},L={r[2]}"
            plt.annotate(label, (1 - r[3], r[4]))
    plt.xlabel("1 - Recall@1")
    plt.ylabel("Latency (ms/query)")
    plt.title("Latency vs Recall on SIFT1M (HNSW vs DiskANN)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    print(f"✅ Saved plot to {outpng}")


# ----------------- Main -----------------
def main():
    xb = read_fvecs("sift/sift_base.fvecs")
    xq = read_fvecs("sift/sift_query.fvecs")
    gt = read_ivecs("sift/sift_groundtruth.ivecs")
    print(f"✅ Loaded SIFT1M: base={xb.shape} query={xq.shape}")

    results = []
    results += bench_hnsw(xb, xq, gt)
    results += bench_diskann(xb, xq, gt)

    plot_latency_vs_recall(results, "part3_latency_vs_recall.png")


if __name__ == "__main__":
    main()
