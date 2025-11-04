import faiss, numpy as np, os, time, matplotlib.pyplot as plt

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
    return np.mean(I_pred[:,0] == I_true[:,0])

def timer():
    return time.time()

def run_one_dataset(path, name):
    xb = read_fvecs(os.path.join(path, "sift_base.fvecs"))
    xq = read_fvecs("sift/sift_query.fvecs")
    gt = read_ivecs("sift/sift_groundtruth.ivecs")
    d = xb.shape[1]
    efC, efS = 200, 100

    qps_points, build_points = [], []
    print(f"\n📦 Dataset {name}: {xb.shape[0]} vectors")

    for M in [4, 8, 12, 24, 48]:
        index = faiss.IndexHNSWFlat(d, M)
        index.metric_type = faiss.METRIC_L2
        index.hnsw.efConstruction = efC

        t0 = timer()
        index.add(xb)
        build_time = timer() - t0

        index.hnsw.efSearch = efS
        t1 = timer()
        D, I = index.search(xq, 1)
        query_time = timer() - t1

        recall = recall_at_1(I, gt)
        qps = xq.shape[0] / query_time

        qps_points.append((recall, qps, f"M={M}"))
        build_points.append((recall, build_time, f"M={M}"))
        print(f"M={M:<2} Recall@1={recall:.3f} QPS={qps:.1f} Build={build_time:.1f}s")

    return qps_points, build_points

def plot_curves(all_curves, title, ylabel, outfile):
    plt.figure(figsize=(8,6))
    for label, points in all_curves.items():
        xs=[p[0] for p in points]
        ys=[p[1] for p in points]
        annos=[p[2] for p in points]
        plt.plot(xs, ys, marker='o', label=label)
        for x,y,a in points: plt.annotate(a, (x,y))
    plt.xlabel("1-Recall@1")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile,dpi=200)
    print(f"✅ Saved {outfile}")

def main():
    base_dir = "datasets"
    datasets = {
        "100k": os.path.join(base_dir, "sift_100k"),
        "300k": os.path.join(base_dir, "sift_300k"),
        "500k": os.path.join(base_dir, "sift_500k"),
        "1M":   "sift"
    }

    qps_curves, build_curves = {}, {}

    for name, path in datasets.items():
        qps_pts, build_pts = run_one_dataset(path, name)
        qps_curves[name] = qps_pts
        build_curves[name] = build_pts

    plot_curves(qps_curves, "HNSW: QPS vs Recall (scaling)", "QPS", "part2_qps_vs_recall.png")
    plot_curves(build_curves, "HNSW: Build Time vs Recall (scaling)", "Build time (s)", "part2_build_vs_recall.png")

if __name__ == "__main__":
    main()
