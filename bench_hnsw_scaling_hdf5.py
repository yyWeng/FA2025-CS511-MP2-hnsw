import os, time, h5py
import numpy as np
import faiss
import matplotlib.pyplot as plt


def recall_at_1(I_pred, I_true):
    return np.mean(I_pred[:, 0] == I_true[:, 0])


def qps(nq, seconds):
    return nq / seconds if seconds > 0 else float("inf")


def load_hdf5_dataset(path):
    with h5py.File(path, "r") as f:
        xb = np.array(f["train"], dtype="float32")
        xq = np.array(f["test"], dtype="float32")
        gt = np.array(f["neighbors"], dtype="int32")
    print(f"✅ Loaded {path} base={xb.shape} query={xq.shape}")
    return xb, xq, gt


# 🔧 新增：根据数据集类型自动选择度量方式与预处理
def pick_metric_and_preprocess(path, xb, xq):
    p = path.lower()

    if "dot" in p:
        # lastfm 等 dot product 数据集
        metric = faiss.METRIC_INNER_PRODUCT
        return metric, xb, xq

    elif "angular" in p or "cosine" in p:
        # angular/cosine 数据集需要 L2 归一化 + inner product
        def l2norm(a):
            n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
            return a / n

        xb = l2norm(xb)
        xq = l2norm(xq)
        metric = faiss.METRIC_INNER_PRODUCT
        return metric, xb, xq

    else:
        # 默认使用 L2
        metric = faiss.METRIC_L2
        return metric, xb, xq


def run_one_dataset(name, path, efC, efS):
    xb, xq, gt = load_hdf5_dataset(path)
    metric, xb, xq = pick_metric_and_preprocess(path, xb, xq)
    d = xb.shape[1]
    print(f"\n📦 {name}: dim={d}, nbase={xb.shape[0]} nquery={xq.shape[0]} metric={metric}")

    qps_points, build_points = [], []

    for M in [4, 8, 12, 24, 48]:
        index = faiss.IndexHNSWFlat(d, M)
        index.metric_type = metric
        index.hnsw.efConstruction = efC

        t0 = time.time()
        index.add(xb)
        build_time = time.time() - t0

        index.hnsw.efSearch = efS
        t1 = time.time()
        D, I = index.search(xq, 1)
        search_time = time.time() - t1

        recall = recall_at_1(I, gt)
        throughput = qps(xq.shape[0], search_time)
        qps_points.append((recall, throughput, f"M={M}"))
        build_points.append((recall, build_time, f"M={M}"))
        print(f"  M={M:<2} Recall={recall:.3f}  QPS={throughput:.1f}  Build={build_time:.1f}s")

    return name, qps_points, build_points


def plot_curves(curves, title, ylabel, outfile):
    plt.figure(figsize=(8, 6))
    for label, points in curves:
        xs = [p[0] for p in points]   # x轴: Recall@1  ← 不再用 1 - recall
        ys = [p[1] for p in points]
        ann = [p[2] for p in points]
        plt.plot(xs, ys, "-o", label=label)
        for x, y, a in zip(xs, ys, ann):
            plt.annotate(a, (x, y))
    plt.xlabel("Recall@1")            # ← 更新横轴名称
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print(f"✅ Saved {outfile}")



def main():
    datasets = {
        "COCO-I2I": ("datasets/coco-i2i-512-angular.hdf5", 200, 100),
        "COCO-T2I": ("datasets/coco-t2i-512-angular.hdf5", 200, 100),
        "GLOVE-25": ("datasets/glove-25-angular.hdf5", 200, 100),
        "LASTFM-64": ("datasets/lastfm-64-dot.hdf5", 200, 100),
    }

    qps_curves, build_curves = [], []

    for name, (path, efC, efS) in datasets.items():
        if not os.path.exists(path):
            print(f"⚠️  Missing {path}, skip {name}")
            continue
        res = run_one_dataset(name, path, efC, efS)
        if res:
            name, qps_pts, build_pts = res
            qps_curves.append((name, qps_pts))
            build_curves.append((name, build_pts))

    plot_curves(qps_curves, "HNSW QPS vs Recall@1 (All Datasets)", "QPS", "part2_qps_vs_recall.png")
    plot_curves(build_curves, "HNSW Build Time vs Recall@1 (All Datasets)", "Build Time (s)", "part2_build_vs_recall.png")

if __name__ == "__main__":
    main()
