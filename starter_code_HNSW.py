import faiss
import numpy as np
import os

# ---- 辅助函数：读取 fvecs/ivecs 文件 ----
def read_fvecs(fname):
    fv = np.fromfile(fname, dtype=np.int32)
    d = fv[0]
    fv = fv.reshape(-1, d + 1)
    return fv[:, 1:].view('float32')

def read_ivecs(fname):
    iv = np.fromfile(fname, dtype=np.int32)
    d = iv[0]
    iv = iv.reshape(-1, d + 1)
    return iv[:, 1:]

# ---- 主函数 ----
def evaluate_hnsw():
    # 数据路径
    data_dir = "sift"  # 你的 sift 文件夹
    base_path = os.path.join(data_dir, "sift_base.fvecs")
    query_path = os.path.join(data_dir, "sift_query.fvecs")
    
    # 加载数据
    print("📂 Loading data...")
    xb = read_fvecs(base_path)
    xq = read_fvecs(query_path)
    print(f"✅ base shape: {xb.shape}, query shape: {xq.shape}")

    d = xb.shape[1]

    # ---- 创建 HNSW 索引 ----
    print("⚙️  Building HNSW index...")
    index = faiss.IndexHNSWFlat(d, 16)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 200

    index.add(xb)
    print("✅ Index built and populated.")

    # ---- 查询第一个 query ----
    print("🔍 Running search for the first query vector...")
    D, I = index.search(xq[:1], 10)

    # ---- 输出前 10 个近邻索引 ----
    print("Top 10 approximate nearest neighbors:")
    print(I[0])

    # ---- 写入 output.txt ----
    output_path = os.path.join(os.getcwd(), "output.txt")
    with open(output_path, "w") as f:
        for idx in I[0]:
            f.write(f"{idx}\n")
    print(f"💾 Results saved to {output_path}")

# ---- 主程序入口 ----
if __name__ == "__main__":
    evaluate_hnsw()
