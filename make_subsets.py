import numpy as np, os, struct

def read_fvecs(fname):
    fv = np.fromfile(fname, dtype=np.int32)
    d = fv[0]
    fv = fv.reshape(-1, d + 1)
    return fv[:, 1:].view('float32')

def write_fvecs(fname, arr):
    n, d = arr.shape
    with open(fname, "wb") as f:
        for i in range(n):
            f.write(struct.pack("i", d))
            f.write(arr[i].astype("float32").tobytes())

# 读取原始 SIFT1M 数据
xb = read_fvecs("sift/sift_base.fvecs")

# 创建 datasets 目录
os.makedirs("datasets", exist_ok=True)

for size in [100000, 300000, 500000]:
    subset = xb[:size]
    outdir = f"datasets/sift_{size//1000}k"
    os.makedirs(outdir, exist_ok=True)
    write_fvecs(f"{outdir}/sift_base.fvecs", subset)
    print(f"✅ Created subset: {outdir}/sift_base.fvecs ({size} vectors)")

# 1M 直接复制
os.makedirs("datasets/sift_1m", exist_ok=True)
os.system("cp sift/sift_base.fvecs datasets/sift_1m/")
print("✅ Copied 1M dataset..")
