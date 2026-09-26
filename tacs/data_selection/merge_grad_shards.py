import argparse
import os
import re
import torch

def list_shards(dir_path: str):
    files = []
    for name in os.listdir(dir_path):
        if re.match(r"grads-\d+\.pt$", name):
            files.append(name)
    # sort by the numeric offset inside filename
    files.sort(key=lambda x: int(re.search(r"grads-(\d+)\.pt", x).group(1)))
    return [os.path.join(dir_path, f) for f in files]

def merge(dir_path: str, out_name: str = "all_orig.pt"):
    shards = list_shards(dir_path)
    if not shards:
        raise FileNotFoundError(f"No shards like grads-*.pt found under {dir_path}")
    tensors = []
    for p in shards:
        t = torch.load(p, map_location="cpu")
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        tensors.append(t.float())
    merged = torch.cat(tensors, dim=0)
    outp = os.path.join(dir_path, out_name)
    torch.save(merged, outp)
    return outp, merged.shape

def main():
    ap = argparse.ArgumentParser(description="Merge gradient shards grads-*.pt into a single all_orig.pt")
    ap.add_argument("dir", help="Directory containing grads-*.pt shards")
    ap.add_argument("--out", default="all_orig.pt", help="Output filename inside the directory")
    args = ap.parse_args()
    outp, shape = merge(args.dir, args.out)
    print(f"Wrote {outp} with shape {tuple(shape)}")

if __name__ == "__main__":
    main()
