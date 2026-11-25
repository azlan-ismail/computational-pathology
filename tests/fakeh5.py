import numpy as np, os, types, sys
class File:
    def __init__(self, path, mode="r"):
        self.path = path; self.mode = mode
        self.d = dict(np.load(path + ".npz")) if mode == "r" else {}
    def __enter__(self): return self
    def __exit__(self, *a):
        if self.mode == "w": np.savez(self.path + ".npz", **self.d)
    def __getitem__(self, k): return self.d[k]
    def __setitem__(self, k, v): self.d[k] = np.asarray(v)
m = types.ModuleType("h5py"); m.File = File; sys.modules["h5py"] = m
