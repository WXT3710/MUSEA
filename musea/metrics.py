import numpy as np

def sign(z):
    return np.where(z >= 0, 1.0, -1.0).astype(np.float32)

def unit(z):
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)

def map_score(q, g, ql, gl, k, binary=True):
    q = sign(q) if binary else unit(q.astype(np.float32))
    g = sign(g) if binary else unit(g.astype(np.float32))
    values = []
    for x, label in zip(q, ql):
        scores = 0.5 * (q.shape[1] - g @ x) if binary else -(g @ x)
        order = np.argsort(scores)[:k]
        pos = np.flatnonzero(gl[order] == label)
        values.append(np.mean(np.arange(1, len(pos) + 1) / (pos + 1)) if len(pos) else 0.0)
    return float(np.mean(values))
