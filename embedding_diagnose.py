import numpy as np

cache_path = "embedding_cache.npz"

def load_cache():
    data = np.load(cache_path, allow_pickle=True)
    segments = list(data["segments"])
    embeddings = data["embeddings"]
    return segments, embeddings

def report_similarity_spread(embeddings):
    n = embeddings.shape[0]
    sims = embeddings @ embeddings.T
    mask = ~np.eye(n, dtype=bool)
    off_diagonal = sims[mask]
    print(f"segments: {n}")
    print(f"pairwise cosine similarity min: {off_diagonal.min():.4f}")
    print(f"pairwise cosine similarity max: {off_diagonal.max():.4f}")
    print(f"pairwise cosine similarity mean: {off_diagonal.mean():.4f}")
    print(f"pairwise cosine similarity std: {off_diagonal.std():.4f}")
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    values = np.percentile(off_diagonal, percentiles)
    for p, v in zip(percentiles, values):
        print(f"percentile {p}: {v:.4f}")
    nearest_neighbor_sims = []
    for i in range(n):
        row = sims[i].copy()
        row[i] = -np.inf
        nearest_neighbor_sims.append(row.max())
    nearest_neighbor_sims = np.array(nearest_neighbor_sims)
    print(f"mean nearest-neighbor similarity: {nearest_neighbor_sims.mean():.4f}")
    print(f"min nearest-neighbor similarity: {nearest_neighbor_sims.min():.4f}")
    print(f"max nearest-neighbor similarity: {nearest_neighbor_sims.max():.4f}")

def main():
    segments, embeddings = load_cache()
    report_similarity_spread(embeddings)

if __name__ == "__main__":
    main()
