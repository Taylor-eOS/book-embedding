import numpy as np
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform

cache_path = "embedding_cache.npz"
preview_chars = 60

def load_cache():
    data = np.load(cache_path, allow_pickle=True)
    segments = list(data["segments"])
    embeddings = data["embeddings"]
    return segments, embeddings

def compute_cosine_distance_matrix(embeddings):
    distances = pdist(embeddings, metric="cosine")
    return squareform(distances)

def compute_linear_order(distance_matrix):
    n_samples = distance_matrix.shape[0]
    perplexity = max(5, min(30, n_samples // 3))
    perplexity = min(perplexity, n_samples - 1)
    reducer = TSNE(n_components=1, perplexity=perplexity, metric="precomputed", init="random", random_state=0)
    positions = reducer.fit_transform(distance_matrix)
    order = np.argsort(positions[:, 0])
    return order

def print_ordered_segments(segments, order):
    print("\nSegments ordered by meaning:\n")
    for rank, idx in enumerate(order):
        preview = segments[idx][:preview_chars].replace("\n", " ")
        print(f"{rank:4d}  (orig #{idx:4d})  {preview}")

def main():
    segments, embeddings = load_cache()
    if len(segments) < 3:
        print("Need at least 3 segments to produce a meaningful ordering.")
        return
    distance_matrix = compute_cosine_distance_matrix(embeddings)
    order = compute_linear_order(distance_matrix)
    print_ordered_segments(segments, order)

if __name__ == "__main__":
    main()
