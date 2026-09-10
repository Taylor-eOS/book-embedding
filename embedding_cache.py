import os
import numpy as np
from sentence_transformers import SentenceTransformer

input_path = "input.txt"
model_name = "Qwen/Qwen3-Embedding-0.6B"
SPLIT_ON = "\n\n"
BATCH_SIZE = 16
cache_path = "embedding_cache.npz"

def load_segments():
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    segments = [s.strip() for s in text.split(SPLIT_ON) if s.strip()]
    return segments

def load_cached():
    if not os.path.exists(cache_path):
        return None
    cached = np.load(cache_path, allow_pickle=True)
    segments = list(cached["segments"])
    embeddings = cached["embeddings"]
    return segments, embeddings

def save_cache(segments, embeddings):
    np.savez(
        cache_path,
        segments=np.array(segments, dtype=object),
        embeddings=embeddings,
    )

def embed_segments_whole(segments, model):
    embeddings = model.encode(
        segments,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
    )
    return embeddings

def get_segments_and_embeddings():
    cached = load_cached()
    if cached is not None:
        print("Loaded segments and embeddings from cache.")
        return cached
    segments = load_segments()
    model = SentenceTransformer(model_name, device="cpu")
    embeddings = embed_segments_whole(segments, model)
    save_cache(segments, embeddings)
    print("Computed segments and embeddings, saved to cache.")
    return segments, embeddings

if __name__ == "__main__":
    segments, embeddings = get_segments_and_embeddings()
    print(f"{len(segments)} segments, embedding dim {embeddings.shape[1]}")
