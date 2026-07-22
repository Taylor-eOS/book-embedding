import pysbd
import numpy as np
from sentence_transformers import SentenceTransformer

input_path = "input.txt"
model_name = "Qwen/Qwen3-Embedding-0.6B"
language = "en"

def main():
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    segmenter = pysbd.Segmenter(language=language, clean=False)
    sentences = [s.strip() for s in segmenter.segment(text) if s.strip()]
    if len(sentences) < 2:
        print("Need at least two sentences to compare.")
        return
    model = SentenceTransformer(model_name, device="cpu")
    embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=True)
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, np.nan)
    mean_similarity = np.nanmean(similarity, axis=1)
    order = np.argsort(mean_similarity)
    print("\nSentences ranked from most to least different (by average similarity to all others):\n")
    for idx in order:
        print(f"{mean_similarity[idx]:.4f} {sentences[idx][:40]}")
    most_different_idx = order[0]
    print("\nMost different sentence:")
    print(sentences[most_different_idx])

if __name__ == "__main__":
    main()
