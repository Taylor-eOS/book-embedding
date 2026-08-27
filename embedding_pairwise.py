import pysbd
import numpy as np
from sentence_transformers import SentenceTransformer
embeddings
input_path = "input.txt"
model_name = "Qwen/Qwen3-Embedding-0.6B"
language = "en"
BATCH_SIZE = 8
top_n = 30
min_sentence_distance = 1

def main():
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    segmenter = pysbd.Segmenter(language=language, clean=False)
    sentences = [s.strip() for s in segmenter.segment(text) if s.strip()]
    if len(sentences) < 2:
        print("Need at least 2 sentences to compare.")
        return
    model = SentenceTransformer(model_name, device="cpu")
    embeddings = model.encode(sentences, normalize_embeddings=True, show_progress_bar=True, batch_size=BATCH_SIZE)
    similarity_matrix = embeddings @ embeddings.T
    n = len(sentences)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) < min_sentence_distance:
                continue
            pairs.append((float(similarity_matrix[i, j]), i, j))
    pairs.sort(key=lambda p: p[0], reverse=True)
    print(f"\nTop {top_n} most similar sentence pairs (possible conceptual repetitions):\n")
    for score, i, j in pairs[:top_n]:
        print(f"{score:.4f}  [{i}] {sentences[i][:80]}")
        print(f"          [{j}] {sentences[j][:80]}\n")

if __name__ == "__main__":
    main()
