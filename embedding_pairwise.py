import pysbd
import numpy as np
from sentence_transformers import SentenceTransformer

input_path = "input.txt"
report_path = "report.txt"
model_name = "Qwen/Qwen3-Embedding-0.6B"
language = "en"
BATCH_SIZE = 16
SENTENCE_DISTANCE = 1
NUM_RESULTS = 10
PRINTED_CONTEXT = 80
REPORT_CONTEXT = 300

def load_sentences():
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    segmenter = pysbd.Segmenter(language=language, clean=False)
    return [s.strip() for s in segmenter.segment(text) if s.strip()]

def embed_sentences(sentences):
    model = SentenceTransformer(model_name, device="cpu")
    return model.encode(sentences, normalize_embeddings=True, show_progress_bar=True, batch_size=BATCH_SIZE)

def compute_pairs(embeddings):
    similarity_matrix = embeddings @ embeddings.T
    n = len(embeddings)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) < SENTENCE_DISTANCE:
                continue
            pairs.append((float(similarity_matrix[i, j]), i, j))
    pairs.sort(key=lambda p: p[0], reverse=True)
    return pairs

def print_top_pairs(pairs, sentences):
    print(f"\nTop {NUM_RESULTS} most similar sentence pairs (possible conceptual repetitions):\n")
    for score, i, j in pairs[:NUM_RESULTS]:
        print(f"{score:.4f}  [{i}] {sentences[i][:PRINTED_CONTEXT]}")
        print(f"          [{j}] {sentences[j][:PRINTED_CONTEXT]}\n")

def write_report(pairs, sentences):
    scores = np.array([p[0] for p in pairs])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Embedding pairwise similarity report\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Sentences: {len(sentences)}\n")
        f.write(f"Pairs compared: {len(pairs)}\n")
        f.write(f"Similarity min/mean/max: {scores.min():.4f} / {scores.mean():.4f} / {scores.max():.4f}\n")
        f.write("\nFull sentence list:\n")
        for idx, s in enumerate(sentences):
            f.write(f"[{idx}] {s}\n")
        f.write("\nAll pairs, sorted by similarity (descending):\n\n")
        for score, i, j in pairs:
            f.write(f"{score:.4f}  [{i}] {sentences[i][:REPORT_CONTEXT]}\n")
            f.write(f"          [{j}] {sentences[j][:REPORT_CONTEXT]}\n\n")
    print(f"Report file created: {report_path}")

def main():
    sentences = load_sentences()
    if len(sentences) < 2:
        print("Need at least 2 sentences to compare.")
        return
    embeddings = embed_sentences(sentences)
    pairs = compute_pairs(embeddings)
    print_top_pairs(pairs, sentences)
    write_report(pairs, sentences)

if __name__ == "__main__":
    main()
