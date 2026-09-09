import numpy as np
from embedding_cache import get_segments_and_embeddings

context_size = 3

def main():
    segments, embeddings = get_segments_and_embeddings()
    if len(segments) < context_size + 1:
        print(f"Need at least {context_size + 1} segments to compare.")
        return
    scores = []
    indices = []
    for i in range(context_size, len(segments)):
        context_embeddings = embeddings[i - context_size:i]
        context_mean = context_embeddings.mean(axis=0)
        context_mean = context_mean / np.linalg.norm(context_mean)
        similarity = float(embeddings[i] @ context_mean)
        scores.append(similarity)
        indices.append(i)
    order = np.argsort(scores)
    print(f"\nSegments ranked from most to least different than the previous {context_size} segments:\n")
    for pos in order:
        idx = indices[pos]
        print(f"{scores[pos]:.4f} {segments[idx][:40]}")
    most_different_pos = order[0]
    most_different_idx = indices[most_different_pos]
    print("\nMost different segment from its preceding context:")
    print(segments[most_different_idx])

if __name__ == "__main__":
    main()
