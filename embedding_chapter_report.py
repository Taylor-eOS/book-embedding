import numpy as np
from embedding_report import detect_chapter_boundaries

cache_path = "embedding_cache.npz"
TOP_OUTLIER_COUNT = 20
LOCAL_NEIGHBOR_WINDOW = 3
DRIFT_WINDOW = 5
SNIPPET_CHARS = 70

def load_cache():
    data = np.load(cache_path, allow_pickle=True)
    segments = list(data["segments"])
    embeddings = data["embeddings"]
    return segments, embeddings

def normalize_rows(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms

def boundaries_to_chapter_ranges(boundaries, total_segments):
    ranges = []
    start = 0
    for b in boundaries:
        ranges.append((start, b))
        start = b
    ranges.append((start, total_segments))
    return ranges

def cosine_distance(a, b):
    return float(1.0 - np.dot(a, b))

def compute_chapter_centroids(embeddings_normalized, chapter_ranges):
    centroids = []
    for start, end in chapter_ranges:
        chapter_vectors = embeddings_normalized[start:end]
        centroid = chapter_vectors.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            centroid_norm = 1.0
        centroids.append(centroid / centroid_norm)
    return centroids

def compute_centroid_distances(embeddings_normalized, chapter_ranges, centroids):
    distances = np.zeros(embeddings_normalized.shape[0])
    for chapter_index, (start, end) in enumerate(chapter_ranges):
        centroid = centroids[chapter_index]
        for i in range(start, end):
            distances[i] = cosine_distance(embeddings_normalized[i], centroid)
    return distances

def compute_local_neighbor_distances(embeddings_normalized, chapter_ranges, window):
    distances = np.zeros(embeddings_normalized.shape[0])
    for start, end in chapter_ranges:
        for i in range(start, end):
            neighbor_low = max(start, i - window)
            neighbor_high = min(end, i + window + 1)
            neighbor_indices = [j for j in range(neighbor_low, neighbor_high) if j != i]
            if not neighbor_indices:
                distances[i] = 0.0
                continue
            local_distances = [cosine_distance(embeddings_normalized[i], embeddings_normalized[j]) for j in neighbor_indices]
            distances[i] = float(np.mean(local_distances))
    return distances

def compute_running_drift(embeddings_normalized, chapter_ranges, window):
    drift = np.zeros(embeddings_normalized.shape[0])
    for start, end in chapter_ranges:
        for i in range(start, end):
            running_low = start
            running_high = i + 1
            running_vectors = embeddings_normalized[running_low:running_high]
            running_centroid = running_vectors.mean(axis=0)
            running_norm = np.linalg.norm(running_centroid)
            if running_norm == 0:
                running_norm = 1.0
            running_centroid = running_centroid / running_norm
            window_low = max(start, i - window)
            if window_low >= i:
                drift[i] = 0.0
                continue
            window_vectors = embeddings_normalized[window_low:i]
            window_centroid = window_vectors.mean(axis=0)
            window_norm = np.linalg.norm(window_centroid)
            if window_norm == 0:
                window_norm = 1.0
            window_centroid = window_centroid / window_norm
            drift[i] = cosine_distance(running_centroid, window_centroid)
    return drift

def zscore(values):
    values = np.asarray(values, dtype=float)
    std = values.std()
    if std == 0:
        return np.zeros_like(values)
    return (values - values.mean()) / std

def chapter_index_for_segment(segment_index, chapter_ranges):
    for chapter_index, (start, end) in enumerate(chapter_ranges):
        if start <= segment_index < end:
            return chapter_index
    return len(chapter_ranges) - 1

def snippet(segments, index):
    text = segments[index].strip().replace("\n", " ")
    if len(text) <= SNIPPET_CHARS:
        return text
    return text[:SNIPPET_CHARS] + "..."

def print_chapter_overview(chapter_ranges, centroid_distances):
    print("\nCHAPTER OVERVIEW (average and max centroid distance per chapter)")
    for chapter_index, (start, end) in enumerate(chapter_ranges):
        chapter_distances = centroid_distances[start:end]
        print(f"chapter {chapter_index:>3}  segments {start:>5}-{end - 1:<5}  " f"count {end - start:>4}  " f"mean {chapter_distances.mean():.4f}  " f"max {chapter_distances.max():.4f}")

def print_outlier_segments(segments, chapter_ranges, centroid_distances, local_distances, combined_score):
    print(f"\nTOP {TOP_OUTLIER_COUNT} SEGMENTS THAT FIT POORLY WITH THEIR CHAPTER")
    print("(ranked by combined centroid + local neighbor distance)\n")
    order = np.argsort(combined_score)[::-1][:TOP_OUTLIER_COUNT]
    for index in order:
        chapter_index = chapter_index_for_segment(int(index), chapter_ranges)
        print(f"segment {index:>5}  chapter {chapter_index:>3}  " f"centroid_dist {centroid_distances[index]:.4f}  " f"local_dist {local_distances[index]:.4f}  " f"combined_z {combined_score[index]:.4f}")
        print(f"    {snippet(segments, int(index))}")

def print_drift_segments(segments, chapter_ranges, drift):
    print(f"\nTOP {TOP_OUTLIER_COUNT} POINTS OF SHARPEST LOCAL MEANING DRIFT")
    print("(running chapter centroid vs. the preceding local window)\n")
    order = np.argsort(drift)[::-1][:TOP_OUTLIER_COUNT]
    for index in order:
        chapter_index = chapter_index_for_segment(int(index), chapter_ranges)
        print(f"segment {index:>5}  chapter {chapter_index:>3}  drift {drift[index]:.4f}")
        print(f"    {snippet(segments, int(index))}")

def main():
    segments, embeddings = load_cache()
    if len(segments) < 2:
        print("Need at least 2 segments to analyze chapter coherence.")
        return
    boundaries = detect_chapter_boundaries(len(segments))
    chapter_ranges = boundaries_to_chapter_ranges(boundaries, len(segments))
    embeddings_normalized = normalize_rows(embeddings)
    centroids = compute_chapter_centroids(embeddings_normalized, chapter_ranges)
    centroid_distances = compute_centroid_distances(embeddings_normalized, chapter_ranges, centroids)
    local_distances = compute_local_neighbor_distances(embeddings_normalized, chapter_ranges, LOCAL_NEIGHBOR_WINDOW)
    drift = compute_running_drift(embeddings_normalized, chapter_ranges, DRIFT_WINDOW)
    combined_score = zscore(centroid_distances) + zscore(local_distances)
    print_chapter_overview(chapter_ranges, centroid_distances)
    print_outlier_segments(segments, chapter_ranges, centroid_distances, local_distances, combined_score)
    print_drift_segments(segments, chapter_ranges, drift)

if __name__ == "__main__":
    main()
