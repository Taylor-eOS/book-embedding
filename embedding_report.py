import os
import numpy as np
import ruptures as rpt
from embedding_cache import get_segments_and_embeddings, input_path, SPLIT_ON

MIN_SEGMENT_SIZE = 2
EXPLORATORY_PENALTIES = [1, 2, 4, 8, 16, 32]
NEAR_BOUNDARY_WINDOW = 2
TOP_JUMP_COUNT = 30
TOP_ELSEWHERE_COUNT = 8
CHAPTER_MARKER = SPLIT_ON + " "

def detect_chapter_boundaries(total_segments):
    if not os.path.exists(input_path):
        print(f"\nNo input file found at '{input_path}'.")
        print("Proceeding without existing chapter boundaries.\n")
        return []
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    raw_pieces = text.split(SPLIT_ON)
    boundaries = []
    segment_count = 0
    for i, piece in enumerate(raw_pieces[:-1]):
        if not piece.strip():
            continue
        segment_count += 1
        next_piece = raw_pieces[i + 1]
        if next_piece.startswith(" ") and next_piece.strip():
            boundaries.append(segment_count)
    return curate_chapter_boundaries(boundaries, total_segments)

def curate_chapter_boundaries(boundaries, total_segments):
    curated = []
    seen = set()
    previous = 0
    for b in boundaries:
        if b in seen:
            raise ValueError(f"duplicate chapter boundary at segment {b}")
        if b <= previous:
            raise ValueError(f"chapter boundary at segment {b} is not strictly after the previous boundary at {previous}")
        if b >= total_segments:
            raise ValueError(f"chapter boundary at segment {b} falls at or beyond the last segment ({total_segments - 1})")
        seen.add(b)
        curated.append(b)
        previous = b
    print(f"\ndetected {len(curated)} chapter boundaries from '{input_path}' via the '{CHAPTER_MARKER!r}' marker, verified against {total_segments} segments:")
    print(", ".join(str(b) for b in curated))
    return curated

def compute_adjacent_distances(embeddings):
    return np.linalg.norm(embeddings[1:] - embeddings[:-1], axis=1)

def print_distance_stats(diffs):
    print(f"\nadjacent-segment distance stats over {len(diffs)} pairs:")
    print(f"min {diffs.min():.4f}  max {diffs.max():.4f}  mean {diffs.mean():.4f}  std {diffs.std():.4f}")

def percentile_rank(value, population):
    return float((population < value).sum()) / len(population) * 100.0

def is_near_existing_boundary(segment_index, existing_boundaries):
    for b in existing_boundaries:
        if abs(segment_index - b) <= NEAR_BOUNDARY_WINDOW:
            return True
    return False

def print_top_jumps(segments, diffs, existing_boundaries):
    print(f"\ntop {TOP_JUMP_COUNT} largest meaning-shifts in the book (boundary is before the listed segment):\n")
    order = np.argsort(diffs)[::-1][:TOP_JUMP_COUNT]
    for idx in sorted(order):
        segment_index = idx + 1
        distance = diffs[idx]
        pct = percentile_rank(distance, diffs)
        near = "existing chapter break nearby" if is_near_existing_boundary(segment_index, existing_boundaries) else "NOT near an existing chapter break"
        before_snippet = segments[segment_index - 1][-50:].replace("\n", " ")
        after_snippet = segments[segment_index][:50].replace("\n", " ")
        print(f"segment {segment_index:>4}  distance {distance:.4f}  (top {100 - pct:.1f}% of all gaps)  [{near}]")
        print(f"    ...{before_snippet} || {after_snippet}...")

def print_existing_boundary_report(segments, diffs, existing_boundaries):
    if not existing_boundaries:
        return
    print(f"\nof your {len(existing_boundaries)} existing chapter boundaries, these show the weakest meaning-shift")
    print("in the embedding signal, meaning they are candidates to merge with the surrounding chapter")
    print("(this is a report ON your existing boundaries, not a search for missing ones):\n")
    ranked_positions = []
    for b in existing_boundaries:
        distance = diffs[b - 1]
        pct = percentile_rank(distance, diffs)
        ranked_positions.append((b, distance, pct))
    ranked_positions.sort(key=lambda x: x[1])
    weak_count = sum(1 for _, _, pct in ranked_positions if pct < 50.0)
    print(f"{weak_count} of {len(existing_boundaries)} existing boundaries sit below the median adjacent-distance ")
    print("(meaning the model sees less of a topic shift there than at a typical adjacent-segment pair)\n")
    print("weakest existing boundaries (candidates to reconsider or merge):")
    for b, distance, pct in ranked_positions[:15]:
        before_snippet = segments[b - 1][-50:].replace("\n", " ")
        after_snippet = segments[b][:50].replace("\n", " ")
        print(f"  segment {b:>4}  distance {distance:.4f}  (top {100 - pct:.1f}% of all gaps)")
        print(f"    ...{before_snippet} || {after_snippet}...")

def print_pelt_elsewhere_detail(segments, diffs, elsewhere_penalties_by_segment):
    ranked = []
    for b, penalties in elsewhere_penalties_by_segment.items():
        distance = diffs[b - 1] if 0 <= b - 1 < len(diffs) else float("nan")
        ranked.append((b, distance, max(penalties)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    shown = ranked[:TOP_ELSEWHERE_COUNT]
    print(f"\ntop {len(shown)} of {len(ranked)} PELT boundaries NOT near an existing chapter break, pooled across all penalties tested:")
    print("ranked by embedding-distance strength, so these are the strongest candidates for a missed chapter break;")
    print("'found at penalty <=' shows the loosest (largest) penalty that still detected it, since a boundary that survives")
    print("a high penalty is a stronger signal than one that only appears when PELT is set to find almost everything:\n")
    for b, distance, max_penalty in shown:
        pct = percentile_rank(distance, diffs) if not np.isnan(distance) else float("nan")
        if b < 1 or b >= len(segments):
            print(f"  segment {b:>4}  (at edge of book, no snippet)  found at penalty <= {max_penalty}")
            continue
        before_snippet = segments[b - 1][-50:].replace("\n", " ")
        after_snippet = segments[b][:50].replace("\n", " ")
        print(f"  segment {b:>4}  distance {distance:.4f}  (top {100 - pct:.1f}% of all gaps)  found at penalty <= {max_penalty}")
        print(f"    ...{before_snippet} || {after_snippet}...")

def run_exploratory_pelt(segments, diffs, embeddings, existing_boundaries):
    algo = rpt.Pelt(model="l2", min_size=MIN_SEGMENT_SIZE, jump=1)
    algo.fit(embeddings)
    print("\nexploratory PELT boundaries at a few sensitivities (not tuned to match your chapters):\n")
    elsewhere_penalties_by_segment = {}
    for penalty in EXPLORATORY_PENALTIES:
        breakpoints = algo.predict(pen=penalty)
        if breakpoints and breakpoints[-1] == len(embeddings):
            breakpoints = breakpoints[:-1]
        near_existing = sum(1 for b in breakpoints if is_near_existing_boundary(b, existing_boundaries))
        away_from_existing = len(breakpoints) - near_existing
        print(f"penalty {penalty:>4}: {len(breakpoints):>4} boundaries, {near_existing} near an existing chapter break, {away_from_existing} elsewhere")
        for b in breakpoints:
            if not is_near_existing_boundary(b, existing_boundaries):
                elsewhere_penalties_by_segment.setdefault(b, []).append(penalty)
    if elsewhere_penalties_by_segment:
        print_pelt_elsewhere_detail(segments, diffs, elsewhere_penalties_by_segment)

def main():
    segments, embeddings = get_segments_and_embeddings()
    if len(segments) < MIN_SEGMENT_SIZE * 2:
        print(f"Need at least {MIN_SEGMENT_SIZE * 2} segments to analyze meaning shifts.")
        return
    existing_boundaries = detect_chapter_boundaries(len(segments))
    diffs = compute_adjacent_distances(embeddings)
    print_distance_stats(diffs)
    print_top_jumps(segments, diffs, existing_boundaries)
    print_existing_boundary_report(segments, diffs, existing_boundaries)
    run_exploratory_pelt(segments, diffs, embeddings, existing_boundaries)

if __name__ == "__main__":
    main()
