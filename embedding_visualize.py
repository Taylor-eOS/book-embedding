import numpy as np
import pygame
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import colorsys

cache_path = "embedding_cache.npz"
window_width = 1200
window_height = 800
icon_font_size = 13
icon_padding = 4
label_max_words = 12
font_size = 14
min_clusters = 2
max_clusters_fraction = 0.15
initial_clusters_fraction = 0.05
neighbor_lines_per_point = 3
neighbor_line_max_alpha = 140

def load_cache():
    data = np.load(cache_path, allow_pickle=True)
    segments = list(data["segments"])
    embeddings = data["embeddings"]
    return segments, embeddings

def compute_cosine_distance_matrix(embeddings):
    distances = pdist(embeddings, metric="cosine")
    return squareform(distances)

def reduce_to_2d(embeddings, distance_matrix):
    n_samples = embeddings.shape[0]
    perplexity = max(5, min(30, n_samples // 3))
    perplexity = min(perplexity, n_samples - 1)
    reducer = TSNE(n_components=2, perplexity=perplexity, metric="precomputed", init="random", random_state=0)
    coords = reducer.fit_transform(distance_matrix)
    return coords

def choose_initial_cluster_count(distance_matrix, linkage_matrix, n_samples):
    upper = max(min_clusters + 1, int(n_samples * max_clusters_fraction))
    upper = min(upper, n_samples - 1)
    target = max(min_clusters, int(n_samples * initial_clusters_fraction))
    target = min(target, upper)
    search_low = max(min_clusters, target - 5)
    search_high = min(upper, target + 5)
    best_k = target
    best_score = -1.0
    for k in range(search_low, search_high + 1):
        labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(distance_matrix, labels, metric="precomputed")
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def compute_linkage(embeddings):
    return linkage(embeddings, method="ward")

def clusters_for_k(linkage_matrix, k):
    labels = fcluster(linkage_matrix, t=k, criterion="maxclust")
    labels = labels - labels.min()
    return labels, len(set(labels))

def compute_max_cluster_count(n_samples):
    upper = max(min_clusters + 1, int(n_samples * max_clusters_fraction))
    upper = min(upper, n_samples - 1)
    return upper

def compute_nearest_neighbors(distance_matrix, k):
    n_samples = distance_matrix.shape[0]
    neighbor_lists = []
    for i in range(n_samples):
        row = distance_matrix[i].copy()
        row[i] = np.inf
        nearest = np.argsort(row)[:k]
        neighbor_lists.append(list(nearest))
    return neighbor_lists

def make_labels(segments):
    labels = []
    for s in segments:
        words = s.split()
        label = " ".join(words[:label_max_words])
        labels.append(label)
    return labels

def color_for_cluster(cluster_id):
    golden = 0.6180339887498949
    hue = (cluster_id * golden) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.72, 0.88)
    return (int(r * 255), int(g * 255), int(b * 255))

def compute_view_transform(coords, width, height, padding=60):
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    scale = min((width - 2 * padding) / span_x, (height - 2 * padding) / span_y)
    offset_x = padding - min_x * scale
    offset_y = padding - min_y * scale
    return scale, offset_x, offset_y

def to_screen(point, scale, offset_x, offset_y, pan_x, pan_y, zoom):
    x = point[0] * scale * zoom + offset_x * zoom + pan_x
    y = point[1] * scale * zoom + offset_y * zoom + pan_y
    return int(x), int(y)

def find_nearest_point(mouse_pos, screen_points, icon_sizes, margin=6):
    best_i = None
    best_d = None
    for i, (sx, sy) in enumerate(screen_points):
        half_w = icon_sizes[i][0] / 2 + margin
        half_h = icon_sizes[i][1] / 2 + margin
        if abs(sx - mouse_pos[0]) > half_w or abs(sy - mouse_pos[1]) > half_h:
            continue
        d = ((sx - mouse_pos[0]) ** 2 + (sy - mouse_pos[1]) ** 2) ** 0.5
        if best_d is None or d < best_d:
            best_d = d
            best_i = i
    return best_i

def measure_icon_sizes(icon_font, n_points):
    digit_w, digit_h = icon_font.size("000")
    box_w = digit_w + icon_padding * 2
    box_h = digit_h + icon_padding * 2
    return [(box_w, box_h) for _ in range(n_points)]

def draw_neighbor_lines_batched(screen, screen_points, neighbor_lists, distance_matrix, highlight_index):
    overlay = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
    max_dist = distance_matrix.max()
    if max_dist <= 0:
        max_dist = 1.0
    indices_to_draw = range(len(screen_points)) if highlight_index is None else [highlight_index]
    for i in indices_to_draw:
        for j in neighbor_lists[i]:
            d = distance_matrix[i][j]
            closeness = 1.0 - min(d / max_dist, 1.0)
            alpha = int(neighbor_line_max_alpha * closeness)
            if alpha <= 0:
                continue
            color = (255, 255, 255, alpha) if highlight_index is None else (255, 255, 0, alpha)
            pygame.draw.line(overlay, color, screen_points[i], screen_points[j], 1)
    screen.blit(overlay, (0, 0))

def draw_scene(screen, font, small_font, icon_font, coords, labels, colors, icon_sizes, scale, offset_x, offset_y, pan_x, pan_y, zoom, hover_index, selected_index, cluster_count, silhouette, neighbor_lists, distance_matrix, show_all_lines):
    screen.fill((15, 15, 20))
    screen_points = [to_screen(p, scale, offset_x, offset_y, pan_x, pan_y, zoom) for p in coords]
    highlight_index = selected_index if selected_index is not None else hover_index
    if show_all_lines:
        draw_neighbor_lines_batched(screen, screen_points, neighbor_lists, distance_matrix, None)
    elif highlight_index is not None:
        draw_neighbor_lines_batched(screen, screen_points, neighbor_lists, distance_matrix, highlight_index)
    for i, (sx, sy) in enumerate(screen_points):
        box_w, box_h = icon_sizes[i]
        if i == hover_index:
            box_w += 4
            box_h += 4
        box_rect = pygame.Rect(0, 0, box_w, box_h)
        box_rect.center = (sx, sy)
        pygame.draw.rect(screen, colors[i], box_rect, border_radius=4)
        border_color = (255, 255, 255) if i != selected_index else (255, 255, 0)
        border_width = 1 if i != selected_index else 3
        pygame.draw.rect(screen, border_color, box_rect, border_width, border_radius=4)
        number_surface = icon_font.render(str(i), True, (15, 15, 20))
        number_rect = number_surface.get_rect(center=(sx, sy))
        screen.blit(number_surface, number_rect)
    if selected_index is not None:
        sx, sy = screen_points[selected_index]
        label_text = f"{labels[selected_index]}"
        text_surface = font.render(label_text, True, (255, 255, 255))
        box_rect = text_surface.get_rect()
        box_x = min(max(sx + 12, 0), window_width - box_rect.width - 10)
        box_y = min(max(sy - 10, 0), window_height - box_rect.height - 10)
        background_rect = pygame.Rect(box_x - 5, box_y - 3, box_rect.width + 10, box_rect.height + 6)
        pygame.draw.rect(screen, (30, 30, 40), background_rect)
        pygame.draw.rect(screen, (255, 255, 0), background_rect, 1)
        screen.blit(text_surface, (box_x, box_y))
        neighbor_text = "nearest: " + ", ".join(str(n) for n in neighbor_lists[selected_index])
        neighbor_surface = small_font.render(neighbor_text, True, (200, 200, 200))
        screen.blit(neighbor_surface, (box_x, box_y + box_rect.height + 6))
    hint_text = f"clusters: {cluster_count} (silhouette {silhouette:.3f})   +/- to change cluster count, click a point for text and neighbors, drag to pan, scroll to zoom, L to toggle all links, R to reset"
    hint_surface = small_font.render(hint_text, True, (140, 140, 140))
    screen.blit(hint_surface, (10, window_height - 20))
    pygame.display.flip()

def run_visualizer():
    segments, embeddings = load_cache()
    distance_matrix = compute_cosine_distance_matrix(embeddings)
    coords = reduce_to_2d(embeddings, distance_matrix)
    linkage_matrix = compute_linkage(embeddings)
    n_samples = embeddings.shape[0]
    max_k = compute_max_cluster_count(n_samples)
    current_k = choose_initial_cluster_count(distance_matrix, linkage_matrix, n_samples)
    cluster_ids, cluster_count = clusters_for_k(linkage_matrix, current_k)
    silhouette = silhouette_score(distance_matrix, cluster_ids, metric="precomputed")
    neighbor_lists = compute_nearest_neighbors(distance_matrix, neighbor_lines_per_point)
    labels = make_labels(segments)
    colors = [color_for_cluster(c) for c in cluster_ids]
    scale, offset_x, offset_y = compute_view_transform(coords, window_width, window_height)
    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Segment embedding space (t-SNE cosine layout, hierarchical clusters)")
    font = pygame.font.SysFont("monospace", font_size)
    small_font = pygame.font.SysFont("monospace", 11)
    icon_font = pygame.font.SysFont("monospace", icon_font_size, bold=True)
    icon_sizes = measure_icon_sizes(icon_font, len(coords))
    pan_x, pan_y = 0.0, 0.0
    zoom = 1.0
    dragging = False
    last_mouse_pos = (0, 0)
    hover_index = None
    selected_index = None
    show_all_lines = False
    clock = pygame.time.Clock()
    running = True
    while running:
        screen_points = [to_screen(p, scale, offset_x, offset_y, pan_x, pan_y, zoom) for p in coords]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    pan_x, pan_y, zoom = 0.0, 0.0, 1.0
                elif event.key == pygame.K_l:
                    show_all_lines = not show_all_lines
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    current_k = min(current_k + 1, max_k)
                    cluster_ids, cluster_count = clusters_for_k(linkage_matrix, current_k)
                    silhouette = silhouette_score(distance_matrix, cluster_ids, metric="precomputed")
                    colors = [color_for_cluster(c) for c in cluster_ids]
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    current_k = max(current_k - 1, min_clusters)
                    cluster_ids, cluster_count = clusters_for_k(linkage_matrix, current_k)
                    silhouette = silhouette_score(distance_matrix, cluster_ids, metric="precomputed")
                    colors = [color_for_cluster(c) for c in cluster_ids]
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    last_mouse_pos = event.pos
                    clicked = find_nearest_point(event.pos, screen_points, icon_sizes)
                    selected_index = clicked
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    dx = event.pos[0] - last_mouse_pos[0]
                    dy = event.pos[1] - last_mouse_pos[1]
                    pan_x += dx
                    pan_y += dy
                    last_mouse_pos = event.pos
                hover_index = find_nearest_point(event.pos, screen_points, icon_sizes)
            elif event.type == pygame.MOUSEWHEEL:
                zoom_factor = 1.1 if event.y > 0 else (1 / 1.1)
                zoom *= zoom_factor
        draw_scene(screen, font, small_font, icon_font, coords, labels, colors, icon_sizes, scale, offset_x, offset_y, pan_x, pan_y, zoom, hover_index, selected_index, cluster_count, silhouette, neighbor_lists, distance_matrix, show_all_lines)
        clock.tick(60)
    pygame.quit()

if __name__ == "__main__":
    run_visualizer()

