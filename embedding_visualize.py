import numpy as np
import pygame
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

cache_path = "embedding_cache.npz"
window_width = 1200
window_height = 800
icon_font_size = 13
icon_padding = 4
label_max_words = 12
font_size = 14
tsne_perplexity = 6
min_clusters = 6
max_clusters = 24

cluster_palette = [
    (230, 90, 70),
    (70, 150, 230),
    (90, 200, 120),
    (230, 190, 70),
    (180, 90, 220),
    (70, 210, 210),
    (230, 130, 180),
    (150, 150, 90),
    (240, 150, 60),
    (110, 110, 230),
    (200, 200, 200),
    (100, 200, 160),
    (220, 100, 140),
    (140, 180, 230),
    (190, 140, 90),
    (120, 220, 80),
    (210, 70, 130),
    (80, 130, 100),
    (240, 200, 140),
    (160, 100, 200),
    (60, 180, 230),
    (200, 150, 60),
    (100, 100, 160),
    (220, 220, 90),
]

def load_cache():
    data = np.load(cache_path, allow_pickle=True)
    segments = list(data["segments"])
    embeddings = data["embeddings"]
    return segments, embeddings

def reduce_to_2d(embeddings):
    n_samples = embeddings.shape[0]
    perplexity = min(tsne_perplexity, max(5, n_samples - 1))
    reducer = TSNE(n_components=2, perplexity=perplexity, metric="cosine", init="pca", random_state=0)
    coords = reducer.fit_transform(embeddings)
    return coords

def choose_cluster_count(embeddings):
    n_samples = embeddings.shape[0]
    lower = min(min_clusters, n_samples - 1)
    upper = min(max_clusters, n_samples - 1)
    if upper < lower:
        return lower
    best_k = lower
    best_score = -1.0
    for k in range(lower, upper + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, metric="cosine")
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def compute_clusters(embeddings):
    k = choose_cluster_count(embeddings)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = kmeans.fit_predict(embeddings)
    return labels, k

def make_labels(segments):
    labels = []
    for s in segments:
        words = s.split()
        label = " ".join(words[:label_max_words])
        if len(words) > label_max_words:
            label += "..."
        labels.append(label)
    return labels

def color_for_cluster(cluster_id):
    return cluster_palette[cluster_id % len(cluster_palette)]

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

def draw_scene(screen, font, small_font, icon_font, coords, labels, colors, icon_sizes, scale, offset_x, offset_y, pan_x, pan_y, zoom, hover_index, selected_index, cluster_count):
    screen.fill((15, 15, 20))
    screen_points = [to_screen(p, scale, offset_x, offset_y, pan_x, pan_y, zoom) for p in coords]
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
        label_text = f"{selected_index}: {labels[selected_index]}"
        text_surface = font.render(label_text, True, (255, 255, 255))
        box_rect = text_surface.get_rect()
        box_x = min(max(sx + 12, 0), window_width - box_rect.width - 10)
        box_y = min(max(sy - 10, 0), window_height - box_rect.height - 10)
        background_rect = pygame.Rect(box_x - 5, box_y - 3, box_rect.width + 10, box_rect.height + 6)
        pygame.draw.rect(screen, (30, 30, 40), background_rect)
        pygame.draw.rect(screen, (255, 255, 0), background_rect, 1)
        screen.blit(text_surface, (box_x, box_y))
    hint_surface = small_font.render(f"clusters: {cluster_count}   click a point for text, drag to pan, scroll to zoom, R to reset", True, (140, 140, 140))
    screen.blit(hint_surface, (10, window_height - 20))
    pygame.display.flip()

def run_visualizer():
    segments, embeddings = load_cache()
    coords = reduce_to_2d(embeddings)
    cluster_ids, cluster_count = compute_clusters(embeddings)
    labels = make_labels(segments)
    colors = [color_for_cluster(c) for c in cluster_ids]
    scale, offset_x, offset_y = compute_view_transform(coords, window_width, window_height)
    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Segment embedding space (t-SNE cosine layout, k-means clusters)")
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
                elif event.key == pygame.K_ESCAPE:
                    running = False
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
        draw_scene(screen, font, small_font, icon_font, coords, labels, colors, icon_sizes, scale, offset_x, offset_y, pan_x, pan_y, zoom, hover_index, selected_index, cluster_count)
        clock.tick(60)
    pygame.quit()

if __name__ == "__main__":
    run_visualizer()
