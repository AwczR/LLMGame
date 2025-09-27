#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid Pattern CLI Environment (v3: porous blob)
- Terminal N x M numeric grid.
- Adds random noise after initialization.
- Overlays a shape made of a fixed digit (1..9 or user-specified).
- Shape types: hollow rect / L / plus / random blob (now supports porosity/holes).
- Control the shape with WASD keys (press Enter after each key).
- Movement clamps at borders (invalid outward moves do nothing).
"""
import argparse
import random
import sys
from collections import deque
from typing import List, Tuple, Optional, Set

def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

def make_grid(rows: int, cols: int, fill: int = 0) -> List[List[int]]:
    return [[fill for _ in range(cols)] for _ in range(rows)]

def add_noise(grid: List[List[int]], noise_prob: float, rng: random.Random):
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            if rng.random() < noise_prob:
                grid[r][c] = rng.randint(1, 9)

def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))

def overlay(base: List[List[int]], shape: List[List[int]], top: int, left: int) -> List[List[int]]:
    rows, cols = len(base), len(base[0])
    H, W = len(shape), len(shape[0])
    out = [row[:] for row in base]
    for i in range(H):
        for j in range(W):
            d = shape[i][j]
            if d != 0:
                r, c = top + i, left + j
                if 0 <= r < rows and 0 <= c < cols:
                    out[r][c] = d
    return out

# ------------------ Shape generators (0 = transparent) ------------------
def shape_hollow_rect(h: int, w: int, d: int) -> List[List[int]]:
    h = max(3, h); w = max(3, w)
    shp = [[0]*w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if i == 0 or i == h-1 or j == 0 or j == w-1:
                shp[i][j] = d
    return shp

def shape_L(h: int, w: int, d: int) -> List[List[int]]:
    h = max(3, h); w = max(3, w)
    shp = [[0]*w for _ in range(h)]
    for i in range(h):
        shp[i][0] = d
    for j in range(w):
        shp[h-1][j] = d
    return shp

def shape_plus(size: int, d: int) -> List[List[int]]:
    size = max(3, size)
    if size % 2 == 0:
        size += 1
    mid = size // 2
    shp = [[0]*size for _ in range(size)]
    for i in range(size):
        shp[i][mid] = d
    for j in range(size):
        shp[mid][j] = d
    return shp

# --------------- Helpers for porous blob generation --------------------
NEI4 = [(-1,0),(1,0),(0,-1),(0,1)]

def is_connected(cells: Set[Tuple[int,int]]) -> bool:
    """Check 4-connectedness of a set of (r,c) cells."""
    if not cells:
        return True
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in NEI4:
            nb = (r+dr, c+dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(cells)

def carve_holes(cells: Set[Tuple[int,int]], rng: random.Random, porosity: float, hole_chunk_max: int) -> Set[Tuple[int,int]]:
    """
    Remove cells (possibly in small clusters) while keeping the remaining set connected.
    porosity in [0, 0.9] roughly means removing that fraction of cells at most.
    hole_chunk_max controls the max size of each removed cluster.
    """
    porosity = max(0.0, min(0.9, porosity))
    if porosity <= 0.0 or len(cells) <= 4:
        return cells

    target_remove = int(porosity * len(cells))
    removed = 0
    attempts = 0
    max_attempts = max(100, target_remove * 30)
    hole_chunk_max = max(1, int(hole_chunk_max))

    cells_cur = set(cells)
    while removed < target_remove and attempts < max_attempts:
        attempts += 1
        seed = rng.choice(tuple(cells_cur))
        cluster_size = rng.randint(1, hole_chunk_max)
        # grow a candidate cluster inside current set
        cluster = {seed}
        frontier = [seed]
        while len(cluster) < cluster_size and frontier:
            r, c = frontier.pop(rng.randrange(len(frontier)))
            nbrs = [(r+dr, c+dc) for dr,dc in NEI4]
            rng.shuffle(nbrs)
            for nb in nbrs:
                if nb in cells_cur and nb not in cluster:
                    cluster.add(nb)
                    frontier.append(nb)
                    if len(cluster) >= cluster_size:
                        break
        proposal = cells_cur - cluster
        if not proposal:
            continue
        if is_connected(proposal):
            cells_cur = proposal
            removed += len(cluster)
    return cells_cur

def shape_random_blob(max_h: int, max_w: int, d: int, rng: random.Random,
                      area: Optional[int] = None,
                      porosity: float = 0.0,
                      hole_chunk_max: int = 1) -> List[List[int]]:
    """
    Create a connected blob within bounds, then carve connectivity-preserving holes.
    - Start with random-walk blob (connected).
    - Carve holes by removing cells/patches while ensuring the remainder stays connected.
    """
    H = max(3, max_h)
    W = max(3, max_w)
    h = rng.randint(3, H)
    w = rng.randint(3, W)

    # target area
    if area is None:
        target = max(5, int(0.3 * h * w))  # ~30% density initial
    else:
        target = clamp(area, 3, h*w)

    # random walk to get connected set
    visited: Set[Tuple[int,int]] = set()
    cur = (h // 2, w // 2)
    visited.add(cur)
    while len(visited) < target:
        if rng.random() < 0.2 and len(visited) > 1:
            cur = rng.choice(tuple(visited))
        dr, dc = rng.choice(NEI4)
        nr = clamp(cur[0] + dr, 0, h-1)
        nc = clamp(cur[1] + dc, 0, w-1)
        cur = (nr, nc)
        visited.add(cur)

    # carve holes (increase sparsity) but keep connectivity
    visited = carve_holes(visited, rng, porosity=porosity, hole_chunk_max=hole_chunk_max)

    # crop to bbox
    rows = [r for r,c in visited]
    cols = [c for r,c in visited]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    shp = [[0]*ww for _ in range(hh)]
    for (r, c) in visited:
        shp[r - r0][c - c0] = d
    return shp

def random_shape(rows: int, cols: int, rng: random.Random, digit: Optional[int] = None,
                 shape_type: str = "auto",
                 max_h: Optional[int] = None, max_w: Optional[int] = None,
                 blob_area: Optional[int] = None,
                 blob_porosity: float = 0.0,
                 blob_hole_chunk: int = 1) -> Tuple[str, List[List[int]]]:
    d = digit if (digit is not None) else rng.randint(1, 9)

    if shape_type == "rect":
        h = rng.randint(3, max(3, min(6, rows if max_h is None else max_h)))
        w = rng.randint(3, max(3, min(10, cols if max_w is None else max_w)))
        shp = shape_hollow_rect(h, w, d)
        return f"hollow_rect({h}x{w}) digit={d}", shp
    elif shape_type == "L":
        h = rng.randint(4, max(4, min(8, rows if max_h is None else max_h)))
        w = rng.randint(4, max(4, min(10, cols if max_w is None else max_w)))
        shp = shape_L(h, w, d)
        return f"L({h}x{w}) digit={d}", shp
    elif shape_type == "plus":
        size = rng.randint(3, max(3, min(9, rows if max_h is None else max_h, cols if max_w is None else max_w)))
        shp = shape_plus(size, d)
        return f"plus({size}) digit={d}", shp
    elif shape_type == "blob":
        H = rows if max_h is None else min(max_h, rows)
        W = cols if max_w is None else min(max_w, cols)
        shp = shape_random_blob(H, W, d, rng, area=blob_area, porosity=blob_porosity, hole_chunk_max=blob_hole_chunk)
        return f"blob(max {H}x{W}, area~{blob_area if blob_area else 'auto'}, porosity={blob_porosity:.2f}, chunk<= {blob_hole_chunk}) digit={d}", shp
    else:
        # auto: pick among rect/L/plus/blob
        choice = rng.choice(["rect","L","plus","blob"])
        return random_shape(rows, cols, rng, digit=d, shape_type=choice, max_h=max_h, max_w=max_w,
                            blob_area=blob_area, blob_porosity=blob_porosity, blob_hole_chunk=blob_hole_chunk)

def render_ascii(grid: List[List[int]]) -> str:
    lines = []
    for row in grid:
        lines.append(" ".join(str(v) for v in row))
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Grid Pattern CLI Environment (v3: porous blob)")
    parser.add_argument("--rows", type=int, default=15, help="Grid rows (default: 15)")
    parser.add_argument("--cols", type=int, default=30, help="Grid cols (default: 30)")
    parser.add_argument("--noise-prob", type=float, default=0.15, help="Noise probability per cell in [0,1] (default: 0.15)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--digit", type=int, default=None, help="Fixed digit for shape (1..9). If omitted, random each run.")
    parser.add_argument("--shape-type", type=str, default="auto", choices=["auto","rect","L","plus","blob"],
                        help="Shape type: auto/random choice or one of rect/L/plus/blob (blob supports porosity).")
    parser.add_argument("--shape-max-h", type=int, default=None, help="Max shape height (used by blob/auto).")
    parser.add_argument("--shape-max-w", type=int, default=None, help="Max shape width (used by blob/auto).")
    parser.add_argument("--blob-area", type=int, default=None, help="Approx target filled cells for blob. Defaults to ~30%% density.")
    parser.add_argument("--blob-porosity", type=float, default=0.0, help="Fraction [0..0.9] of cells to remove as holes (connectivity preserved).")
    parser.add_argument("--blob-hole-chunk", type=int, default=1, help="Max hole chunk size per removal (1=single-cell holes).")
    args = parser.parse_args()

    if args.rows < 3 or args.cols < 3:
        print("rows and cols must be >= 3")
        sys.exit(1)
    if args.digit is not None and not (1 <= args.digit <= 9):
        print("--digit must be in 1..9")
        sys.exit(1)
    if not (0.0 <= args.blob_porosity <= 0.9):
        print("--blob-porosity must be in [0, 0.9]")
        sys.exit(1)

    rng = random.Random(args.seed)

    # Base grid + noise
    base = make_grid(args.rows, args.cols, fill=0)
    add_noise(base, args.noise_prob, rng)

    # Shape
    shape_name, shp = random_shape(
        args.rows, args.cols, rng,
        digit=args.digit,
        shape_type=args.shape_type,
        max_h=args.shape_max_h,
        max_w=args.shape_max_w,
        blob_area=args.blob_area,
        blob_porosity=args.blob_porosity,
        blob_hole_chunk=args.blob_hole_chunk
    )
    H, W = len(shp), len(shp[0])

    # Initial position: random but fit
    top = rng.randint(0, max(0, args.rows - H))
    left = rng.randint(0, max(0, args.cols - W))

    help_text = (
        "Controls: w=up, s=down, a=left, d=right, q=quit\n"
        "Enter one key then press Enter.\n"
        "Movement clamps at borders; invalid outward moves do not move."
    )

    while True:
        view = overlay(base, shp, top, left)
        clear_screen()
        print(f"Grid: {args.rows}x{args.cols} | Noise Prob: {args.noise_prob:.2f} | Seed: {args.seed}")
        print(f"Shape: {shape_name} | Top-Left: ({top},{left}) | Size: {H}x{W}")
        print(help_text)
        print("-" * (args.cols * 2 - 1))
        print(render_ascii(view))
        print("-" * (args.cols * 2 - 1))

        try:
            cmd = input("Move (w/a/s/d) or q: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not cmd:
            continue
        if cmd[0] == 'q':
            print("Bye.")
            break

        new_top, new_left = top, left
        if cmd[0] == 'w':
            new_top = clamp(top - 1, 0, args.rows - H)
        elif cmd[0] == 's':
            new_top = clamp(top + 1, 0, args.rows - H)
        elif cmd[0] == 'a':
            new_left = clamp(left - 1, 0, args.cols - W)
        elif cmd[0] == 'd':
            new_left = clamp(left + 1, 0, args.cols - W)

        top, left = new_top, new_left

if __name__ == "__main__":
    main()
