#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GridEnv 库：
- 维护一个 rows x cols 的数字矩阵（默认填 0），加入噪声（1..9）。
- 叠加一个由同一数字组成的图形（矩形/L/十字/随机连通块 blob）。
- 提供类方法：初始化(reset)、移动(move)、获取当前矩阵(get_matrix)。
- blob 支持“孔洞”参数：在保持 4-连通性的前提下随机挖洞，制造稀疏形状。

新增（兼容原代码）：
- get_shape_digit()：返回当前形状的数字。
- get_target_canonical()：返回“最小外接矩形 + 形状处为 d，非形状处为 -1”的目标矩阵。
- verify_prediction(pred)：校验 LLM 输出是否正确（尺寸、最小性、掩码与数字）。
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Set, Dict, Any
import random
from collections import deque

NEI4 = [(-1,0),(1,0),(0,-1),(0,1)]

def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))

def make_grid(rows: int, cols: int, fill: int = 0) -> List[List[int]]:
    return [[fill for _ in range(cols)] for _ in range(rows)]

def add_noise(grid: List[List[int]], noise_prob: float, rng: random.Random):
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
        for c in range(cols):
            if rng.random() < noise_prob:
                grid[r][c] = rng.randint(1, 9)

def overlay(base: List[List[int]], shape: List[List[int]], top: int, left: int) -> List[List[int]]:
    """将 shape 覆盖到 base 上（shape 的 0 表示透明，不覆盖）。"""
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

# ---------- 形状生成（0 表示透明，d 表示该数字像素） ----------
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

# ---------- blob（连通块）辅助：保持连通性的打孔 ----------
def _is_connected(cells: Set[Tuple[int,int]]) -> bool:
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
                seen.add(nb); q.append(nb)
    return len(seen) == len(cells)

def _carve_holes(cells: Set[Tuple[int,int]], rng: random.Random, porosity: float, hole_chunk_max: int) -> Set[Tuple[int,int]]:
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
        seed = random_choice_tuple(cells_cur, rng)
        cluster_size = rng.randint(1, hole_chunk_max)
        # 在当前集合里长出一个待删除的小簇
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
        if proposal and _is_connected(proposal):
            cells_cur = proposal
            removed += len(cluster)
    return cells_cur

def random_choice_tuple(s: Set[Tuple[int,int]], rng: random.Random) -> Tuple[int,int]:
    return rng.choice(tuple(s))

def shape_random_blob(max_h: int, max_w: int, d: int, rng: random.Random,
                      area: Optional[int] = None,
                      porosity: float = 0.0,
                      hole_chunk_max: int = 1) -> List[List[int]]:
    H = max(3, max_h)
    W = max(3, max_w)
    h = rng.randint(3, H)
    w = rng.randint(3, W)

    # 目标像素数
    if area is None:
        target = max(5, int(0.3 * h * w))
    else:
        target = clamp(area, 3, h*w)

    # 随机游走生成连通块
    visited: Set[Tuple[int,int]] = set()
    cur = (h // 2, w // 2)
    visited.add(cur)
    while len(visited) < target:
        if rng.random() < 0.2 and len(visited) > 1:
            cur = random_choice_tuple(visited, rng)
        dr, dc = rng.choice(NEI4)
        nr = clamp(cur[0] + dr, 0, h-1)
        nc = clamp(cur[1] + dc, 0, w-1)
        cur = (nr, nc)
        visited.add(cur)

    # 打孔：保持连通
    visited = _carve_holes(visited, rng, porosity=porosity, hole_chunk_max=hole_chunk_max)

    # 裁剪到外接矩形
    rows = [r for r,_ in visited]
    cols = [c for _,c in visited]
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
    """返回 (形状描述, 形状矩阵)。"""
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
        choice = rng.choice(["rect","L","plus","blob"])
        return random_shape(rows, cols, rng, digit=d, shape_type=choice, max_h=max_h, max_w=max_w,
                            blob_area=blob_area, blob_porosity=blob_porosity, blob_hole_chunk=blob_hole_chunk)

class GridEnv:
    """
    交互环境类：
    - reset()：重新生成底板、噪声和图形，并随机摆放。
    - move(dir)：用 'w/a/s/d' 或 'up/down/left/right' 移动；越界时不变；返回是否发生移动。
    - get_matrix()：返回当前可见矩阵（已叠加形状）。
    - get_info()：返回形状信息、位置等元数据。
    新增：
    - get_shape_digit()
    - get_target_canonical()
    - verify_prediction(pred)
    """
    def __init__(self,
                 rows: int = 15,
                 cols: int = 30,
                 noise_prob: float = 0.15,
                 seed: Optional[int] = None,
                 digit: Optional[int] = None,
                 shape_type: str = "auto",
                 shape_max_h: Optional[int] = None,
                 shape_max_w: Optional[int] = None,
                 blob_area: Optional[int] = None,
                 blob_porosity: float = 0.0,
                 blob_hole_chunk: int = 1):
        self.rows = rows
        self.cols = cols
        self.noise_prob = noise_prob
        self.seed = seed
        self.rng = random.Random(seed)
        self.digit = digit
        self.shape_type = shape_type
        self.shape_max_h = shape_max_h
        self.shape_max_w = shape_max_w
        self.blob_area = blob_area
        self.blob_porosity = blob_porosity
        self.blob_hole_chunk = blob_hole_chunk

        self.base: Optional[List[List[int]]] = None
        self.shape: Optional[List[List[int]]] = None
        self.shape_name = ""
        self.H = self.W = 0
        self.top = self.left = 0

        self.reset()

    # --- API ---
    def reset(self) -> List[List[int]]:
        """重新生成底板与形状，随机放置。返回当前矩阵视图。"""
        if self.rows < 3 or self.cols < 3:
            raise ValueError("rows and cols must be >= 3")
        if self.digit is not None and not (1 <= self.digit <= 9):
            raise ValueError("digit must be in 1..9")
        if not (0.0 <= self.blob_porosity <= 0.9):
            raise ValueError("blob_porosity must be in [0, 0.9]")

        self.base = make_grid(self.rows, self.cols, fill=0)
        add_noise(self.base, self.noise_prob, self.rng)
        self.shape_name, self.shape = random_shape(
            self.rows, self.cols, self.rng,
            digit=self.digit, shape_type=self.shape_type,
            max_h=self.shape_max_h, max_w=self.shape_max_w,
            blob_area=self.blob_area, blob_porosity=self.blob_porosity,
            blob_hole_chunk=self.blob_hole_chunk
        )
        self.H, self.W = len(self.shape), len(self.shape[0])
        self.top = self.rng.randint(0, max(0, self.rows - self.H))
        self.left = self.rng.randint(0, max(0, self.cols - self.W))
        return self.get_matrix()

    def move(self, direction: str) -> bool:
        """
        用 'w/a/s/d' 或 'up/down/left/right' 控制。
        越界时位置不变。返回 bool 表示是否实际移动。
        """
        direction = direction.lower().strip()
        new_top, new_left = self.top, self.left
        if direction in ("w", "up"):
            new_top = clamp(self.top - 1, 0, self.rows - self.H)
        elif direction in ("s", "down"):
            new_top = clamp(self.top + 1, 0, self.rows - self.H)
        elif direction in ("a", "left"):
            new_left = clamp(self.left - 1, 0, self.cols - self.W)
        elif direction in ("d", "right"):
            new_left = clamp(self.left + 1, 0, self.cols - self.W)
        moved = (new_top != self.top) or (new_left != self.left)
        self.top, self.left = new_top, new_left
        return moved

    def get_matrix(self) -> List[List[int]]:
        """返回叠加了形状的当前矩阵（深拷贝）。"""
        return overlay(self.base, self.shape, self.top, self.left)

    # -------- 新增：用于 LLM 判题的工具 --------
    def get_shape_digit(self) -> int:
        """返回当前形状的数字。若构造时 digit=None，则从 self.shape 推断。"""
        if self.digit is not None:
            return self.digit
        # 从形状里找任意非 0 的值
        if self.shape is None:
            raise RuntimeError("shape not initialized")
        for i in range(self.H):
            for j in range(self.W):
                if self.shape[i][j] != 0:
                    return self.shape[i][j]
        # 理论不会走到
        return 0

    def get_target_canonical(self) -> List[List[int]]:
        """
        返回 LLM 需要输出的目标矩阵：
        - 尺寸为形状的最小外接矩形 (H x W)。
        - 形状像素处为 digit，非形状处为 -1。
        """
        if self.shape is None:
            raise RuntimeError("shape not initialized")
        d = self.get_shape_digit()
        out = []
        for i in range(self.H):
            row = []
            for j in range(self.W):
                row.append(d if self.shape[i][j] != 0 else -1)
            out.append(row)
        return out

    @staticmethod
    def _trim_neg1_border(mat: List[List[int]]) -> List[List[int]]:
        """裁去仅由 -1 组成的外框，保持最小外接矩形。"""
        if not mat or not mat[0]:
            return mat
        R, C = len(mat), len(mat[0])
        top, bottom = 0, R-1
        left, right = 0, C-1

        def row_all_neg1(r): return all(x == -1 for x in mat[r])
        def col_all_neg1(c): return all(mat[r][c] == -1 for r in range(R))

        while top <= bottom and row_all_neg1(top): top += 1
        while top <= bottom and row_all_neg1(bottom): bottom -= 1
        while left <= right and col_all_neg1(left): left += 1
        while left <= right and col_all_neg1(right): right -= 1

        if top > bottom or left > right:
            return []  # 全是 -1 的非法情况
        return [row[left:right+1] for row in mat[top:bottom+1]]

    def verify_prediction(self, pred: List[List[int]], max_report: int = 20) -> Dict[str, Any]:
        """
        校验 LLM 的输出：
        - 自动裁去四周仅含 -1 的边框，检查是否已最小。
        - 尺寸必须与真实形状 (H, W) 一致。
        - 取值只能是 {-1, digit}。
        - 与 ground truth 掩码逐元素一致。
        返回 dict：{ok, reason, digit, expected_size, pred_size, mismatches:[(i,j,gt,pred),...]}
        """
        if self.shape is None:
            raise RuntimeError("shape not initialized")
        d = self.get_shape_digit()
        gt = self.get_target_canonical()

        # 基本合法性
        if not isinstance(pred, list) or not pred or not isinstance(pred[0], list):
            return {"ok": False, "reason": "prediction must be 2D list", "digit": d}

        # 裁去 -1 外框，确保最小
        pred_trim = GridEnv._trim_neg1_border(pred)

        # 尺寸比对
        Rh, Rw = len(gt), len(gt[0])
        Ph = len(pred_trim) if pred_trim else 0
        Pw = len(pred_trim[0]) if pred_trim and pred_trim[0] else 0
        if Ph != Rh or Pw != Rw:
            return {
                "ok": False,
                "reason": "size mismatch",
                "digit": d,
                "expected_size": (Rh, Rw),
                "pred_size": (Ph, Pw)
            }

        # 逐元素检查
        mism = []
        for i in range(Rh):
            for j in range(Rw):
                v = pred_trim[i][j]
                if v not in (-1, d):
                    mism.append((i, j, gt[i][j], v))
                elif v != gt[i][j]:
                    mism.append((i, j, gt[i][j], v))
                if len(mism) >= max_report:
                    break
            if len(mism) >= max_report:
                break

        if mism:
            return {
                "ok": False,
                "reason": "value mismatch",
                "digit": d,
                "expected_size": (Rh, Rw),
                "pred_size": (Ph, Pw),
                "mismatches": mism
            }

        # 额外最小性校验：裁剪后四边都应包含至少一个 digit
        def edge_has_d(mat):
            top_has = any(x == d for x in mat[0])
            bot_has = any(x == d for x in mat[-1])
            left_has = any(row[0] == d for row in mat)
            right_has = any(row[-1] == d for row in mat)
            return top_has and bot_has and left_has and right_has
        if not edge_has_d(pred_trim):
            # 虽尺寸对，但边界不含 d，说明内部有 -1 包围错误
            return {
                "ok": False,
                "reason": "non-minimal or hollow border",
                "digit": d,
                "expected_size": (Rh, Rw),
                "pred_size": (Ph, Pw)
            }

        return {
            "ok": True,
            "reason": "pass",
            "digit": d,
            "expected_size": (Rh, Rw),
            "pred_size": (Ph, Pw)
        }

    def get_info(self) -> dict:
        return {
            "rows": self.rows, "cols": self.cols, "noise_prob": self.noise_prob, "seed": self.seed,
            "shape_name": self.shape_name, "shape_size": (self.H, self.W),
            "digit": self.get_shape_digit(),
            "shape_type": self.shape_type,
            "position": (self.top, self.left),
            "blob": {"area": self.blob_area, "porosity": self.blob_porosity, "hole_chunk": self.blob_hole_chunk}
        }