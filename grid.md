# GridEnv 使用文档

## 目录结构
```
grid_env.py   # 库（环境与形状生成）
main.py       # 运行入口（命令行交互，WASD）
```

---

## 快速上手

### 以代码方式使用
```python
from grid_env import GridEnv

# 创建环境
env = GridEnv(
    rows=20, cols=40, noise_prob=0.2, seed=7,
    shape_type="blob", shape_max_h=10, shape_max_w=14,
    blob_area=80, blob_porosity=0.5, blob_hole_chunk=3
)

# 读取当前矩阵（已叠加形状）
mat = env.get_matrix()

# 移动（'w'/'a'/'s'/'d' 或 'up'/'down'/'left'/'right'）
moved = env.move('d')   # True/False：是否真的移动了（边界不动）

# 元信息
info = env.get_info()   # 字典：尺寸/噪声/形状描述/位置等

# 重新随机初始化（底板+形状+位置）
mat = env.reset()
```

### 以命令行方式使用
```bash
python main.py --rows 20 --cols 40 --noise-prob 0.2 \
  --shape-type blob --shape-max-h 10 --shape-max-w 14 \
  --blob-area 80 --blob-porosity 0.5 --blob-hole-chunk 3 \
  --seed 7
```
终端中用 `w/a/s/d`（回车确认）移动，`q` 退出。

---

## API 说明（grid_env.GridEnv）

### 构造函数
```python
GridEnv(
  rows: int = 15,
  cols: int = 30,
  noise_prob: float = 0.15,
  seed: int | None = None,
  digit: int | None = None,          # 1..9，缺省为随机
  shape_type: str = "auto",          # "auto" | "rect" | "L" | "plus" | "blob"
  shape_max_h: int | None = None,    # 形状最高上限（含 blob）
  shape_max_w: int | None = None,    # 形状最宽上限（含 blob）
  blob_area: int | None = None,      # blob 期望填充像素数（默认约30%密度）
  blob_porosity: float = 0.0,        # [0..0.9] 打孔比例（保持连通）
  blob_hole_chunk: int = 1           # 每次打孔的最大连通块大小（1=小孔，>1=洞簇）
)
```

### 方法
- `reset() -> List[List[int]]`  
  重新生成噪声底板与形状，并随机摆放。返回当前矩阵（深拷贝）。
- `move(direction: str) -> bool`  
  方向：`'w'/'a'/'s'/'d'` 或 `'up'/'down'/'left'/'right'`。  
  触边后继续朝外移动**不生效**（位置不变）。返回是否实际移动。
- `get_matrix() -> List[List[int]]`  
  返回**叠加了形状**的当前矩阵（深拷贝）。
- `get_info() -> dict`  
  返回元数据：`rows/cols/noise_prob/seed/shape_name/shape_size/digit/shape_type/position/blob{...}`。

---

## 形状与参数

- `shape_type="rect"`：空心矩形（最小 3×3），随机高宽（受 `shape_max_h/w` 约束与内置上限）。
- `shape_type="L"`：L 形（最小 3×3）。
- `shape_type="plus"`：十字形（保证奇数边，最小 3）。
- `shape_type="blob"`：**随机连通块**（随机游走生成）  
  - `blob_porosity`：在保持**4-连通**前提下“打孔”，让图形有大量空隙；`0.0` 表示不打孔。  
  - `blob_hole_chunk`：每次移除的孔洞连通块上限（1=单点孔，越大孔洞越成簇）。  
  - `blob_area`：目标像素数（若未设，默认约 30% 密度）。
- `shape_type="auto"`：在上述四类中随机抽取（`shape_max_*` 同样生效）。

**覆盖规则**：形状矩阵中 `0` 表示透明；非零（默认同一 `digit`）会**覆盖**底板对应位置的数字。  
**噪声**：底板初始为全 0；按 `noise_prob` 概率将格子改为 `1..9` 的随机数。  
**边界移动**：使用 clamp；超界请求会被截断到合法位置，若已在边界则不动。

---

## 示例

### 固定数字 6 的空心矩形
```python
env = GridEnv(rows=12, cols=24, noise_prob=0.1, digit=6, shape_type="rect")
```

### 稀疏多孔的随机连通块
```python
env = GridEnv(
    rows=20, cols=40, noise_prob=0.2, seed=0,
    shape_type="blob", shape_max_h=10, shape_max_w=14,
    blob_area=80, blob_porosity=0.6, blob_hole_chunk=4
)
```

### 仅通过 CLI 快速体验（随机形状）
```bash
python main.py --rows 18 --cols 36 --noise-prob 0.15 --seed 42 --shape-type auto
```

---

## 常见注意事项
- `rows/cols >= 3`；`digit` 需在 `1..9`。  
- `blob_porosity` 必须在 `[0, 0.9]`；数值越大，孔洞越多，但仍保持整体连通。  
- 若未设置 `shape_max_h/w`，形状最大上限默认受网格尺寸和各形状的内置上限共同约束。  
- 设定相同的 `seed` + 相同参数，可复现初始状态（底板与形状及其初始位置）。

---
