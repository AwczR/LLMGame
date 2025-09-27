#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行入口：
- 以 ASCII 方式显示矩阵。
- WASD 控制移动；到边界再向外移动时不动；q 退出。
"""
import argparse
import sys
from grid_env import GridEnv  # 引用库

def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

def render_ascii(grid):
    return "\n".join(" ".join(str(v) for v in row) for row in grid)

def parse_args():
    p = argparse.ArgumentParser(description="Grid Pattern CLI Environment (library + main)")
    p.add_argument("--rows", type=int, default=15, help="Grid rows")
    p.add_argument("--cols", type=int, default=30, help="Grid cols")
    p.add_argument("--noise-prob", type=float, default=0.15, help="Noise probability per cell in [0,1]")
    p.add_argument("--seed", type=int, default=None, help="Random seed")

    p.add_argument("--digit", type=int, default=None, help="Fixed digit for shape (1..9); omit for random")
    p.add_argument("--shape-type", type=str, default="auto", choices=["auto","rect","L","plus","blob"],
                   help="Shape type")
    p.add_argument("--shape-max-h", type=int, default=None, help="Max shape height (for blob/auto too)")
    p.add_argument("--shape-max-w", type=int, default=None, help="Max shape width (for blob/auto too)")
    p.add_argument("--blob-area", type=int, default=None, help="Approx target filled cells for blob")
    p.add_argument("--blob-porosity", type=float, default=0.0, help="Hole ratio [0..0.9], connectivity preserved")
    p.add_argument("--blob-hole-chunk", type=int, default=1, help="Max hole chunk size (1=single-cell holes)")
    return p.parse_args()

def main():
    args = parse_args()
    env = GridEnv(rows=args.rows, cols=args.cols, noise_prob=args.noise_prob, seed=args.seed,
                  digit=args.digit, shape_type=args.shape_type,
                  shape_max_h=args.shape_max_h, shape_max_w=args.shape_max_w,
                  blob_area=args.blob_area, blob_porosity=args.blob_porosity,
                  blob_hole_chunk=args.blob_hole_chunk if hasattr(args, "blob-hole-chunk") else args.blob_hole_chunk) 

    # 初始化并显示
    grid = env.get_matrix()  # __init__ 已自动 reset
    while True:
        info = env.get_info()
        clear_screen()
        print(f"Grid: {info['rows']}x{info['cols']} | Noise Prob: {info['noise_prob']:.2f} | Seed: {info['seed']}")
        print(f"Shape: {info['shape_name']} | Size: {info['shape_size'][0]}x{info['shape_size'][1]} | Pos: {info['position']}")
        print("Controls: w=up, s=down, a=left, d=right, q=quit  (press Enter after key)")
        print("-" * (info['cols'] * 2 - 1))
        print(render_ascii(grid))
        print("-" * (info['cols'] * 2 - 1))

        try:
            cmd = input("Move (w/a/s/d) or q: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye."); break

        if not cmd:
            continue
        if cmd[0] == 'q':
            print("Bye."); break

        env.move(cmd[0])
        grid = env.get_matrix()

if __name__ == "__main__":
    main()