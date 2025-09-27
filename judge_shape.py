#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修改版 main：
- 初始化环境，渲染并显示。
- 写死一份“预测矩阵 pred”，模拟 LLM 输出。
- 调用 env.verify_prediction(pred)，打印判定结果。
"""

import argparse
import sys
from grid_env import GridEnv

def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

def render_ascii(grid):
    return "\n".join(" ".join(str(v) for v in row) for row in grid)

def parse_args():
    p = argparse.ArgumentParser(description="Grid Pattern CLI Environment (library + main)")
    p.add_argument("--rows", type=int, default=15)
    p.add_argument("--cols", type=int, default=30)
    p.add_argument("--noise-prob", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--digit", type=int, default=None)
    p.add_argument("--shape-type", type=str, default="blob", choices=["auto","rect","L","plus","blob"])
    p.add_argument("--shape-max-h", type=int, default=None)
    p.add_argument("--shape-max-w", type=int, default=None)
    p.add_argument("--blob-area", type=int, default=None)
    p.add_argument("--blob-porosity", type=float, default=0.0)
    p.add_argument("--blob-hole-chunk", type=int, default=1)
    return p.parse_args()

def main():
    args = parse_args()
    env = GridEnv(rows=args.rows, cols=args.cols, noise_prob=args.noise_prob, seed=args.seed,
                  digit=args.digit, shape_type=args.shape_type,
                  shape_max_h=args.shape_max_h, shape_max_w=args.shape_max_w,
                  blob_area=args.blob_area, blob_porosity=args.blob_porosity,
                  blob_hole_chunk=args.blob_hole_chunk)

    # 初始化并显示当前矩阵
    grid = env.get_matrix()
    info = env.get_info()
    clear_screen()
    print(f"Grid: {info['rows']}x{info['cols']} | Noise Prob: {info['noise_prob']:.2f} | Seed: {info['seed']}")
    print(f"Shape: {info['shape_name']} | Size: {info['shape_size'][0]}x{info['shape_size'][1]} | Pos: {info['position']}")
    print("-" * (info['cols'] * 2 - 1))
    print(render_ascii(grid))
    print("-" * (info['cols'] * 2 - 1))

    # 写一个“预测矩阵”，这里直接拿标准答案来模拟正确预测
    pred = env.get_target_canonical()

    print('answer:')
    print(render_ascii(pred))

    # 判定
    result = env.verify_prediction(pred)
    print("Prediction (simulated):")
    print(render_ascii(pred))
    print("Judgement:", "PASS" if result.get("ok") else "FAIL")
    print("Detail:", result)

if __name__ == "__main__":
    main()