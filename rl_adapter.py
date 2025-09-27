#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import json
import numpy as np
import torch
from grid_env import GridEnv

ACTIONS = ["w", "a", "s", "d", "stop"]

def obs_to_text(grid: List[List[int]]) -> str:
    return "\n".join(" ".join(str(v) for v in row) for row in grid)

class AutoCurriculum:
    def __init__(self, cfg: Dict[str, Any]):
        c = cfg["curriculum"]
        self.p_target = c["p_target"]; self.kp = c["kp"]; self.window = c["window"]
        self.rows_min, self.cols_min = c["rows_min"], c["cols_min"]
        self.rows_max, self.cols_max = c["rows_max"], c["cols_max"]
        self.noise_min, self.noise_max = c["noise_min"], c["noise_max"]
        self.blob_porosity_min = c["blob_porosity_min"]; self.blob_porosity_max = c["blob_porosity_max"]
        self.d = 0.0; self.hist: List[int] = []

    def push(self, ok: bool) -> Tuple[float, float]:
        self.hist.append(1 if ok else 0)
        if len(self.hist) > self.window: self.hist.pop(0)
        p = sum(self.hist)/max(1,len(self.hist)); e = self.p_target - p
        self.d = max(0.0, min(1.0, self.d + self.kp * e))
        return p, self.d

    def map_params(self) -> Dict[str, Any]:
        rows = round(self.rows_min + (self.rows_max - self.rows_min)*self.d)
        cols = round(self.cols_min + (self.cols_max - self.cols_min)*self.d)
        noise = self.noise_min + (self.noise_max - self.noise_min)*self.d
        poro  = self.blob_porosity_min + (self.blob_porosity_max - self.blob_porosity_min)*self.d
        return dict(rows=rows, cols=cols, noise=noise, porosity=poro)

class Episode:
    """
    统一接口：每步调用一次统一提示词，模型返回：
      - {"action": "..."} 继续观察
      - {"matrix": [[...]]} 结束并判分
    若达到 max_steps 仍未给出 matrix，则直接终止且奖励=0（不计步惩罚与格式奖励）。
    """
    def __init__(self, env: GridEnv, max_steps: int, step_penalty: float,
                 fmt_bonus_action: float, fmt_bonus_predict: float):
        self.env = env
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.fmt_bonus_action = fmt_bonus_action
        self.fmt_bonus_predict = fmt_bonus_predict
        self.steps = 0

    def reset(self):
        self.env.reset(); self.steps = 0

    def current_obs_text(self) -> str:
        return obs_to_text(self.env.get_matrix())

    def rollout(self, policy) -> Tuple[float, Dict[str, Any]]:
        """
        policy 接口：
          decide(obs_text) -> (kind:str, payload:any, seq_logp:float, raw_text:str, fmt_ok:bool)
            kind="action", payload in {"w","a","s","d","stop"}
            kind="final",  payload is List[List[int]]
        """
        self.reset()
        action_logps: List[float] = []
        fmt_ok_actions = 0
        terminated_by_max = False
        final_mat: List[List[int]] = []
        pred_logp = 0.0
        fmt_ok_pred = False

        for _ in range(self.max_steps):
            obs_text = self.current_obs_text()
            kind, payload, lp, raw, fmt_ok = policy.decide(obs_text)
            if kind == "action":
                if fmt_ok: fmt_ok_actions += 1
                a = payload if payload in ACTIONS else "stop"
                action_logps.append(lp)
                if a == "stop":
                    # 允许 stop 后再次用同一提示词产出最终矩阵（由策略自行在下一步返回 final）
                    pass
                else:
                    self.env.move(a); self.steps += 1
            elif kind == "final":
                final_mat = payload; pred_logp = lp; fmt_ok_pred = fmt_ok
                break
            else:
                # 无法解析，视为继续观察（不移动），记录logp但不加步
                action_logps.append(lp)

        else:
            # for 循环自然结束，说明达到 max_steps 且没有 final
            terminated_by_max = True

        if terminated_by_max:
            R = 0.0  # 不给任何奖励，也不计格式/步数
            ok = False
        else:
            judge = self.env.verify_prediction(final_mat)
            ok = bool(judge.get("ok"))
            R = (1.0 if ok else 0.0) - self.step_penalty * self.steps
            # 格式奖励（仅在非超步终止时计入）
            R += self.fmt_bonus_action * (1 if fmt_ok_actions > 0 else 0)
            R += self.fmt_bonus_predict * (1 if fmt_ok_pred else 0)

        info = {
            "ok": ok,
            "R": R,
            "steps": self.steps,
            "action_logps": action_logps,
            "pred_logp": pred_logp,
            "terminated_by_max": terminated_by_max,
            "fmt_ok_actions": fmt_ok_actions,
            "fmt_ok_predict": fmt_ok_pred,
        }
        return R, info

# --------- JSON helpers ----------
def parse_unified_json(s: str) -> tuple[str, Optional[str], List[List[int]], bool]:
    """
    返回: (kind, action_or_none, matrix, fmt_ok)
      kind in {"action","final","invalid"}
    """
    try:
        obj = json.loads(s)
    except Exception:
        return "invalid", None, [], False
    if isinstance(obj, dict) and "action" in obj and obj["action"] in ACTIONS:
        return "action", obj["action"], [], True
    if isinstance(obj, dict) and "matrix" in obj and isinstance(obj["matrix"], list):
        mat = obj["matrix"]
        if all(isinstance(r, list) and all(isinstance(v, int) for v in r) for r in mat):
            if mat and all(len(r) == len(mat[0]) for r in mat):
                return "final", None, mat, True
    return "invalid", None, [], False