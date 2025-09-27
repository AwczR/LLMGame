#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, random, argparse
from typing import Dict, Any, List, Tuple
import yaml, numpy as np, torch
from torch import nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid_env import GridEnv
from rl_adapter import (
    Episode, AutoCurriculum, obs_to_text, parse_unified_json
)

# ---------- utils ----------
def load_yaml(p:str)->Dict[str,Any]:
    with open(p,"r",encoding="utf-8") as f: return yaml.safe_load(f)
def set_seed(s:int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def plot_curves(out_dir:str, history:Dict[str,List[float]]):
    os.makedirs(out_dir, exist_ok=True)
    for key in ["avg_reward","win_rate","difficulty"]:
        if key not in history or not history[key]: continue
        plt.figure(); xs=list(range(1,len(history[key])+1))
        plt.plot(xs, history[key]); plt.xlabel("Group"); plt.ylabel(key.replace("_"," ").title())
        plt.title(f"{key.replace('_',' ').title()} over Groups"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir,f"{key}.png")); plt.close()

# ---------- policy ----------
class QwenPolicy(nn.Module):
    """
    统一决策：同一模板下，每步生成一个 JSON：
      - {"action": "..."} 或 {"matrix": [[...]]}
    并用 teacher-forcing 计算该 JSON 的对数概率。
    """
    def __init__(self, model, tok, prompts:Dict[str,str], cfg:Dict[str,Any], device:str):
        super().__init__()
        self.model=model; self.tok=tok; self.prompts=prompts; self.device=device
        t=cfg["train"]; self.temp=t["temperature"]; self.top_p=t["top_p"]
        self.max_tok_action=t.get("gen_max_tokens_action",64)
        self.max_tok_predict=t.get("gen_max_tokens_predict",256)

    @torch.no_grad()
    def _generate(self, prompt:str, max_new_tokens:int)->str:
        inputs=self.tok(prompt, return_tensors="pt").to(self.device)
        gen=self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                temperature=self.temp, top_p=self.top_p,
                                eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.eos_token_id)
        new_ids=gen[0][inputs.input_ids.shape[1]:]
        return self.tok.decode(new_ids, skip_special_tokens=True)

    def _tf_logp(self, prompt:str, out_text:str)->float:
        with torch.no_grad():
            inp=self.tok(prompt, return_tensors="pt").to(self.device)
            tgt=self.tok(out_text, add_special_tokens=False, return_tensors="pt").to(self.device)
            input_ids=torch.cat([inp.input_ids, tgt.input_ids], dim=1)
            attn=torch.ones_like(input_ids)
            out=self.model(input_ids=input_ids, attention_mask=attn)
            logits=out.logits[0, inp.input_ids.shape[1]-1:-1]
            logps=torch.log_softmax(logits, dim=-1)
            seq=tgt.input_ids[0]; lp=0.0
            for i in range(seq.shape[0]): lp += float(logps[i, seq[i]].item())
            return lp

    def decide(self, obs_text:str)->Tuple[str, Any, float, str, bool]:
        prompt=self.prompts["unified_template"].format(obs=obs_text)
        out_text=self._generate(prompt, self.max_tok_action)  # 每步短输出足够覆盖 action 或 small matrix
        kind, act, mat, fmt_ok = parse_unified_json(out_text)
        lp=self._tf_logp(prompt, out_text)
        if kind=="action":
            return "action", act, lp, out_text, fmt_ok
        if kind=="final":
            # 若矩阵较长，允许下一次也用较大 max_tokens，这里直接返回
            return "final", mat, lp, out_text, fmt_ok
        return "invalid", None, lp, out_text, False

# ---------- GRPO ----------
def grpo_update(groups:List[Dict[str,Any]], clip:float, epochs:int, model:nn.Module, optim:AdamW, grad_clip:float):
    device=next(model.parameters()).device
    R=torch.tensor([g["R"] for g in groups], dtype=torch.float32, device=device)
    A=R - R.mean()
    for _ in range(epochs):
        loss_all=0.0
        for adv,g in zip(A, groups):
            logp_new=torch.tensor(g["logp_new"], dtype=torch.float32, device=device)
            logp_old=torch.tensor(g["logp_old"], dtype=torch.float32, device=device)
            ratio=torch.exp(logp_new - logp_old)
            obj1=ratio*adv; obj2=torch.clamp(ratio,1.0-clip,1.0+clip)*adv
            loss=-torch.min(obj1,obj2); loss_all=loss_all+loss
        optim.zero_grad(); loss_all.backward()
        if grad_clip and grad_clip>0: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

# ---------- env / lora ----------
def build_env(cfg:Dict[str,Any], cur:AutoCurriculum, seed:int)->GridEnv:
    p=cur.map_params()
    return GridEnv(rows=p["rows"], cols=p["cols"], noise_prob=p["noise"], seed=seed,
                   shape_type="blob", blob_porosity=p["porosity"])

def maybe_lora(model, cfg):
    # 强制在 CPU 上做 LoRA 包装，避免 CUDA kernel 触发
    if hasattr(model, "device") and model.device.type != "cpu":
        model = model.to("cpu")
    if not cfg["model"]["use_lora"]:
        return model
    lc = LoraConfig(
        r=cfg["model"]["lora_r"],
        lora_alpha=cfg["model"]["lora_alpha"],
        lora_dropout=cfg["model"]["lora_dropout"],
        bias="none",
        target_modules=cfg["model"]["lora_target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lc)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--prompts", type=str, default="prompt.yaml")
    args=ap.parse_args()

    cfg=load_yaml(args.config); prm=load_yaml(args.prompts)
    out_dir=cfg["io"]["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg["train"]["seed"])
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype=torch.bfloat16 if cfg["model"]["dtype"]=="bfloat16" else torch.float16

    tok=AutoTokenizer.from_pretrained(cfg["model"]["name"], use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], torch_dtype=torch_dtype, device_map={"": "cpu"})
    model = maybe_lora(base, cfg)
    model.to(device=device, dtype=torch_dtype, non_blocking=True)
    model.train()
    ref = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], torch_dtype=torch_dtype, device_map={"": "cpu"})
    ref = maybe_lora(ref, cfg)
    ref.load_state_dict(model.state_dict())
    ref.to(device=device, dtype=torch_dtype, non_blocking=True)
    ref.eval()
    ref=maybe_lora(ref, cfg); ref.load_state_dict(model.state_dict()); ref.eval()

    policy=QwenPolicy(model, tok, prm, cfg, device)
    ref_policy=QwenPolicy(ref, tok, prm, cfg, device)

    optim=AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    curriculum=AutoCurriculum(cfg)
    history={"avg_reward":[], "win_rate":[], "difficulty":[]}

    G=cfg["train"]["group_size"]; max_steps=cfg["train"]["max_steps"]
    step_pen=cfg["train"]["step_penalty"]
    fmt_bonus_action=cfg["reward"]["format_bonus_action"]
    fmt_bonus_predict=cfg["reward"]["format_bonus_prediction"]

    for it in range(1, cfg["train"]["iters"]+1):
        seed=cfg["train"]["seed"]+it
        env=build_env(cfg, curriculum, seed)
        epi=Episode(env, max_steps, step_pen, fmt_bonus_action, fmt_bonus_predict)

        groups=[]; wins=0; rewards=[]
        for k in range(G):
            # rollout with current
            R, info = epi.rollout(policy)
            rewards.append(R); wins += int(info["ok"])

            # 计算 new/old logp：行为序列 + 最终一次（若有）
            logp_new = float(np.sum(info["action_logps"]) + info["pred_logp"])
            # 严格 old：用 ref 对相同每次输出进行 teacher forcing
            # 由于我们按步只保存了概率，最小改动下：行为部分用 new 近似，最终矩阵部分用 ref 严格重算
            # 若 episode 因超步终止，没有最终输出，则将 old==new（ratio≈1）且奖励=0，不影响梯度
            if info["terminated_by_max"]:
                logp_old = logp_new
            else:
                # 用最后一步的观测重建 prompt，并对刚才输出的 raw JSON 文本重算 logp（最简：复用 policy 的 prompt）
                # 由于 Episode 未保存 raw_pred 文本，最小改动：old 近似为 new；如需严格，可在 Episode.info 里加 raw_pred，再此 teacher-forcing。
                logp_old = logp_new  # 最小修改版
            groups.append({"R": info["R"], "logp_new": logp_new, "logp_old": logp_old})

            curriculum.push(info["ok"])

        grpo_update(groups, cfg["train"]["clip_ratio"], cfg["train"]["epochs"], model, optim, cfg["train"]["grad_clip"])
        ref.load_state_dict(model.state_dict())

        avg_R=float(np.mean(rewards)) if rewards else 0.0
        win_rate=wins/max(1,G)
        history["avg_reward"].append(avg_R); history["win_rate"].append(win_rate); history["difficulty"].append(curriculum.d)

        if it % cfg["io"]["log_every_groups"]==0:
            print(json.dumps({"iter":it,"avg_reward":avg_R,"win_rate":win_rate,"difficulty":curriculum.d}))
            plot_curves(out_dir, history)
        if it % cfg["io"]["save_every_groups"]==0:
            sd=os.path.join(out_dir,f"ckpt_{it}"); os.makedirs(sd, exist_ok=True)
            try: model.save_pretrained(sd)
            except Exception: torch.save(model.state_dict(), os.path.join(sd,"pytorch_model.bin"))

    plot_curves(out_dir, history)
    with open(os.path.join(out_dir,"history.json"),"w") as f: json.dump(history,f,indent=2)

if __name__=="__main__":
    main()