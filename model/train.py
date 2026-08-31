"""FP16 training of the diffusion-distance ResMLP, with purely self-supervised
online data generation (random walks) and periodic beam-search validation.

Default budget: `cfg.total_samples` (state, depth) pairs per agent.
If `max_hours > 0`, training instead adaptively targets that wall-clock budget.
Supports `resume=True` to continue from the last saved per-agent training state
(model, optimizer, LR-scheduler, scaler, RNG, step), which is essential for long
multi-hour runs.  Uses a cosine LR schedule for better long-run convergence.
"""
from __future__ import annotations

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.resmlp import build_model
from cube import scramble as S
from config import Config


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_criterion(cfg):
    if cfg.loss == "mse":
        return nn.MSELoss()
    return nn.SmoothL1Loss(beta=1.0)   # huber


def build_validation_states(cfg, device):
    return S.generate_scrambles(cfg.val_scrambles, min_len=20, max_len=cfg.k_max,
                                device=device)


@torch.no_grad()
def validate(model, val_states, cfg, device):
    from solve.beam import beam_search
    model.eval()
    solved, lens = 0, []
    for i in range(val_states.shape[0]):
        moves = beam_search(model, val_states[i], beam=cfg.val_beam,
                            max_depth=cfg.max_depth, device=device)
        if moves is not None:
            solved += 1
            lens.append(len(moves))
    model.train()
    avg_len = float(np.mean(lens)) if lens else float("nan")
    return solved, avg_len, len(lens)


def _estimate_step_sec(model, device, cfg, burn_in=20):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16 and device.type == "cuda")
    crit = make_criterion(cfg)
    t0 = time.time()
    for _ in range(burn_in):
        x, y = S.make_batch(cfg.num_walks, cfg.k_max, device=device)
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=cfg.fp16 and device.type == "cuda"):
            out = model(x)
            loss = crit(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
    return (time.time() - t0) / burn_in


def _save_train_checkpoint(path, model, opt, sched, scaler, step, cfg, agent_i,
                           best_avg_len, device):
    tmp = {
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "scheduler_state": sched.state_dict(),
        "scaler_state": scaler.state_dict(),
        "step": step,
        "config": cfg,
        "agent_i": agent_i,
        "best_avg_len": best_avg_len,
        "rng_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        tmp["rng_cuda"] = torch.cuda.get_rng_state()
    torch.save(tmp, path)


def _save_model(path, model, cfg, agent_i, steps, best_avg_len):
    torch.save({"model_state": model.state_dict(), "config": cfg, "agent_i": agent_i,
                "steps": steps, "best_avg_len": best_avg_len}, path)


def train(cfg: Config, max_hours=0.0, max_steps=None, resume=False):
    device = cfg.device
    ckpt_dir = os.path.join(cfg.project_dir, cfg.out_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    base_steps = cfg.num_steps

    for agent_i in range(cfg.num_agents):
        set_seed(cfg.seed + agent_i * 1000)
        model = build_model(cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16 and device.type == "cuda")
        crit = make_criterion(cfg)
        model.train()

        steps = max_steps if max_steps else base_steps
        if max_hours > 0 and steps == base_steps:
            dt = _estimate_step_sec(model, device, cfg)
            steps = int(max_hours * 3600 / max(dt, 1e-6))
            print(f"[agent{agent_i}] est. step time {dt*1000:.1f} ms -> {steps} steps for {max_hours}h")

        sched = CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=cfg.lr * 0.02)

        # resumable state
        start_step = 0
        best_avg_len = float("inf")
        last_path = os.path.join(ckpt_dir, f"agent{agent_i}_train.pt")
        if resume and os.path.isfile(last_path):
            ck = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state"])
            opt.load_state_dict(ck["optimizer_state"])
            if ck.get("scheduler_state") is not None:
                sched.load_state_dict(ck["scheduler_state"])
            if ck.get("scaler_state") is not None:
                scaler.load_state_dict(ck["scaler_state"])
            start_step = int(ck["step"])
            best_avg_len = ck.get("best_avg_len", float("inf")) or float("inf")
            # map_location may have moved the CPU RNG state onto cuda; bring it back
            try:
                torch.set_rng_state(ck["rng_cpu"].cpu())
            except Exception as e:
                print("  warn: could not restore CPU RNG:", e)
            if device.type == "cuda" and ck.get("rng_cuda") is not None:
                try:
                    torch.cuda.set_rng_state(ck["rng_cuda"].cpu())
                except Exception as e:
                    print("  warn: could not restore CUDA RNG:", e)
            print(f"[agent{agent_i}] resumed from step {start_step}/{steps}")

        val_states = build_validation_states(cfg, device)
        t_start = time.time()
        running, count = 0.0, 0
        print(f"[agent{agent_i}] training {steps} steps, batch_pairs={cfg.batch_pairs}, "
              f"params={model.num_params()/1e6:.2f}M, device={device}", flush=True)

        for step in range(start_step, steps + 1):
            x, y = S.make_batch(cfg.num_walks, cfg.k_max, device=device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=cfg.fp16 and device.type == "cuda"):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            sched.step()

            running += loss.item()
            count += 1
            if step % cfg.log_every == 0 and step > 0:
                print(f"  step {step}/{steps}  loss {running/count:.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  ({time.time()-t_start:.0f}s)", flush=True)
                running, count = 0.0, 0

            if step and (step % cfg.validate_every == 0 or step == steps):
                solved, avg_len, n_ok = validate(model, val_states, cfg, device)
                print(f"  [val@step {step}] solved {solved}/{val_states.shape[0]}  "
                      f"avg_len {avg_len:.1f} ({n_ok} solved)", flush=True)
                if avg_len == avg_len and avg_len < best_avg_len:
                    best_avg_len = avg_len
                    _save_model(cfg.ckpt_path(tag=cfg.model_tag(agent_i)), model, cfg,
                                agent_i, step, best_avg_len)

            if step and step % cfg.save_every == 0:
                _save_train_checkpoint(last_path, model, opt, sched, scaler, step,
                                       cfg, agent_i, best_avg_len, device)

        _save_model(os.path.join(ckpt_dir, f"{cfg.model_tag(agent_i)}.pt"), model, cfg,
                    agent_i, steps, best_avg_len)
        _save_train_checkpoint(last_path, model, opt, sched, scaler, steps, cfg,
                               agent_i, best_avg_len, device)
        print(f"[agent{agent_i}] done -> {os.path.join(ckpt_dir, f'agent{agent_i}.pt')}  "
              f"best_avg_len {best_avg_len:.2f}", flush=True)