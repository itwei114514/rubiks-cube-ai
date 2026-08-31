"""A/B compare 1e8 vs 1e10 models on the SAME fixed 40 scrambles, same params
(single agent, beam=val_beam=256, max_depth=100)."""
from __future__ import annotations

import os, sys, json, time
import torch, numpy as np

sys.path.insert(0, r"F:\VScode_program\.vscode\Rubik's Cube_agent")
from config import Config
from cube import scramble as S
from solve.agent import CubeAgent

BZ = r"F:\VScode_program\.vscode\Rubik's Cube_agent\ckpt\1e8_backup"
LOGS = r"F:\VScode_program\.vscode\Rubik's Cube_agent\logs"

cfg = Config()
device = cfg.device
BEAM = cfg.val_beam          # 256 (same as in-training validation)
MAX_DEPTH = cfg.max_depth    # 100
SEED = 2026

# fixed, identical 40 scrambles
rng = np.random.default_rng(SEED)
states = S.generate_scrambles(40, min_len=20, max_len=cfg.k_max, device=device, rng=rng)
os.makedirs(BZ, exist_ok=True)
torch.save(states.cpu(), os.path.join(BZ, "compare_40states.pt"))

models = {
    "1e8 (100M/agent)": os.path.join(BZ, "agent0_1e8.pt"),
    "1e10 @step40000": os.path.join(BZ, "..", "agent0.pt"),   # current training best
}

def run(path):
    agent = CubeAgent([path], cfg)
    solved, lens, times = 0, [], []
    t0 = time.time()
    for i in range(states.shape[0]):
        s = time.time()
        sol = agent.solve(states[i], beam=BEAM, max_depth=MAX_DEPTH)
        dt = time.time() - s
        times.append(dt)
        if sol is not None:
            solved += 1
            lens.append(len(sol))
    return {
        "n": states.shape[0], "solved": solved, "solve_rate": solved / states.shape[0],
        "avg_len": (sum(lens) / len(lens)) if lens else None,
        "avg_time_s": (sum(times) / len(times)) if times else None,
        "total_s": time.time() - t0,
    }

out = {}
print(f"=== A/B compare (beam={BEAM}, max_depth={MAX_DEPTH}, {states.shape[0]} scrambles) ===", flush=True)
for name, path in models.items():
    if not os.path.isfile(path):
        print("MISSING model:", path); continue
    r = run(path)
    out[name] = r
    print(f"{name:22s} solved={r['solved']}/{r['n']} ({r['solve_rate']*100:.0f}%) "
          f"avg_len={r['avg_len']:.1f} avg_time={r['avg_time_s']:.1f}s", flush=True)

with open(os.path.join(LOGS, "compare_1e8_vs_1e10.json"), "w") as f:
    json.dump(out, f, indent=2, default=str)
print("saved", os.path.join(LOGS, "compare_1e8_vs_1e10.json"), flush=True)