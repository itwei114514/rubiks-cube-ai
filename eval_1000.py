"""Eval the 1e10 model on 1000 deep scrambles with beam=5120."""
from __future__ import annotations

import os, sys, json, time
import torch, numpy as np

sys.path.insert(0, r"F:\VScode_program\.vscode\Rubik's Cube_agent")
from config import Config
from cube import scramble as S
from solve.agent import CubeAgent

cfg = Config()
device = cfg.device
BEAM = 5120
MAX_DEPTH = 100
N = 1000
SEED = 2026
LOGS = r"F:\VScode_program\.vscode\Rubik's Cube_agent\logs"
BZ = r"F:\VScode_program\.vscode\Rubik's Cube_agent\ckpt\1e8_backup"

rng = np.random.default_rng(SEED)
states = S.generate_scrambles(N, min_len=20, max_len=cfg.k_max, device=device, rng=rng)
sp = os.path.join(BZ, "eval1000_deep.pt")
torch.save(states.cpu(), sp)
agent = CubeAgent([r"F:\VScode_program\.vscode\Rubik's Cube_agent\ckpt\agent0.pt"], cfg)

solved, lens, times = 0, [], []
t0 = time.time()
print(f"start {N} cubes, beam={BEAM}, max_depth={MAX_DEPTH}", flush=True)
for i in range(states.shape[0]):
    s = time.time()
    sol = agent.solve(states[i], beam=BEAM, max_depth=MAX_DEPTH)
    dt = time.time() - s
    times.append(dt)
    if sol is not None:
        solved += 1
        lens.append(len(sol))
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N} solved={solved} ({100*solved/(i+1):.1f}%) "
              f"avg_len={sum(lens)/len(lens) if lens else float('nan'):.1f} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

res = {
    "beam": BEAM, "n": N, "max_depth": MAX_DEPTH, "solved": solved,
    "solve_rate": solved / N,
    "avg_len": sum(lens)/len(lens) if lens else None,
    "avg_time_s": sum(times)/len(times) if times else None,
    "total_s": time.time() - t0,
}
os.makedirs(LOGS, exist_ok=True)
with open(os.path.join(LOGS, "eval1000_beam5120.json"), "w") as f:
    json.dump(res, f, indent=2)
print("RESULT", json.dumps(res), flush=True)