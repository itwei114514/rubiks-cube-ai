"""Re-run on 100 deep scrambles, beam=2048, record per-cube solution lengths, then
compute statistics (worst 5%, variance, etc.)."""
from __future__ import annotations

import os, sys, json, time
import torch, numpy as np

sys.path.insert(0, r"F:\VScode_program\.vscode\Rubik's Cube_agent")
from config import Config
from cube import scramble as S
from solve.agent import load_agent

cfg = Config()
cfg.beam = 2048
device = cfg.device
N = 100
SEED = 2026
BEAM = 2048
MAX_DEPTH = 100
LOGS = r"F:\VScode_program\.vscode\Rubik's Cube_agent\logs"
DATA = r"F:\VScode_program\.vscode\Rubik's Cube_agent\data"

rng = np.random.default_rng(SEED)
states = S.generate_scrambles(N, min_len=20, max_len=cfg.k_max, device=device, rng=rng)
os.makedirs(DATA, exist_ok=True)
torch.save(states.cpu(), os.path.join(DATA, "eval100_deep.pt"))

agent = load_agent(cfg)
records = []
t0 = time.time()
print(f"start {N} cubes, beam={BEAM}, max_depth={MAX_DEPTH}", flush=True)
for i in range(states.shape[0]):
    s = time.time()
    sol = agent.solve(states[i], beam=BEAM, max_depth=MAX_DEPTH)
    dt = time.time() - s
    records.append({"i": i, "len": (len(sol) if sol is not None else None), "time": dt})
    if (i + 1) % 20 == 0:
        solved = sum(1 for r in records if r["len"] is not None)
        print(f"  {i+1}/{N} solved={solved} elapsed={time.time()-t0:.0f}s", flush=True)

lens = [r["len"] for r in records if r["len"] is not None]
n_fail = N - len(lens)
os.makedirs(LOGS, exist_ok=True)
with open(os.path.join(LOGS, "eval100_beam2048_percube.json"), "w") as f:
    json.dump({"beam": BEAM, "N": N, "records": records}, f, indent=2)

if lens:
    arr = np.array(lens)
    worst5 = int(np.ceil(len(arr) * 0.05))          # number of cubes in worst 5%
    worst = np.sort(arr)[-worst5:] if worst5 > 0 else arr
    stats = {
        "solved": len(lens), "failed": n_fail, "solve_rate": len(lens) / N,
        "mean": float(arr.mean()), "median": float(np.median(arr)),
        "std": float(arr.std()), "var": float(arr.var()),
        "min": int(arr.min()), "max": int(arr.max()),
        "p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)), "p99": float(np.percentile(arr, 99)),
        "worst5pct_avg": float(worst.mean()), "worst5pct_max": int(worst.max()),
        "worst5pct_n": int(len(worst)), "total_s": time.time() - t0,
        "avg_time_s": float(np.mean([r["time"] for r in records])),
    }
    print("STATS", json.dumps(stats, indent=2), flush=True)
    with open(os.path.join(LOGS, "eval100_beam2048_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
else:
    print("STATS: no cubes solved", flush=True)