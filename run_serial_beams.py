"""Serial beam-width comparison (small scale). Runs beams one at a time.
Usage: python run_serial_beams.py --n 20 --max_depth 100 [--states PATH]
"""
from __future__ import annotations

import os, sys, json, time, argparse
import torch, numpy as np

sys.path.insert(0, r"F:\VScode_program\.vscode\Rubik's Cube_agent")
from config import Config
from cube import scramble as S
from solve.agent import CubeAgent

BEAMS = [4096, 8192, 20480]           # 1x, 2x, 5x of 2^12
BZ = r"F:\VScode_program\.vscode\Rubik's Cube_agent\ckpt\1e8_backup"
LOGS = r"F:\VScode_program\.vscode\Rubik's Cube_agent\logs"


def run_one(agent, states, beam, max_depth):
    solved, lens, times = 0, [], []
    t0 = time.time()
    print(f"[beam={beam}] start {states.shape[0]} cubes, max_depth={max_depth}", flush=True)
    for i in range(states.shape[0]):
        s = time.time()
        sol = agent.solve(states[i], beam=beam, max_depth=max_depth)
        dt = time.time() - s
        times.append(dt)
        if sol is not None:
            solved += 1
            lens.append(len(sol))
        # per-cube progress so we can monitor clearly
        print(f"[beam={beam}] cube {i+1}/{states.shape[0]} solved={solved} "
              f"len={len(sol) if sol else '--'} elapsed={time.time()-t0:.0f}s", flush=True)
    total = time.time() - t0
    return {
        "beam": beam, "n": states.shape[0], "max_depth": max_depth, "solved": solved,
        "solve_rate": solved / states.shape[0],
        "avg_len": (sum(lens) / len(lens)) if lens else None,
        "avg_time_s": (sum(times) / len(times)) if times else None,
        "total_time_s": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max_depth", type=int, default=100)
    ap.add_argument("--states", type=str, default=None)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    cfg = Config()
    device = cfg.device

    if args.states and os.path.isfile(args.states):
        states = torch.load(args.states, map_location="cpu").to(device)
        if states.shape[0] > args.n:
            states = states[:args.n]
    else:
        rng = np.random.default_rng(args.seed)
        states = S.generate_scrambles(args.n, min_len=20, max_len=cfg.k_max, device=device, rng=rng)
        os.makedirs(BZ, exist_ok=True)
        p = os.path.join(BZ, f"eval_{args.n}states.pt")
        torch.save(states.cpu(), p)
        print("saved states ->", p, flush=True)

    agent = CubeAgent([os.path.join(BZ, "agent0_1e8.pt"),
                       os.path.join(BZ, "agent1_1e8.pt")], cfg)
    os.makedirs(LOGS, exist_ok=True)
    all_res = {}
    for beam in BEAMS:
        r = run_one(agent, states, beam, args.max_depth)
        all_res[beam] = r
        with open(os.path.join(LOGS, f"serial_beam{beam}.json"), "w") as f:
            json.dump(r, f, indent=2)
        with open(os.path.join(LOGS, "serial_results.json"), "w") as f:
            json.dump(all_res, f, indent=2)
        print("DONE", json.dumps(r), flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()