"""Run a single beam-width evaluation on the shared 100-scramble set (1e8 weights).
Usage: python eval_beams.py --beam 4096 --out logs/beam4096.json
"""
from __future__ import annotations

import os, sys, json, time, argparse
import torch
sys.path.insert(0, r"F:\VScode_program\.vscode\Rubik's Cube_agent")
from config import Config
from cube import state as C
from solve.agent import CubeAgent

MAX_DEPTH = 200
BZ = r"F:\VScode_program\.vscode\Rubik's Cube_agent\ckpt\1e8_backup"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, required=True)
    ap.add_argument("--states", type=str, default=os.path.join(BZ, "eval_100states.pt"))
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    cfg = Config()
    agent = CubeAgent([os.path.join(BZ, "agent0_1e8.pt"),
                       os.path.join(BZ, "agent1_1e8.pt")], cfg)
    states = torch.load(args.states, map_location="cpu").to(cfg.device)

    solved, lens, times = 0, [], []
    t0 = time.time()
    print(f"[beam={args.beam}] start {states.shape[0]} cubes, max_depth={MAX_DEPTH}", flush=True)
    for i in range(states.shape[0]):
        s = time.time()
        sol = agent.solve(states[i], beam=args.beam, max_depth=MAX_DEPTH)
        dt = time.time() - s
        times.append(dt)
        if sol is not None:
            solved += 1
            lens.append(len(sol))
        if (i + 1) % 10 == 0:
            print(f"[beam={args.beam}] {i+1}/{states.shape[0]} solved={solved} "
                  f"avg_len={sum(lens)/len(lens) if lens else float('nan'):.1f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    total = time.time() - t0
    res = {
        "beam": args.beam,
        "n": states.shape[0],
        "solved": solved,
        "solve_rate": solved / states.shape[0],
        "avg_len": (sum(lens) / len(lens)) if lens else None,
        "avg_time_s": (sum(times) / len(times)) if times else None,
        "total_time_s": total,
    }
    outdir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(outdir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("RESULT", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()