"""Evaluate the trained agent on a set of arbitrary scrambles."""
from __future__ import annotations

import argparse
import time
import torch
import numpy as np

from config import Config
from cube import scramble as S
from solve.agent import load_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=100, help="number of scrambles to test")
    ap.add_argument("--beam", type=int, default=None)
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--num_agents", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    if args.smoke:
        args.num = 5
        cfg.beam = 128
        cfg.max_depth = 30
    if args.beam:
        cfg.beam = args.beam
    if args.max_depth:
        cfg.max_depth = args.max_depth
    if args.num_agents:
        cfg.num_agents = args.num_agents

    device = cfg.device
    agent = load_agent(cfg)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    states = S.generate_scrambles(args.num, min_len=20, max_len=cfg.k_max, device=device)

    print(f"Evaluating {args.num} scrambles, beam={cfg.beam}, agents={cfg.num_agents}, "
          f"max_depth={cfg.max_depth} ...", flush=True)
    solved, lens, times = 0, [], []
    for i in range(states.shape[0]):
        t0 = time.time()
        moves = agent.solve(states[i], beam=cfg.beam, max_depth=cfg.max_depth)
        dt = time.time() - t0
        times.append(dt)
        if moves is not None:
            solved += 1
            lens.append(len(moves))
    avg_len = float(np.mean(lens)) if lens else float("nan")
    avg_time = float(np.mean(times))
    print(f"\n=== RESULTS ===")
    print(f"solved: {solved}/{args.num}  ({100*solved/args.num:.1f}%)")
    print(f"avg solution length: {avg_len:.1f}  (target <= 100)")
    print(f"avg solve time: {avg_time:.2f}s/cube  (beam={cfg.beam})")


if __name__ == "__main__":
    main()