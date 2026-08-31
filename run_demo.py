"""Interactive demo: solve a cubing scramble (or a random one)."""
from __future__ import annotations

import argparse
import torch
import numpy as np

from config import Config
from cube import state as C, scramble as S
from solve.agent import load_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scramble", type=str, default=None,
                    help="moves, e.g. \"R U R' U'\" (standard notation)")
    ap.add_argument("--beam", type=int, default=None)
    ap.add_argument("--max_depth", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.beam:
        cfg.beam = args.beam
    if args.max_depth:
        cfg.max_depth = args.max_depth

    agent = load_agent(cfg)
    device = cfg.device

    if args.scramble:
        moves = C.parse_scramble(args.scramble)
    else:
        moves = C.parse_scramble(C.random_scramble_str(cfg.k_max))
    start = C.apply_moves(torch.tensor(C.solved_stickers(), dtype=torch.int64), moves)

    print(f"scramble ({len(moves)} moves): {' '.join(moves)}")
    sol = agent.solve(start, beam=cfg.beam, max_depth=cfg.max_depth)
    if sol is None:
        print("No solution found within max_depth.")
        return
    print(f"solution ({len(sol)} moves): {' '.join(sol)}")

    # verify
    end = C.apply_moves(start, sol)
    print("valid:", bool(C.is_solved(end)))


if __name__ == "__main__":
    main()