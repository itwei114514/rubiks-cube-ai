"""Train the diffusion-distance ResMLP agent.

Examples
--------
# full run using 1e10 total training samples across all agents
python train_main.py --total_data 10000000000 --num_walks 1024 --resume

# fill the full 5h budget adaptively
python train_main.py --max_hours 5

# quick smoke test
python train_main.py --smoke
"""
from __future__ import annotations

import argparse
from config import Config
from model.train import train


def make_config(args) -> Config:
    cfg = Config()
    if args.smoke:
        cfg.num_walks = 16
        cfg.k_max = 6
        cfg.total_samples = 4096
        cfg.n1 = 128
        cfg.n2 = 64
        cfg.n_res = 1
        cfg.num_agents = 1
        cfg.beam = 8
        cfg.val_beam = 8
        cfg.val_scrambles = 8
        cfg.log_every = 20
        cfg.validate_every = 200
        cfg.save_every = 50
    if args.num_agents:
        cfg.num_agents = args.num_agents
    if args.total_data is not None:
        per = max(1, int(args.total_data) // max(1, cfg.num_agents))
        cfg.total_samples = per
        print(f"[total_data] {args.total_data} across {cfg.num_agents} agent(s) "
              f"-> {per} samples per agent (total {per * cfg.num_agents})")
    if args.total_samples:
        cfg.total_samples = args.total_samples
    if args.num_walks:
        cfg.num_walks = args.num_walks
    if args.beam:
        cfg.beam = args.beam
    if args.k_max:
        cfg.k_max = args.k_max
    if args.val_scrambles:
        cfg.val_scrambles = args.val_scrambles
    if args.validate_every:
        cfg.validate_every = args.validate_every
    if args.save_every:
        cfg.save_every = args.save_every
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_hours", type=float, default=0.0)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--total_data", type=int, default=None,
                    help="total training samples across ALL agents")
    ap.add_argument("--total_samples", type=int, default=None,
                    help="training samples per agent (overrides total_data)")
    ap.add_argument("--num_walks", type=int, default=None)
    ap.add_argument("--num_agents", type=int, default=None)
    ap.add_argument("--beam", type=int, default=None)
    ap.add_argument("--k_max", type=int, default=None)
    ap.add_argument("--val_scrambles", type=int, default=None)
    ap.add_argument("--validate_every", type=int, default=None)
    ap.add_argument("--save_every", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = make_config(args)
    print("Config:", cfg)
    train(cfg, max_hours=args.max_hours, max_steps=args.max_steps, resume=args.resume)


if __name__ == "__main__":
    main()