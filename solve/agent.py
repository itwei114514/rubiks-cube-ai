"""High-level agent: loads trained models and solves an arbitrary 3x3 state.

A states can be given as a kociemba facelet string or an int (54,) sticker array.
"""
from __future__ import annotations

import os
import torch
from cube import state as C
from model.resmlp import build_model
from solve.multiagent import solve_ensemble


class CubeAgent:
    def __init__(self, model_paths, cfg=None):
        self.cfg = cfg
        self.models = []
        self.model_cfgs = []
        self.device = (cfg.device if cfg is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        for path in model_paths:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            stored_cfg = ckpt.get("config", cfg)
            model = build_model(stored_cfg)
            model.load_state_dict(ckpt["model_state"])
            model.to(self.device)
            model.eval()
            self.models.append(model)
            self.model_cfgs.append(stored_cfg)

    @property
    def num_models(self):
        return len(self.models)

    def solve(self, state, beam=None, max_depth=None):
        beam = beam or self.cfg.beam
        max_depth = max_depth or self.cfg.max_depth
        return solve_ensemble(self.models, state, beam=beam, max_depth=max_depth,
                              device=self.device)

    @staticmethod
    def state_from_facelet(s):
        return C.from_kociemba(s)


def load_agent(cfg, tags="012"):
    """Load `cfg.num_agents` checkpoints named {prefix}{i}.pt."""
    model_paths = []
    if os.path.exists(cfg.primary_ckpt):
        # the 1e10 primary model is the default
        model_paths.append(cfg.primary_ckpt)
    else:
        for i in range(cfg.num_agents):
            p = cfg.ckpt_path(tag=cfg.model_tag(i))
            if os.path.exists(p):
                model_paths.append(p)
    if not model_paths:
        raise FileNotFoundError("No agent checkpoints found in " + cfg.out_dir)
    return CubeAgent(model_paths, cfg)