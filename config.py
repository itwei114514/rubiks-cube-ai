"""Central configuration for the 3x3 Rubik's Cube agent."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import torch


@dataclass
class Config:
    # ---- network (ResMLP) ----
    input_dim: int = 324          # 54 stickers x 6 colors one-hot
    n1: int = 1024                # first layer width
    n2: int = 768                 # residual block width
    n_res: int = 4                # number of residual blocks
    output_dim: int = 1           # predicted diffusion distance (scalar)

    # ---- cube / data ----
    n_moves: int = 12             # QTM quarter-turn action set
    k_max: int = 60               # maximum random-walk length (user: K_max=60)
    num_walks: int = 256          # random walks per training step
    total_samples: int = 100_000_000  # total (state, depth) pairs consumed (user: 1e8)
    val_scrambles: int = 200      # validation states
    val_beam: int = 256           # beam used during training validation (fast)

    # ---- training ----
    fp16: bool = True
    lr: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "huber"           # "mse" or "huber"
    grad_clip: float = 5.0
    log_every: int = 100
    validate_every: int = 500
    save_every: int = 5000
    seed: int = 0

    # ---- solving / inference ----
    beam: int = 4096              # user: beam = 2^12
    num_agents: int = 1           # user: agent = 2
    max_depth: int = 100          # solution length limit (user: <= 100 steps)

    # ---- io ----
    out_dir: str = "ckpt"
    project_dir: str = os.path.dirname(os.path.abspath(__file__))

    @property
    def batch_pairs(self) -> int:
        return self.num_walks * self.k_max

    @property
    def num_steps(self) -> int:
        return max(1, self.total_samples // self.batch_pairs)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def ckpt_path(self, tag: str = "agent") -> str:
        return os.path.join(self.project_dir, self.out_dir, f"{tag}.pt")

    def model_tag(self, idx: int = 0) -> str:
        """Tag naming the primary serving model.  The 1e10 model is the default
        and is always named `agent_1e10` when there is a single agent."""
        if self.num_agents == 1:
            return "agent_1e10"
        return f"agent{idx}"

    @property
    def primary_ckpt(self) -> str:
        return os.path.join(self.project_dir, self.out_dir, "agent_1e10.pt")