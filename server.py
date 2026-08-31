"""Web UI backend: serves the 3D cube page and solves scrambles with the trained
diffusion-distance ResMLP + beam search agent.

Run (from the project directory, using the project venv):
    python server.py
Then open http://127.0.0.1:8000/
"""
from __future__ import annotations

import os
import sys
import threading
import collections

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from config import Config
from cube import state as C
from solve.agent import CubeAgent

app = FastAPI(title="Rubik's Cube AI")
cfg = Config()
agent = CubeAgent([cfg.primary_ckpt], cfg)


class SolveReq(BaseModel):
    scramble: str = ""
    beam: int = 4096


class FaceletReq(BaseModel):
    facelet: str = ""
    beam: int = 4096


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE, "ui", "index.html"))


app.mount("/ui", StaticFiles(directory=os.path.join(BASE, "ui")), name="ui")


@app.get("/api/scramble")
def scramble(n: int = 22):
    return {"scramble": C.random_scramble_str(n)}


@app.post("/api/solve")
def solve(req: SolveReq):
    moves = C.parse_scramble(req.scramble) if req.scramble.strip() else []
    st = torch.tensor(C.solved_stickers(), dtype=torch.int64)
    st = C.apply_moves(st, moves)
    beam = max(1, int(req.beam))
    sol = agent.solve(st, beam=beam, max_depth=cfg.max_depth)
    if sol is None:
        return {"moves": [], "length": 0, "solved": False,
                "error": "未找到解（100 步窗口内）"}
    end = C.apply_moves(st, sol)
    solved = bool(C.is_solved(end))
    return {"moves": sol, "length": len(sol), "solved": solved}


@app.post("/api/solve_facelet")
def solve_facelet(req: FaceletReq):
    s = req.facelet.strip().upper()
    if len(s) != 54:
        return {"moves": [], "length": 0, "solved": False,
                "error": f"面片数应为 54，实际 {len(s)}"}
    if not set(s) <= set("URFDLB"):
        return {"moves": [], "length": 0, "solved": False,
                "error": "包含非法颜色字符"}
    cnt = collections.Counter(s)
    if not all(cnt[c] == 9 for c in "URFDLB"):
        return {"moves": [], "length": 0, "solved": False,
                "error": "每种颜色应恰好出现 9 次（简单合法性检查未通过）"}
    st = C.from_kociemba(s)
    beam = max(1, int(req.beam))
    sol = agent.solve(st, beam=beam, max_depth=cfg.max_depth)
    if sol is None:
        return {"moves": [], "length": 0, "solved": False,
                "error": "未找到解（可能无解，或 100 步窗口内未找到）"}
    end = C.apply_moves(st, sol)
    solved = bool(C.is_solved(end))
    return {"moves": sol, "length": len(sol), "solved": solved}


@app.post("/api/shutdown")
def shutdown():
    """Gracefully stop the web server (used by the in-page close button)."""
    threading.Timer(0.3, lambda: os._exit(0)).start()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)