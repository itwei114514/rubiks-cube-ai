"""GPU beam search guided by the learned diffusion-distance ResMLP heuristic.

The search is implemented as a growing node list (states + parent pointers +
creating move).  At each depth the beam's states are expanded by all 12 moves,
scored by the network, and the top-`beam` *distinct* states (lowest predicted
distance) become the next beam.  A **global visited set** prevents re-expanding
states already seen (cutting cycles / duplicate work).  A solution is
reconstructed by tracing parent pointers when the solved state is reached.
"""
from __future__ import annotations

import torch
from cube import state as C
from cube.state import state_to_input

MOVE_INV_IDX = [C.MOVE_INDEX[C._INVERSE_TABLE[name]] for name in C.MOVE_NAMES]
MOVE_INV_IDX = torch.tensor(MOVE_INV_IDX, dtype=torch.long)


def _reconstruct(parent, move, node_id, final_move):
    seq = []
    if final_move is not None:
        seq.append(C.MOVE_NAMES[int(final_move)])
    cur = int(node_id)
    while int(parent[cur]) >= 0:
        seq.append(C.MOVE_NAMES[int(move[cur])])
        cur = int(parent[cur])
    seq.reverse()
    return seq


@torch.inference_mode()
def beam_search(model, init_state, beam=4096, max_depth=100, device=None,
                return_stats=False):
    """Find a move sequence solving the cube from `init_state`.

    init_state: int tensor (...,54) or (54,).
    Returns a list of move names, or None if not found within max_depth.
    With return_stats=True returns (moves, solution_depth, n_expansions).
    """
    device = device or next(model.parameters()).device
    model.eval()

    if init_state.dim() == 1:
        init_state = init_state.unsqueeze(0)
    init_state = init_state.to(device).long()

    if bool(C.is_solved(init_state).any()):
        return [] if not return_stats else ([], 0, 0)

    states = init_state.clone()
    parent = torch.tensor([-1], dtype=torch.long, device=device)
    move = torch.tensor([-1], dtype=torch.long, device=device)
    beam_ids = torch.tensor([0], dtype=torch.long, device=device)
    visited = {bytes(init_state[0].tolist())}   # global dedup set

    n_exp = 0
    for depth in range(max_depth):
        cur = states[beam_ids]
        n = cur.shape[0]
        expanded = C.apply_all_moves(cur).reshape(-1, 54)          # (B*12, 54)
        exp_parent = beam_ids.repeat_interleave(C.NUM_MOVES)
        exp_move = torch.arange(C.NUM_MOVES, device=device).repeat(n)
        n_exp += expanded.shape[0]

        h = model(state_to_input(expanded))                         # (B*12,)

        sm = C.is_solved(expanded)
        if bool(sm.any()):
            j = int(sm.nonzero()[0, 0])
            moves = _reconstruct(parent, move, exp_parent[j], exp_move[j])
            if return_stats:
                return moves, depth + 1, n_exp
            return moves

        # order candidates, then pick the first `beam` distinct, non-undo states
        order = torch.argsort(h, stable=True)
        exp_cpu = expanded.cpu()
        if depth >= 1:
            pm = move[exp_parent]                                   # (B*12,)
            keep = (exp_move != MOVE_INV_IDX.to(device)[pm]).cpu()
        else:
            keep = torch.ones(expanded.shape[0], dtype=torch.bool)

        sel_state, sel_parent, sel_move = [], [], []
        seen_in_depth = set()
        for idx in order.tolist():
            if not bool(keep[idx]):
                continue
            key = bytes(exp_cpu[idx].tolist())
            if key in visited or key in seen_in_depth:
                continue
            visited.add(key)
            seen_in_depth.add(key)
            sel_state.append(idx)
            sel_parent.append(exp_parent[idx])
            sel_move.append(exp_move[idx])
            if len(sel_state) >= beam:
                break

        if not sel_state:
            return None if not return_stats else (None, depth, n_exp)

        sel_state = torch.tensor(sel_state, device=device)
        new_ids = torch.arange(len(states), len(states) + len(sel_state), device=device)
        states = torch.cat([states, expanded[sel_state]], dim=0)
        parent = torch.cat([parent, torch.stack(sel_parent)], dim=0)
        move = torch.cat([move, torch.stack(sel_move)], dim=0)
        beam_ids = new_ids

    return None if not return_stats else (None, max_depth, n_exp)