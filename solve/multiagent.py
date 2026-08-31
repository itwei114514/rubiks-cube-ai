"""Multi-agent ensemble: several independently-trained ResMLP heuristics are
each run with beam search; the shortest resulting solution wins.  Because each
agent's distance estimate errs differently, the ensemble reliably improves
solution length and success rate over a single model."""
from __future__ import annotations

from .beam import beam_search


def solve_ensemble(models, init_state, beam=4096, max_depth=100, device=None,
                   return_stats=False):
    """Run each model's beam search and return the best (shortest) solution."""
    best = None
    best_len = None
    stats = []
    for model in models:
        res = beam_search(model, init_state, beam=beam, max_depth=max_depth,
                          device=device, return_stats=return_stats)
        if return_stats:
            moves, depth, n_visited = res
            stats.append((depth, n_visited))
        else:
            moves = res
        if moves is not None:
            if best_len is None or len(moves) < best_len:
                best = moves
                best_len = len(moves)
    if return_stats:
        return best, stats
    return best