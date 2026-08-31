"""3x3 Rubik's Cube state and move mechanics.

The cube is described by 54 stickers, each identified by its grid position
(a cell on the unit cube surface) plus its outward face normal.  A face turn is
a 90-degree geometric rotation of that layer's cells; this is the most reliable
way to define moves (it cannot get adjacency conventions wrong).

Colors are integers 0..5 mapping to ["U","D","L","R","F","B"] (this is the
kociemba face order, used so that ``to_kociemba()`` needs no remapping).
"""
from __future__ import annotations

import numpy as np
import torch

COLORS = "URFDLB"          # index -> color letter
COLOR_IDX = {c: i for i, c in enumerate(COLORS)}
N_STICKERS = 54
N_COLORS = 6
STATE_DIM = 54 * 6         # 324-dimensional one-hot input

# Face order used by kociemba's 54-char facelet string: U, R, F, D, L, B.
# Each face grid is drawn from outside as
#     0 1 2
#     3 4 5
#     6 7 8
# Coordinates: +x=right, +y=up, +z=front.

# face -> list of (position, normal) for its 9 stickers, in facelet order.
# normals are unit vectors along the face axis.
X, Y, Z = 0, 1, 2
def _cell_list(axis, sign, order):
    """Return the 9 (x,y,z) grid positions of a face, in facelet order.

    `keep` is the face coordinates; `order` describes how the other two axes
    sweep in facelet-major order (outer index = rows as you look at the face).
    """
    # For each face we need the tangent axes and their sweep order such that the
    # facelet string matches kociemba.  We encode faces explicitly below instead.
    pass

# Build faces explicitly.  In kociemba's layout:
#   U (y=+1), viewed from above with F=facing viewer: rows -> z from -1..1, cols -> x -1..1
#   D (y=-1), viewed from below: mirrored (rows z -1..1, cols x 1..-1)
#   R (x=+1), viewed from right: rows y 1..-1, cols z -1..1
#   L (x=-1), viewed from left: rows y 1..-1, cols z 1..-1
#   F (z=+1), viewed from front: rows y 1..-1, cols x -1..1
#   B (z=-1), viewed from back: rows y 1..-1, cols x 1..-1
def _face_faces():
    """Explicit face grids.  Axes: X=right, Y=up, Z=front.
    Each face: (letter, axis, sign, row_axis, col_axis, row_values, col_values).
    Continuity with shared edges fixes these uniquely."""
    return [
        # U (y=+1): rows sweep z back->front, cols sweep x left->right
        ("U", Y, +1, Z, X, [-1, 0, 1], [-1, 0, 1]),
        # R (x=+1): rows sweep y top->bottom, cols sweep z back->front
        ("R", X, +1, Y, Z, [1, 0, -1], [1, 0, -1]),
        # F (z=+1): rows sweep y top->bottom, cols sweep x left->right
        ("F", Z, +1, Y, X, [1, 0, -1], [-1, 0, 1]),
        # D (y=-1): rows sweep z front->back, cols sweep x left->right
        ("D", Y, -1, Z, X, [1, 0, -1], [-1, 0, 1]),
        # L (x=-1): rows sweep y top->bottom, cols sweep z back->front
        ("L", X, -1, Y, Z, [1, 0, -1], [-1, 0, 1]),
        # B (z=-1): rows sweep y top->bottom, cols sweep x left->right
        ("B", Z, -1, Y, X, [1, 0, -1], [1, 0, -1]),
    ]


# kociemba face order U,R,F,D,L,B
_FACE_ORDER = ["U", "R", "F", "D", "L", "B"]
_POSITIONS, _NORMALS = [], []
for face, axis, sign, rax, cax, rows, cols in _face_faces():
    normal = [0, 0, 0]
    normal[axis] = sign
    for r in rows:
        for c in cols:
            p = [0, 0, 0]
            p[axis] = sign
            p[rax] = r
            p[cax] = c
            _POSITIONS.append(tuple(p))
            _NORMALS.append(tuple(normal))

assert len(_POSITIONS) == 54, len(_POSITIONS)

# Map (position, normal) -> index, for finding target cells after a rotation.
_CELL_INDEX = {}
for i, (p, n) in enumerate(zip(_POSITIONS, _NORMALS)):
    _CELL_INDEX[(p, n)] = i
assert len(_CELL_INDEX) == 54

# Rotation matrices (right-hand rule) about each axis, for +/-90 degrees.
def rot_x(theta):
    c, s = int(round(np.cos(theta))), int(round(np.sin(theta)))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def rot_y(theta):
    c, s = int(round(np.cos(theta))), int(round(np.sin(theta)))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rot_z(theta):
    c, s = int(round(np.cos(theta))), int(round(np.sin(theta)))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

_rot = {"x": rot_x, "y": rot_y, "z": rot_z}
# MOVE_DEFS[face] = (axis, sign_of_normal, clockwise_direction)
# Clockwise turn of a face is "rotate about its axis by -90 (for +axis faces) or
# +90 (for -axis faces)" under the right-hand rule.
_MOVE_SPEC = {
    "U": (Y, +1), "D": (Y, -1),
    "R": (X, +1), "L": (X, -1),
    "F": (Z, +1), "B": (Z, -1),
}

def _turn_perm(face):
    """Return perm[target_index] = source_index for a clockwise turn of `face`."""
    axis, nsign = _MOVE_SPEC[face]
    angle = -np.pi / 2 if nsign > 0 else +np.pi / 2   # clockwise viewed from outside
    R = _rot["xyz"[axis]](angle)
    perm = list(range(54))   # off-layer stickers stay in place
    for src, (p, n) in enumerate(zip(_POSITIONS, _NORMALS)):
        # does this sticker lie on the turned layer?
        if p[axis] != nsign:
            continue
        pp = np.array(p)
        nn = np.array(n)
        q = tuple((_rot["xyz"[axis]](angle) @ pp).astype(int))
        qn = tuple((_rot["xyz"[axis]](angle) @ nn).astype(int))
        tgt = _CELL_INDEX[(q, qn)]
        perm[tgt] = src
    return perm

# Build the 12 move permutation tables (QTM quarter turns).
# Move names: U U' D D' L L' R R' F F' B B'
def _invert_perm(perm):
    inv = [0] * 54
    for t, s in enumerate(perm):
        inv[s] = t
    return inv


MOVE_NAMES = []
for face in "UDLRFB":
    MOVE_NAMES.append(face)
    MOVE_NAMES.append(face + "'")
# For inverse we use the permutation of the opposite geometric turn.
_INVERSE_TABLE = {}
for face in "UDLRFB":
    _INVERSE_TABLE[face] = face + "'"
    _INVERSE_TABLE[face + "'"] = face
    _INVERSE_TABLE[face + "2"] = face + "2"   # half turn is its own inverse

_MOVE_PERM = {}
for name in MOVE_NAMES:
    face = name[0]
    base = _turn_perm(face)
    if name.endswith("'"):
        base = _invert_perm(base)
    _MOVE_PERM[name] = base


# index -> name for the 12 moves
MOVE_INDEX = {name: i for i, name in enumerate(MOVE_NAMES)}
NUM_MOVES = 12

# Tensor move tables (target_index -> source_index), shape (12, 54).
MOVE_PERM = np.array([_MOVE_PERM[name] for name in MOVE_NAMES], dtype=np.int64)


# ----------------------------------------------------------------------------
# State construction / conversions
# ----------------------------------------------------------------------------
def solved_stickers():
    """Return a length-54 int array with color 0..5 per kociemba face order."""
    s = np.zeros(54, dtype=np.int64)
    for i, face in enumerate(_FACE_ORDER):
        s[i * 9:(i + 1) * 9] = COLOR_IDX[face]
    return s


def solved_state():
    """Return the solved state as a (1, 54) int64 tensor."""
    return torch.tensor(solved_stickers(), dtype=torch.int64).unsqueeze(0)


def is_solved(stickers):
    """stickers: int array of shape (..., 54). Returns bool tensor."""
    if isinstance(stickers, np.ndarray):
        stickers = torch.from_numpy(stickers)
    sol = torch.tensor(solved_stickers(), dtype=stickers.dtype, device=stickers.device)
    return torch.all(stickers == sol, dim=-1)


def apply_move(stickers, move):
    """Apply a single move to a state tensor (..., 54) -> new (..., 54).

    Supports quarter turns ("R", "R'") and half turns ("R2", kociemba/HTM).
    """
    if isinstance(move, int):
        perm = torch.tensor(MOVE_PERM[move], device=stickers.device)
        return stickers[..., perm]
    if isinstance(move, str):
        base = move[0]
        if len(move) > 1 and move.endswith("2") and not move.endswith("'2"):
            # half turn: apply the base quarter turn twice
            return apply_move(apply_move(stickers, base), base)
        if base in _MOVE_PERM:
            perm = torch.tensor(_MOVE_PERM[move], device=stickers.device)
            return stickers[..., perm]
    raise ValueError("unknown move: %r" % (move,))


def apply_moves(stickers, moves):
    """Apply a sequence of move names to (...,54) state tensor."""
    out = stickers
    for m in moves:
        out = apply_move(out, m)
    return out


def apply_all_moves(stickers):
    """Expand each state by all 12 moves. Input (..., 54) -> (..., 12, 54)."""
    perm = torch.tensor(MOVE_PERM, device=stickers.device)       # (12,54)
    return stickers[..., perm]                                    # broadcast over leading dims


def inverse_moves(moves):
    """Moves that undo a sequence (reverse order + invert each)."""
    return [_INVERSE_TABLE[m] for m in reversed(moves)]


def state_to_input(stickers):
    """Convert int sticker array (..., 54) to one-hot (..., 54, 6) then flatten
    to (..., 324) float. Matching the DeepCubeA / EfficientCube convention."""
    if isinstance(stickers, np.ndarray):
        stickers = torch.from_numpy(stickers)
    onehot = torch.nn.functional.one_hot(stickers.long(), num_classes=6)  # (...,54,6)
    return onehot.reshape(*stickers.shape[:-1], STATE_DIM).float()


def parse_scramble(s):
    """Parse a move string like \"R U R' U'\" into a list of move names."""
    return s.split()


def random_scramble_str(num_moves, rng=None):
    """Return a standard-notation scramble of `num_moves` QTM moves (no immediate
    self-cancel / same-face repeats)."""
    faces = list("UDLRFB")
    rng = rng or np.random.default_rng()
    moves = []
    last_face = None
    for _ in range(num_moves):
        allowed = [f for f in faces if f != last_face]
        f = allowed[rng.integers(len(allowed))]
        moves.append(f + "'" if rng.integers(2) else f)
        last_face = f
    return " ".join(moves)


def to_kociemba(stickers):
    """Return a 54-char string in kociemba face order, for verification."""
    if isinstance(stickers, torch.Tensor):
        stickers = stickers.cpu().numpy().reshape(-1, 54)
    if stickers.ndim == 2:
        stickers = stickers[0]
    return "".join(COLORS[int(c)] for c in stickers)


def from_kociemba(s):
    """Build an int stick array (1,54) from a kociemba facelet string."""
    return torch.tensor([[COLOR_IDX[c] for c in s]], dtype=torch.int64)


