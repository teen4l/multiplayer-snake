import queue
import random
import time
import threading
from dataclasses import dataclass
from enum import StrEnum, auto, Enum
from itertools import cycle, count
from typing import Iterable

import av
import cv2
import numpy as np
import streamlit as st
from numpy._typing import NDArray  # noqa
from streamlit_webrtc import create_video_source_track

from utils.color import theme_primary_rgb

BOARD_H = BOARD_W = 128


class Direction(StrEnum):
    UP = 'up'
    DOWN = 'down'
    LEFT = 'left'
    RIGHT = 'right'


DIRECTION_TO_MOVE: dict[Direction, tuple[int, int]] = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0,-1),
    Direction.RIGHT: (0, 1),
}
MOVE_TO_DIRECTION = {v: k for k, v in DIRECTION_TO_MOVE.items()}


def is_opposite(d1: Direction, d2: Direction) -> bool:
    m1 = DIRECTION_TO_MOVE[d1]; m2 = DIRECTION_TO_MOVE[d2]
    return (m1[0] + m2[0], m1[1] + m2[1]) == (0, 0)


class ColorSequence:
    def __init__(self, colors: Iterable[tuple[int, int, int]]):
        # ensure we always have at least one color
        _list = [(int(r)&255, int(g)&255, int(b)&255) for (r,g,b) in colors]
        if not _list:
            _list = [(255, 255, 255)]
        self._colors = tuple(_list)
        self._iter = cycle(self._colors)

    def __iter__(self):
        # independent round-robin iterator
        return cycle(self._colors)

    def next_color(self) -> tuple[int, int, int]:
        return next(self._iter)


class SnakeSegment:
    def __init__(self, color: tuple[int, int, int], position: tuple[int, int]):
        self.color = (int(color[0])&255, int(color[1])&255, int(color[2])&255)
        self.position = (int(position[0]), int(position[1]))


NEW_SNAKE_SEGMENTS = 5
FOOD_SPAWN_PROB_ON_DEATH = 0.5


class Snake:
    """
    Array-first snake:
      - pos: (L,2) int32 of (row, col), head at index 0
      - rgb: (L,3) uint8 per-segment color
      - snake_direction: Direction
      - _pending_grow: number of segments to add on next moves

    Notes:
      - move(...) uses slice ops and wraps with modulo.
      - Optionally updates OCC_SNAKE (head set; tail cleared only if no growth).
      - die(...) returns flats of cells that should spawn food (no Food objects).
    """
    __slots__ = ("color_sequence", "pos", "rgb", "snake_direction", "_pending_grow")

    def __init__(self, color_sequence: ColorSequence, initial_state: Iterable[SnakeSegment]):
        self.color_sequence = color_sequence

        # Build arrays once from initial_state (head at index 0)
        pos_list: list[tuple[int, int]] = []
        rgb_list: list[tuple[int, int, int]] = []
        for seg in initial_state:
            pos_list.append((int(seg.position[0]), int(seg.position[1])))
            rgb_list.append((int(seg.color[0]) & 255, int(seg.color[1]) & 255, int(seg.color[2]) & 255))

        if not pos_list:
            raise ValueError("Snake must have at least one segment")

        self.pos = np.asarray(pos_list, dtype=np.int32)        # (L,2)
        self.rgb = np.asarray(rgb_list, dtype=np.uint8)        # (L,3)

        self.snake_direction = self._infer_direction()
        self._pending_grow = 0

    # -------- Views compatible with existing rendering --------
    def positions(self) -> np.ndarray:
        """(L,2) int32 view."""
        return self.pos

    def colors(self) -> np.ndarray:
        """(L,3) uint8 view."""
        return self.rgb

    # -------- Helpers --------
    @staticmethod
    def _delta_from_direction(direction: Direction) -> tuple[int, int]:
        dr, dc = DIRECTION_TO_MOVE[direction]
        return int(dr), int(dc)

    def _infer_direction(self) -> Direction:
        if self.pos.shape[0] >= 2:
            head = self.pos[0]
            neck = self.pos[1]
            delta = (int(head[0] - neck[0]), int(head[1] - neck[1]))
            return MOVE_TO_DIRECTION.get(delta, Direction.UP)
        return Direction.UP

    def head_flat(self) -> int:
        return int(self.pos[0, 0] * BOARD_W + self.pos[0, 1])

    def tail_flat(self) -> int:
        return int(self.pos[-1, 0] * BOARD_W + self.pos[-1, 1])

    # -------- Core step --------
    def move(self, to: Direction | None = None) -> tuple[int | None, int]:
        """
        Advance one step. Updates OCC_SNAKE (sets new head; clears old tail if no growth).
        Returns (vacated_tail_flat_or_None, new_head_flat).
        """
        global OCC  # uses unified occupancy if requested
        if to is None or is_opposite(self.snake_direction, to):
            to = self.snake_direction
        dr, dc = self._delta_from_direction(to)

        # Remember pre-move tail cell for possible OCC clear
        old_tail_flat = self.tail_flat()

        # Shift body and move head
        # Save old head to compute new head quickly
        # (pos[1:] <- pos[:-1]) then update head
        self.pos[1:] = self.pos[:-1]
        self.pos[0, 0] = (self.pos[0, 0] + dr) % BOARD_H
        self.pos[0, 1] = (self.pos[0, 1] + dc) % BOARD_W
        new_head_flat = self.head_flat()

        # Apply growth: if pending, re-append old tail cell and add a color
        grew = False
        if self._pending_grow > 0:
            # Append the previous tail position to keep length+1
            # Reconstruct (r,c) from old_tail_flat to avoid saving a copy
            r_tail, c_tail = divmod(old_tail_flat, BOARD_W)
            self.pos = np.vstack([self.pos, np.array([[r_tail, c_tail]], dtype=np.int32)])
            new_col = np.asarray(self.color_sequence.next_color(), dtype=np.uint8)[None, :]
            self.rgb = np.vstack([self.rgb, new_col])
            self._pending_grow -= 1
            grew = True

        # Update direction
        self.snake_direction = to

        # Maintain occupancy
        # Set head bit
        OCC[new_head_flat] |= OCC_SNAKE
        # Clear tail only if we actually vacated it (i.e., no growth this tick)
        if not grew:
            OCC[old_tail_flat] &= _OCC_CLR_SNAKE

        return (None if grew else old_tail_flat), new_head_flat

    def eat(self) -> None:
        """Schedule growth by one segment on the next moves (idempotent per call)."""
        self._pending_grow += 1

    def die_spawn_food_flats(self) -> np.ndarray:
        """
        Clear all segments and return an array of flat indices where food should spawn.
        """
        if self.pos.size == 0:
            return np.empty(0, dtype=np.int32)

        flats = self.pos[:, 0] * BOARD_W + self.pos[:, 1]
        # Vectorized Bernoulli
        # If you want reproducibility, pass in an np.random.Generator and use .random()
        p = FOOD_SPAWN_PROB_ON_DEATH
        mask = (np.random.random(flats.shape[0]) < p)
        spawn = flats[mask].astype(np.int32, copy=False)

        # Clear internal state
        self.pos = np.empty((0, 2), dtype=np.int32)
        self.rgb = np.empty((0, 3), dtype=np.uint8)
        self._pending_grow = 0
        return spawn

    @classmethod
    def new(cls, colors: Iterable[tuple[int, int, int]]):
        """
        Build a new snake of length NEW_SNAKE_SEGMENTS at a random location and direction.
        (No occupancy check here; caller should validate against OCC.)
        """
        head_h = random.randint(0, BOARD_H - 1)
        head_w = random.randint(0, BOARD_W - 1)
        body_direction = random.choice(list(Direction))
        d_h, d_w = DIRECTION_TO_MOVE[body_direction]
        snake_colors = ColorSequence(colors)

        # Generate segments (head first)
        pos = []
        rgb = []
        for i in range(NEW_SNAKE_SEGMENTS):
            r = (head_h + i * d_h) % BOARD_H
            c = (head_w + i * d_w) % BOARD_W
            pos.append((r, c))
            rgb.append(snake_colors.next_color())

        # Create through the standard initializer to reuse logic
        segments = [SnakeSegment(rgb[i], pos[i]) for i in range(NEW_SNAKE_SEGMENTS)]
        return cls(snake_colors, segments)


def _clip_mask(r: NDArray, c: NDArray, H: int, W: int) -> NDArray[np.bool_]:
    return (r >= 0) & (r < H) & (c >= 0) & (c < W)


def render_snakes_into(snakes: Iterable[Snake], frame: NDArray[np.uint8]) -> None:
    H = W = BOARD_H
    for s in snakes:
        pos = s.positions()
        if pos.ndim != 2 or pos.shape[1] != 2 or pos.size == 0:
            continue

        r_all = pos[:, 0]
        c_all = pos[:, 1]
        m = _clip_mask(r_all, c_all, H, W)
        if not np.any(m):
            continue

        r = r_all[m]; c = c_all[m]
        cols = s.colors()  # (L,3) uint8
        frame[r, c, :3] = cols[m, :3]


def render_food_arrays_into(food_flats: np.ndarray, food_rgbs: np.ndarray, frame: NDArray[np.uint8]) -> None:
    """
    Draw food RGB into an existing RGBA frame from the array state.
    Assumes caller cleared `frame`. Does NOT set alpha.

    Args:
        food_flats: (N,) int32 of flat indices r*BOARD_W + c
        food_rgbs: (N,3) uint8 per-food RGB
        frame: (H,W,4) uint8 RGBA buffer
    """
    if food_flats.size == 0:
        return

    # Guard: ensure shapes match
    if food_rgbs.ndim != 2 or food_rgbs.shape[1] != 3 or food_rgbs.shape[0] != food_flats.shape[0]:
        raise ValueError(f"render_food_arrays_into: shape mismatch "
                         f"{food_flats.shape=} {food_rgbs.shape=}")

    # Convert flats → (r,c)
    r = (food_flats // BOARD_W).astype(np.int32, copy=False)
    c = (food_flats %  BOARD_W).astype(np.int32, copy=False)

    # Clip mask (defensive; OCC should already guarantee validity)
    m = (r >= 0) & (r < BOARD_H) & (c >= 0) & (c < BOARD_W)
    if not np.any(m):
        return

    r = r[m]; c = c[m]
    frame[r, c, :3] = food_rgbs[m, :3]


def finalize_alpha(frame: NDArray[np.uint8]) -> None:
    """
    Set A=255 where any RGB channel is non-zero, else 0. In-place.
    """
    alpha_mask = frame[..., :3].any(axis=2)
    frame[..., 3] = np.where(alpha_mask, 255, 0).astype(np.uint8)


TICK_RATE = 12  # Hz

# Published frame/timestamp; the thread will overwrite these.
LAST_RENDER: float = time.time()  # wall clock
LAST_FRAME: np.ndarray = np.zeros((BOARD_H, BOARD_W, 4), dtype=np.uint8)
FRAME_LOCK = threading.RLock()
OBSERVED_FRAME_RATE: float = 0.0  # EMA of FPS

# ---- Board geometry ----
N_CELLS: int = BOARD_H * BOARD_W

# ---- Unified occupancy grid (bitmask) ----
# 0 = empty, 1 = snake, 2 = food (you can OR them if needed)
OCC_EMPTY: int = 0
OCC_SNAKE: int = 1 << 0
OCC_FOOD:  int = 1 << 1
OCC: np.ndarray = np.zeros(N_CELLS, dtype=np.uint8)  # flat-indexed occupancy

_OCC_CLR_SNAKE = np.uint8((~OCC_SNAKE) & 0xFF)
_OCC_CLR_FOOD  = np.uint8((~OCC_FOOD)  & 0xFF)

# ---- Array-backed food state ----
# Flats are r*BOARD_W + c
FOOD_FLAT:  np.ndarray = np.empty(0, dtype=np.int32)      # (N,)
FOOD_RGB:   np.ndarray = np.empty((0, 3), dtype=np.uint8) # (N,3)
FOOD_BIRTH: np.ndarray = np.empty(0, dtype=np.float64)    # (N,)

# ---- Global snake state (objects for now; will refactor later) ----
STATE_LOCK = threading.RLock()
ACTIVE_SNAKES: dict[str, Snake] = {}
SNAKE_MOVES: dict[str, Direction] = {}

def state_lock(func):
    def wrapper(*args, **kwargs):
        with STATE_LOCK:
            return func(*args, **kwargs)
    return wrapper

FOOD_LIFESPAN_FRAMES = 128
FOOD_PER_FRAME = 1

_LAST_ID = 0


def _generate_snake_id() -> str:
    global _LAST_ID
    id_ = f'snake-{_LAST_ID}'
    _LAST_ID += 1
    return id_



# ---------- Food management (array + OCC) ----------
def add_food(*, new_n: int = 0, now: float | None = None) -> None:
    """
    Spawn `new_n` food items into free cells (neither snake nor food).
    Updates FOOD_FLAT/FOOD_RGB/FOOD_BIRTH and OCC (sets OCC_FOOD bit).
    Assumes OCC_SNAKE is already maintained elsewhere.
    """
    global FOOD_FLAT, FOOD_RGB, FOOD_BIRTH, OCC

    if new_n <= 0:
        return
    if now is None:
        now = time.time()

    # Free cells = fully empty (no snake, no food)
    free_flats = np.flatnonzero(OCC == OCC_EMPTY)
    if free_flats.size == 0:
        return

    n = int(min(new_n, free_flats.size))
    sel = np.random.choice(free_flats.size, size=n, replace=False)
    flats = free_flats[sel].astype(np.int32, copy=False)

    rgb = np.array(theme_primary_rgb(), dtype=np.uint8)
    rgbs = np.repeat(rgb[None, :], n, axis=0)

    # Append to arrays
    FOOD_FLAT  = np.concatenate([FOOD_FLAT, flats])
    FOOD_RGB   = np.concatenate([FOOD_RGB,  rgbs])
    FOOD_BIRTH = np.concatenate([FOOD_BIRTH, np.full(n, now, dtype=np.float64)])

    # Update occupancy
    OCC[flats] |= OCC_FOOD


def remove_expired_food(now: float | None = None) -> None:
    """
    Remove foods older than FOOD_LIFESPAN_FRAMES (mapped to seconds via TICK_RATE).
    Compacts FOOD_* arrays and clears OCC_FOOD for removed cells.
    """
    global FOOD_FLAT, FOOD_RGB, FOOD_BIRTH, OCC

    if FOOD_FLAT.size == 0:
        return
    if now is None:
        now = time.time()

    max_age_sec = FOOD_LIFESPAN_FRAMES / float(TICK_RATE)
    keep = (now - FOOD_BIRTH) <= max_age_sec
    if np.all(keep):
        return

    to_clear = FOOD_FLAT[~keep]
    OCC[to_clear] &= _OCC_CLR_FOOD  # clear the food bit

    FOOD_FLAT  = FOOD_FLAT[keep]
    FOOD_RGB   = FOOD_RGB[keep]
    FOOD_BIRTH = FOOD_BIRTH[keep]


def resolve_collisions(now: float | None = None) -> None:
    """
    Resolve collisions on the board.

    Rules:
      - Snake vs snake: at any cell with >=2 snake segments, the entry with the
        LARGEST segment index wins. If there is a tie on the largest segment
        index, ALL snakes in that cell are removed.
      - Snake vs food: every snake present on a cell containing food 'eats'
        (grow next tick), and that food is removed.

    Assumes:
      - Called under STATE_LOCK.
      - OCC is up-to-date (OCC_SNAKE and OCC_FOOD bits).
      - ACTIVE_SNAKES contain current snakes.
      - FOOD_FLAT/FOOD_RGB/FOOD_BIRTH mirror OCC_FOOD (<=1 food per cell).

      Requires (to be implemented):
      - _remove_snake_and_spawn_food(snake_id: str, now: float) -> None
        (array/OCC-based removal + spawning, no Food objects)
    """
    global FOOD_FLAT, FOOD_RGB, FOOD_BIRTH, OCC

    if now is None:
        now = time.time()
    if not ACTIVE_SNAKES:
        return

    # --------- Aggregate all snake segments into flat arrays ---------
    snake_ids = list(ACTIVE_SNAKES.keys())
    id2idx = {sid: i for i, sid in enumerate(snake_ids)}

    flats_lst = []
    segs_lst = []
    sidx_lst = []

    for sid, s in ACTIVE_SNAKES.items():
        pos = np.asarray(s.positions(), dtype=np.int32)  # (L, 2)
        if pos.ndim != 2 or pos.shape[0] == 0:
            continue
        L = pos.shape[0]
        flats = pos[:, 0] * BOARD_W + pos[:, 1]
        flats_lst.append(flats)
        segs_lst.append(np.arange(L, dtype=np.int32))                # 0..L-1
        sidx_lst.append(np.full(L, id2idx[sid], dtype=np.int32))     # dense snake idx

    if not flats_lst:
        return

    all_flat = np.concatenate(flats_lst)
    all_seg  = np.concatenate(segs_lst)
    all_sid  = np.concatenate(sidx_lst)

    # Sort by flat cell to create contiguous runs per cell
    order = np.argsort(all_flat, kind="stable")
    f_sorted   = all_flat[order]
    seg_sorted = all_seg[order]
    sid_sorted = all_sid[order]

    uniq, start_idx, counts = np.unique(f_sorted, return_index=True, return_counts=True)

    # --------- Snake–snake collisions (largest seg index wins) ---------
    to_remove: set[str] = set()

    multi = counts > 1
    for s0, cnt in zip(start_idx[multi], counts[multi]):
        seg_run = seg_sorted[s0:s0 + cnt]
        sid_run = sid_sorted[s0:s0 + cnt]

        max_seg = seg_run.max()
        winners_mask: NDArray = (seg_run == max_seg)  # noqa
        winners_count = int(winners_mask.sum())

        if winners_count >= 2:
            # tie on the largest seg id -> remove all snakes in this cell
            for k in range(cnt):
                to_remove.add(snake_ids[sid_run[k]])
        else:
            # unique winner: remove everyone else
            for k in range(cnt):
                if seg_run[k] != max_seg:
                    to_remove.add(snake_ids[sid_run[k]])

    # --------- Snake–food: all snakes on a food cell eat; remove that food ---------
    # We iterate all occupied cells (including non-collided) and check the OCC_FOOD bit.
    food_cells_to_clear: list[int] = []

    for cell, s0, cnt in zip(uniq, start_idx, counts):
        if (OCC[cell] & OCC_FOOD) == 0:
            continue  # no food at this cell

        # All snakes present on this cell eat (even if they will be removed later this tick).
        for k in range(s0, s0 + cnt):
            sid = snake_ids[sid_sorted[k]]
            if sid in ACTIVE_SNAKES:
                ACTIVE_SNAKES[sid].eat()

        # Mark food for deletion from arrays/OCC
        food_cells_to_clear.append(int(cell))

    if food_cells_to_clear:
        food_cells_to_clear: NDArray = np.array(food_cells_to_clear, dtype=np.int32)
        # Clear OCC_FOOD for these cells
        OCC[food_cells_to_clear] &= _OCC_CLR_FOOD
        if FOOD_FLAT.size:
            # Remove matching rows from FOOD_* arrays
            mask_keep = ~np.isin(FOOD_FLAT, food_cells_to_clear)
            if not np.all(mask_keep):  # at least one removal
                FOOD_FLAT  = FOOD_FLAT[mask_keep]
                FOOD_RGB   = FOOD_RGB[mask_keep]
                FOOD_BIRTH = FOOD_BIRTH[mask_keep]

    # --------- Apply snake removals in batch ---------
    if to_remove:
        for sid in to_remove:
            #  array/OCC-based removal that also spawns food from the dead snake's segments
            _remove_snake_and_spawn_food(sid, now)


def _remove_snake_and_spawn_food(snake_id: str, now: float | None = None) -> None:
    """
    Remove a snake from the board, clear its OCC_SNAKE bits, and spawn food on a
    subset of its former cells (per-segment Bernoulli). Spawns only into EMPTY cells
    after this snake is cleared. Updates FOOD_* arrays and OCC_FOOD.

    Assumes caller holds STATE_LOCK.
    """
    global ACTIVE_SNAKES, SNAKE_MOVES
    global OCC, FOOD_FLAT, FOOD_RGB, FOOD_BIRTH

    if now is None:
        now = time.time()

    s = ACTIVE_SNAKES.get(snake_id)
    if s is None:
        # Already gone; ensure move state is cleaned up
        SNAKE_MOVES.pop(snake_id, None)
        return

    # Compute all current segment flats BEFORE clearing snake internals
    if s.pos.size:
        seg_flats = (s.pos[:, 0] * BOARD_W + s.pos[:, 1]).astype(np.int32, copy=False)
    else:
        seg_flats = np.empty(0, dtype=np.int32)

    # Clear this snake's occupancy
    if seg_flats.size:
        OCC[seg_flats] &= _OCC_CLR_SNAKE

    # Decide where to spawn food (vectorized inside; returns flats). This clears s.pos/rgb.
    spawn_flats = s.die_spawn_food_flats()

    # Remove snake from registries
    ACTIVE_SNAKES.pop(snake_id, None)
    SNAKE_MOVES.pop(snake_id, None)

    # Filter spawn targets to EMPTY cells (no snake, no existing food); deduplicate
    if spawn_flats.size:
        spawn_flats = np.unique(spawn_flats)
        empty_mask = (OCC[spawn_flats] == OCC_EMPTY)
        spawn_flats = spawn_flats[empty_mask]

    if spawn_flats.size:
        rgb = np.array(theme_primary_rgb(), dtype=np.uint8)
        rgbs = np.repeat(rgb[None, :], spawn_flats.size, axis=0)

        # Append to food arrays
        FOOD_FLAT = np.concatenate([FOOD_FLAT, spawn_flats])
        FOOD_RGB = np.concatenate([FOOD_RGB, rgbs])
        FOOD_BIRTH = np.concatenate([FOOD_BIRTH, np.full(spawn_flats.size, now, dtype=np.float64)])

        # Set occupancy bits
        OCC[spawn_flats] |= OCC_FOOD


# Snake management ----------
def new_snake(colors: Iterable[tuple[int, int, int]]) -> Snake | None:
    """
    Create a new snake of length NEW_SNAKE_SEGMENTS placed entirely on EMPTY cells.
    Uses the unified occupancy grid (OCC). Does NOT mutate OCC; caller should
    set OCC_SNAKE bits when the snake is actually added to the game state.
    """
    snake_colors = ColorSequence(colors)

    # Candidate head cells = fully empty cells
    free_flats = np.flatnonzero(OCC == OCC_EMPTY)
    if free_flats.size == 0:
        return None

    # Try heads in random order
    head_candidates = free_flats.copy()
    np.random.shuffle(head_candidates)

    idx = np.arange(NEW_SNAKE_SEGMENTS, dtype=np.int32)

    for head_flat in head_candidates:
        hr, hc = divmod(int(head_flat), BOARD_W)

        # Try directions in random order per head
        directions: list[Direction] = list(Direction)  # noqa
        random.shuffle(directions)

        for d in directions:
            dr, dc = DIRECTION_TO_MOVE[d]
            rr = (hr + idx * int(dr)) % BOARD_H
            cc = (hc + idx * int(dc)) % BOARD_W
            flats = rr * BOARD_W + cc

            # All cells for this snake must be empty
            if not np.all(OCC[flats] == OCC_EMPTY):
                continue

            # Build initial segments (head at index 0)
            segments = [
                SnakeSegment(snake_colors.next_color(), (int(r), int(c)))
                for r, c in zip(rr.tolist(), cc.tolist())
            ]
            return Snake(snake_colors, segments)

    # No valid placement found
    return None

def move_snakes():
    for snake_id, snake in ACTIVE_SNAKES.items():
        snake.move(SNAKE_MOVES.get(snake_id))


# ---------------- Job plumbing ----------------
class JobType(Enum):
    SUBMIT_SNAKE = auto()
    REMOVE_SNAKE = auto()
    SUBMIT_MOVE = auto()

@dataclass
class Job:
    kind: JobType
    snake_id: str | None = None
    snake: Snake | None = None
    direction: Direction | None = None

# Priority: lower is served first
_JOB_PRIORITY = {
    JobType.SUBMIT_MOVE: 0,
    JobType.REMOVE_SNAKE: 1,
    JobType.SUBMIT_SNAKE: 2,
}
_JOB_COUNTER = count()

# items are tuples: (priority:int, seq:int, job:Job)
_JOBS: queue.PriorityQueue[tuple[int,int,Job]] = queue.PriorityQueue(maxsize=10000)
_WORKER: threading.Thread | None = None
_WORKER_STARTED = threading.Event()


def _ensure_worker():
    global _WORKER
    if _WORKER and _WORKER.is_alive():
        return

    def _worker():
        while True:
            prio, _, job = _JOBS.get()
            try:
                if job.kind is JobType.SUBMIT_SNAKE:
                    _apply_submit_snake(job.snake_id, job.snake)
                elif job.kind is JobType.REMOVE_SNAKE:
                    _apply_remove_snake(job.snake_id)
                elif job.kind is JobType.SUBMIT_MOVE:
                    _apply_submit_move(job.snake_id, job.direction)
            except Exception as e:
                print(f'Job failed: {job}, {e}')
            finally:
                _JOBS.task_done()

    _WORKER = threading.Thread(target=_worker, name="state-worker", daemon=True)
    _WORKER.start()
    _WORKER_STARTED.set()

def _put_job_nowait(job: Job) -> None:
    prio = _JOB_PRIORITY[job.kind]
    _JOBS.put_nowait((prio, next(_JOB_COUNTER), job))

def _put_job_blocking(job: Job) -> None:
    prio = _JOB_PRIORITY[job.kind]
    _JOBS.put((prio, next(_JOB_COUNTER), job))


# ---------------- State mutations (locked) ----------------
def _apply_submit_snake_nolock(snake_id: str, snake: "Snake") -> None:
    """
    Place a new snake if all its cells are currently empty.
    Sets OCC_SNAKE bits and registers the snake & its initial direction.
    Assumes STATE_LOCK is held.
    """
    # Validate placement against OCC
    if snake.pos.size == 0:
        raise ValueError("submit_snake: empty snake")

    flats = (snake.pos[:, 0] * BOARD_W + snake.pos[:, 1]).astype(np.int32, copy=False)
    if not np.all(OCC[flats] == OCC_EMPTY):
        raise ValueError("submit_snake: placement conflicts with occupied cells")

    # Mark occupancy
    OCC[flats] |= OCC_SNAKE

    # Register
    ACTIVE_SNAKES[snake_id] = snake
    SNAKE_MOVES[snake_id] = snake.snake_direction


@state_lock
def _apply_submit_snake(snake_id: str, snake: "Snake") -> None:
    _apply_submit_snake_nolock(snake_id, snake)


def _apply_remove_snake_nolock(snake_id: str, *, now: float | None = None) -> None:
    """
    Remove a snake and spawn food on some of its cells (array/OCC path).
    Assumes STATE_LOCK is held.
    """
    if now is None:
        now = time.time()
    if snake_id in ACTIVE_SNAKES:
        _remove_snake_and_spawn_food(snake_id, now)
    SNAKE_MOVES.pop(snake_id, None)


@state_lock
def _apply_remove_snake(snake_id: str) -> None:
    _apply_remove_snake_nolock(snake_id, now=time.time())


def _apply_submit_move_nolock(snake_id: str, direction: Direction) -> None:
    """
    Update desired direction for a snake. No validation here;
    the Snake.move() guards against 180° reversals.
    Assumes STATE_LOCK is held.
    """
    if snake_id in ACTIVE_SNAKES:
        SNAKE_MOVES[snake_id] = direction


@state_lock
def _apply_submit_move(snake_id: str, direction: Direction) -> None:
    _apply_submit_move_nolock(snake_id, direction)


# ---------------- Public async API (non-blocking) ----------------
def submit_snake(snake: "Snake") -> str:
    """Return ID immediately; worker will add the snake shortly."""
    _ensure_worker()
    snake_id = _generate_snake_id()
    _put_job_nowait(Job(kind=JobType.SUBMIT_SNAKE, snake_id=snake_id, snake=snake))
    return snake_id

def remove_snake(snake_id: str) -> None:
    _ensure_worker()
    _put_job_nowait(Job(kind=JobType.REMOVE_SNAKE, snake_id=snake_id))

def submit_move(snake_id: str, direction: Direction) -> None:
    """
    Low-latency move submission.
    Fast path: try a non-blocking lock and write immediately.
    Fallback: enqueue with highest priority if the lock is briefly busy.
    """
    got = STATE_LOCK.acquire(blocking=False)
    if got:
        try:
            _apply_submit_move_nolock(snake_id, direction)
            return
        finally:
            STATE_LOCK.release()

    _ensure_worker()
    try:
        _put_job_nowait(Job(kind=JobType.SUBMIT_MOVE, snake_id=snake_id, direction=direction))
    except queue.Full:
        _put_job_blocking(Job(kind=JobType.SUBMIT_MOVE, snake_id=snake_id, direction=direction))

# main rendering thread ---------

@st.cache_resource
def run_rendering_thread():
    """
    Starts a single background renderer (per Streamlit server).
    Recomputes LAST_FRAME from ACTIVE_SNAKES and FOOD_* arrays at TICK_RATE FPS.
    Returns (thread, stop_event).
    """
    stop_event = threading.Event()

    # Preallocate double buffers (no per-frame np.zeros)
    H = W = BOARD_H
    frame_a = np.zeros((H, W, 4), dtype=np.uint8)
    frame_b = np.zeros_like(frame_a)
    use_a = True  # published buffer; we render into the other one

    # Publish an initial frame
    global LAST_FRAME, LAST_RENDER, OBSERVED_FRAME_RATE
    LAST_FRAME = frame_a
    LAST_RENDER = time.time()
    OBSERVED_FRAME_RATE = 0.0

    def _loop():
        nonlocal use_a, frame_a, frame_b
        global LAST_FRAME, LAST_RENDER, OBSERVED_FRAME_RATE

        dt = 1.0 / float(TICK_RATE)
        next_tick = time.perf_counter()  # drift-free scheduler

        while not stop_event.is_set():
            t0 = time.perf_counter()
            now_wall = time.time()

            # ---- Update game state & snapshot references under one lock ----
            with STATE_LOCK:
                remove_expired_food(now=now_wall)
                add_food(new_n=FOOD_PER_FRAME, now=now_wall)

                # move_snakes() must call snake.move(update_occ=True) internally
                move_snakes()

                # array-first, "largest segment id wins"
                resolve_collisions(now=now_wall)

                # Snapshots for rendering (copies so we can release the lock)
                snakes_snapshot = tuple(ACTIVE_SNAKES.values())  # shallow; arrays are read-only here
                food_flats = FOOD_FLAT.copy()
                food_rgbs  = FOOD_RGB.copy()

            # ---- Choose back buffer, clear, render, and finalize alpha ----
            write_frame = frame_b if use_a else frame_a
            write_frame.fill(0)

            render_snakes_into(snakes_snapshot, write_frame)
            render_food_arrays_into(food_flats, food_rgbs, write_frame)
            finalize_alpha(write_frame)

            # ---- Publish by snapshot (never mutated after publish) ----
            pub = write_frame.copy()
            with FRAME_LOCK:
                LAST_FRAME = pub
                LAST_RENDER = now_wall
            use_a = not use_a

            # ---- Pace to target tick rate (drift-free) ----
            next_tick += dt
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # If we're behind, resync to now to avoid spiral of death
                next_tick = time.perf_counter()

            # ---- Update observed FPS (EMA) ----
            loop_dt = time.perf_counter() - t0
            if loop_dt > 0.0:
                curr_fps = 1.0 / loop_dt
                alpha = 0.9 if OBSERVED_FRAME_RATE != 0.0 else 0.0
                OBSERVED_FRAME_RATE = alpha * OBSERVED_FRAME_RATE + (1.0 - alpha) * curr_fps

    thread = threading.Thread(target=_loop, name='snake-render', daemon=True)
    thread.start()
    return thread, stop_event


# Cached video frame to avoid rebuilding when LAST_RENDER hasn't advanced
CACHE_LOCK = threading.RLock()
_CACHED_RENDER_TS: float = -1.0
_CACHED_SCALE: int = -1
_CACHED_AVFRAME: av.VideoFrame | None = None


def _video_source_callback(*args) -> av.VideoFrame:
    """
    Build a frame from the current LAST_FRAME in state, with caching.
    If the renderer hasn't published a newer frame (LAST_RENDER unchanged),
    return the previously built av.VideoFrame to skip all conversions.
    The cache is also invalidated when the pixel scale changes.
    """
    global _CACHED_RENDER_TS, _CACHED_SCALE, _CACHED_AVFRAME

    scale = int(st.session_state.get("frame_scale", 6))

    # Take a consistent snapshot of the published frame and its timestamp
    with FRAME_LOCK:
        snap = LAST_FRAME
        render_ts = LAST_RENDER

    # Fast path: reuse cached frame if nothing new was rendered and scale is unchanged
    with CACHE_LOCK:
        if (
            _CACHED_AVFRAME is not None
            and render_ts <= _CACHED_RENDER_TS
            and scale == _CACHED_SCALE
        ):
            return _CACHED_AVFRAME

    # Otherwise (new render or scale change), rebuild the frame
    frame = snap
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[:, :, :3]  # drop alpha (RGB)

    # Convert RGB -> BGR for OpenCV/av 'bgr24'
    frame_bgr = frame[:, :, ::-1]

    # Pixel-perfect upscale
    if scale > 1:
        h, w = frame_bgr.shape[:2]
        frame_bgr = cv2.resize(
            frame_bgr,
            (w * scale, h * scale),
            interpolation=cv2.INTER_NEAREST,
        )

    avframe = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")

    # Update cache
    with CACHE_LOCK:
        _CACHED_RENDER_TS = render_ts
        _CACHED_SCALE = scale
        _CACHED_AVFRAME = avframe

    return avframe


_WEbrtc_FPS = int(2 * float(TICK_RATE))
VIDEO_SOURCE_TRACK = create_video_source_track(
    _video_source_callback, key="video_source_track", fps=_WEbrtc_FPS
)

