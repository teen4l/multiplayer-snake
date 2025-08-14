import queue
import random
import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import StrEnum, auto, Enum
from itertools import cycle
from typing import Iterable, Iterator

import numpy as np
import streamlit as st
from numpy._typing import NDArray  # noqa

from utils.color import theme_primary_rgb

BOARD_H = BOARD_W = 128
ALL_POSITIONS = {(h, w) for h in range(BOARD_H) for w in range(BOARD_W)}


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
    def __init__(self, color_sequence: ColorSequence, initial_state: Iterable[SnakeSegment]):
        self.color_sequence = color_sequence
        self.state = deque(initial_state)  # head at index 0
        if not self.state:
            raise ValueError("Snake must have at least one segment")
        self.snake_direction = self._infer_direction()
        self._pending_grow = 0  # number of segments to add after next moves

    def _infer_direction(self) -> Direction:
        if len(self.state) >= 2:
            head = self.state[0].position
            neck = self.state[1].position
            delta = (head[0] - neck[0], head[1] - neck[1])  # points from neck -> head
            return MOVE_TO_DIRECTION.get(delta, Direction.UP)
        return Direction.UP

    def move(self, to: Direction | None = None) -> None:
        # by default, move in the direction of the snake
        if to is None:
            to = self.snake_direction

        # prevent 180° turns; keep current direction instead
        direction = self.snake_direction if is_opposite(self.snake_direction, to) else to
        dr, dc = DIRECTION_TO_MOVE[direction]

        # snapshot old positions (for shifting + growth)
        old_positions = [seg.position for seg in self.state]
        old_tail_pos = old_positions[-1]

        # move head with wrap-around
        head_r, head_c = old_positions[0]
        new_head_pos = ((head_r + dr) % BOARD_H, (head_c + dc) % BOARD_W)
        self.state[0].position = new_head_pos

        # shift body
        for i in range(1, len(self.state)):
            self.state[i].position = old_positions[i - 1]

        # apply growth (new segment appears where tail used to be)
        if self._pending_grow > 0:
            self.state.append(SnakeSegment(self.color_sequence.next_color(), old_tail_pos))
            self._pending_grow -= 1

        self.snake_direction = direction

    def eat(self):
        # grow by one segment on the next move
        # IMPORTANT: eat before moving on one tick
        self._pending_grow += 1

    def die(self) -> list['Food']:
        foods = []
        for segment in self.state:
            if random.random() < FOOD_SPAWN_PROB_ON_DEATH:
                foods.append(Food(segment.position))
        self.state.clear()
        return foods

    def positions(self) -> list[tuple[int, int]]:
        return [segment.position for segment in self.state]

    def colors(self) -> list[tuple[int, int, int]]:
        return [segment.color for segment in self.state]

    @classmethod
    def new(cls, colors: Iterable[tuple[int, int, int]]):
        head_h, head_w = random.randint(0, BOARD_H - 1), random.randint(0, BOARD_W - 1)
        body_direction = random.choice(list(Direction))
        d_h, d_w = DIRECTION_TO_MOVE[body_direction]
        snake_colors = ColorSequence(colors)
        segments = [
            SnakeSegment(snake_colors.next_color(), (head_h + i * d_h, head_w + i * d_w))
            for i in range(NEW_SNAKE_SEGMENTS)
        ]
        return cls(snake_colors, segments)


class Food:
    def __init__(
        self,
        pos: tuple[int, int] | None = None,
        rgb: tuple[int, int, int] | None = None,
    ):
        self.created_at = time.time()
        if pos is None:
            self._pos = (random.randrange(0, BOARD_H), random.randrange(0, BOARD_W))
        else:
            self._pos = (int(pos[0]), int(pos[1]))
        self._rgb = tuple(int(x) for x in (rgb if rgb is not None else theme_primary_rgb()))
        assert len(self._rgb) == 3

    def position(self) -> tuple[int, int]:
        return self._pos  # (row, col)

    def color(self) -> tuple[int, int, int]:
        return self._rgb  # noqa uint8 RGB


def _clip_mask(r: NDArray, c: NDArray, H: int, W: int) -> NDArray[np.bool_]:
    return (r >= 0) & (r < H) & (c >= 0) & (c < W)


def render_snakes_into(snakes: Iterable["Snake"], frame: NDArray[np.uint8]) -> None:
    """
    Draw snakes' RGB into an existing RGBA frame.
    Assumes `frame` was cleared by caller. Does NOT set alpha.
    """
    H = W = BOARD_H
    for s in snakes:
        pos = np.asarray(s.positions(), dtype=np.int32)
        if pos.ndim != 2 or pos.shape[1] != 2 or pos.size == 0:
            continue

        r_all = pos[:, 0]
        c_all = pos[:, 1]
        m = _clip_mask(r_all, c_all, H, W)
        if not np.any(m):
            continue

        r = r_all[m]
        c = c_all[m]

        cols = np.asarray(s.colors(), dtype=np.uint8)
        if cols.ndim == 1:
            # Solid color snake
            frame[r, c, :3] = cols[:3]
        else:
            # Per-segment colors aligned with positions; apply same mask
            frame[r, c, :3] = cols[m, :3]


def render_food_into(foods: Iterable["Food"], frame: NDArray[np.uint8]) -> None:
    """
    Draw food RGB into an existing RGBA frame.
    Assumes `frame` was cleared by caller. Does NOT set alpha.
    """
    # Fast exit
    try:
        # If foods is a dict-like view/iterable, len may be cheap; otherwise ignore
        if not foods:
            return
    except Exception:
        pass

    rows: list[int] = []
    cols: list[int] = []
    rgbs: list[tuple[int, int, int]] = []

    for f in foods:
        rr, cc = f.position()
        rows.append(rr)
        cols.append(cc)
        rgbs.append(f.color())

    if not rows:
        return

    H = W = BOARD_H
    r_all = np.asarray(rows, dtype=np.int32)
    c_all = np.asarray(cols, dtype=np.int32)
    rgb_all = np.asarray(rgbs, dtype=np.uint8)

    m = _clip_mask(r_all, c_all, H, W)
    if np.any(m):
        r = r_all[m]
        c = c_all[m]
        frame[r, c, :3] = rgb_all[m, :3]


def finalize_alpha(frame: NDArray[np.uint8]) -> None:
    """
    Set A=255 where any RGB channel is non-zero, else 0. In-place.
    """
    alpha_mask = frame[..., :3].any(axis=2)
    frame[..., 3] = np.where(alpha_mask, 255, 0).astype(np.uint8)


TICK_RATE = 8  # Hz

# Published frame/timestamp; the thread will overwrite these.
LAST_RENDER: float = time.time()  # wall clock
LAST_FRAME: np.ndarray = np.zeros((BOARD_H, BOARD_W, 4), dtype=np.uint8)
OBSERVED_FRAME_RATE: float = 0.0  # EMA of FPS

# Global state
STATE_LOCK = threading.RLock()
ACTIVE_SNAKES: dict[str, Snake] = {}
SNAKE_MOVES: dict[str, Direction] = {}
ACTIVE_FOOD: dict[str, Food] = {}

def state_lock(func):
    def wrapper(*args, **kwargs):
        with STATE_LOCK:
            return func(*args, **kwargs)
    return wrapper

FOOD_LIFESPAN_FRAMES = 128
FOOD_PER_FRAME = 1


class EntityType(StrEnum):
    FOOD = 'food'
    SNAKE = 'snake'


_LAST_ID = {EntityType.FOOD: 0, EntityType.SNAKE: 0}


def _generate_id(e_type: EntityType) -> str:
    id_ = f'{e_type.value}-{_LAST_ID[e_type]}'
    _LAST_ID[e_type] += 1
    return id_


def _get_id_type(id_: str) -> EntityType:
    return EntityType(id_.split('-')[0])


@state_lock
def get_available_positions() -> set[tuple[int, int]]:
    restricted_positions = set()
    for snake in ACTIVE_SNAKES.values():
        restricted_positions.update(snake.positions())

    for food in ACTIVE_FOOD.values():
        restricted_positions.add(food.position())

    return ALL_POSITIONS - restricted_positions


# ---------- Food management ----------
def add_food(*, new_n: int | None = None, new_food: Iterable[Food] | None = None) -> None:
    available_positions = get_available_positions()

    if new_n:
        for _ in range(new_n):
            if not available_positions:
                continue

            pos = random.choice(list(available_positions))
            available_positions.remove(pos)
            ACTIVE_FOOD[_generate_id(EntityType.FOOD)] = Food(pos)

    if new_food:
        for food in new_food:
            if not available_positions:
                continue

            pos = food.position()
            if pos in available_positions:
                ACTIVE_FOOD[_generate_id(EntityType.FOOD)] = food
                available_positions.remove(pos)


def remove_expired_food(now: float | None = None) -> None:
    """Remove foods older than FOOD_LIFESPAN_FRAMES (in frame units mapped from wall time)."""
    if now is None:
        now = time.time()
    max_age_sec = FOOD_LIFESPAN_FRAMES / float(TICK_RATE)
    # in place filter
    keep: dict[str, Food] = {}
    for fid, f in ACTIVE_FOOD.items():
        if (now - f.created_at) <= max_age_sec:
            keep[fid] = f

    # single-writer assumption
    ACTIVE_FOOD.clear()
    ACTIVE_FOOD.update(keep)


# Collisions detection ---------
def resolve_collisions():
    positions2id = defaultdict(list)

    for fid, food in ACTIVE_FOOD.items():
        positions2id[food.position()].append(fid)

    for sid, snake in ACTIVE_SNAKES.items():
        for segment_id, segment in enumerate(snake.state):
            positions2id[segment.position].append((segment_id, sid))

    def filter_snakes(ids: Iterable[str | tuple[int, str]]) -> Iterator[tuple[int, str]]:
        for id_ in ids:
            if isinstance(id_, tuple):
                yield id_

    def filter_food(ids: Iterable[str | tuple[int, str]]) -> Iterator[str]:
        for id_ in ids:
            if isinstance(id_, str):
                yield id_

    def resolve_snakes(ids: Iterable[str | tuple[int, str]]) -> list[str | tuple[int, str]]:
        # ids at specific pos should include 1-2, max 3 entities, so list ops are the most efficient
        ids = list(ids)
        current_snakes = list(filter_snakes(ids))
        if not len(current_snakes) or len(current_snakes) == 1:
            # no snake collisions
            return ids

        remove_snakes = set()

        # resolve collision by favoring the snake with the lowest segment id
        # if equal, both snakes are removed (tie)
        sorted_snakes = sorted(current_snakes, key=lambda x: x[0], reverse=True)
        last_top_idx = sorted_snakes.index(sorted_snakes[-1])

        if last_top_idx != (len(sorted_snakes) - 1):
            # all snakes are removed. all but top are eaten and the top are removed due to tie
            remove_snakes.update(sorted_snakes)
        else:
            # the last snake eats all others
            remove_snakes.update(sorted_snakes[:-1])

        for removed_snake in remove_snakes:
            _, id_ = removed_snake
            _apply_remove_snake(id_)

        return ids

    def resolve_food(ids: Iterable[str | tuple[int, str]]) -> list[str | tuple[int, str]]:
        # ids at specific pos should include 1-2, max 3 entities, so list ops are the most efficient
        ids = list(ids)
        current_snakes = list(filter_snakes(ids))
        if not len(current_snakes):
            # no snakes to eat food
            return ids

        current_food = list(filter_food(ids))
        if not len(current_food):
            # no food to eat
            return ids

        for food_id in current_food:
            if food_id in ACTIVE_FOOD:
                # every snake eats (should never happen, but this is more reliable)
                for snake_segment_descriptor in current_snakes:
                    _, snake_id = snake_segment_descriptor

                    if snake_id in ACTIVE_SNAKES:
                        ACTIVE_SNAKES[snake_id].eat()

                ACTIVE_FOOD.pop(food_id)
                ids.remove(food_id)

        return ids

    for ids_at_pos in positions2id.values():
        ids_at_pos = resolve_snakes(ids_at_pos)
        resolve_food(ids_at_pos)


# Snake management ----------
def new_snake(colors: Iterable[tuple[int, int, int]]) -> Snake | None:
    snake_colors = ColorSequence(colors)
    global_valid_positions = get_available_positions()
    available_positions = list(global_valid_positions)  # for head placement
    random.shuffle(available_positions)

    def _is_valid_positions(positions: list[tuple[int, int]]) -> bool:
        result = True
        for pos in positions:
            if pos not in global_valid_positions:
                result = False
        return result

    while available_positions:
        head_h, head_w = available_positions.pop()
        available_directions: list[Direction] = list(Direction)  # noqa
        random.shuffle(available_directions)

        while available_directions:
            body_direction = available_directions.pop()
            d_h, d_w = DIRECTION_TO_MOVE[body_direction]
            snake_positions = [(head_h + i * d_h, head_w + i * d_w) for i in range(NEW_SNAKE_SEGMENTS)]
            if not _is_valid_positions(snake_positions):
                continue

            segments = [SnakeSegment(snake_colors.next_color(), pos) for pos in snake_positions]
            return Snake(snake_colors, segments)

    return None  # failed to find a valid position

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

_JOBS: queue.Queue[Job] = queue.Queue(maxsize=10000)
_WORKER: threading.Thread | None = None
_WORKER_STARTED = threading.Event()

def _ensure_worker():
    global _WORKER
    if _WORKER and _WORKER.is_alive():
        return
    def _worker():
        while True:
            job = _JOBS.get()
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

# ---------------- State mutations (locked) ----------------
@state_lock
def _apply_submit_snake(snake_id: str, snake: "Snake") -> None:
    ACTIVE_SNAKES[snake_id] = snake
    SNAKE_MOVES[snake_id] = snake.snake_direction

@state_lock
def _apply_remove_snake(snake_id: str) -> None:
    if snake_id in ACTIVE_SNAKES:
        new_food = ACTIVE_SNAKES[snake_id].die()
        add_food(new_food=new_food)
        ACTIVE_SNAKES.pop(snake_id, None)
    SNAKE_MOVES.pop(snake_id, None)

@state_lock
def _apply_submit_move(snake_id: str, direction: "Direction") -> None:
    if snake_id in ACTIVE_SNAKES:
        SNAKE_MOVES[snake_id] = direction

# ---------------- Public async API (non-blocking) ----------------
def submit_snake(snake: "Snake") -> str:
    """Return ID immediately; worker will add snake soon after."""
    _ensure_worker()
    snake_id = _generate_id(EntityType.SNAKE)
    _JOBS.put_nowait(Job(kind=JobType.SUBMIT_SNAKE, snake_id=snake_id, snake=snake))
    return snake_id

def remove_snake(snake_id: str) -> None:
    _ensure_worker()
    _JOBS.put_nowait(Job(kind=JobType.REMOVE_SNAKE, snake_id=snake_id))

def submit_move(snake_id: str, direction: "Direction") -> None:
    _ensure_worker()
    _JOBS.put_nowait(Job(kind=JobType.SUBMIT_MOVE, snake_id=snake_id, direction=direction))

def is_snake_alive(snake_id: str) -> bool:
    return snake_id in ACTIVE_SNAKES

def move_snakes():
    for snake_id, snake in ACTIVE_SNAKES.items():
        snake.move(SNAKE_MOVES.get(snake_id))


# main rendering thread ---------

@st.cache_resource
def run_rendering_thread():
    """
    Starts a single background renderer (per Streamlit server).
    Recomputes LAST_FRAME from ACTIVE_SNAKES and ACTIVE_FOOD at TICK_RATE FPS.
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
        global LAST_FRAME, LAST_RENDER, OBSERVED_FRAME_RATE
        nonlocal use_a, frame_a, frame_b
        dt = 1.0 / float(TICK_RATE)

        while not stop_event.is_set():
            t0 = time.perf_counter()

            # ---- Update game state & snapshot references under one lock ----
            now_wall = time.time()
            with STATE_LOCK:
                remove_expired_food(now=now_wall)
                add_food(new_n=FOOD_PER_FRAME)
                move_snakes()
                resolve_collisions()

                snakes_snapshot = tuple(ACTIVE_SNAKES.values())
                food_snapshot   = tuple(ACTIVE_FOOD.values())

            # ---- Choose back buffer, clear, render, and finalize alpha ----
            write_frame = frame_b if use_a else frame_a
            write_frame.fill(0)

            render_snakes_into(snakes_snapshot, write_frame)  # no allocations
            render_food_into(food_snapshot, write_frame)      # no allocations
            finalize_alpha(write_frame)                       # set RGBA[3] in one pass

            # ---- Publish by swapping buffers (no copy) ----
            LAST_FRAME = write_frame
            LAST_RENDER = now_wall
            use_a = not use_a

            # ---- Pace to target tick rate ----
            elapsed = time.perf_counter() - t0
            remaining = dt - elapsed
            if remaining > 0.0:
                time.sleep(remaining)

            # ---- Update observed FPS (EMA) ----
            loop_dt = time.perf_counter() - t0
            if loop_dt > 0.0:
                curr_fps = 1.0 / loop_dt
                alpha = 0.9 if OBSERVED_FRAME_RATE != 0.0 else 0.0
                OBSERVED_FRAME_RATE = alpha * OBSERVED_FRAME_RATE + (1.0 - alpha) * curr_fps

    thread = threading.Thread(target=_loop, name='snake-render', daemon=True)
    thread.start()
    return thread, stop_event
