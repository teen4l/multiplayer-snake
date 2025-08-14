import base64
import io
import random

import numpy as np
import streamlit as st
import streamlit_hotkeys as hotkeys
from PIL import Image

from state import TICK_RATE, Direction, submit_move

MAX_RENDER_RATE = TICK_RATE
PERIOD = 1.0 / float(MAX_RENDER_RATE)
DIRECTION_TO_KEYS = {
    Direction.UP: ['ArrowUp', 'W'],
    Direction.DOWN: ['ArrowDown', 'S'],
    Direction.LEFT: ['ArrowLeft', 'A'],
    Direction.RIGHT: ['ArrowRight', 'D']
}


@st.fragment  # isolate hotkeys to avoid unnecessary re-render
def init_hotkeys() -> None:
    hotkeys_list = []
    for direction, keys in DIRECTION_TO_KEYS.items():
        for key in keys:
            hotkeys_list.append(hotkeys.hk(
                str(direction), key,
                ignore_repeat=False,
                help=f'Move {direction.name.lower()}',
                prevent_default=True
            ))

    hotkeys.activate(hotkeys_list, key='game-hotkeys')


def show_frame(container, frame_np, width=None):
    # ensure proper dtype
    if frame_np.dtype != np.uint8:
        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
    # encode PNG -> base64
    buf = io.BytesIO()
    Image.fromarray(frame_np).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    style = f' style="width:{width}px;"' if width else ""
    container.markdown(
        f'<img src="data:image/png;base64,{b64}"{style}>',
        unsafe_allow_html=True,
    )


class GameUI:

    def __init__(self):
        if 'session' not in st.session_state:
            st.session_state.session = 'menu'
        if 'frame_scale' not in st.session_state:
            st.session_state.frame_scale = 6

        self.placeholders_ready = False

        self.game_info = None
        self.game_screen = None

    def init_placeholders(self):
        if self.placeholders_ready:
            return

        with st.container():
            self.game_info = st.empty()
            self.game_screen = st.empty()

        self.placeholders_ready = True

    @st.fragment(run_every=PERIOD)
    def render(self):
        # import here to refresh on each render
        from state import ACTIVE_SNAKES, LAST_FRAME, OBSERVED_FRAME_RATE, TICK_RATE

        self.init_placeholders()

        # collect all key events
        moves = []
        for direction in DIRECTION_TO_KEYS:
            if hotkeys.pressed(str(direction), key='game-hotkeys'):
                moves.append(direction)

        if 'snake' in st.session_state and st.session_state.session == 'playing' and len(moves):
            # in case multiple keys are pressed simultaneously
            submit_move(st.session_state.snake, Direction(random.choice(moves)))

        self.game_info.caption(
            f'Currently playing: {len(ACTIVE_SNAKES)}, frame rate: {OBSERVED_FRAME_RATE:.1f}',
            help=f'Frame rate is limited by the server and Streamlit capabilities. Target frame rate: {TICK_RATE}'
        )

        frame_scale = st.session_state.frame_scale
        frame = np.repeat(LAST_FRAME, frame_scale, axis=0)
        frame = np.repeat(frame, frame_scale, axis=1)

        show_frame(self.game_screen, frame)

    @classmethod
    def init(cls):
        if 'game_ui' in st.session_state:
            return st.session_state.game_ui
        return cls()