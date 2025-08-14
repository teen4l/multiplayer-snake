import os
import random
import tempfile

import numpy as np
import streamlit as st
import streamlit_hotkeys as hotkeys
from PIL import Image

from state import TICK_RATE, Direction, submit_move

MAX_RENDER_RATE = TICK_RATE * 2
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
                help=f'Move {direction.name.lower()}'
            ))

    hotkeys.activate(hotkeys_list, key='game-hotkeys')


class GameUI:

    def __init__(self):
        if 'session' not in st.session_state:
            st.session_state.session = 'menu'
        if 'frame_scale' not in st.session_state:
            st.session_state.frame_scale = 5

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

        tmp_dir = tempfile.gettempdir()
        frame_path = os.path.join(tmp_dir, 'game_frame.png')
        staging_path = frame_path + ".tmp"

        Image.fromarray(frame).save(staging_path, format='PNG')
        os.replace(staging_path, frame_path)

        self.game_screen.image(frame_path)

    @classmethod
    def init(cls):
        if 'game_ui' in st.session_state:
            return st.session_state.game_ui
        return cls()