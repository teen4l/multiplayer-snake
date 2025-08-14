import random

import streamlit as st
import streamlit_hotkeys as hotkeys

from state import Direction, submit_move

MAX_RENDER_RATE = 20  # keep it snappy
PERIOD = 1.0 / float(MAX_RENDER_RATE)
DIRECTION_TO_KEYS = {
    Direction.UP: ['ArrowUp', 'W'],
    Direction.DOWN: ['ArrowDown', 'S'],
    Direction.LEFT: ['ArrowLeft', 'A'],
    Direction.RIGHT: ['ArrowRight', 'D']
}


# module-level cache: scale -> {"html": str, "last_checked": float}
_FRAME_CACHE = {}

def get_frame(frame_scale: int = 6):
    """
    Return a cached <img> unless state.LAST_RENDER advanced since we last checked.
    """
    from state import LAST_FRAME, LAST_RENDER  # LAST_RENDER: float timestamp (monotonic)
    cache = _FRAME_CACHE.get(frame_scale)
    last_checked = cache["last_checked"] if cache else float("-inf")

    # Re-render only if there's a newer frame
    if cache is None or LAST_RENDER > last_checked:
        import io, base64
        import numpy as np
        from PIL import Image

        frame = LAST_FRAME
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        # scale width by integer factor; keep pixels crisp
        width_px = int(frame.shape[1] * frame_scale)
        style = f'style="width:{width_px}px; image-rendering: pixelated;"'
        html = f'<img src="data:image/png;base64,{b64}" {style}>'

        _FRAME_CACHE[frame_scale] = {"html": html, "last_checked": float(LAST_RENDER)}
        return html

    # No new render since the last check -> serve cached HTML
    return cache["html"]


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

        self.game_screen.markdown(get_frame(st.session_state.frame_scale), unsafe_allow_html=True)

    @classmethod
    def init(cls):
        if 'game_ui' in st.session_state:
            return st.session_state.game_ui
        return cls()