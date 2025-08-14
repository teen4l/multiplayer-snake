import streamlit as st
import streamlit_hotkeys as hotkeys

from state import new_snake, submit_snake, remove_snake
from utils.color import random_valid_hex, hex_to_rgb, validate_color


MAX_SEGMENTS = 12


def increment_segments():
    st.session_state.snake_segments += 1


def validate_selection(color_idx: int):
    if validate_color(st.session_state[f'picked_color_{color_idx}']):
        st.session_state.color_warnings.discard(color_idx)
    else:
        st.session_state.color_warnings.add(color_idx)


def create_snake(*_, **__):
    rgb_colors = []
    for segment_idx in range(st.session_state.snake_segments):
        rgb_colors.append(hex_to_rgb(st.session_state[f'picked_color_{segment_idx}']))

    snake = new_snake(rgb_colors)
    if not snake:
        return

    st.session_state.snake = submit_snake(snake)
    st.session_state.session = 'playing'


def kill_snake(*_, **__):
    if 'snake' not in st.session_state:
        return
    remove_snake(st.session_state.snake)
    del st.session_state['snake']
    st.session_state.session = 'menu'


def restart_game(*_, **__):
    kill_snake()
    create_snake()


@st.fragment
def activate_restart_hotkey():
    hotkeys.activate(hotkeys.hk('restart', 'R', help='Restart the game'), key='controls')
    hotkeys.on_pressed('restart', restart_game, key='controls')


class ControlsUI:

    def __init__(self):
        if 'session' not in st.session_state:
            st.session_state.session = 'menu'

        if 'snake_segments' not in st.session_state:
            st.session_state.snake_segments = 3

        if 'color_warnings' not in st.session_state:
            st.session_state.color_warnings = set()

        if 'frame_scale' not in st.session_state:
            st.session_state.frame_scale = 6

        self.placeholders_ready = False

        self.color_instructions = None
        self.color_columns = None
        self.info_panel = None
        self.frame_scale_slider = None
        self.session_buttons_columns = None
        self.controls_divider = None

    def init_placeholders(self):
        if self.placeholders_ready:
            return

        with st.container():
            self.color_instructions = st.empty()
            self.color_columns = st.empty()
            self.info_panel = st.empty()
            self.frame_scale_slider = st.empty()
            self.session_buttons_columns = st.empty()
            self.controls_divider = st.empty()

        self.placeholders_ready = True

    @st.fragment(run_every=1)
    def render(self):
        self.init_placeholders()
        self.color_instructions.caption('Pick the color for your snake')
        snake_colors: list[str] = []

        for idx, col in enumerate(self.color_columns.columns([1] * MAX_SEGMENTS, vertical_alignment='bottom')):
            with col:
                if idx == st.session_state.snake_segments:
                    st.button(
                        ':material/add:',
                        on_click=increment_segments,
                        disabled=(st.session_state.session == 'playing')
                    )

                if idx < st.session_state.snake_segments:
                    if f'picked_color_{idx}' not in st.session_state:
                        st.session_state[f'picked_color_{idx}'] = random_valid_hex()

                    snake_colors.append(st.color_picker(
                        f'Color of the segment {idx}',
                        key=f'picked_color_{idx}',
                        label_visibility='collapsed',
                        on_change=validate_selection,
                        help=f'Segment {idx + 1}',
                        args=(idx,),
                        disabled=(st.session_state.session == 'playing')
                    ))

        if len(st.session_state.color_warnings):
            warnings = list(st.session_state.color_warnings)
            self.info_panel.warning(f':material/warning: The segment {warnings[0] + 1} is too dark or too white')
        else:
            self.info_panel.success(':material/check: Your snake is perfect!')

        # session controls

        self.frame_scale_slider.slider(
            min_value=1,
            max_value=10,
            key='frame_scale',
            step=1,
            label='Gaming screen scale',
            help='Slide to adjust the scale of the gaming screen',
            disabled=(st.session_state.session == 'playing'),
        )

        col_join, col_give_up = self.session_buttons_columns.columns([1, 1])
        with col_join:
            st.button(
                'Join',
                type='primary',
                disabled=(st.session_state.session == 'playing') or len(st.session_state.color_warnings),
                use_container_width=True,
                on_click=create_snake,
                kwargs={'session': 'playing'}
            )
        with col_give_up:
            st.button(
                'Give Up',
                type='secondary',
                disabled=(st.session_state.session == 'menu'),
                use_container_width=True,
                on_click=kill_snake,
                kwargs={'session': 'menu'},
            )

        self.controls_divider.divider()

    @classmethod
    def init(cls):
        if 'controls_ui' in st.session_state:
            return st.session_state.controls_ui
        return cls()