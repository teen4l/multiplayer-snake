import random

import av
import cv2
import numpy as np
import streamlit as st
import streamlit_hotkeys as hotkeys
from streamlit_webrtc import create_video_source_track, webrtc_streamer, WebRtcMode, VideoHTMLAttributes

from state import run_rendering_thread, TICK_RATE, Direction, submit_move, VIDEO_SOURCE_TRACK
from ui.controls import ControlsUI, restart_game

# startup background threads (once for all users)

DIRECTION_TO_KEYS = {
    Direction.UP: ['ArrowUp', 'W'],
    Direction.DOWN: ['ArrowDown', 'S'],
    Direction.LEFT: ['ArrowLeft', 'A'],
    Direction.RIGHT: ['ArrowRight', 'D']
}

run_rendering_thread()

expander_ph = st.empty()

# initialize the UI
controls_ui = ControlsUI()

with expander_ph.expander('How to play?'):
    st.markdown('Click `Start` button to initiate the game streaming, then click on `Join`!  \n >Hint: if the '
                'controls are not working, try clicking outside of the game window.')
    hotkeys.legend(key='hotkeys')

controls_ui.render()


webrtc_streamer(
    key="player",
    mode=WebRtcMode.RECVONLY,
    source_video_track=VIDEO_SOURCE_TRACK,
    media_stream_constraints={"video": True, "audio": False},
    video_receiver_size=12,
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, style={"width": "100%"})
)

@st.fragment()
def isolate_hotkeys():
    hotkeys_list = [hotkeys.hk('restart', 'R', help='Restart the game')]

    for direction, keys in DIRECTION_TO_KEYS.items():
        for key in keys:
            hotkeys_list.append(hotkeys.hk(
                str(direction), key,
                ignore_repeat=False,
                help=f'Move {direction.name.lower()}',
                prevent_default=True
            ))

    hotkeys.activate(hotkeys_list, key='hotkeys')
    hotkeys.on_pressed('restart', restart_game, key='hotkeys')

    # collect all key events to move the snake
    moves = []
    for direction in DIRECTION_TO_KEYS:
        if hotkeys.pressed(str(direction), key='hotkeys'):
            moves.append(direction)

    if 'snake' in st.session_state and st.session_state.session == 'playing' and len(moves):
        # in case multiple keys are pressed simultaneously
        submit_move(st.session_state.snake, Direction(random.choice(moves)))

isolate_hotkeys()
