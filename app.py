import streamlit as st
import streamlit_hotkeys as hotkeys

from state import run_rendering_thread
from ui.controls import ControlsUI, activate_restart_hotkey
from ui.game import GameUI, init_hotkeys

# startup background threads (once for all users)

run_rendering_thread()

expander_ph = st.empty()

# initialize the UI
controls_ui = ControlsUI.init()
game_ui = GameUI.init()

with st.sidebar:
    # move hotkey managers to the sidebar
    activate_restart_hotkey()
    init_hotkeys()

with expander_ph.expander('How to play?'):
    hotkeys.legend(key='controls')
    st.markdown('Use the following keys to control the snake:')
    hotkeys.legend(key='game-hotkeys')
    st.markdown('**Recommended**: expand the game screen (hover the top-right corner of the game screen for the button '
                'to appear), press `R` to start playing!')

controls_ui.render()
game_ui.render()
