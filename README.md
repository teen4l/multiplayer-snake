# Multiplayer Snake (Streamlit)

A small real-time **multiplayer Snake** that runs in the browser with a **Python/Streamlit** backend. It’s primarily a demo of the **`streamlit-hotkeys`** ([PyPI](https://pypi.org/project/streamlit-hotkeys/), [GitHub](https://github.com/viktor-shcherb/streamlit-hotkeys)) component for low-latency keyboard input on Streamlit.

## Demo
Check it out on [Streamlit Community Cloud](https://multiplayer-snake.streamlit.app/).

## Features

* **Multiplayer**: everyone shares one arena.
* **Controls**: Arrow keys or WASD (via `streamlit-hotkeys`).
* **Crisp pixels**: server renders a NumPy frame → PNG, shown with `image-rendering: pixelated`.
* **Fixed tick rate** for consistent gameplay.

## Quickstart

```bash
git clone https://github.com/viktor-shcherb/multiplayer-snake
cd multiplayer-snake
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL in your browser (open several tabs for multiple players).

**Controls:** Arrow keys / WASD. Make sure the page has focus.

## How it works (high level)

* **Shared state** lives in the Python process. Each Streamlit session (tab) is a thread that reads/writes that state.
* **Input** arrives via `streamlit-hotkeys` and updates per-player direction.
* **Game loop** advances at a fixed tick, moves snakes, resolves collisions, and draws the board into a NumPy array.
* **Rendering** encodes the array to PNG (Pillow), embeds it as a base64 `<img>` with `image-rendering: pixelated` for nearest-neighbor scaling.

## Contributing

Contributions are very welcome—issues, discussions, and PRs.

Potential improvements:

* Input/UI: on-screen buttons or mobile touch controls; configurable key bindings.
* Gameplay: scoring, UI overlays, bots, teams, power-ups, and special abilities.
* Net/rendering: smarter diff-based updates; client-side `<canvas>` renderer; WebRTC streaming.
* Robustness: better state sync/locking; tests; linting; CI.
* Observability: FPS/tick metrics, profiling hooks.
* Docs: clearer examples of using `streamlit-hotkeys` in other apps.

**Note:** Performance on **Streamlit Community Cloud** is currently poor and needs investigation (possible culprits: PNG encoding overhead, rerun frequency, cache invalidation, bandwidth/CPU limits). Help profiling and optimizing this is especially appreciated.
