import colorsys
import random

import streamlit as st


def hex_to_rgb(hex_color: str):
    s = hex_color.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:             # #rgb
        s = "".join(c*2 for c in s)
    elif len(s) == 4:           # #rgba
        s = "".join(c*2 for c in s)
    if len(s) not in (6, 8):
        raise ValueError(f"Invalid hex: {hex_color!r}")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    # ignore alpha if present
    return r, g, b


def validate_color(
    hex_color: str,
    l_range=(0.25, 0.85),  # reject too dark (<0.25) or too light (>0.85)
    s_min=0.35             # reject washed-out/grayish colors on dark UI
) -> bool:
    """
    Return True if the color is suitable for a dark theme:
    - Lightness L in [l_range[0], l_range[1]]
    - Saturation S >= s_min
    Based on HSL (via colorsys.rgb_to_hls; note it returns H, L, S).
    """
    try:
        r, g, b = hex_to_rgb(hex_color)
    except ValueError:
        return False

    rf, gf, bf = r/255.0, g/255.0, b/255.0
    h, l, s = colorsys.rgb_to_hls(rf, gf, bf)  # colorsys gives H,L,S

    if not (l_range[0] <= l <= l_range[1]):
        return False
    if s < s_min:
        return False
    return True


def random_valid_hex(
    validate=validate_color,
    l_range=(0.25, 0.85),   # keep in sync with your validator
    s_min=0.35,
    max_tries=50,
) -> str:
    """
    Sample H, L, S inside the validator's acceptance region, convert to RGB hex,
    and confirm with `validate`. Raises if it somehow can't satisfy after retries.
    """
    for _ in range(max_tries):
        h = random.random()                           # [0, 1)
        l = random.uniform(*l_range)                  # within accepted lightness
        s = random.uniform(max(1e-6, s_min), 1.0)     # at/above min saturation

        r, g, b = colorsys.hls_to_rgb(h, l, s)        # colorsys uses H, L, S
        ri, gi, bi = (int(round(255*x)) for x in (r, g, b))
        hex_str = f"#{ri:02x}{gi:02x}{bi:02x}"
        if validate(hex_str):
            return hex_str

    raise RuntimeError("Could not generate a valid color; relax thresholds?")


def theme_primary_rgb() -> tuple[int, int, int]:
    try:
        hex_col = st.get_option("theme.primaryColor") or "#F63366"
    except Exception:
        hex_col = "#F63366"
    return hex_to_rgb(hex_col)
