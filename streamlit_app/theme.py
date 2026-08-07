"""Shared chart theme for the Streamlit UI's 5 tabs.

Per the dataviz skill's color-formula: categorical hues are assigned in a
fixed order and never cycled, magnitude uses one sequential hue (blue,
light -> dark), and pass/fail state uses a fixed status palette that is
never reused for a data series and is always paired with an icon + label
(never color alone). Values are the skill's validated default palette --
retarget a brand palette by swapping this file's hexes only; no chart code
in streamlit_app/*.py references a raw hex.
"""

from __future__ import annotations

# Fixed order -- slot N is always the Nth series, never reassigned by a filter.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Single hue, light -> dark, for continuous magnitude (heatmaps, gauges).
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#184f95"]

# Fixed, never themed, never reused for a categorical series.
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
STATUS_ICON = {"good": "✅", "warning": "⚠️", "serious": "\U0001f7e0", "critical": "\U0001f534"}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

PLOTLY_TEMPLATE = "plotly_white"


def status_badge(status: str, label: str) -> str:
    """Icon + label markdown -- status color never carries meaning alone."""
    icon = STATUS_ICON.get(status, "•")
    return f"{icon} **{label}**"
