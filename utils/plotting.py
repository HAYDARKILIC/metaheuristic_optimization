"""Consistent matplotlib styling and figure saving."""

from pathlib import Path

import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"

PALETTE = {
    "blue": "#1f4e79",
    "orange": "#d9822b",
    "green": "#3a7d44",
    "red": "#b23a48",
    "purple": "#6a4c93",
    "grey": "#6c757d",
    "teal": "#2a9d8f",
}


def set_style():
    """Apply the curriculum-wide matplotlib style."""
    plt.rcParams.update(
        {
            "figure.figsize": (7.5, 4.5),
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.prop_cycle": plt.cycler(color=list(PALETTE.values())),
            "legend.frameon": False,
            "font.size": 10.5,
        }
    )


def savefig(name, fig=None, dpi=150):
    """Save a figure into the repository-level ``figures/`` directory.

    Parameters
    ----------
    name : str
        File name, e.g. ``"w1_utility_curves.png"``. Prefix with the week.
    fig : matplotlib.figure.Figure, optional
        Defaults to the current figure.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig = fig or plt.gcf()
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
