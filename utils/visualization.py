"""
Funksionet e vizualizimit me Matplotlib.

"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

# Ngjyra të buta, të dallueshme për vlerat 1..9 (indeksi 0 = qelizë bosh)
_VALUE_COLORS = [
    "#ffffff",  # 0 – bosh
    "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3",
    "#fdb462", "#b3de69", "#fccde5", "#bc80bd",
]


def plot_value_matrix(
    matrix: list[list[int]],
    title: str = "",
    highlight: tuple[int, int] | None = None,
    max_value: int | None = None,
):
    """
    Vizaton një matricë me vlera 0..9 si rrjetë me ngjyra.

    Çdo vlerë ka ngjyrën e vet (0 = e bardhë), numri shfaqet në qelizë.
    `highlight` = (rresht, kolonë) e qelizës që theksohet me kornizë të kuqe.
    """
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if n_rows else 0
    if max_value is None:
        max_value = max((v for row in matrix for v in row), default=0)
    max_value = max(1, min(max_value, len(_VALUE_COLORS) - 1))
    cmap = ListedColormap(_VALUE_COLORS[: max_value + 1])

    size = max(2.5, 0.6 * max(n_rows, n_cols))
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=max_value)

    for i in range(n_rows):
        for j in range(n_cols):
            value = matrix[i][j]
            if value:
                ax.text(j, i, str(value), ha="center", va="center",
                        fontsize=14, fontweight="bold", color="#222222")

    ax.set_xticks([x - 0.5 for x in range(1, n_cols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n_rows)], minor=True)
    ax.grid(which="minor", color="#555555", linewidth=1)
    ax.tick_params(which="both", length=0)
    ax.set_xticks(range(n_cols), [str(j + 1) for j in range(n_cols)])
    ax.set_yticks(range(n_rows), [str(i + 1) for i in range(n_rows)])

    if highlight is not None:
        r, c = highlight
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor="red", linewidth=3))
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_latin_square(matrix: list[list[int]], title: str = "Latin Square",
                      highlight: tuple[int, int] | None = None):
    """Matricë Latin Square me ngjyrë për çdo vlerë 1..n."""
    return plot_value_matrix(matrix, title=title, highlight=highlight, max_value=len(matrix))