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


# ---------------------------------------------------------------------------
# Pema e kërkimit
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    "generated": "#d9d9d9",
    "explored": "#9ecae1",
    "duplicate": "#fdd0a2",
    "invalid": "#fb6a4a",
    "goal": "#31a354",
    "solution": "#a1d99b",
    "backtrack": "#bdbdbd",
}


def plot_search_tree(graph, title: str = "Search Tree", font_size: int = 7,
                     show_labels: bool = True):
    """
    Vizaton pemën reale të kërkimit (graf NetworkX nga SearchTreeRecorder).
    Ngjyrat tregojnë statusin e çdo nyjeje; skaji mban emrin e veprimit.
    """
    import networkx as nx
    from matplotlib.patches import Patch

    from utils.search_tree import STATUS_LABELS, tree_layout

    positions = tree_layout(graph)
    n_nodes = graph.number_of_nodes()
    max_depth = max((-y for _, y in positions.values()), default=0)
    width = max(6.0, min(28.0, 0.9 * max((x for x, _ in positions.values()), default=0) + 2))
    height = max(3.0, min(20.0, 1.3 * (max_depth + 1)))
    fig, ax = plt.subplots(figsize=(width, height))

    colors = [_STATUS_COLORS.get(graph.nodes[n].get("status"), "#ffffff") for n in graph.nodes]
    node_size = 900 if n_nodes <= 40 else 450 if n_nodes <= 120 else 200
    nx.draw_networkx_edges(graph, positions, ax=ax, arrows=False, edge_color="#888888")
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=colors,
                           node_size=node_size, edgecolors="#444444", linewidths=0.6)
    if show_labels and n_nodes <= 120:
        labels = {n: graph.nodes[n].get("label", str(n)) for n in graph.nodes}
        nx.draw_networkx_labels(graph, positions, labels=labels, ax=ax, font_size=font_size)

    used = {graph.nodes[n].get("status") for n in graph.nodes}
    handles = [Patch(facecolor=_STATUS_COLORS[s], edgecolor="#444444", label=STATUS_LABELS[s])
               for s in _STATUS_COLORS if s in used]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Missionaries and Cannibals – lumi
# ---------------------------------------------------------------------------

def plot_river_state(state, total_missionaries: int, total_cannibals: int,
                     step: int | None = None, crossing: tuple[int, int] | None = None,
                     direction: str | None = None, valid: bool = True):
    """
    Vizaton dy brigjet, lumin dhe varkën.
    state = (M_left, C_left, Boat) me Boat = 1 majtas, 0 djathtas.
    Misionarët = rrathë blu (M), kanibalët = trekëndësha të kuq (K).
    Një gjendje e pavlefshme vizatohet me sfond të kuq.
    """
    m_left, c_left, boat = state
    m_right, c_right = total_missionaries - m_left, total_cannibals - c_left

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    bank_color = "#e5d3a1" if valid else "#f4a3a3"
    ax.add_patch(plt.Rectangle((0, 0), 3, 4, color=bank_color))
    ax.add_patch(plt.Rectangle((7, 0), 3, 4, color=bank_color))
    ax.add_patch(plt.Rectangle((3, 0), 4, 4, color="#9ecae1"))

    def draw_people(x0, missionaries, cannibals):
        for k in range(missionaries):
            ax.plot(x0 + 0.5 + 0.7 * (k % 3), 3.2 - 0.6 * (k // 3), "o",
                    markersize=16, color="#2171b5")
            ax.text(x0 + 0.5 + 0.7 * (k % 3), 3.2 - 0.6 * (k // 3), "M",
                    ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        for k in range(cannibals):
            ax.plot(x0 + 0.5 + 0.7 * (k % 3), 1.6 - 0.6 * (k // 3), "^",
                    markersize=17, color="#cb181d")
            ax.text(x0 + 0.5 + 0.7 * (k % 3), 1.5 - 0.6 * (k // 3), "K",
                    ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    draw_people(0.1, m_left, c_left)
    draw_people(7.1, m_right, c_right)

    boat_x = 3.3 if boat == 1 else 5.7
    ax.add_patch(plt.Rectangle((boat_x, 1.8), 1.0, 0.4, color="#8c510a"))
    ax.text(boat_x + 0.5, 1.55, "Varka", ha="center", fontsize=8)

    ax.text(1.5, 3.8, f"Majtas: {m_left}M, {c_left}K", ha="center", fontsize=9)
    ax.text(8.5, 3.8, f"Djathtas: {m_right}M, {c_right}K", ha="center", fontsize=9)

    title_parts = []
    if step is not None:
        title_parts.append(f"Hapi {step}")
    if crossing is not None:
        title_parts.append(f"kaluan {crossing[0]}M + {crossing[1]}K")
    if direction:
        title_parts.append(direction)
    title_parts.append("gjendje e vlefshme" if valid else "GJENDJE E PAVLEFSHME")
    ax.set_title(" · ".join(title_parts), fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Workforce Scheduling – shpërndarja e ngarkesës
# ---------------------------------------------------------------------------

def plot_workload(workload_rows: list[dict], shift_names: list[str],
                  title: str = "Shpërndarja e ngarkesës"):
    """Shtylla të grumbulluara: turnet e secilit punëtor sipas llojit të turnit."""
    workers = [row["Punëtori"] for row in workload_rows]
    colors = ["#fdd49e", "#9ecae1", "#3f007d"]
    fig, ax = plt.subplots(figsize=(max(4.5, 0.7 * len(workers) + 2), 3.5))
    bottom = [0] * len(workers)
    for k, name in enumerate(shift_names):
        values = [row[name] for row in workload_rows]
        ax.bar(workers, values, bottom=bottom, label=name, color=colors[k % len(colors)],
               edgecolor="#444444", linewidth=0.5)
        bottom = [b + v for b, v in zip(bottom, values)]
    for x, total in enumerate(bottom):
        ax.text(x, total + 0.1, str(total), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Numri i turneve")
    ax.set_ylim(0, max(bottom, default=0) + 1.5)
    ax.legend(fontsize=8, ncol=len(shift_names), loc="upper center")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
