"""
Funksionet e vizualizimit me Matplotlib dhe NetworkX.

Parimet:
  * çdo funksion kthen një `matplotlib.figure.Figure` (Streamlit: `st.pyplot(fig)`);
  * titujt dhe etiketat janë në shqip, me stil të njëtrajtshëm;
  * funksionet VETËM vizatojnë të dhënat që u jepen nga algoritmet –
    nuk ka logjikë kërkimi këtu dhe nuk krijohen gjendje të sajuara;
  * vlerat origjinale dallohen nga ato të gjeneruara, gjendjet e pavlefshme
    nga të vlefshmet, dhe rruga e zgjidhjes theksohet.

Përdoret backend-i "Agg" (pa dritare), që funksionon edhe në teste.
Pas shfaqjes, figura mbyllet me `plt.close(fig)` për të liruar kujtesën.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ---------------------------------------------------------------------------
# Stili i përbashkët
# ---------------------------------------------------------------------------

TITLE_SIZE = 12
LABEL_SIZE = 9
GRID_COLOR = "#555555"
ORIGINAL_BG = "#e8e8e8"       # qeliza origjinale (shifrat fillestare)
ORIGINAL_TEXT = "#111111"
GENERATED_TEXT = "#1f5fbf"    # vlera të vendosura nga algoritmi
VALID_COLOR = "#a1d99b"
INVALID_COLOR = "#fb6a4a"
HIGHLIGHT_COLOR = "#d62728"

# Ngjyra të buta, të dallueshme për vlerat 1..9 (indeksi 0 = qelizë bosh)
_VALUE_COLORS = [
    "#ffffff",
    "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3",
    "#fdb462", "#b3de69", "#fccde5", "#bc80bd",
]

_SHIFT_COLORS = ["#fdd49e", "#9ecae1", "#6a51a3"]


def _finish(fig, ax, title: str | None):
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE)
    fig.tight_layout()
    return fig


def _grid_axes(ax, n_rows: int, n_cols: int, row_labels=None, col_labels=None):
    """Vija rrjete dhe etiketa 1..n (ose të dhëna) për një matricë."""
    ax.set_xticks([x - 0.5 for x in range(1, n_cols)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, n_rows)], minor=True)
    ax.grid(which="minor", color=GRID_COLOR, linewidth=1)
    ax.tick_params(which="both", length=0, labelsize=LABEL_SIZE)
    ax.set_xticks(range(n_cols), col_labels or [str(j + 1) for j in range(n_cols)])
    ax.set_yticks(range(n_rows), row_labels or [str(i + 1) for i in range(n_rows)])


def _highlight_cell(ax, cell):
    if cell is not None:
        r, c = cell
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor=HIGHLIGHT_COLOR, linewidth=3))


# ---------------------------------------------------------------------------
# Matrica me vlera (bazë për Latin Square)
# ---------------------------------------------------------------------------

def plot_value_matrix(
    matrix: list[list[int]],
    title: str = "",
    highlight: tuple[int, int] | None = None,
    max_value: int | None = None,
):
    """
    Matricë me vlera 0..9 si rrjetë me ngjyra: çdo vlerë ka ngjyrën e vet,
    0 = qelizë bosh (e bardhë). `highlight` = qeliza që theksohet me të kuqe.
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
            if matrix[i][j]:
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                        fontsize=14, fontweight="bold", color="#222222")
    _grid_axes(ax, n_rows, n_cols)
    _highlight_cell(ax, highlight)
    return _finish(fig, ax, title)


def plot_latin_square(matrix: list[list[int]], title: str = "Latin Square",
                      highlight: tuple[int, int] | None = None):
    """Latin Square (i plotë ose i pjesshëm) me ngjyrë për çdo vlerë 1..n."""
    return plot_value_matrix(matrix, title=title, highlight=highlight, max_value=len(matrix))


# ---------------------------------------------------------------------------
# Sudoku
# ---------------------------------------------------------------------------

def plot_sudoku(board: list[list[int]], original: list[list[int]] | None = None,
                title: str = "Sudoku", highlight: tuple[int, int] | None = None):
    """
    Tabela Sudoku 9×9. Shifrat origjinale: të zeza në sfond gri;
    shifrat e vendosura nga algoritmi: blu. Vija të trasha ndajnë nënrrjetat 3×3.
    """
    original = original or board
    fig, ax = plt.subplots(figsize=(4.6, 4.9))
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(8.5, -0.5)
    for r in range(9):
        for c in range(9):
            value = board[r][c]
            is_original = original[r][c] != 0
            if is_original:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=ORIGINAL_BG))
            if value:
                ax.text(c, r, str(value), ha="center", va="center", fontsize=13,
                        fontweight="bold" if is_original else "normal",
                        color=ORIGINAL_TEXT if is_original else GENERATED_TEXT)
    for k in range(10):
        width = 2.2 if k % 3 == 0 else 0.6
        ax.axhline(k - 0.5, color="#222222", linewidth=width)
        ax.axvline(k - 0.5, color="#222222", linewidth=width)
    _highlight_cell(ax, highlight)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    handles = [Patch(facecolor=ORIGINAL_BG, label="Shifër origjinale"),
               Patch(facecolor="white", edgecolor=GENERATED_TEXT, label="Vendosur nga BFS")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
              ncol=2, fontsize=8, frameon=False)
    return _finish(fig, ax, title)


# ---------------------------------------------------------------------------
# Social Golfers
# ---------------------------------------------------------------------------

def plot_golfers_schedule(schedule: list[list[tuple[int, ...]]],
                          title: str = "Orari i Social Golfers",
                          max_weeks: int | None = None):
    """
    Tabelë javë × grupe: çdo qelizë përmban golferët e grupit.
    `schedule` = rezultati i solve_social_golfers (golferët 1..n).
    """
    weeks = schedule[:max_weeks] if max_weeks else schedule
    n_weeks = len(weeks)
    n_groups = len(weeks[0]) if weeks else 0
    fig, ax = plt.subplots(figsize=(max(5.0, 1.15 * n_groups + 1), max(1.6, 0.48 * n_weeks + 1)))
    shading = [[(w % 2) for _ in range(n_groups)] for w in range(n_weeks)]
    ax.imshow(shading or [[0]], cmap=ListedColormap(["#f7fbff", "#deebf7"]), vmin=0, vmax=1,
              aspect="auto")
    for w, week in enumerate(weeks):
        for j, group in enumerate(week):
            ax.text(j, w, "-".join(str(x) for x in group), ha="center", va="center",
                    fontsize=8 if len(group) > 3 else 9)
    _grid_axes(ax, max(n_weeks, 1), max(n_groups, 1),
               row_labels=[f"Java {w + 1}" for w in range(n_weeks)] or [""],
               col_labels=[f"Grupi {j + 1}" for j in range(n_groups)] or [""])
    return _finish(fig, ax, title)


def plot_pair_matrix(pair_matrix: list[list[int]],
                     title: str = "Çiftet e golferëve sipas javës"):
    """
    Matricë n×n: qeliza (a, b) = java kur a dhe b luajtën bashkë,
    0 = nuk kanë luajtur, −1 = çift i përsëritur (shkelje, me të kuqe).
    Të dhënat vijnë nga `pair_week_matrix` i detyrës.
    """
    n = len(pair_matrix)
    max_week = max((v for row in pair_matrix for v in row), default=0)
    cmap = plt.get_cmap("viridis", max(max_week, 1))
    fig, ax = plt.subplots(figsize=(max(4.0, 0.16 * n + 2.5), max(3.6, 0.16 * n + 2)))
    masked = [[float("nan") if v <= 0 else v for v in row] for row in pair_matrix]
    image = ax.imshow(masked, cmap=cmap, vmin=0.5, vmax=max(max_week, 1) + 0.5)
    for a in range(n):
        for b in range(n):
            if pair_matrix[a][b] == -1:
                ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, color=INVALID_COLOR))
    ax.set_xlabel("Golferi", fontsize=LABEL_SIZE)
    ax.set_ylabel("Golferi", fontsize=LABEL_SIZE)
    step = 1 if n <= 16 else 4
    ax.set_xticks(range(0, n, step), [str(k + 1) for k in range(0, n, step)], fontsize=7)
    ax.set_yticks(range(0, n, step), [str(k + 1) for k in range(0, n, step)], fontsize=7)
    if max_week:
        bar = fig.colorbar(image, ax=ax, ticks=range(1, max_week + 1), fraction=0.046)
        bar.set_label("Java", fontsize=LABEL_SIZE)
    return _finish(fig, ax, title)


# ---------------------------------------------------------------------------
# 8-Puzzle
# ---------------------------------------------------------------------------

def _draw_puzzle(ax, state, goal, title=None, title_size=10):
    for i, value in enumerate(state):
        r, c = divmod(i, 3)
        if value == 0:
            color = "#525252"
        elif goal is not None and goal[i] == value:
            color = VALID_COLOR           # pllakë në vendin e qëllimit
        else:
            color = "#fdd0a2"             # pllakë jo në vendin e qëllimit
        ax.add_patch(plt.Rectangle((c, 2 - r), 0.95, 0.95, color=color))
        if value:
            ax.text(c + 0.475, 2 - r + 0.475, str(value), ha="center", va="center",
                    fontsize=14, fontweight="bold")
    ax.set_xlim(-0.05, 3)
    ax.set_ylim(-0.05, 3)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=title_size)


def plot_eight_puzzle(state, goal=None, title: str = "8-Puzzle"):
    """
    Një tabelë 8-Puzzle. Gri e errët = pozicioni bosh; jeshile = pllakë në
    vendin e saj në Goal State; portokalli = pllakë jo në vend.
    """
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    _draw_puzzle(ax, state, goal)
    return _finish(fig, ax, title)


def plot_eight_puzzle_path(path, actions, goal=None, max_boards: int = 12,
                           title: str = "Rruga e zgjidhjes"):
    """
    Tabelat e rrugës së zgjidhjes, njëra pas tjetrës (maks. `max_boards`,
    nëse rruga është më e gjatë shfaqen të parat dhe e fundit).
    `path` dhe `actions` vijnë nga rezultati i BFS.
    """
    indices = list(range(len(path)))
    if len(indices) > max_boards:
        indices = indices[: max_boards - 1] + [len(path) - 1]
    cols = min(6, len(indices))
    rows = -(-len(indices) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.9 * cols, 2.35 * rows + 0.3), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for slot, k in enumerate(indices):
        label = "Fillimi" if k == 0 else f"Hapi {k}: {actions[k - 1]}"
        _draw_puzzle(axes.flat[slot], path[k], goal, label, title_size=8)
    if len(path) > max_boards:
        title = f"{title} (shfaqen {len(indices)} nga {len(path)} gjendje)"
    fig.suptitle(title, fontsize=TITLE_SIZE)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Missionaries and Cannibals – lumi
# ---------------------------------------------------------------------------

def plot_river_state(state, total_missionaries: int, total_cannibals: int,
                     step: int | None = None, crossing: tuple[int, int] | None = None,
                     direction: str | None = None, valid: bool = True):
    """
    Dy brigjet, lumi dhe varka. state = (M_left, C_left, Boat), Boat = 1 majtas.
    Misionarët = rrathë blu (M), kanibalët = trekëndësha të kuq (K).
    Gjendja e pavlefshme vizatohet me brigje të kuqe.
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
        for k in range(max(0, missionaries)):
            x, y = x0 + 0.5 + 0.7 * (k % 3), 3.2 - 0.6 * (k // 3)
            ax.plot(x, y, "o", markersize=16, color="#2171b5")
            ax.text(x, y, "M", ha="center", va="center", color="white", fontsize=8,
                    fontweight="bold")
        for k in range(max(0, cannibals)):
            x, y = x0 + 0.5 + 0.7 * (k % 3), 1.6 - 0.6 * (k // 3)
            ax.plot(x, y, "^", markersize=17, color="#cb181d")
            ax.text(x, y - 0.1, "K", ha="center", va="center", color="white", fontsize=8,
                    fontweight="bold")

    draw_people(0.1, m_left, c_left)
    draw_people(7.1, m_right, c_right)

    boat_x = 3.3 if boat == 1 else 5.7
    ax.add_patch(plt.Rectangle((boat_x, 1.8), 1.0, 0.4, color="#8c510a"))
    ax.text(boat_x + 0.5, 1.55, "Varka", ha="center", fontsize=8)
    if direction:
        arrow = "→" if "majtas →" in direction else "←"
        ax.text(5, 2.7, arrow, ha="center", va="center", fontsize=26, color="#08519c")

    ax.text(1.5, 3.8, f"Majtas: {m_left}M, {c_left}K", ha="center", fontsize=9)
    ax.text(8.5, 3.8, f"Djathtas: {m_right}M, {c_right}K", ha="center", fontsize=9)

    parts = []
    if step is not None:
        parts.append(f"Hapi {step}")
    if crossing is not None:
        parts.append(f"kaluan {crossing[0]}M + {crossing[1]}K")
    if direction:
        parts.append(direction)
    parts.append("gjendje e vlefshme" if valid else "GJENDJE E PAVLEFSHME")
    return _finish(fig, ax, " · ".join(parts))


# ---------------------------------------------------------------------------
# Workforce Scheduling
# ---------------------------------------------------------------------------

def plot_workforce_schedule(schedule: list[list[list[int]]], shift_names: list[str],
                            title: str = "Orari javor i punëtorëve"):
    """
    Tabelë ditë × turne; çdo qelizë = punëtorët e turnit (P1, P2, ...),
    ngjyra sipas llojit të turnit. `schedule` = rezultati i DFS (punëtorët 1..W).
    """
    n_days = len(schedule)
    n_shifts = len(shift_names)
    fig, ax = plt.subplots(figsize=(max(4.5, 1.9 * n_shifts + 1.5), max(2.2, 0.45 * n_days + 1)))
    colors = ListedColormap(_SHIFT_COLORS[:n_shifts])
    ax.imshow([list(range(n_shifts)) for _ in range(n_days)], cmap=colors,
              vmin=0, vmax=max(n_shifts - 1, 1), aspect="auto")
    for d, day in enumerate(schedule):
        for s, workers in enumerate(day):
            ax.text(s, d, ", ".join(f"P{w}" for w in workers), ha="center", va="center",
                    fontsize=9, color="white" if s == 2 else "#222222")
    _grid_axes(ax, n_days, n_shifts, row_labels=[f"Dita {d + 1}" for d in range(n_days)],
               col_labels=list(shift_names))
    return _finish(fig, ax, title)


def plot_workload(workload_rows: list[dict], shift_names: list[str],
                  title: str = "Shpërndarja e ngarkesës"):
    """Shtylla të grumbulluara: turnet e secilit punëtor sipas llojit të turnit."""
    workers = [row["Punëtori"] for row in workload_rows]
    fig, ax = plt.subplots(figsize=(max(4.5, 0.7 * len(workers) + 2), 3.5))
    bottom = [0] * len(workers)
    for k, name in enumerate(shift_names):
        values = [row[name] for row in workload_rows]
        ax.bar(workers, values, bottom=bottom, label=name,
               color=_SHIFT_COLORS[k % len(_SHIFT_COLORS)], edgecolor="#444444", linewidth=0.5)
        bottom = [b + v for b, v in zip(bottom, values)]
    for x, total in enumerate(bottom):
        ax.text(x, total + 0.1, str(total), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Numri i turneve", fontsize=LABEL_SIZE)
    ax.set_ylim(0, max(bottom, default=0) + 1.5)
    ax.legend(fontsize=8, ncol=len(shift_names), loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, ax, title)


# ---------------------------------------------------------------------------
# Pema e kërkimit
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    "generated": "#d9d9d9",
    "explored": "#9ecae1",
    "duplicate": "#fdd0a2",
    "invalid": INVALID_COLOR,
    "backtrack": "#bdbdbd",
    "solution": VALID_COLOR,
    "goal": "#31a354",
}

DEFAULT_MAX_DISPLAY_NODES = 150


def plot_search_tree(graph, title: str = "Search Tree", font_size: int = 7,
                     show_labels: bool = True,
                     max_display_nodes: int = DEFAULT_MAX_DISPLAY_NODES):
    """
    Pema reale e kërkimit (graf NetworkX nga SearchTreeRecorder).
    Ngjyra = statusi i nyjes; skajet e rrugës së zgjidhjes janë të trasha jeshile.
    Nëse grafi ka më shumë se `max_display_nodes` nyje, shfaqen nyjet e para
    sipas renditjes së gjenerimit (id-të), gjë që ruan lidhjet reale prind–fëmijë.
    """
    from utils.search_tree import STATUS_LABELS, tree_layout

    total_nodes = graph.number_of_nodes()
    if total_nodes > max_display_nodes:
        keep = sorted(graph.nodes)[:max_display_nodes]
        graph = graph.subgraph(keep).copy()
        title = f"{title} (shfaqen {max_display_nodes} nga {total_nodes} nyje)"

    import networkx as nx

    positions = tree_layout(graph)
    n_nodes = graph.number_of_nodes()
    max_depth = max((-y for _, y in positions.values()), default=0)
    width = max(6.0, min(28.0, 0.9 * max((x for x, _ in positions.values()), default=0) + 2))
    height = max(3.0, min(20.0, 1.3 * (max_depth + 1)))
    fig, ax = plt.subplots(figsize=(width, height))

    on_path = {n for n in graph.nodes if graph.nodes[n].get("status") in ("solution", "goal")}
    path_edges = [(u, v) for u, v in graph.edges if u in on_path and v in on_path]
    other_edges = [e for e in graph.edges if e not in set(path_edges)]
    nx.draw_networkx_edges(graph, positions, edgelist=other_edges, ax=ax, arrows=False,
                           edge_color="#9a9a9a", width=0.8)
    nx.draw_networkx_edges(graph, positions, edgelist=path_edges, ax=ax, arrows=False,
                           edge_color="#238b45", width=2.6)

    colors = [_STATUS_COLORS.get(graph.nodes[n].get("status"), "#ffffff") for n in graph.nodes]
    node_size = 900 if n_nodes <= 40 else 450 if n_nodes <= 120 else 200
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
    ax.axis("off")
    return _finish(fig, ax, title)
