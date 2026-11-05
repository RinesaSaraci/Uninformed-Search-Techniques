"""
Ndërfaqja Streamlit për Detyrën 2 – Sudoku Solver (BFS me Backtracking).
"""

import pandas as pd
import streamlit as st

from tasks.sudoku import (
    InvalidSudokuError,
    compute_landscape,
    copy_grid,
    get_puzzle,
    is_complete_solution,
    list_difficulties,
    objective_function,
    preserves_clues,
    solution_states,
    solve_sudoku,
    validate_grid,
)
from ui.common import (
    init_state,
    model_section,
    page_header,
    show_errors,
    show_figure,
    show_statistics,
    show_termination,
)
from utils.statistics import format_big_number
from utils.visualization import plot_sudoku

DIFFICULTY_LABELS = dict(list_difficulties())
COLUMNS = [str(c) for c in range(1, 10)]

DEFAULTS = {
    "sd_difficulty": "easy",
    "sd_max_explored": 200_000,
    "sd_max_generated": 500_000,
    "sd_timeout": 20.0,
    "sd_use_mrv": True,
}


def _init() -> None:
    init_state(DEFAULTS)
    if "sd_original" not in st.session_state:
        st.session_state["sd_original"] = get_puzzle(st.session_state["sd_difficulty"])
        st.session_state["sd_grid"] = copy_grid(st.session_state["sd_original"])
        st.session_state["sd_editor_version"] = 0


def _clear_results() -> None:
    for key in ("sd_result", "sd_errors", "sd_validation", "sd_step"):
        st.session_state.pop(key, None)


def _load_puzzle() -> None:
    """Gjenero enigmën: ngarkon enigmën e verifikuar të vështirësisë së zgjedhur."""
    puzzle = get_puzzle(st.session_state["sd_difficulty"])
    st.session_state["sd_original"] = puzzle
    st.session_state["sd_grid"] = copy_grid(puzzle)
    st.session_state["sd_editor_version"] += 1
    _clear_results()


def _reset() -> None:
    """Rivendos: kthen enigmën e fundit të ngarkuar dhe fshin rezultatet."""
    st.session_state["sd_grid"] = copy_grid(st.session_state["sd_original"])
    st.session_state["sd_editor_version"] += 1
    _clear_results()


def _clear_board() -> None:
    """Tabelë bosh për futje manuale."""
    st.session_state["sd_original"] = [[0] * 9 for _ in range(9)]
    st.session_state["sd_grid"] = [[0] * 9 for _ in range(9)]
    st.session_state["sd_editor_version"] += 1
    _clear_results()


def _grid_to_frame(grid) -> pd.DataFrame:
    return pd.DataFrame(grid, columns=COLUMNS, index=[str(r) for r in range(1, 10)])


def _frame_to_grid(frame: pd.DataFrame):
    """Konverton tabelën e redaktuar në listë 9×9; qelizat bosh → 0."""
    grid = []
    for _, row in frame.iterrows():
        values = []
        for value in row.tolist():
            if value is None or pd.isna(value):
                values.append(0)
            elif isinstance(value, float) and value.is_integer():
                values.append(int(value))
            else:
                values.append(value if isinstance(value, int) else value)
        grid.append(values)
    return grid


def _current_grid():
    edited = st.session_state.get(f"sd_editor_{st.session_state['sd_editor_version']}_df")
    return edited if edited is not None else st.session_state["sd_grid"]


def _solve() -> None:
    grid = _current_grid()
    st.session_state.pop("sd_validation", None)
    st.session_state.pop("sd_step", None)
    errors = validate_grid(grid)
    if errors:
        st.session_state.pop("sd_result", None)
        st.session_state["sd_errors"] = ["Enigma e futur nuk është e vlefshme:"] + errors
        return
    try:
        result = solve_sudoku(
            grid,
            max_explored=st.session_state["sd_max_explored"],
            max_generated=st.session_state["sd_max_generated"],
            timeout_seconds=float(st.session_state["sd_timeout"]),
            use_mrv=st.session_state["sd_use_mrv"],
            trace_limit=200,
        )
        st.session_state["sd_result"] = result
        st.session_state["sd_errors"] = []
    except InvalidSudokuError as error:
        st.session_state.pop("sd_result", None)
        st.session_state["sd_errors"] = error.errors


def _validate() -> None:
    """Valido: kontrollon tabelën aktuale dhe, nëse ka, zgjidhjen e gjetur."""
    grid = _current_grid()
    errors = validate_grid(grid)
    report = {"grid_errors": errors}
    if not errors:
        report["grid_objective"] = objective_function(grid)
    result = st.session_state.get("sd_result")
    if result is not None and result.solution is not None:
        report["solution_complete"] = is_complete_solution(result.solution)
        report["solution_preserves"] = preserves_clues(result.puzzle, result.solution)
    st.session_state["sd_validation"] = report


def _theory_sections() -> None:
    with st.expander("1. Përshkrimi i problemit"):
        st.write(
            "Sudoku është një tabelë 9×9 e ndarë në nëntë nënrrjeta 3×3. Disa qeliza janë "
            "të plotësuara që në fillim (shifrat origjinale). Duhet të plotësohen qelizat "
            "bosh me shifrat 1–9 ashtu që asnjë shifër të mos përsëritet në rresht, kolonë "
            "ose nënrrjetë 3×3. Programi ofron tri nivele: E lehtë, Mesatare, E vështirë."
        )
    with st.expander("2–3. Modelimi dhe reprezentimi i problemit"):
        model_section(
            variables="qelizat bosh të tabelës (0 = bosh).",
            domain="{1, 2, ..., 9}.",
            constraints=["Asnjë vlerë e përsëritur në rresht.",
                         "Asnjë vlerë e përsëritur në kolonë.",
                         "Asnjë vlerë e përsëritur në nënrrjetën 3×3."],
            initial="enigma origjinale.",
            goal="të 81 qelizat janë plotësuar dhe të gjitha kufizimet plotësohen.",
        )
    with st.expander("4. Objective Function"):
        st.markdown(
            "- **f(s)** = numri i shkeljeve të kufizimeve: çifte qelizash me të njëjtën vlerë "
            "në të njëjtin rresht, kolonë ose nënrrjetë.\n"
            "- **F(s)** = numri i qelizave të plotësuara."
        )
        st.latex(r"s^* :\; f(s^*) = 0 \;\wedge\; F(s^*) = 81")
        st.write("BFS krijon vetëm gjendje me f(s) = 0; çdo nivel e rrit F(s) me 1.")
    with st.expander("6. Implementimi i algoritmit (BFS me Backtracking)"):
        st.markdown(
            "1. Radha FIFO (`deque`) inicializohet me enigmën origjinale.\n"
            "2. Merret gjendja më e vjetër (`popleft`) dhe kontrollohet Goal Test.\n"
            "3. Zgjidhet një qelizë bosh (MRV – qeliza me më pak vlera të mundshme).\n"
            "4. Për çdo vlerë të vlefshme krijohet një gjendje fëmijë dhe shtohet **në fund** të radhës.\n"
            "5. Vlerat që shkelin kufizimet **refuzohen** pa krijuar gjendje.\n"
            "6. **Backtracking:** nëse një qelizë nuk ka asnjë vlerë të mundshme (dead end), "
            "dega braktiset dhe kërkimi vazhdon me alternativat e tjera në radhë.\n\n"
            "Çdo kalim cakton saktësisht një qelizë, prandaj niveli i BFS = numri i caktimeve. "
            "MRV zgjedh vetëm qelizën; radha mbetet FIFO."
        )


def _landscape(grid) -> None:
    with st.expander("5. Search Space / Landscape Calculation", expanded=True):
        errors = validate_grid(grid)
        if errors:
            st.warning("Landscape nuk mund të llogaritet: tabela aktuale nuk është e vlefshme.")
            return
        land = compute_landscape(grid)
        c = st.columns(5)
        c[0].metric("Rrjeta", land["grid_size"])
        c[1].metric("Qeliza gjithsej", land["total_cells"])
        c[2].metric("Qeliza fikse", land["fixed_cells"])
        c[3].metric("Qeliza bosh (k)", land["empty_cells"])
        c[4].metric("Madhësia e domenit", land["domain_size"])
        st.latex(rf"9^k = 9^{{{land['empty_cells']}}} \approx 10^{{{land['unrestricted_upper_bound_log10']:.1f}}}")
        st.write(f"Kufiri i sipërm pa kufizime: **{format_big_number(land['unrestricted_upper_bound'])}** "
                 "mbushje të mundshme.")
        st.write("Pas filtrimit fillestar (vetëm vlerat e lejuara në çdo qelizë): "
                 f"**{format_big_number(land['initial_candidate_product'])}**.")
        if land["cells_without_candidates"]:
            st.warning(f"{land['cells_without_candidates']} qeliza bosh nuk kanë asnjë vlerë të "
                       "mundshme: enigma nuk ka zgjidhje (BFS do ta provojë).")
        st.caption("Këto janë kufij teorikë. Numri real i gjendjeve të gjeneruara dhe të "
                   "eksploruara shfaqet te statistikat pas ekzekutimit.")


def _board_input():
    st.subheader("Tabela")
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.selectbox("Vështirësia", list(DIFFICULTY_LABELS),
                 format_func=lambda key: DIFFICULTY_LABELS[key], key="sd_difficulty")
    c2.button("Gjenero enigmën", on_click=_load_puzzle, key="sd_load")
    c3.button("Tabelë bosh (futje manuale)", on_click=_clear_board,
              key="sd_clear")

    left, right = st.columns([1, 1])
    with left:
        st.caption("Ndryshoni qelizat drejtpërdrejt (0 ose bosh = qelizë bosh). Për futje "
                   "manuale jepni një enigmë me shumë shifra fillestare: BFS nuk mund ta "
                   "plotësojë një tabelë (pothuajse) bosh brenda kufijve, sepse radha shpërthen.")
        editor_key = f"sd_editor_{st.session_state['sd_editor_version']}"
        edited = st.data_editor(
            _grid_to_frame(st.session_state["sd_grid"]),
            key=editor_key,
            column_config={col: st.column_config.NumberColumn(col, min_value=0, max_value=9,
                                                              step=1, width="small")
                           for col in COLUMNS},
        )
        grid = _frame_to_grid(edited)
        st.session_state[f"{editor_key}_df"] = grid
    with right:
        show_figure(plot_sudoku(grid, original=st.session_state["sd_original"],
                                title="Tabela aktuale"))
    return grid


def _controls() -> None:
    with st.expander("Kufijtë e kërkimit"):
        c1, c2, c3 = st.columns(3)
        c1.number_input("Maks. gjendje të eksploruara", min_value=100, max_value=2_000_000,
                        step=10_000, key="sd_max_explored")
        c2.number_input("Maks. gjendje të gjeneruara", min_value=100, max_value=5_000_000,
                        step=10_000, key="sd_max_generated")
        c3.number_input("Timeout (sekonda)", min_value=1.0, max_value=120.0, step=1.0,
                        key="sd_timeout")
        st.checkbox("Përdor MRV për zgjedhjen e qelizës (radha mbetet FIFO)", key="sd_use_mrv")
        st.caption("Pa MRV, BFS mund të gjenerojë qindra mijëra gjendje edhe për enigma të "
                   "mesme – kufijtë e mbrojnë ekzekutimin. Algoritmi nuk ndërrohet kurrë.")
    b1, b2, b3 = st.columns(3)
    b1.button("▶ Zgjidh me BFS", type="primary", on_click=_solve,
              key="sd_solve")
    b2.button("✔ Valido", on_click=_validate, key="sd_validate")
    b3.button("↺ Rivendos", on_click=_reset, key="sd_reset")


def _validation_report() -> None:
    report = st.session_state.get("sd_validation")
    if report is None:
        return
    st.subheader("10. Validimi")
    if report["grid_errors"]:
        st.error("Tabela aktuale ka shkelje:")
        for message in report["grid_errors"]:
            st.write(f"- {message}")
    else:
        obj = report["grid_objective"]
        st.success(f"Tabela aktuale nuk ka shkelje (f(s) = {obj['violations']}); "
                   f"qeliza të plotësuara F(s) = {obj['filled']}/81"
                   + (" – është zgjidhje e plotë." if obj["is_goal"] else "."))
    if "solution_complete" in report:
        ok = report["solution_complete"] and report["solution_preserves"]
        (st.success if ok else st.error)(
            f"Zgjidhja e BFS: e plotë dhe e vlefshme = {report['solution_complete']}, "
            f"ruan shifrat origjinale = {report['solution_preserves']}.")


def _results() -> None:
    show_errors(st.session_state.get("sd_errors", []))
    result = st.session_state.get("sd_result")
    if result is None:
        return
    st.subheader("7. Ekzekutimi dhe rezultatet")
    show_termination(result.stats)
    if result.solved:
        st.success(result.message)
    elif result.proven_unsolvable:
        st.error(result.message)
    else:
        st.warning(result.message)

    st.subheader("8. Paraqitja vizuale")
    left, right = st.columns(2)
    with left:
        show_figure(plot_sudoku(result.puzzle, title="Enigma origjinale"))
    with right:
        if result.solution is not None:
            show_figure(plot_sudoku(result.solution, original=result.puzzle,
                                    title="Zgjidhja (blu = vendosur nga BFS)"))
        else:
            show_figure(plot_sudoku(result.deepest_board, original=result.puzzle,
                                    title="Gjendja më e thellë e arritur (e pjesshme)"))

    if result.solution is not None and result.solution_path:
        with st.expander("Hapat e zgjidhjes (gjendjet e ndërmjetme)"):
            states = solution_states(result.puzzle, result.solution_path)
            step = st.slider("Hapi", 0, len(states) - 1, 0, key="sd_step")
            highlight = result.solution_path[step - 1][:2] if step > 0 else None
            caption = ("Enigma origjinale" if step == 0 else
                       f"Hapi {step}: qeliza ({highlight[0] + 1}, {highlight[1] + 1}) = "
                       f"{result.solution_path[step - 1][2]}")
            col, _ = st.columns([1, 1])
            with col:
                show_figure(plot_sudoku(states[step], original=result.puzzle, title=caption,
                                        highlight=highlight))

    st.subheader("9. Statistikat e performancës")
    io, fo = result.initial_objective, result.final_objective
    c = st.columns(4)
    c[0].metric("Madhësia maks. e radhës", result.max_queue_size)
    c[1].metric("Thellësia e zgjidhjes", result.solution_depth if result.solution_depth is not None else "–")
    c[2].metric("Qeliza bosh fillimisht", result.initial_empty)
    c[3].metric("Qeliza bosh të mbetura", result.remaining_empty)
    st.caption(f"Objective: fillimi f(s) = {io['violations']}, F(s) = {io['filled']} → "
               f"fundi f(s) = {fo['violations']}, F(s) = {fo['filled']}.")
    show_statistics(result.stats, extra={
        "Madhësia maksimale e radhës": (result.max_queue_size,
                                        "Numri më i madh i gjendjeve në radhë njëkohësisht."),
        "Dead ends (degë të braktisura)": (result.dead_ends,
                                           "Gjendje ku një qelizë nuk kishte asnjë vlerë."),
        "Thellësia e zgjidhjes": (result.solution_depth, "Numri i caktimeve deri te zgjidhja."),
    })
    if st.checkbox("Shfaq hapat e kërkimit BFS (200 zgjerimet e para)", key="sd_show_trace"):
        st.dataframe(
            [{"Radha e zgjerimit": e["order"], "Niveli BFS": e["depth"],
              "Qeliza": f"({e['cell'][0] + 1}, {e['cell'][1] + 1})" if e["cell"] else "–",
              "Vlerat e mundshme": ", ".join(map(str, e["candidates"])) or "asnjë",
              "Madhësia e radhës": e["queue_size"],
              "Dead end": "Po" if e["dead_end"] else "Jo"} for e in result.trace],
            hide_index=True, 
        )
        st.caption("Nivelet e BFS nuk zvogëlohen kurrë: kjo tregon renditjen FIFO.")


def render() -> None:
    _init()
    page_header("Detyra 2 – Sudoku Solver", "Breadth-First Search (BFS) me Backtracking",
                "Zgjidhja e Sudoku-së me radhë FIFO, nivel pas niveli.")
    _theory_sections()
    grid = _board_input()
    _landscape(grid)
    _controls()
    _validation_report()
    _results()
