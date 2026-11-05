"""
Ndërfaqja Streamlit për Detyrën 3 – Latin Square (IDDFS me Backtracking).
"""

import streamlit as st

from tasks.latin_square import (
    MAX_N,
    MIN_N,
    InvalidLatinSquareConfigError,
    compute_landscape,
    format_matrix,
    iterations_to_rows,
    objective_function,
    solution_states,
    solve_latin_square,
    verify_latin_square,
)
from ui.common import (
    init_state,
    model_section,
    page_header,
    reset_state,
    show_errors,
    show_figure,
    show_statistics,
    show_termination,
    show_validation,
)
from utils.statistics import format_big_number, format_count
from utils.visualization import plot_latin_square

DEFAULTS = {
    "ls_n": 5,
    "ls_symmetry": True,
    "ls_max_nodes": 2_000_000,
    "ls_timeout": 20.0,
}
RESULT_KEYS = ("ls_result", "ls_errors", "ls_validation", "ls_step")


def _run() -> None:
    st.session_state.pop("ls_validation", None)
    st.session_state.pop("ls_step", None)        # rrëshqitësi rifillon për rezultatin e ri
    try:
        st.session_state["ls_result"] = solve_latin_square(
            n=st.session_state["ls_n"],
            max_nodes=st.session_state["ls_max_nodes"],
            timeout_seconds=float(st.session_state["ls_timeout"]),
            symmetry_breaking=st.session_state["ls_symmetry"],
        )
        st.session_state["ls_errors"] = []
    except InvalidLatinSquareConfigError as error:
        st.session_state.pop("ls_result", None)
        st.session_state["ls_errors"] = error.errors


def _validate() -> None:
    result = st.session_state.get("ls_result")
    if result is not None and result.solution is not None:
        st.session_state["ls_validation"] = verify_latin_square(result.solution, result.n)


def _theory_sections() -> None:
    with st.expander("1. Përshkrimi i problemit"):
        st.write(
            "Një Latin Square është një rrjetë n×n e mbushur me n numra të ndryshëm, ku secili "
            "shfaqet saktësisht një herë në çdo rresht dhe në çdo kolonë. Për hyrjen n, "
            "programi duhet të printojë një matricë n×n me numrat 1..n që plotëson këtë kusht."
        )
    with st.expander("2–3. Modelimi dhe reprezentimi i problemit"):
        model_section(
            variables="çdo qelizë X[i][j] e matricës (n² variabla).",
            domain="{1, 2, ..., n}.",
            constraints=["Çdo rresht përmban secilin numër 1..n saktësisht një herë.",
                         "Çdo kolonë përmban secilin numër 1..n saktësisht një herë."],
            initial="matrica bosh n×n.",
            goal="të gjitha qelizat janë caktuar dhe rreshtat e kolonat janë të vlefshme.",
        )
    with st.expander("4. Objective Function"):
        st.markdown(
            "- **f(s)** = numri i shkeljeve: çifte qelizash me të njëjtën vlerë në të njëjtin "
            "rresht ose kolonë.\n- **F(s)** = numri i qelizave të caktuara."
        )
        st.latex(r"s^* :\; f(s^*) = 0 \;\wedge\; F(s^*) = n^2")
    with st.expander("6. Implementimi i algoritmit (IDDFS me Backtracking)"):
        st.markdown(
            "- **Thellësia** = numri i qelizave të caktuara (rresht pas rreshti); qëllimi është "
            "gjithmonë në thellësinë n².\n"
            "- Për **limit = 0, 1, 2, ...** ekzekutohet një **Depth-Limited DFS** nga matrica bosh; "
            "nyjet në thellësinë e limitit nuk zgjerohen (*cutoff*).\n"
            "- Nëse qëllimi nuk gjendet, limiti rritet me 1 dhe kërkimi **rifillon nga rrënja**.\n"
            "- Vlera që përsëritet në rresht/kolonë **refuzohet**; pas dështimit të një dege, "
            "caktimi **zhbëhet (Backtracking)** dhe provohet vlera tjetër.\n"
            "- **Thyerja e simetrisë** (opsionale): rreshti dhe kolona e parë kufizohen në 1..n. "
            "Çdo Latin Square kthehet në këtë formë vetëm duke ndërruar rreshta dhe kolona, "
            "prandaj nuk humbet ekzistenca e zgjidhjes. Edhe këto qeliza i cakton kërkimi.\n"
            "- Matrica gjendet gjithmonë nga kërkimi – **nuk përdoret ndërtim ciklik**."
        )


def _configuration() -> None:
    st.subheader("Konfigurimi")
    c1, c2, c3 = st.columns(3)
    c1.number_input(f"Madhësia n ({MIN_N}–{MAX_N})", min_value=MIN_N, max_value=MAX_N, step=1,
                    key="ls_n")
    c2.number_input("Kufiri i nyjeve", min_value=100, max_value=10_000_000, step=100_000,
                    key="ls_max_nodes")
    c3.number_input("Timeout (sekonda)", min_value=1.0, max_value=120.0, step=1.0,
                    key="ls_timeout")
    st.checkbox("Thyerja e simetrisë (forma e reduktuar)", key="ls_symmetry")
    st.caption("Pa thyerje simetrie, IDDFS për n ≥ 5 zakonisht arrin kufijtë. "
               "Me thyerje, n ≤ 6 zgjidhet; n = 7 zakonisht ndërpritet nga kufijtë.")


def _landscape() -> None:
    with st.expander("5. Search Space / Landscape Calculation", expanded=True):
        land = compute_landscape(st.session_state["ls_n"])
        c = st.columns(4)
        c[0].metric("Dimensionet", land["dimensions"])
        c[1].metric("Qeliza (n²)", land["cells"])
        c[2].metric("Madhësia e domenit", land["domain_size"])
        c[3].metric("Thellësia e qëllimit", land["goal_depth"])
        n = land["n"]
        rows = [
            {"Hapësira": f"Caktime pa kufizime n^(n²) = {n}^{n * n}",
             "Madhësia": format_big_number(land["unrestricted_assignments"])},
            {"Hapësira": f"Vetëm rreshtat si permutacione (n!)^n = ({n}!)^{n}",
             "Madhësia": format_big_number(land["row_permutation_bound"])},
            {"Hapësira": "Latin Squares të vlefshme (numër i saktë)",
             "Madhësia": format_count(land["latin_squares"]) if land["latin_squares"] else "–"},
            {"Hapësira": "Latin Squares të reduktuara (numër i saktë)",
             "Madhësia": format_count(land["reduced_latin_squares"])
             if land["reduced_latin_squares"] else "–"},
        ]
        st.dataframe(rows, hide_index=True)
        st.caption("Hapësira pa kufizime përfshin të gjitha mbushjet (edhe të pavlefshmet); "
                   "Latin Squares të vlefshme janë vetëm një pjesë shumë e vogël e saj. "
                   "Numrat e saktë (OEIS A002860, A000315) shfaqen vetëm për n ≤ 7.")


def _results() -> None:
    show_errors(st.session_state.get("ls_errors", []))
    result = st.session_state.get("ls_result")
    if result is None:
        return
    st.subheader("7. Ekzekutimi dhe rezultatet")
    show_termination(result.stats)
    (st.success if result.solved else st.warning)(result.message)

    c = st.columns(4)
    c[0].metric("Iteracione", len(result.iterations))
    c[1].metric("Limiti i fundit i thellësisë", result.final_depth_limit)
    c[2].metric("Nyje gjithsej", format_count(result.stats.explored))
    c[3].metric("Koha", f"{result.stats.elapsed_seconds:.3f} s")

    if result.solution is not None:
        st.subheader("8. Paraqitja vizuale")
        left, right = st.columns([2, 1])
        with left:
            show_figure(plot_latin_square(result.solution,
                                          title=f"Latin Square {result.n}×{result.n}"))
        with right:
            st.markdown("**Dalja (format teksti):**")
            st.code(format_matrix(result.solution), language=None)
            obj = objective_function(result.solution, result.n)
            st.write(f"Objective: f(s) = {obj['violations']}, F(s) = {obj['filled']}/{result.n ** 2}")

        st.subheader("10. Validimi i pavarur")
        validation = st.session_state.get("ls_validation", (result.is_valid,
                                                            result.validation_errors))
        show_validation(validation[0], validation[1],
                        "Matrica ka dimensione të sakta, vlera 1..n, dhe çdo rresht e kolonë "
                        "përmban secilën vlerë saktësisht një herë.")

        with st.expander("Gjendjet e ndërmjetme (rruga e iteracionit të fundit)"):
            states = solution_states(result.n, result.solution_path)
            step = st.slider("Thellësia (qeliza të caktuara)", 0, len(states) - 1,
                             len(states) - 1, key="ls_step")
            highlight = result.solution_path[step - 1][:2] if step > 0 else None
            col, _ = st.columns([1, 1])
            with col:
                show_figure(plot_latin_square(states[step], title=f"Thellësia {step}",
                                              highlight=highlight))

    st.subheader("9. Statistikat e performancës")
    if st.checkbox("Shfaq iteracionet e IDDFS", value=True, key="ls_show_iterations"):
        rows = iterations_to_rows(result.iterations)
        st.dataframe(rows, hide_index=True)
        st.bar_chart({"Nyje të eksploruara": [r["Të eksploruara"] for r in rows]},
                     x_label="Iteracioni (limiti = iteracioni − 1)",
                     y_label="Nyje të eksploruara")
        st.caption("Çdo iteracion rifillon nga matrica bosh, prandaj nyjet e cekëta "
                   "rivizitohen – kjo është sjellja normale e IDDFS.")
    show_statistics(result.stats, extra={
        "Iteracione": (len(result.iterations), "Numri i Depth-Limited DFS të ekzekutuara."),
        "Limiti i fundit i thellësisë": (result.final_depth_limit, "Limiti i iteracionit të fundit."),
        "Cutoffs gjithsej": (sum(it.cutoffs for it in result.iterations),
                             "Nyje të ndalura nga limiti i thellësisë."),
    })


def render() -> None:
    init_state(DEFAULTS)
    page_header("Detyra 3 – Latin Square",
                "Iterative Deepening Depth-First Search (IDDFS) me Backtracking",
                "Ndërtimi i një Latin Square n×n me thellim iterativ.")
    _theory_sections()
    _configuration()
    _landscape()
    b1, b2, b3 = st.columns(3)
    b1.button("▶ Gjenero zgjidhjen", type="primary", on_click=_run, 
              key="ls_run")
    b2.button("✔ Valido zgjidhjen", on_click=_validate,
              key="ls_validate", disabled="ls_result" not in st.session_state)
    b3.button("↺ Rivendos", on_click=reset_state, args=(DEFAULTS, RESULT_KEYS),
              key="ls_reset")
    _results()
