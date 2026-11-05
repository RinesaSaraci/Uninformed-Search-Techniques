"""
Ndërfaqja Streamlit për Detyrën 1 – Social Golfers Problem (DFS me Backtracking).
"""

import streamlit as st

from tasks.social_golfers import (
    PIVOT_RULES,
    InvalidConfigurationError,
    compute_landscape,
    pair_week_matrix,
    schedule_to_rows,
    solve_social_golfers,
    verify_schedule,
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
from utils.validation import validate_golfer_config
from utils.visualization import plot_golfers_schedule, plot_pair_matrix

DEFAULTS = {
    "sg_n": 32,
    "sg_g": 8,
    "sg_p": 4,
    "sg_weeks": 10,
    "sg_max_nodes": 200_000,
    "sg_timeout": 20.0,
    "sg_use_seed": True,
    "sg_seed": 1,
    "sg_pivot": "mrv",
}
RESULT_KEYS = ("sg_result", "sg_errors", "sg_validation")

PIVOT_LABELS = {
    "mrv": "MRV – golferi me më pak partnerë të mundshëm (fail-first)",
    "lowest": "Golferi me numrin më të vogël",
}


def _run() -> None:
    st.session_state.pop("sg_validation", None)
    try:
        result = solve_social_golfers(
            num_golfers=st.session_state["sg_n"],
            num_groups=st.session_state["sg_g"],
            group_size=st.session_state["sg_p"],
            requested_weeks=st.session_state["sg_weeks"],
            max_nodes=st.session_state["sg_max_nodes"],
            timeout_seconds=float(st.session_state["sg_timeout"]),
            seed=st.session_state["sg_seed"] if st.session_state["sg_use_seed"] else None,
            pivot_rule=st.session_state["sg_pivot"],
            trace_limit=300,
        )
        st.session_state["sg_result"] = result
        st.session_state["sg_errors"] = []
    except InvalidConfigurationError as error:
        st.session_state.pop("sg_result", None)
        st.session_state["sg_errors"] = error.errors


def _validate() -> None:
    result = st.session_state.get("sg_result")
    if result is not None:
        st.session_state["sg_validation"] = verify_schedule(
            result.schedule, result.num_golfers, result.num_groups, result.group_size)


def _theory_sections() -> None:
    with st.expander("1. Përshkrimi i problemit"):
        st.write(
            "n golferë luajnë golf një herë në javë, të ndarë në g grupe me nga p golferë "
            "(n = g × p). Qëllimi është të planifikohen sa më shumë javë, ashtu që asnjë "
            "çift golferësh të mos luajë në të njëjtin grup më shumë se një herë. "
            "Konfigurimi standard: 32 golferë, 8 grupe, 4 golferë për grup."
        )
    with st.expander("2–3. Modelimi dhe reprezentimi i problemit"):
        model_section(
            variables="grupet e çdo jave G[w][j] (java w, grupi j).",
            domain="kombinimet e p golferëve nga n golferë, pra C(n, p) grupe të mundshme.",
            constraints=[
                "Çdo golfer shfaqet saktësisht një herë në javë.",
                "Çdo grup ka saktësisht p golferë.",
                "Asnjë çift golferësh nuk luan bashkë më shumë se një herë nëpër javë.",
            ],
            initial="orar bosh (java 1 fiksohet në formën kanonike {1..p}, {p+1..2p}, ... – "
                    "thyerje e simetrisë që nuk humb zgjidhje).",
            goal="numri i kërkuar i javëve të plota pa shkelur kufizimet.",
        )
    with st.expander("4. Objective Function"):
        st.latex(r"\max\; w \quad \text{me kusht që çdo çift } \{a,b\} \text{ të takohet } \le 1 \text{ herë}")
        st.write("Maksimizohet numri i javëve të vlefshme w pa shkelur asnjë kufizim.")
    with st.expander("6. Implementimi i algoritmit (DFS me Backtracking)"):
        st.markdown(
            "- Një nyje e pemës = orar i pjesshëm; një degë = shtimi i një grupi në javën aktuale.\n"
            "- Grupet gjenerohen golfer pas golferi vetëm për nyjen aktuale (pa gjeneruar "
            "paraprakisht C(n,p) grupe apo ndarje javore).\n"
            "- Grupi që krijon një çift të përsëritur **refuzohet**.\n"
            "- **Forward checking:** çdo golfer i pacaktuar duhet të ketë ende ≥ p−1 partnerë të mundshëm.\n"
            "- Kur nuk ka alternativë, grupi i fundit **zhbëhet (Backtracking)** – edhe nëpër javë.\n"
            "- Strukturat: bitmask-a për golferët e përdorur dhe për partnerët e takuar."
        )


def _landscape_section() -> None:
    with st.expander("5. Search Space / Landscape Calculation", expanded=True):
        errors = validate_golfer_config(st.session_state["sg_n"], st.session_state["sg_g"],
                                        st.session_state["sg_p"])
        if errors:
            show_errors(errors)
            return
        land = compute_landscape(st.session_state["sg_n"], st.session_state["sg_g"],
                                 st.session_state["sg_p"])
        rows = [
            {"Madhësia": "Golferë gjithsej (n)", "Vlera": land["num_golfers"]},
            {"Madhësia": "Numri i grupeve (g)", "Vlera": land["num_groups"]},
            {"Madhësia": "Madhësia e grupit (p)", "Vlera": land["group_size"]},
            {"Madhësia": "Grupe të mundshme C(n,p)", "Vlera": format_count(land["possible_groups"])},
            {"Madhësia": "Çifte në grup C(p,2)", "Vlera": land["pairs_per_group"]},
            {"Madhësia": "Çifte në javë g·C(p,2)", "Vlera": land["pairs_per_week"]},
            {"Madhësia": "Çifte të mundshme C(n,2)", "Vlera": land["total_pairs"]},
            {"Madhësia": "Ndarje të ndryshme të një jave",
             "Vlera": format_big_number(land["weekly_partitions"])},
        ]
        st.dataframe([{k: str(v) for k, v in r.items()} for r in rows], hide_index=True,
                     use_container_width=True)
        st.latex(
            rf"w \le \left\lfloor \frac{{C(n,2)}}{{g\cdot C(p,2)}} \right\rfloor = "
            rf"\left\lfloor \frac{{{land['total_pairs']}}}{{{land['pairs_per_week']}}} "
            rf"\right\rfloor = {land['pair_bound']}"
        )
        st.caption(
            f"Kufiri sipas golferit: ⌊(n−1)/(p−1)⌋ = {land['golfer_bound']}. "
            f"Kufiri i sipërm teorik = {land['upper_bound']} javë."
        )
        st.info(
            f"**{land['upper_bound']} javë është kufi i sipërm teorik**, jo rezultat i "
            "garantuar: ai tregon se më shumë javë janë të pamundura, por nuk provon që "
            "ky numër arrihet. Vetëm një orar i ndërtuar dhe i validuar e provon."
        )


def _configuration() -> None:
    st.subheader("Konfigurimi")
    c1, c2, c3 = st.columns(3)
    c1.number_input("Numri i golferëve (n)", min_value=2, max_value=64, step=1, key="sg_n")
    c2.number_input("Numri i grupeve (g)", min_value=1, max_value=32, step=1, key="sg_g")
    c3.number_input("Madhësia e grupit (p)", min_value=2, max_value=16, step=1, key="sg_p")
    c4, c5, c6 = st.columns(3)
    c4.number_input("Numri i kërkuar i javëve", min_value=1, max_value=63, step=1, key="sg_weeks")
    c5.number_input("Kufiri i nyjeve", min_value=100, max_value=5_000_000, step=10_000,
                    key="sg_max_nodes")
    c6.number_input("Timeout (sekonda)", min_value=1.0, max_value=120.0, step=1.0,
                    key="sg_timeout")
    with st.expander("Opsione të avancuara të renditjes"):
        st.selectbox("Zgjedhja e golferit pivot", PIVOT_RULES,
                     format_func=lambda key: PIVOT_LABELS[key], key="sg_pivot")
        st.checkbox("Renditje e rastësishme e kandidatëve (me seed)", key="sg_use_seed")
        st.number_input("Seed", min_value=0, max_value=10_000, step=1, key="sg_seed",
                        disabled=not st.session_state["sg_use_seed"])
        st.caption(
            "Renditja ndikon vetëm në cilën degë provohet e para; algoritmi mbetet DFS me "
            "Backtracking. Me seed = 1, për 32-8-4 arrihen 7 javë menjëherë; renditja "
            "deterministe arrin 5."
        )


def _week_cards(result) -> None:
    st.markdown("#### Grupet sipas javëve")
    tabs = st.tabs([f"Java {w}" for w in range(1, result.weeks_constructed + 1)])
    per_row = 4
    for tab, week in zip(tabs, result.schedule):
        with tab:
            for start in range(0, len(week), per_row):
                cols = st.columns(per_row)
                for offset, group in enumerate(week[start:start + per_row]):
                    with cols[offset]:
                        with st.container(border=True):
                            st.markdown(f"**Grupi {start + offset + 1}**")
                            st.write(", ".join(str(x) for x in group))


def _results() -> None:
    show_errors(st.session_state.get("sg_errors", []))
    result = st.session_state.get("sg_result")
    if result is None:
        return

    st.subheader("7. Ekzekutimi dhe rezultatet")
    show_termination(result.stats)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Javë të ndërtuara", result.weeks_constructed)
    m2.metric("Javë të kërkuara", result.requested_weeks)
    m3.metric("Kufiri i sipërm teorik", result.upper_bound)
    m4.metric("Objektivi i arritur", "Po" if result.target_reached else "Jo")
    if result.optimality_proven:
        st.success(result.optimality_message)
    else:
        st.warning(result.optimality_message)

    st.subheader("10. Validimi i pavarur")
    validation = st.session_state.get("sg_validation")
    if validation is None:
        show_validation(result.is_valid, result.validation_errors,
                        "Orari kaloi validimin e pavarur: çdo grup ka p golferë, çdo golfer "
                        "shfaqet një herë në javë dhe asnjë çift nuk përsëritet.")
    else:
        show_validation(validation[0], validation[1],
                        "Rivalidimi (i ekzekutuar tani) konfirmon që orari është i vlefshëm.")

    st.subheader("8. Paraqitja vizuale")
    if result.weeks_constructed:
        _week_cards(result)
        st.markdown("#### Tabela e plotë e orarit")
        st.dataframe(schedule_to_rows(result.schedule), hide_index=True, use_container_width=True)
        col_a, col_b = st.columns([3, 2])
        with col_a:
            show_figure(plot_golfers_schedule(result.schedule))
        with col_b:
            show_figure(plot_pair_matrix(pair_week_matrix(result.schedule, result.num_golfers)))
            st.caption("Çdo qelizë me ngjyrë = java kur çifti luajti bashkë; "
                       "qelizat e bardha = çifte që ende nuk janë takuar.")

    st.subheader("9. Statistikat e performancës")
    show_statistics(result.stats, extra={
        "Javë të plota të ndërtuara gjatë kërkimit": (result.stats.valid,
                                                     "Përfshin javën 1 kanonike."),
        "Nyja kur u gjet orari më i mirë": (result.best_found_at_explored,
                                            "Numri i nyjeve të eksploruara në atë moment."),
        "Koha kur u gjet orari më i mirë (s)": (round(result.best_found_at_seconds, 4),
                                                "Sekonda nga fillimi i kërkimit."),
    })
    if st.checkbox("Shfaq hapat e kërkimit (300 ngjarjet e para)", key="sg_show_trace"):
        labels = {"place": "Vendos grupin", "prune": "Krasit (forward checking)",
                  "backtrack": "Backtracking"}
        st.dataframe(
            [{"Ngjarja": labels.get(e["event"], e["event"]), "Java": e["week"],
              "Grupi #": e["group_index"], "Golferët": str(e["group"]),
              "Thellësia": e["depth"]} for e in result.trace],
            hide_index=True, use_container_width=True,
        )


def render() -> None:
    init_state(DEFAULTS)
    page_header(
        "Detyra 1 – Social Golfers Problem",
        "Depth-First Search (DFS) me Backtracking",
        "Ndërtimi i orarit javor të golferëve pa përsëritur asnjë çift.",
    )
    _theory_sections()
    _configuration()
    _landscape_section()

    b1, b2, b3 = st.columns(3)
    b1.button("▶ Ekzekuto algoritmin", type="primary", on_click=_run,
              use_container_width=True, key="sg_run")
    b2.button("✔ Valido zgjidhjen", on_click=_validate, use_container_width=True,
              key="sg_validate", disabled="sg_result" not in st.session_state)
    b3.button("↺ Rivendos", on_click=reset_state, args=(DEFAULTS, RESULT_KEYS),
              use_container_width=True, key="sg_reset")
    _results()
