"""
Ndërfaqja Streamlit për Detyrën 4 – Search Trees:
8-Puzzle (BFS), Missionaries and Cannibals (BFS), Rotating Workforce Scheduling (DFS).
"""

import streamlit as st

from tasks.search_trees import eight_puzzle as ep
from tasks.search_trees import missionaries_cannibals as mc
from tasks.search_trees import workforce_scheduling as wf
from ui.common import (
    init_state,
    page_header,
    reset_state,
    show_errors,
    show_figure,
    show_statistics,
    show_termination,
    show_validation,
)
from utils.statistics import format_big_number
from utils.visualization import (
    plot_eight_puzzle,
    plot_eight_puzzle_path,
    plot_river_state,
    plot_search_tree,
    plot_workforce_schedule,
    plot_workload,
)

SECTIONS = ("8-Puzzle", "Missionaries and Cannibals", "Rotating Workforce Scheduling")


def _tree_section(recorder, title: str, key: str) -> None:
    """Pema reale e kërkimit (nën-bashkësi e kufizuar e nyjeve të gjeneruara)."""
    st.markdown("#### Search Tree")
    if recorder is None or not recorder.nodes:
        st.info("Pema nuk u regjistrua (numri i nyjeve të pemës = 0).")
        return
    summary = recorder.summary()
    show_labels = st.checkbox("Shfaq etiketat e nyjeve", value=True, key=f"{key}_labels")
    show_figure(plot_search_tree(recorder.to_networkx(), title=title, show_labels=show_labels))
    note = (f"Pema përmban {summary['recorded_nodes']} gjendje reale të gjeneruara gjatë "
            "ekzekutimit, me lidhjet e vërteta prind–fëmijë.")
    if summary["truncated"]:
        note += (" **Vizualizim i pjesshëm:** u arrit kufiri i nyjeve të pemës, prandaj "
                 "gjendjet e mëvonshme nuk shfaqen (statistikat përfshijnë të gjitha).")
    st.caption(note)


# ===========================================================================
# 8-Puzzle
# ===========================================================================

EP_DEFAULTS = {
    "ep_mode": "predefined",
    "ep_choice": "default",
    "ep_custom": "1 2 3 4 0 6 7 5 8",
    "ep_max_explored": 200_000,
    "ep_timeout": 20.0,
    "ep_tree_nodes": 60,
}
EP_RESULTS = ("ep_result", "ep_errors", "ep_step")


def _ep_start():
    """Gjendja fillestare e zgjedhur; kthen (gjendja, gabimet)."""
    if st.session_state["ep_mode"] == "predefined":
        return ep.PREDEFINED_STATES[st.session_state["ep_choice"]]["state"], []
    try:
        return ep.parse_state(st.session_state["ep_custom"]), []
    except ep.InvalidPuzzleError as error:
        return None, error.errors


def _ep_run() -> None:
    st.session_state.pop("ep_step", None)
    start, errors = _ep_start()
    if errors:
        st.session_state.pop("ep_result", None)
        st.session_state["ep_errors"] = errors
        return
    try:
        st.session_state["ep_result"] = ep.solve_eight_puzzle(
            start, max_explored=st.session_state["ep_max_explored"],
            timeout_seconds=float(st.session_state["ep_timeout"]),
            tree_nodes=st.session_state["ep_tree_nodes"])
        st.session_state["ep_errors"] = []
    except ep.InvalidPuzzleError as error:
        st.session_state.pop("ep_result", None)
        st.session_state["ep_errors"] = error.errors


def _eight_puzzle() -> None:
    init_state(EP_DEFAULTS)
    st.header("8-Puzzle – Breadth-First Search")
    with st.expander("Përshkrimi, reprezentimi, veprimet dhe kufizimet"):
        st.markdown(
            "**Përshkrimi:** tabelë 3×3 me pllakat 1–8 dhe një pozicion bosh. Pllaka fqinje "
            "rrëshqet në vendin bosh. Qëllimi: renditja 1–8 me vendin bosh në fund, me "
            "**numrin minimal të lëvizjeve**.\n\n"
            "**Reprezentimi i gjendjes:** tuple i pandryshueshëm me 9 numra (0 = bosh), "
            "p.sh. `(1, 2, 3, 4, 0, 6, 7, 5, 8)`.\n\n"
            "**Veprimet:** lëvizja e pozicionit bosh **Lart, Poshtë, Majtas, Djathtas** "
            "(vetëm brenda tabelës).\n\n"
            "**Kufizimet:** gjendja duhet të jetë permutacion i 0–8; vetëm lëvizjet e ligjshme. "
            "Gjendja është e zgjidhshme vetëm kur **numri i inversioneve është çift** – kjo "
            "kontrollohet para kërkimit.\n\n"
            "**Pse BFS:** çdo lëvizje ka kosto 1, prandaj BFS (radhë FIFO, bashkësi vizitimi, "
            "prindër) gjen rrugën më të shkurtër. Hapësira e arritshme: 9!/2 = 181 440 gjendje."
        )

    st.radio("Gjendja fillestare", ("predefined", "custom"), horizontal=True, key="ep_mode",
             format_func=lambda m: "E paracaktuar" if m == "predefined" else "E personalizuar")
    if st.session_state["ep_mode"] == "predefined":
        st.selectbox("Zgjidhni gjendjen", list(ep.PREDEFINED_STATES), key="ep_choice",
                     format_func=lambda k: ep.PREDEFINED_STATES[k]["label"])
    else:
        st.text_input("9 numra (0 = bosh), p.sh. 1 2 3 4 0 6 7 5 8", key="ep_custom")

    start, errors = _ep_start()
    left, right, info = st.columns([1, 1, 2])
    if start is not None:
        with left:
            show_figure(plot_eight_puzzle(start, goal=ep.GOAL_STATE, title="Initial State"))
        with info:
            inversions = ep.count_inversions(start)
            if ep.is_solvable(start):
                st.success(f"Inversione: {inversions} (çift) → gjendja është e zgjidhshme.")
            else:
                st.error(f"Inversione: {inversions} (tek) → gjendja NUK është e zgjidhshme; "
                         "BFS nuk do të niset.")
            st.caption("Jeshile = pllakë në vendin e Goal State; portokalli = jo në vend.")
    else:
        with info:
            show_errors(errors)
    with right:
        show_figure(plot_eight_puzzle(ep.GOAL_STATE, goal=ep.GOAL_STATE, title="Goal State"))

    with st.expander("Kufijtë e kërkimit dhe pema"):
        c1, c2, c3 = st.columns(3)
        c1.number_input("Maks. gjendje të eksploruara", min_value=100, max_value=500_000,
                        step=10_000, key="ep_max_explored")
        c2.number_input("Timeout (sekonda)", min_value=1.0, max_value=120.0, step=1.0,
                        key="ep_timeout")
        c3.slider("Nyje në pemë (vizualizim)", 10, 150, key="ep_tree_nodes")

    b1, b2 = st.columns(2)
    b1.button("▶ Ekzekuto BFS", type="primary", on_click=_ep_run, 
              key="ep_run")
    b2.button("↺ Rivendos", on_click=reset_state, args=(EP_DEFAULTS, EP_RESULTS),
               key="ep_reset")

    show_errors(st.session_state.get("ep_errors", []))
    result = st.session_state.get("ep_result")
    if result is None:
        return
    show_termination(result.stats)
    (st.success if result.solved else st.warning)(result.message)
    if result.solved:
        st.markdown("#### Rruga e zgjidhjes")
        st.write(" → ".join(result.actions) if result.actions else "Gjendja fillestare është qëllimi.")
        show_figure(plot_eight_puzzle_path(result.path, result.actions, goal=ep.GOAL_STATE))
        if len(result.path) > 1:
            step = st.slider("Shiko një hap", 0, len(result.path) - 1, len(result.path) - 1,
                             key="ep_step")
            col, _ = st.columns([1, 3])
            with col:
                label = "Fillimi" if step == 0 else f"Hapi {step}: {result.actions[step - 1]}"
                show_figure(plot_eight_puzzle(result.path[step], goal=ep.GOAL_STATE, title=label))
        legal = all(ep.is_legal_transition(a, b) for a, b in zip(result.path, result.path[1:]))
        show_validation(legal and ep.is_goal(result.path[-1]), ["Rruga përmban kalim të paligjshëm."],
                        f"Çdo kalim në rrugë është lëvizje e ligjshme dhe gjendja përfundimtare "
                        f"është Goal State ({len(result.actions)} lëvizje).")
    _tree_section(result.tree, "Search Tree – 8-Puzzle (BFS)", "ep_tree")
    show_statistics(result.stats, extra={
        "Madhësia maksimale e radhës": (result.max_queue_size, "Kulmi i radhës FIFO."),
        "Thellësia e zgjidhjes": (result.solution_depth, "Numri minimal i lëvizjeve."),
    })


# ===========================================================================
# Missionaries and Cannibals
# ===========================================================================

MC_DEFAULTS = {
    "mc_m": 3,
    "mc_c": 3,
    "mc_capacity": 2,
    "mc_tree_nodes": 80,
}
MC_RESULTS = ("mc_result", "mc_errors", "mc_step")


def _mc_run() -> None:
    st.session_state.pop("mc_step", None)
    try:
        st.session_state["mc_result"] = mc.solve_missionaries_cannibals(
            st.session_state["mc_m"], st.session_state["mc_c"], st.session_state["mc_capacity"],
            tree_nodes=st.session_state["mc_tree_nodes"])
        st.session_state["mc_errors"] = []
    except mc.InvalidRiverConfigError as error:
        st.session_state.pop("mc_result", None)
        st.session_state["mc_errors"] = error.errors


def _missionaries() -> None:
    init_state(MC_DEFAULTS)
    st.header("Missionaries and Cannibals – Breadth-First Search")
    with st.expander("Përshkrimi, reprezentimi, veprimet dhe kufizimet"):
        st.markdown(
            "**Përshkrimi:** 3 misionarë (M) dhe 3 kanibalë (K) janë në bregun e majtë me një "
            "varkë që mban 1 ose 2 persona. Të gjithë duhet të kalojnë në bregun e djathtë.\n\n"
            "**Reprezentimi:** `(M_majtas, K_majtas, Varka)`, ku Varka = 1 kur varka është majtas "
            "dhe 0 kur është djathtas. Initial State = `(3, 3, 1)`, Goal State = `(0, 0, 0)`.\n\n"
            "**Veprimet:** kalimet (m, k) me 1 ≤ m + k ≤ kapaciteti: "
            "(1,0), (2,0), (0,1), (0,2), (1,1). Varka lëviz gjithmonë te bregu tjetër.\n\n"
            "**Kufizimet (kontrollohen për TË DY brigjet pas çdo kalimi):** numrat brenda "
            "kufijve; në asnjë breg misionarët (nëse ka) nuk janë më pak se kanibalët; "
            "asnjë kalim me 0 persona."
        )
    c1, c2, c3, c4 = st.columns(4)
    c1.number_input("Misionarë", min_value=0, max_value=10, step=1, key="mc_m")
    c2.number_input("Kanibalë", min_value=0, max_value=10, step=1, key="mc_c")
    c3.number_input("Kapaciteti i varkës", min_value=1, max_value=6, step=1, key="mc_capacity")
    c4.slider("Nyje në pemë", 10, 200, key="mc_tree_nodes")
    problem = mc.RiverProblem(st.session_state["mc_m"], st.session_state["mc_c"],
                              st.session_state["mc_capacity"])
    if not problem.validate():
        st.caption(f"Gjendje teorike (M+1)(K+1)·2 = {problem.theoretical_states()}; "
                   f"prej tyre të sigurta: {problem.count_safe_states()}.")

    b1, b2 = st.columns(2)
    b1.button("▶ Ekzekuto BFS", type="primary", on_click=_mc_run,
              key="mc_run")
    b2.button("↺ Rivendos", on_click=reset_state, args=(MC_DEFAULTS, MC_RESULTS),
              key="mc_reset")

    show_errors(st.session_state.get("mc_errors", []))
    result = st.session_state.get("mc_result")
    if result is None:
        return
    show_termination(result.stats)
    p = result.problem
    if not result.solved:
        (st.error if result.stats.termination_reason == "exhausted" else st.warning)(result.message)
    else:
        st.success(result.message)
        st.markdown("#### Rruga e zgjidhjes")
        st.dataframe(mc.path_to_rows(result), hide_index=True)
        step = st.slider("Hapi aktual", 0, len(result.path) - 1, 0, key="mc_step")
        move = result.moves[step - 1] if step > 0 else None
        show_figure(plot_river_state(
            result.path[step], p.missionaries, p.cannibals, step=step,
            crossing=(move["missionaries"], move["cannibals"]) if move else None,
            direction=move["direction"] if move else None,
            valid=p.is_valid_state(result.path[step])))
        all_valid = all(p.is_valid_state(s) for s in result.path)
        legal = all(p.is_legal_transition(a, b) for a, b in zip(result.path, result.path[1:]))
        show_validation(all_valid and legal and p.is_goal(result.path[-1]),
                        ["Rruga përmban gjendje ose kalim të pavlefshëm."],
                        "Çdo gjendje në rrugë është e vlefshme në të dy brigjet, çdo kalim mbart "
                        "1–kapacitet persona dhe gjendja e fundit është Goal State.")
    with st.expander("Shembull: si duket një gjendje e pavlefshme"):
        standard = mc.RiverProblem(3, 3, 2)
        invalid_state = (1, 2, 1)
        show_figure(plot_river_state(invalid_state, 3, 3,
                                     valid=standard.is_valid_state(invalid_state)))
        st.caption("Majtas 1 misionar me 2 kanibalë: misionari është në pakicë → e pavlefshme. "
                   "Në pemë, gjendjet e tilla shfaqen me të kuqe.")
    _tree_section(result.tree, "Search Tree – Missionaries and Cannibals (BFS)", "mc_tree")
    show_statistics(result.stats, extra={
        "Refuzuar si të pavlefshme": (result.rejected_invalid,
                                      "Kalime që shkelin kufijtë ose sigurinë e brigjeve."),
        "Refuzuar si të vizituara": (result.rejected_duplicate, "Gjendje të gjeneruara më parë."),
        "Madhësia maksimale e radhës": (result.max_queue_size, "Kulmi i radhës FIFO."),
        "Thellësia e zgjidhjes": (result.solution_depth, "Numri minimal i kalimeve."),
    })


# ===========================================================================
# Rotating Workforce Scheduling
# ===========================================================================

WF_FIELDS = ("num_workers", "num_days", "num_shifts", "workers_per_shift",
             "max_consecutive_days", "min_rest_hours")
WF_DEFAULTS = {
    "wf_preset": "default",
    **{f"wf_{k}": v for k, v in wf.PRESETS["default"]["config"].items()},
    "wf_max_nodes": 200_000,
    "wf_timeout": 10.0,
    "wf_optimize": True,
    "wf_tree_nodes": 60,
}
WF_RESULTS = ("wf_result", "wf_errors")


def _wf_apply_preset() -> None:
    for key, value in wf.PRESETS[st.session_state["wf_preset"]]["config"].items():
        st.session_state[f"wf_{key}"] = value
    for key in WF_RESULTS:
        st.session_state.pop(key, None)


def _wf_run() -> None:
    try:
        st.session_state["wf_result"] = wf.solve_workforce_schedule(
            **{k: st.session_state[f"wf_{k}"] for k in WF_FIELDS},
            max_nodes=st.session_state["wf_max_nodes"],
            timeout_seconds=float(st.session_state["wf_timeout"]),
            optimize=st.session_state["wf_optimize"],
            tree_nodes=st.session_state["wf_tree_nodes"])
        st.session_state["wf_errors"] = []
    except wf.WorkforceConfigError as error:
        st.session_state.pop("wf_result", None)
        st.session_state["wf_errors"] = error.errors


def _workforce() -> None:
    init_state(WF_DEFAULTS)
    st.header("Rotating Workforce Scheduling – DFS me Backtracking")
    st.info("Ky është një **model i thjeshtuar edukativ** i planifikimit të turneve, "
            "jo një sistem real planifikimi.")
    with st.expander("Përshkrimi, modelimi, veprimet dhe kufizimet"):
        st.markdown(
            "**Përshkrimi:** W punëtorë, D ditë, S turne në ditë (Mëngjes 06–14, Pasdite 14–22, "
            "Natë 22–06), R punëtorë për turn. Parazgjedhja: 6 × 7 × 3 × 2 → 42 caktime.\n\n"
            "**Variables:** caktimi X[d][s][k] – punëtori në vendin k të turnit s, ditën d.  \n"
            "**Domain:** punëtorët P1…PW.  \n"
            "**Initial State:** orar bosh.  \n"
            "**Veprimi:** caktimi i një punëtori në vendin e radhës (ditë → turn → vend).  \n"
            "**Goal Test:** orari i plotë që plotëson kufizimet 1–5.\n\n"
            "**Kufizimet:**\n"
            "1. Çdo turn ka saktësisht R punëtorë.\n"
            "2. Një punëtor nuk punon dy turne në të njëjtën ditë (dita = dita kur fillon turni).\n"
            "3. Ndalohen turnet e njëpasnjëshme me pushim më të vogël se pushimi minimal: "
            "pushimi = fillimi i turnit të ri + 24 − mbarimi i turnit të mëparshëm.\n"
            "4. Natë → Mëngjes të nesërmen ndalohet gjithmonë (0 orë pushim).\n"
            "5. Nuk tejkalohet numri maksimal i ditëve të njëpasnjëshme të punës.\n"
            "6. Ngarkesa shpërndahet sa më drejtë (kufizim i butë – objektivi).\n\n"
            "**Objective:** min imbalance = max(turnet) − min(turnet) për punëtor. "
            "Kufiri i poshtëm teorik: 0 nëse D·S·R pjesëtohet me W, përndryshe 1.\n\n"
            "**DFS me Backtracking:** provohen punëtorët e lirë (më pak të ngarkuarit së pari); "
            "caktimi që shkel kufizimet refuzohet; dega e dështuar zhbëhet. Pas zgjidhjes së parë "
            "kërkimi vazhdon (Branch and Bound) për orar më të drejtë. Optimaliteti pohohet "
            "vetëm kur arrihet kufiri teorik ose kur kërkimi eksploron gjithë hapësirën."
        )

    st.selectbox("Konfigurim i gatshëm", list(wf.PRESETS), key="wf_preset",
                 format_func=lambda k: wf.PRESETS[k]["label"], on_change=_wf_apply_preset)
    c = st.columns(3)
    c[0].number_input("Numri i punëtorëve", min_value=1, max_value=20, step=1, key="wf_num_workers")
    c[1].number_input("Numri i ditëve", min_value=1, max_value=14, step=1, key="wf_num_days")
    c[2].number_input("Numri i turneve (1–3)", min_value=1, max_value=3, step=1, key="wf_num_shifts")
    c = st.columns(3)
    c[0].number_input("Punëtorë për turn", min_value=1, max_value=5, step=1,
                      key="wf_workers_per_shift")
    c[1].number_input("Maks. ditë pune radhazi", min_value=1, max_value=14, step=1,
                      key="wf_max_consecutive_days")
    c[2].number_input("Pushimi minimal (orë)", min_value=0, max_value=24, step=1,
                      key="wf_min_rest_hours")
    with st.expander("Kufijtë e kërkimit"):
        c = st.columns(3)
        c[0].number_input("Kufiri i nyjeve", min_value=10, max_value=2_000_000, step=10_000,
                          key="wf_max_nodes")
        c[1].number_input("Timeout (sekonda)", min_value=1.0, max_value=120.0, step=1.0,
                          key="wf_timeout")
        c[2].slider("Nyje në pemë", 10, 200, key="wf_tree_nodes")
        st.checkbox("Optimizo imbalance (Branch and Bound pas zgjidhjes së parë)",
                    key="wf_optimize")

    cfg = wf.WorkforceConfig(**{k: st.session_state[f"wf_{k}"] for k in WF_FIELDS})
    precheck = wf.validate_config(cfg)
    if precheck:
        st.warning("Konfigurimi nuk mund të ketë zgjidhje (kontroll para kërkimit):")
        for message in precheck:
            st.write(f"- {message}")
    else:
        land = wf.landscape(cfg)
        st.caption(f"Landscape: {land['variables']} variabla, domeni {land['domain_size']} "
                   f"punëtorë → W^(D·S·R) = {format_big_number(land['unrestricted_assignments'])} "
                   f"caktime pa kufizime. Mesatarja: {land['average_shifts_per_worker']:.2f} "
                   f"turne/punëtor; kufiri i poshtëm i imbalance = {land['imbalance_lower_bound']}.")

    b1, b2 = st.columns(2)
    b1.button("▶ Ekzekuto DFS", type="primary", on_click=_wf_run, 
              key="wf_run")
    b2.button("↺ Rivendos", on_click=reset_state, args=(WF_DEFAULTS, WF_RESULTS),
              key="wf_reset")

    show_errors(st.session_state.get("wf_errors", []))
    result = st.session_state.get("wf_result")
    if result is None:
        return
    show_termination(result.stats)
    if result.schedule is None:
        (st.error if result.stats.termination_reason == "exhausted" else st.warning)(result.message)
    else:
        (st.success if result.optimal_proven else st.warning)(result.message)
        cfg = result.config
        names = [s[1] for s in cfg.shifts]
        m = st.columns(4)
        m[0].metric("Imbalance", result.imbalance)
        m[1].metric("Kufiri i poshtëm", cfg.imbalance_lower_bound)
        m[2].metric("Zgjidhje të gjetura", result.solutions_found)
        m[3].metric("Nyja e zgjidhjes së parë", result.first_solution_explored)

        st.markdown("#### Orari javor")
        st.dataframe(wf.schedule_to_rows(result.schedule, cfg), hide_index=True)
        left, right = st.columns(2)
        with left:
            show_figure(plot_workforce_schedule(result.schedule, names))
        with right:
            rows = wf.workload_rows(result.schedule, cfg)
            show_figure(plot_workload(rows, names))
        st.markdown("#### Turnet për punëtor")
        st.dataframe(rows, hide_index=True)

        st.markdown("#### Validimi i të gjitha kufizimeve (i pavarur)")
        st.dataframe(
            [{"Kufizimi": ch["constraint"],
              "Rezultati": ("✅ Plotësohet" if ch["ok"] else
                            ("❌ Shkelet" if ch["hard"] else "⚠️ Jo optimale")),
              "Detaje": "; ".join(ch["details"][:3]) or "–"}
             for ch in result.verification["checks"]],
            hide_index=True)
        show_validation(result.verification["valid"], [],
                        "Orari plotëson të gjitha kufizimet e forta (1–5).")
    _tree_section(result.tree, "Search Tree – Workforce Scheduling (DFS)", "wf_tree")
    show_statistics(result.stats, extra={
        "Orare të plota të gjetura": (result.solutions_found,
                                      "Zgjidhje të vlefshme gjatë kërkimit (Branch and Bound)."),
        "Nyja e zgjidhjes së parë": (result.first_solution_explored,
                                     "Nyje të eksploruara deri te zgjidhja e parë."),
    })


# ===========================================================================

def render() -> None:
    page_header("Detyra 4 – Search Trees",
                "BFS (8-Puzzle, Missionaries and Cannibals) · DFS me Backtracking (Workforce)",
                "Tre probleme me pemë reale kërkimi: çdo nyje e vizatuar është një gjendje e "
                "gjeneruar gjatë ekzekutimit.")
    section = st.radio("Problemi", SECTIONS, horizontal=True, key="st_section")
    st.divider()
    if section == SECTIONS[0]:
        _eight_puzzle()
    elif section == SECTIONS[1]:
        _missionaries()
    else:
        _workforce()
