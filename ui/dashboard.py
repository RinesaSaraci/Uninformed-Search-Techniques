"""
Faqja kryesore (Dashboard) e aplikacionit.
"""

import streamlit as st

DASHBOARD_TITLE = "Teknikat e Kërkimit në Inteligjencën Artificiale"

INTRODUCTION = (
    "Ky aplikacion edukativ demonstron teknikat e kërkimit të painformuar "
    "(Uninformed Search) në katër detyra. Për çdo detyrë paraqiten modelimi, "
    "reprezentimi i problemit, objective function, landscape, ekzekutimi real i "
    "algoritmit, vizualizimi dhe statistikat e matura gjatë ekzekutimit."
)

UNINFORMED_SEARCH = """
Kërkimi i **painformuar** (Uninformed / Blind Search) eksploron hapësirën e gjendjeve
pa asnjë njohuri shtesë për afërsinë ndaj qëllimit. Ai mbështetet vetëm në:
**Initial State**, **veprimet** (funksionin e pasardhësve) dhe **Goal Test**.

| Teknika | Struktura | Karakteristika |
|---|---|---|
| **BFS** – Breadth-First Search | Radhë FIFO | Eksploron nivel pas niveli; gjen rrugën më të shkurtër kur çdo hap ka kosto të njëjtë. Kërkon shumë kujtesë. |
| **DFS** – Depth-First Search | Stekë / rekursion | Shkon në thellësi para se të kthehet; kujtesë e vogël; nuk garanton rrugën më të shkurtër. |
| **IDDFS** – Iterative Deepening DFS | DFS me limit në rritje | Përsërit DFS me limit 0, 1, 2, ...; kujtesë si DFS dhe plotësi si BFS, por rivizitën nyjet e cekëta. |
| **Backtracking** | DFS mbi caktime | Cakton variablat një nga një; kur shkelet një kufizim, zhbën caktimin dhe provon alternativën. |
"""

ASSIGNMENTS = [
    {
        "page": "Detyra 1 – Social Golfers",
        "title": "1. Social Golfers Problem",
        "algorithm": "DFS me Backtracking",
        "description": "32 golferë në 8 grupe me nga 4 çdo javë; asnjë çift nuk luan "
                       "bashkë më shumë se një herë. Synimi: sa më shumë javë.",
    },
    {
        "page": "Detyra 2 – Sudoku",
        "title": "2. Sudoku Solver",
        "algorithm": "BFS me Backtracking",
        "description": "Zgjidhja e Sudoku 9×9 në tri nivele (Easy, Medium, Hard) me "
                       "radhë FIFO dhe braktisje të degëve të pavlefshme.",
    },
    {
        "page": "Detyra 3 – Latin Square",
        "title": "3. Latin Square",
        "algorithm": "IDDFS me Backtracking",
        "description": "Ndërtimi i një matrice n×n ku çdo numër 1..n shfaqet saktësisht "
                       "një herë në çdo rresht dhe kolonë.",
    },
    {
        "page": "Detyra 4 – Search Trees",
        "title": "4. Search Trees",
        "algorithm": "BFS · BFS · DFS me Backtracking",
        "description": "8-Puzzle (BFS), Missionaries and Cannibals (BFS) dhe "
                       "Rotating Workforce Scheduling (DFS me Backtracking).",
    },
]


def _go_to(page: str) -> None:
    st.session_state["page"] = page


def render() -> None:
    """Vizaton faqen kryesore."""
    st.title(DASHBOARD_TITLE)
    st.write(INTRODUCTION)

    with st.expander("Çfarë janë teknikat e kërkimit të painformuar?", expanded=True):
        st.markdown(UNINFORMED_SEARCH)

    st.subheader("Detyrat")
    columns = st.columns(2)
    for index, assignment in enumerate(ASSIGNMENTS):
        with columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"#### {assignment['title']}")
                st.markdown(f"**Algoritmi:** {assignment['algorithm']}")
                st.write(assignment["description"])
                st.button("Hap detyrën", key=f"open_{index}", on_click=_go_to,
                          args=(assignment["page"],))

    st.divider()
    st.caption(
        "Të gjitha statistikat dhe vizualizimet në këtë aplikacion vijnë nga ekzekutimi "
        "real i algoritmeve. Kur një kërkim ndërpritet nga kufiri i nyjeve ose i kohës, "
        "kjo raportohet qartë dhe nuk paraqitet si provë që zgjidhja nuk ekziston."
    )
