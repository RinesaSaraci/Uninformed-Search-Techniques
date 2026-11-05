"""
Pika hyrëse e aplikacionit Streamlit.

Ekzekutimi:
    streamlit run app.py

"""

import streamlit as st

from ui import dashboard, latin_square_ui, search_trees_ui, social_golfers_ui, sudoku_ui

st.set_page_config(
    page_title="Teknikat e Kërkimit në IA",
    page_icon="🔍",
    layout="wide",
)

PAGES = {
    "Ballina": dashboard.render,
    "Detyra 1 – Social Golfers": social_golfers_ui.render,
    "Detyra 2 – Sudoku": sudoku_ui.render,
    "Detyra 3 – Latin Square": latin_square_ui.render,
    "Detyra 4 – Search Trees": search_trees_ui.render,
}

if "page" not in st.session_state:
    st.session_state["page"] = "Ballina"

with st.sidebar:
    st.title("🔍 Navigimi")
    st.radio("Zgjidhni faqen", list(PAGES), key="page")
    st.divider()
    st.caption(
        "Teknikat e kërkimit të painformuar: DFS, BFS, IDDFS dhe Backtracking. "
        "Algoritmet ekzekutohen vetëm kur shtypet butoni përkatës."
    )

PAGES[st.session_state["page"]]()
