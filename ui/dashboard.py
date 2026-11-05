"""
Faqja kryesore (Dashboard) e aplikacionit.
"""

import streamlit as st

DASHBOARD_TITLE = "Teknikat e Kërkimit në Inteligjencën Artificiale"

INTRODUCTION = (
    "Teknikat e kërkimit të painformuar (Uninformed Search) eksplorojnë hapësirën "
    "e gjendjeve pa përdorur njohuri shtesë për afërsinë ndaj qëllimit. "
    "Ato mbështeten vetëm në përkufizimin e problemit: gjendjen fillestare "
    "(Initial State), veprimet e lejuara dhe testin e qëllimit (Goal Test). "
    "Ky aplikacion demonstron DFS, BFS, IDDFS dhe Backtracking në katër detyra."
)


def render() -> None:
    """Vizaton faqen kryesore."""
    st.title(DASHBOARD_TITLE)
    st.write(INTRODUCTION)
