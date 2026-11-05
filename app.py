"""
Pika hyrëse e aplikacionit Streamlit.

Ekzekutimi:
    streamlit run app.py

Ky skedar përmban vetëm konfigurimin e faqes dhe navigimin.
Logjika e algoritmeve ndodhet në paketën `tasks/`, ndërsa
ndërfaqja e secilës detyrë në paketën `ui/`.
"""

import streamlit as st

from ui import dashboard

st.set_page_config(
    page_title="Teknikat e Kërkimit në IA",
    page_icon="🔍",
    layout="wide",
)

dashboard.render()
