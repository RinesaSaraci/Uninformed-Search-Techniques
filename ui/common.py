from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

from utils.statistics import stats_rows, termination_summary


def show_figure(fig) -> None:
    """Shfaq një figurë Matplotlib dhe e mbyll për të liruar kujtesën."""
    st.pyplot(fig)
    plt.close(fig)


def show_errors(errors: list[str]) -> None:
    for message in errors:
        st.error(message)


def show_termination(stats) -> None:
    """Mesazhi i përfundimit me ngjyrë sipas llojit (sukses / info / paralajmërim)."""
    summary = termination_summary(stats)
    if summary["level"] == "success":
        st.success(summary["message"])
    elif summary["level"] == "warning":
        st.warning(summary["message"] + " Rezultati është i pjesshëm.")
    else:
        st.info(summary["message"])


def show_statistics(stats, extra: dict | None = None) -> None:
    """Metrika kryesore + tabela e plotë e statistikave reale të ekzekutimit."""
    cols = st.columns(4)
    cols[0].metric("Gjendje të gjeneruara", f"{stats.generated:,}".replace(",", "."))
    cols[1].metric("Gjendje të eksploruara", f"{stats.explored:,}".replace(",", "."))
    cols[2].metric("Gjendje të refuzuara", f"{stats.rejected:,}".replace(",", "."))
    cols[3].metric("Koha", f"{stats.elapsed_seconds:.3f} s")
    with st.expander("Të gjitha statistikat e ekzekutimit"):
        st.dataframe(stats_rows(stats, extra), hide_index=True, use_container_width=True)
        st.caption("Të gjitha vlerat janë numëruar gjatë ekzekutimit real të algoritmit.")


def show_validation(is_valid: bool, errors: list[str], ok_text: str) -> None:
    if is_valid:
        st.success(f"✅ {ok_text}")
    else:
        st.error("❌ Validimi i pavarur dështoi.")
        for message in errors[:20]:
            st.write(f"- {message}")


def model_section(variables: str, domain: str, constraints: list[str],
                  initial: str, goal: str) -> None:
    """Reprezentimi i problemit në format të njëtrajtshëm."""
    st.markdown(f"**Variables:** {variables}")
    st.markdown(f"**Domain:** {domain}")
    st.markdown("**Constraints:**")
    for c in constraints:
        st.markdown(f"- {c}")
    st.markdown(f"**Initial State:** {initial}")
    st.markdown(f"**Goal State / Goal Test:** {goal}")


def init_state(defaults: dict) -> None:
    """Vendos vlerat fillestare në session_state vetëm nëse mungojnë."""
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_state(defaults: dict, clear: tuple[str, ...] = ()) -> None:
    """Rikthen vlerat fillestare dhe fshin rezultatet (përdoret si on_click)."""
    for key, value in defaults.items():
        st.session_state[key] = value
    for key in clear:
        st.session_state.pop(key, None)


def page_header(title: str, algorithm: str, description: str) -> None:
    st.title(title)
    st.caption(f"Algoritmi: **{algorithm}**")
    st.write(description)
