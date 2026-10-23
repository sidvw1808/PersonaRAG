"""Streamlit UI for PersonaRAG.

    streamlit run app.py
"""

import streamlit as st

from personarag import config
from personarag.engine import PersonaEngine, grounding_report
from personarag.panel import run_panel
from personarag.personas import list_personas
from personarag.prompts import MODES

st.set_page_config(page_title="PersonaRAG", page_icon="🎙️", layout="wide")

GROUNDING_BADGE = {
    "strong": ("🟢", "Strongly grounded in sources"),
    "partial": ("🟡", "Partially grounded"),
    "weak": ("🟠", "Weakly grounded"),
    "ungrounded": ("🔴", "Not grounded in any source: treat as speculation"),
}


@st.cache_resource
def get_engine() -> PersonaEngine:
    return PersonaEngine()


engine = get_engine()
personas = {p.id: p for p in list_personas()}
counts = {pid: engine.store.count(pid) for pid in personas}


def render_sources(hits, grounding):
    icon, label = GROUNDING_BADGE[grounding["level"]]
    st.caption(
        f"{icon} {label} · citation coverage {grounding['citation_coverage']:.0%} · "
        f"top similarity {grounding['top_similarity']:.2f}"
    )
    if not hits:
        return
    cited = set(grounding["cited"])
    with st.expander(f"Sources ({len(cited)} cited of {len(hits)} retrieved)"):
        for i, h in enumerate(hits, 1):
            mark = "**cited**" if i in cited else "retrieved"
            url = h.metadata.get("url")
            title = f"[{h.label}]({url})" if url else h.label
            st.markdown(f"**[{i}]** {title} · {mark} · similarity {h.score:.2f}")
            st.caption(h.text[:600] + ("…" if len(h.text) > 600 else ""))


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🎙️ PersonaRAG")
    st.caption("Evidence-grounded conversations with AI simulations of public figures.")
    persona_id = st.selectbox(
        "Persona",
        list(personas),
        format_func=lambda pid: f"{personas[pid].avatar} {personas[pid].name}"
        + ("" if counts[pid] else "  (no sources yet)"),
    )
    mode = st.radio("Mode", list(MODES), format_func=str.title, horizontal=True)
    st.divider()
    llm_label = config.CLAUDE_MODEL if config.LLM_PROVIDER == "claude" else f"Ollama · {config.OLLAMA_MODEL}"
    st.caption(f"LLM: {llm_label}  \nEmbeddings: {config.EMBED_MODEL}")
    with st.expander(f"Knowledge base · {counts[persona_id]} chunks"):
        for src in engine.store.sources(persona_id):
            st.markdown(f"- {src.get('title')} ({src['chunks']} chunks)")
        if not counts[persona_id]:
            st.info(f"Run `python -m personarag.ingest {persona_id}` to build it.")
    if st.button("Clear conversation"):
        st.session_state.pop(f"chat_{persona_id}", None)
        st.rerun()
    st.warning(
        "This is an AI simulation built from publicly available material. "
        "It is not the real person and does not represent their private views.",
        icon="⚠️",
    )

chat_tab, panel_tab = st.tabs(["💬 Chat", "🧑‍⚖️ Panel"])

# ---------------- Chat ----------------
with chat_tab:
    persona = personas[persona_id]
    st.subheader(f"{persona.avatar} {persona.name}")
    st.caption(persona.tagline)
    key = f"chat_{persona_id}"
    history = st.session_state.setdefault(key, [])

    for turn in history:
        with st.chat_message(turn["role"], avatar=persona.avatar if turn["role"] == "assistant" else None):
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
                render_sources(turn["hits"], turn["grounding"])

    if question := st.chat_input(f"Ask {persona.name} anything…"):
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant", avatar=persona.avatar):
            plain = [{"role": t["role"], "content": t["content"]} for t in history]
            try:
                hits, tokens = engine.stream(persona_id, question, plain, mode)
                text = st.write_stream(tokens)
            except Exception as exc:  # noqa: BLE001 - surface config problems in the UI
                st.error(f"LLM call failed: {exc}")
                st.stop()
            grounding = grounding_report(text, hits)
            render_sources(hits, grounding)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": text, "hits": hits, "grounding": grounding})

# ---------------- Panel ----------------
with panel_tab:
    ready = [pid for pid in personas if counts[pid]]
    chosen = st.multiselect(
        "Panelists",
        list(personas),
        default=ready[:3],
        format_func=lambda pid: f"{personas[pid].avatar} {personas[pid].name}",
    )
    panel_q = st.text_input("Question for the panel", placeholder="Should I leave college to start a company?")
    if st.button("Ask the panel", type="primary", disabled=len(chosen) < 2 or not panel_q):
        with st.spinner("Panelists are thinking…"):
            result = run_panel(engine, chosen, panel_q, mode)
        cols = st.columns(len(result.answers))
        for col, ans in zip(cols, result.answers):
            with col:
                st.markdown(f"### {ans.persona.avatar} {ans.persona.name}")
                st.markdown(ans.text)
                render_sources(ans.hits, ans.grounding)
        st.divider()
        st.markdown("### 🧑‍⚖️ Moderator")
        st.markdown(result.summary)
    elif len(chosen) < 2:
        st.caption("Pick at least two panelists.")
