import streamlit as st


def _render_sources(sources: list):
    if not sources:
        return
    seen, unique = set(), []
    for s in sources:
        key = s.get("topic", "")
        if key not in seen:
            seen.add(key)
            unique.append(s)

    lines = []
    for s in unique:
        pct  = int(s.get("score", 0) * 100)
        bar  = "█" * (pct // 10) + "░" * (10 - pct // 10)
        lines.append(
            f"**{s.get('category','—')}** › {s.get('topic','—')} "
            f"<span style='color:#D4A843;font-family:monospace;font-size:0.75rem'>{bar} {pct}%</span>"
        )
    st.markdown(
        '<div style="background:rgba(46,164,164,0.08);border:1px solid rgba(46,164,164,0.25);'
        'border-radius:8px;padding:10px 14px;margin-top:8px">'
        '<div style="font-size:0.72rem;color:#8B949E;letter-spacing:0.06em;'
        'text-transform:uppercase;margin-bottom:6px">📎 Sources retrieved</div>'
        + "".join(f'<div style="font-size:0.8rem;color:#C9D1D9;margin:3px 0">{l}</div>' for l in lines)
        + "</div>",
        unsafe_allow_html=True,
    )


def page_assistant(stops_df, ai, kb_df):
    st.markdown('<div class="section-title">💬 Logistics AI Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Grounded in your logistics knowledge base · '
        'Only answers travel & logistics questions</div>',
        unsafe_allow_html=True,
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    for col, label, active in [
        (col_s1, "LLM",        bool(ai.llm)),
        (col_s2, "Embeddings", bool(ai.embeddings)),
        (col_s3, "RAG Index",  bool(ai.embeddings) and not kb_df.empty),
    ]:
        color  = "#2EA4A4" if active else "#E74C3C"
        status = (
            ("🟢 Connected" if active else "🔴 Offline")
            if label != "RAG Index"
            else ("🟢 Ready" if active else "⚠️ Keyword fallback")
        )
        col.markdown(
            f'<div class="card" style="padding:10px;text-align:center">'
            f'<span style="font-size:0.8rem;color:#8B949E">{label}</span><br>'
            f'<b style="color:{color}">{status}</b></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    with st.expander("ℹ️ What can I ask?", expanded=False):
        st.markdown("""
**In scope:** route planning, delivery operations, customs, fleet management, freight modes, KPIs, documentation.
**Out of scope:** questions unrelated to travel or logistics are politely declined.
        """)

    if "assistant_history" not in st.session_state:
        st.session_state["assistant_history"] = []

    for msg in st.session_state["assistant_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("sources"):
                    _render_sources(msg["sources"])
                if msg.get("rejected"):
                    st.caption("🚫 Outside the logistics/travel domain.")
                elif msg.get("grounded"):
                    st.caption("✅ Grounded in knowledge base via semantic retrieval.")

    user_input = st.chat_input("Ask about routes, delivery windows, customs, fuel costs…")
    if user_input:
        st.session_state["assistant_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base…"):
                result = ai.get_kb_answer(user_input, kb_df)
            answer_text = result["text"]
            sources     = result.get("sources", [])
            grounded    = result.get("grounded", False)
            rejected    = result.get("rejected", False)
            st.markdown(answer_text)
            if sources:    _render_sources(sources)
            if rejected:   st.caption("🚫 Outside the logistics/travel domain.")
            elif grounded: st.caption("✅ Grounded in knowledge base via semantic retrieval.")
            else:          st.caption("⚠️ Low-confidence retrieval — broader KB context used.")

        st.session_state["assistant_history"].append({
            "role": "assistant", "content": answer_text,
            "sources": sources, "grounded": grounded, "rejected": rejected,
        })

    col_a, _ = st.columns([1, 5])
    with col_a:
        if st.button("🗑️ Clear Chat"):
            st.session_state["assistant_history"] = []
            st.rerun()

    if not kb_df.empty:
        with st.expander(f"📚 Knowledge Base ({len(kb_df)} articles)"):
            st.dataframe(
                kb_df[["category", "topic", "content"]], use_container_width=True,
                column_config={"content": st.column_config.TextColumn("content", width="large")},
            )
