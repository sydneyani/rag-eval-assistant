"""Thin Streamlit client for the CiteGround API.

The API owns retrieval, generation, and tracing. This file only renders the
response. Start the API first:

    python -m citeground.cli.serve

Then:

    python -m streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("CITEGROUND_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="CiteGround", layout="wide")
st.title("CiteGround")
st.caption("Synthetic Helios filings. Not a real issuer and not investment advice.")

question = st.text_area(
    "Question",
    value="What was fiscal 2024 revenue compared with fiscal 2023 revenue?",
    height=80,
)
mode = st.selectbox("Retrieval", ["hybrid", "bm25", "dense"], index=0)
top_k = st.slider("Passages", min_value=1, max_value=10, value=5)

if st.button("Ask", type="primary"):
    try:
        response = httpx.post(
            f"{API_URL}/v1/answer",
            json={"question": question, "mode": mode, "top_k": top_k},
            timeout=60.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        st.error(f"The API at {API_URL} did not answer. Start it with citeground-serve. {exc}")
    else:
        payload = response.json()
        st.markdown(payload["answer"])
        usage = payload["usage"]
        st.caption(
            f"{payload['provider']} · {payload['model']} · {payload['latency_ms']:.0f} ms · "
            f"{usage['prompt_tokens'] + usage['completion_tokens']} tokens · "
            f"${usage['cost_usd']:.6f} ({usage['cost_basis']}) · trace {payload['trace_id'][:8]}"
        )
        st.subheader("Passages")
        for index, hit in enumerate(payload["retrieved"], start=1):
            chunk = hit["chunk"]
            with st.expander(f"[{index}] {chunk['section_title']} · {chunk['section_id']}"):
                st.write(chunk["text"])
