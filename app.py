"""
app.py
------
NLP Text Analysis Dashboard — a Streamlit front end for five classic
preprocessing and analysis steps built on spaCy.

Run with:  streamlit run app.py
"""

import html
import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from spacy import displacy

import nlp_engine as engine
from sample_texts import SAMPLE_TEXTS

# --------------------------------------------------------------------------- #
#  Page configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="NLP Text Analysis Dashboard",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#146C63", "#2E8B7F", "#4FA89A", "#7BC0B3", "#B45309", "#D08A2C",
           "#1F3B52", "#37536B", "#5A7288", "#8AA0B2"]


def load_css() -> None:
    css_path = Path(__file__).parent / "assets" / "styles.css"
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


load_css()


# --------------------------------------------------------------------------- #
#  Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the spaCy pipeline…")
def get_pipeline():
    return engine.load_model()


@st.cache_data(show_spinner=False)
def get_doc_bytes(text: str):
    """Cache the parsed document so repeated clicks do not re-run the pipeline."""
    nlp = get_pipeline()
    return engine.analyse(nlp, text).to_bytes()


def get_doc(text: str):
    from spacy.tokens import Doc

    return Doc(get_pipeline().vocab).from_bytes(get_doc_bytes(text))


# --------------------------------------------------------------------------- #
#  Small render helpers
# --------------------------------------------------------------------------- #
LONG_VALUE_KEYS = {"Cleaned text", "Lemmatised text"}


def stat_cards(stats: dict) -> None:
    """Render the numeric statistics as a responsive grid of cards."""
    cards = "".join(
        f"""<div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value}</div>
            </div>"""
        for label, value in stats.items()
        if label not in LONG_VALUE_KEYS
    )
    st.markdown(f'<div class="stat-grid">{cards}</div>', unsafe_allow_html=True)


def token_strip(items, tone_fn=None) -> None:
    """The dashboard's signature view: the result as a strip of token chips."""
    chips = "".join(
        f'<span class="chip {tone_fn(item) if tone_fn else ""}">'
        f'<span class="chip-text">{html.escape(str(item["text"]))}</span>'
        f'<span class="chip-tag">{html.escape(str(item["tag"]))}</span></span>'
        for item in items
    )
    st.markdown(f'<div class="chip-strip">{chips}</div>', unsafe_allow_html=True)


def download_row(table: pd.DataFrame, name: str) -> None:
    buffer = io.StringIO()
    table.to_csv(buffer, index=False)
    st.download_button(
        "Download result as CSV",
        buffer.getvalue(),
        file_name=f"{name.lower().replace(' ', '_')}_result.csv",
        mime="text/csv",
    )


def html_block(html: str, height: int = 260) -> None:
    st.markdown(
        f'<div class="render-box" style="max-height:{height}px">{html}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
#  Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(
        '<div class="side-brand">◆ NLP Lab</div>'
        '<div class="side-sub">spaCy · en_core_web_sm</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Sample text")
    sample_choice = st.selectbox(
        "Load an example instead of typing",
        ["— none —"] + list(SAMPLE_TEXTS.keys()),
        label_visibility="collapsed",
    )
    def _load_sample():
        if sample_choice != "— none —":
            st.session_state["text_input"] = SAMPLE_TEXTS[sample_choice]
            st.session_state["process"] = None

    st.button("Load sample", use_container_width=True, on_click=_load_sample)

    st.markdown("### Display options")
    show_chips = st.toggle("Show token strip", value=True)
    max_rows = st.slider("Table rows to display", 20, 500, 200, step=20)

    st.markdown("### The five processes")
    for name, meta in engine.PROCESSES.items():
        st.markdown(
            f'<div class="side-note"><b>{name}</b><br>{meta["blurb"]}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="side-foot">Built with Streamlit, spaCy, pandas and Plotly.</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
#  Header
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
      <div class="hero-eyebrow">Natural Language Processing · Assignment Dashboard</div>
      <h1 class="hero-title">Take a sentence apart, layer by layer.</h1>
      <p class="hero-sub">Paste any English text, choose one of five linguistic
      processes, and inspect both the annotated output and the statistics behind it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
#  Input
# --------------------------------------------------------------------------- #
st.session_state.setdefault("text_input", "")
st.session_state.setdefault("process", None)

st.markdown('<div class="section-label">1 · Input text</div>', unsafe_allow_html=True)
text = st.text_area(
    "Input text",
    key="text_input",
    height=180,
    placeholder="Type or paste your text here. For example: "
    "Dr. Ananya Sharma joined Infosys in Bengaluru on 12 March 2021 and now leads "
    "a research team of forty engineers.",
    label_visibility="collapsed",
)

left, right = st.columns([3, 1])
with left:
    st.markdown(
        f'<div class="input-meta">{len(text)} characters · '
        f'{len(text.split())} whitespace-separated words</div>',
        unsafe_allow_html=True,
    )
def _clear_text():
    st.session_state["text_input"] = ""
    st.session_state["process"] = None


with right:
    st.button("Clear text", use_container_width=True, on_click=_clear_text)

# --------------------------------------------------------------------------- #
#  Process buttons
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="section-label">2 · Choose a process</div>', unsafe_allow_html=True
)
button_cols = st.columns(5, gap="small")
for col, name in zip(button_cols, engine.PROCESSES):
    with col:
        if st.button(name, use_container_width=True, key=f"btn_{name}"):
            st.session_state["process"] = name

# --------------------------------------------------------------------------- #
#  Results
# --------------------------------------------------------------------------- #
process = st.session_state["process"]

if process and not text.strip():
    st.warning("Add some text above, then choose a process again.")
    st.session_state["process"] = None
elif process:
    meta = engine.PROCESSES[process]
    doc = get_doc(text)
    result = meta["fn"](doc)
    table, stats, chart = result["table"], result["stats"], result["chart"]

    st.markdown(
        f'<div class="section-label">3 · {process}</div>'
        f'<p class="process-blurb">{meta["blurb"]}</p>',
        unsafe_allow_html=True,
    )

    tab_result, tab_stats, tab_visual = st.tabs(
        ["Result", "Statistics", "Visualisation"]
    )

    # ---------------------------- Result tab ------------------------------- #
    with tab_result:
        if table.empty:
            st.info(meta["empty"])
        else:
            if show_chips:
                if process == "Tokenization":
                    items = [
                        {"text": r["Token"], "tag": r["Type"][:4].upper()}
                        for _, r in table.head(120).iterrows()
                    ]
                    token_strip(items, lambda i: "chip-neutral")
                elif process == "Stopword Removal":
                    items = [
                        {"text": r["Token"], "tag": r["Status"]}
                        for _, r in table.head(120).iterrows()
                    ]
                    token_strip(
                        items,
                        lambda i: "chip-muted" if i["tag"] == "Removed" else "chip-keep",
                    )
                elif process == "POS Tagging":
                    items = [
                        {"text": r["Token"], "tag": r["Coarse tag (POS)"]}
                        for _, r in table.head(120).iterrows()
                    ]
                    token_strip(items, lambda i: "chip-accent")
                elif process == "Lemmatization":
                    items = [
                        {"text": r["Lemma"], "tag": r["Token"]}
                        for _, r in table.head(120).iterrows()
                    ]
                    token_strip(
                        items,
                        lambda i: "chip-accent"
                        if i["text"].lower() != i["tag"].lower()
                        else "chip-neutral",
                    )
                else:
                    items = [
                        {"text": r["Entity"], "tag": r["Label"]}
                        for _, r in table.iterrows()
                    ]
                    token_strip(items, lambda i: "chip-accent")

            st.dataframe(
                table.head(max_rows), use_container_width=True, hide_index=True
            )
            if len(table) > max_rows:
                st.caption(
                    f"Showing the first {max_rows} of {len(table)} rows. "
                    "Raise the limit in the sidebar or download the full CSV."
                )
            download_row(table, process)

        # Processes that produce a rewritten version of the text
        for key in LONG_VALUE_KEYS:
            if key in stats:
                st.markdown(f'<div class="mini-label">{key}</div>', unsafe_allow_html=True)
                st.code(stats[key] or "—", language=None)

    # --------------------------- Statistics tab ---------------------------- #
    with tab_stats:
        stat_cards(stats)
        if chart is not None and not chart.empty:
            st.markdown(
                '<div class="mini-label">Frequency table</div>', unsafe_allow_html=True
            )
            st.dataframe(chart, use_container_width=True, hide_index=True)
            download_row(chart, f"{process} statistics")

    # -------------------------- Visualisation tab -------------------------- #
    with tab_visual:
        if chart is None or chart.empty:
            st.info("There is nothing to plot for this text.")
        else:
            x_col = chart.columns[0]
            if process == "Tokenization":
                fig = px.bar(
                    chart,
                    x=x_col,
                    y="Count",
                    title="How many tokens of each character length",
                )
            elif process == "Stopword Removal":
                fig = px.bar(
                    chart,
                    x="Count",
                    y=x_col,
                    orientation="h",
                    title="Most frequently removed stopwords",
                )
                fig.update_yaxes(autorange="reversed")
            elif process == "POS Tagging":
                fig = px.bar(
                    chart,
                    x=x_col,
                    y="Count",
                    color=x_col,
                    hover_data=["Meaning"],
                    title="Part-of-speech distribution",
                    color_discrete_sequence=PALETTE,
                )
            elif process == "Lemmatization":
                fig = px.bar(
                    chart,
                    x="Count",
                    y=x_col,
                    orientation="h",
                    title="Most frequent content lemmas",
                )
                fig.update_yaxes(autorange="reversed")
            else:
                fig = px.pie(
                    chart,
                    names=x_col,
                    values="Count",
                    hole=0.55,
                    title="Share of each entity label",
                    color_discrete_sequence=PALETTE,
                )

            fig.update_layout(
                template="simple_white",
                showlegend=(process == "Named Entity Recognition"),
                margin=dict(l=10, r=10, t=60, b=10),
                title_font=dict(size=17, color="#10202F"),
                font=dict(family="Inter, sans-serif", color="#37536B"),
                colorway=PALETTE,
            )
            if hasattr(fig.data[0], "marker") and process in (
                "Tokenization",
                "Stopword Removal",
                "Lemmatization",
            ):
                fig.update_traces(marker_color="#146C63")
            st.plotly_chart(fig, use_container_width=True)

        # spaCy's own renderers, where they add something
        if process == "Named Entity Recognition" and doc.ents:
            st.markdown(
                '<div class="mini-label">Entities in context</div>',
                unsafe_allow_html=True,
            )
            html_block(displacy.render(doc, style="ent"), height=320)
        if process == "POS Tagging":
            st.markdown(
                '<div class="mini-label">Dependency parse of the first sentence</div>',
                unsafe_allow_html=True,
            )
            first_sent = list(doc.sents)[0].as_doc()
            svg = displacy.render(
                first_sent,
                style="dep",
                options={
                    "compact": True,
                    "distance": 110,
                    "color": "#10202F",
                    "bg": "#ffffff",
                    "font": "Inter",
                },
            )
            html_block(svg, height=420)
else:
    st.markdown(
        '<div class="empty-state">Enter some text and pick a process to see results here.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="page-foot">NLP Text Analysis Dashboard · spaCy en_core_web_sm · '
    "Streamlit</div>",
    unsafe_allow_html=True,
)
