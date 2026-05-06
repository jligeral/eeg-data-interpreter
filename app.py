"""
EEG Data Interpreter — Streamlit frontend.

Run with:
    streamlit run app.py
"""
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EEG Interpreter",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Pipeline (cached so re-renders don't re-run)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_pipeline():
    """Import pipeline modules once."""
    from preprocessor.preprocess import preprocess
    from feature_extractor.extract import extract
    from reasoning_agent.agent import analyze
    from output_formatter import format_report
    return preprocess, extract, analyze, format_report


def run_pipeline(filepath: str):
    preprocess, extract, analyze, format_report = _load_pipeline()
    with st.spinner("Preprocessing…"):
        feat = extract(preprocess(filepath))
    with st.spinner("Running reasoning agent (this may take 30–60 s)…"):
        reasoning = analyze(feat)
    report = format_report(feat, reasoning)
    return feat, reasoning, report


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _quality_badge(quality: str) -> str:
    return {"good": "🟢 Good", "moderate": "🟡 Moderate", "poor": "🔴 Poor"}.get(
        quality, f"⚪ {quality.capitalize()}"
    )


def _confidence_color(c: float) -> str:
    if c >= 0.70:
        return "green"
    if c >= 0.40:
        return "orange"
    return "red"


def render_objective_findings(report):
    f = report.objective_findings
    st.subheader("Recording Info")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Quality", _quality_badge(f.data_quality))
    c2.metric("Channels", f.n_channels)
    c3.metric("Sampling Rate", f"{f.sampling_rate_hz:.0f} Hz")
    c4.metric("LaBraM Windows", f.n_lead_windows if f.lead_embeddings_available else "N/A")

    if f.confounds:
        st.warning("Artifact flags: " + ", ".join(f.confounds))

    # Key biomarkers table
    st.subheader("Key AD Biomarkers")
    cols = st.columns([3, 2, 2, 2])
    cols[0].markdown("**Biomarker**")
    cols[1].markdown("**Value**")
    cols[2].markdown("**Reference**")
    cols[3].markdown("**Status**")
    st.divider()
    for b in f.biomarkers:
        c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
        c1.write(b.name)
        val = f"{b.value:.3f} {b.unit}" if b.value is not None else "N/A"
        c2.write(val)
        c3.write(b.reference_range)
        if b.value is None:
            c4.write("—")
        elif b.abnormal:
            c4.markdown("🔴 **Abnormal**")
        else:
            c4.markdown("🟢 Normal")
        if b.note:
            st.caption(f"   ↳ {b.note}")

    # Band power chart
    if f.band_power_relative:
        st.subheader("Relative Band Power by Region")
        bands = list(f.band_power_relative.keys())
        regions = list(next(iter(f.band_power_relative.values())).keys())
        fig = go.Figure()
        for region in regions:
            values = [f.band_power_relative[band].get(region, 0) for band in bands]
            fig.add_trace(go.Bar(name=region, x=bands, y=values))
        fig.update_layout(
            barmode="group",
            xaxis_title="Band",
            yaxis_title="Relative Power",
            legend_title="Region",
            height=350,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # LaBraM model stats
    if f.lead_embeddings_available and f.lead_embedding_stats:
        st.subheader("LaBraM Model Stats")
        es = f.lead_embedding_stats
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Inter-window Variance", f"{es.get('inter_window_variance', 0):.4f}",
                   help="Higher = more dynamic instability across windows")
        ec2.metric("Embedding Norm", f"{es.get('embedding_norm', 0):.2f}")
        pca = es.get("pca_top1_variance_ratio")
        ec3.metric("PCA Top-1 Variance", f"{pca:.3f}" if pca else "N/A",
                   help="Lower = more diffuse activation pattern")

        if es.get("channel_block_norms"):
            norms = es["channel_block_norms"]
            channels = list(norms.keys())
            values = list(norms.values())
            fig2 = go.Figure(go.Bar(x=channels, y=values, marker_color="steelblue"))
            fig2.update_layout(
                title="Per-Channel Embedding Activation",
                xaxis_title="Channel",
                yaxis_title="L2 Norm",
                height=280,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)


def render_differential(report):
    dd = report.differential_diagnosis

    if dd.clinical_impression:
        st.info(f"**Clinical impression:** {dd.clinical_impression}")

    if dd.clinical_flags:
        for flag in dd.clinical_flags:
            st.warning(f"⚠️ {flag}")

    for h in dd.hypotheses:
        color = _confidence_color(h.confidence)
        with st.expander(
            f"#{h.rank}  {h.name}  —  {h.confidence:.0%} ({h.confidence_label.upper()})",
            expanded=(h.rank == 1),
        ):
            st.progress(h.confidence, text=f"{h.confidence:.0%} confidence")
            st.write(h.description)

            if h.supporting_evidence or h.contradicting_evidence:
                sc, cc = st.columns(2)
                with sc:
                    if h.supporting_evidence:
                        st.markdown("**Supporting**")
                        for e in h.supporting_evidence:
                            st.markdown(f"✅ {e}")
                with cc:
                    if h.contradicting_evidence:
                        st.markdown("**Against**")
                        for e in h.contradicting_evidence:
                            st.markdown(f"❌ {e}")


def render_uncertainty(report):
    u = report.uncertainty

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(
            "Overall Confidence",
            f"{u.overall_confidence:.0%}",
            delta=u.overall_confidence_label.upper(),
            delta_color="normal" if u.overall_confidence >= 0.5 else "inverse",
        )
    with c2:
        st.write(f"**Data quality:** {u.data_quality_impact}")

    if u.limiting_factors:
        st.subheader("Limiting Factors")
        for lf in u.limiting_factors:
            st.markdown(f"- {lf}")

    if u.caveats:
        st.subheader("Caveats")
        for c in u.caveats:
            st.markdown(f"- ⚠️ {c}")

    if u.what_would_change_assessment:
        st.subheader("What Would Change This Assessment")
        for w in u.what_would_change_assessment:
            st.markdown(f"- {w}")


def render_next_steps(report):
    if not report.next_steps.recommended_actions:
        st.write("No specific next steps recommended.")
        return
    for i, action in enumerate(report.next_steps.recommended_actions, 1):
        st.markdown(f"**{i}.** {action}")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

st.title("🧠 EEG Interpreter")
st.caption("Multiclass EEG reasoning pipeline · Powered by LaBraM + EEG biomarkers + Qwen3:8b")

_EEG_EXTS = {".fif", ".edf", ".bdf", ".set", ".gdf", ".cnt", ".vhdr",
             ".csv", ".tsv", ".npz", ".npy", ".mat"}

_DEMO_DIR  = Path(__file__).parent / "data" / "demo"
_SYNTH_DIR = Path(__file__).parent / "preprocessor" / "data" / "generated"


def _label_for(path: Path) -> str:
    """Infer AD/HC label from parent directory name."""
    parent = path.parent.name.upper()
    if parent == "AD":
        return "🔴 AD"
    if parent == "HC":
        return "🟢 HC"
    return ""


def _list_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(f for f in directory.rglob("*") if f.suffix.lower() in _EEG_EXTS)


# Sidebar
with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Upload EEG file",
        type=["edf", "bdf", "fif", "set", "gdf", "cnt", "vhdr", "csv", "tsv", "npz", "npy", "mat"],
        help="Supported: EDF, BDF, FIF, SET, GDF, CNT, VHDR, CSV, NPZ, MAT",
    )

    st.divider()

    # Real demo files (data/demo/AD/ and data/demo/HC/)
    demo_files = _list_files(_DEMO_DIR)
    if demo_files:
        st.markdown("**Real EEG files** *(ADFTD dataset)*")
        demo_options = {f"{_label_for(f)}  {f.name}": f for f in demo_files}
        selected_demo = st.selectbox("Subject", ["(none)"] + list(demo_options.keys()))
    else:
        selected_demo = "(none)"
        st.markdown("**Real EEG files**")
        st.caption("None found. Run `python scripts/download_demo_data.py` to fetch AD + HC subjects.")

    st.divider()

    # Synthetic test files
    st.markdown("**Synthetic test files**")
    synth_files = _list_files(_SYNTH_DIR)
    synth_options = {f.name: f for f in synth_files}
    selected_synth = st.selectbox("Test file", ["(none)"] + list(synth_options.keys()))

    run_btn = st.button("Run pipeline", type="primary", use_container_width=True)

# Determine input path
input_path = None
if uploaded:
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    input_path = tmp.name
elif selected_demo != "(none)" and demo_files:
    input_path = str(demo_options[selected_demo])
elif selected_synth != "(none)":
    input_path = str(synth_options[selected_synth])

# Run and display
if run_btn:
    if not input_path:
        st.error("Please upload a file or select a test file first.")
        st.stop()

    try:
        feat, reasoning, report = run_pipeline(input_path)
        st.session_state["report"] = report
        st.session_state["feat"] = feat
        st.session_state["reasoning"] = reasoning
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

if "report" in st.session_state:
    report = st.session_state["report"]

    # Error banner if reasoning failed
    if report.reasoning_error:
        st.error(f"Reasoning agent error: {report.reasoning_error}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Objective Findings",
        "🩺 Differential Diagnosis",
        "🎯 Uncertainty & Confidence",
        "➡️ Next Steps",
    ])

    with tab1:
        render_objective_findings(report)
    with tab2:
        render_differential(report)
    with tab3:
        render_uncertainty(report)
    with tab4:
        render_next_steps(report)

    # Downloads
    st.divider()
    dc1, dc2 = st.columns(2)
    with dc1:
        st.download_button(
            "⬇️ Download JSON report",
            data=report.to_json(),
            file_name="eeg_report.json",
            mime="application/json",
        )
    with dc2:
        st.download_button(
            "⬇️ Download text report",
            data=report.to_text(),
            file_name="eeg_report.txt",
            mime="text/plain",
        )
else:
    st.info("Select a file and click **Run pipeline** to begin.")
