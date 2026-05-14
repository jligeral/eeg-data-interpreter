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

st.markdown("""
<style>
/* ── Sub-tab bar ─────────────────────────────────────────── */
div[data-testid="stTabs"] > div > div[role="tablist"] {
    gap: 0;
    border-bottom: 1px solid rgba(0,0,0,0.1);
}
div[data-testid="stTabs"] button[role="tab"] {
    border-radius: 0 !important;
    border-right: 1px solid rgba(0,0,0,0.1) !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 20px !important;
    font-size: 14px;
    color: #666;
    background: transparent;
    margin: 0;
}
div[data-testid="stTabs"] button[role="tab"]:last-child {
    border-right: none !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background-color: rgba(79,102,255,0.07) !important;
    border-bottom: 2px solid #4F66FF !important;
    color: #4F66FF !important;
    font-weight: 600 !important;
}
div[data-testid="stTabs"] button[role="tab"]:hover:not([aria-selected="true"]) {
    background-color: rgba(0,0,0,0.03) !important;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Pipeline (cached so re-renders don't re-run)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_pipeline():
    from preprocessor.preprocess import preprocess
    from feature_extractor.extract import extract
    from reasoning_agent.agent import analyze
    from output_formatter import format_report
    return preprocess, extract, analyze, format_report


def run_pipeline(filepath: str, label: str = ""):
    preprocess, extract, analyze, format_report = _load_pipeline()
    tag = f" ({label})" if label else ""
    with st.spinner(f"Preprocessing{tag}…"):
        feat = extract(preprocess(filepath))
    with st.spinner(f"Running reasoning agent{tag} — 30–60 s…"):
        reasoning = analyze(feat)
    report = format_report(feat, reasoning)
    return feat, reasoning, report


# ---------------------------------------------------------------------------
# Rendering helpers — single-report sections
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
    c4.metric("LEAD Windows", f.n_lead_windows if f.lead_embeddings_available else "N/A")

    if f.confounds:
        st.warning("Artifact flags: " + ", ".join(f.confounds))

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

    if f.psd_by_region and f.psd_freqs:
        st.subheader("Power Spectral Density")
        _BAND_SPANS = [
            (0.5, 4.0,  "delta", "rgba(100,100,220,0.07)"),
            (4.0, 8.0,  "theta", "rgba(80,180,80,0.07)"),
            (8.0, 13.0, "alpha", "rgba(220,170,40,0.10)"),
            (13.0, 30.0,"beta",  "rgba(220,80,80,0.07)"),
        ]
        fig_psd = go.Figure()
        for region, psd_vals in f.psd_by_region.items():
            fig_psd.add_trace(go.Scatter(
                x=f.psd_freqs, y=psd_vals,
                mode="lines", name=region, line=dict(width=2),
            ))
        for lo, hi, band_label, color in _BAND_SPANS:
            fig_psd.add_vrect(
                x0=lo, x1=hi, fillcolor=color, layer="below", line_width=0,
                annotation_text=band_label, annotation_position="top left",
                annotation=dict(font_size=10, font_color="#888"),
            )
        apf = next((b.value for b in f.biomarkers if "alpha peak" in b.name.lower()), None)
        if apf:
            fig_psd.add_vline(x=apf, line_dash="dash", line_color="orange", line_width=1.5)
            fig_psd.add_annotation(
                x=apf, y=0.5, yref="paper",
                text=f"α peak {apf:.1f} Hz",
                showarrow=False,
                xanchor="left", yanchor="middle",
                font=dict(size=10, color="orange"),
                bgcolor="rgba(255,255,255,0.75)",
                borderpad=2,
            )
        fig_psd.update_layout(
            xaxis_title="Frequency (Hz)",
            yaxis_title="Relative Power (area-normalised per region)",
            legend_title="Region",
            height=360,
            margin=dict(t=30, b=20),
        )
        st.plotly_chart(fig_psd, use_container_width=True)

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

    if f.lead_embeddings_available and f.lead_embedding_stats:
        st.subheader("LEAD Embedding Stats")
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
            fig2 = go.Figure(go.Bar(
                x=list(norms.keys()), y=list(norms.values()), marker_color="steelblue"
            ))
            fig2.update_layout(
                title="Per-Channel Embedding Activation",
                xaxis_title="Channel",
                yaxis_title="L2 Norm",
                height=280,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

    if f.lead_ad_per_window and len(f.lead_ad_per_window) > 1:
        st.subheader("AD Probability Over Time")
        window_times = [i * 2 for i in range(len(f.lead_ad_per_window))]
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=window_times, y=f.lead_ad_per_window,
            mode="lines+markers", name="AD probability",
            line=dict(color="crimson", width=2),
            marker=dict(size=4),
            fill="tozeroy", fillcolor="rgba(220,20,60,0.07)",
        ))
        fig_t.add_hline(
            y=0.60, line_dash="dash", line_color="gray", line_width=1,
            annotation_text="0.60 threshold", annotation_position="bottom right",
            annotation=dict(font_size=10, font_color="gray"),
        )
        fig_t.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="AD Probability",
            yaxis=dict(range=[0, 1]),
            height=280,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_t, use_container_width=True)


def render_differential(report):
    dd = report.differential_diagnosis

    if dd.clinical_impression:
        st.info(f"**Clinical impression:** {dd.clinical_impression}")

    if dd.clinical_flags:
        for flag in dd.clinical_flags:
            st.warning(f"⚠️ {flag}")

    for h in dd.hypotheses:
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
# Rendering helpers — multi-file views
# ---------------------------------------------------------------------------

_POSTERIOR_DEFAULT = ["O1", "O2", "P3", "P4", "T5", "T6"]
_WAVEFORM_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22",
]


def _waveform_fig(snapshot: dict, selected: list[str], title: str):
    """Return a stacked-channel Plotly figure for a waveform snapshot."""
    ch_names = snapshot.get("channels", [])
    times = snapshot.get("times", [])
    data = snapshot.get("data", [])

    ch_map = {ch: arr for ch, arr in zip(ch_names, data)}
    ordered = [ch for ch in selected if ch in ch_map]
    if not ordered:
        return None

    import numpy as np
    arrays = [np.array(ch_map[ch]) for ch in ordered]
    pp = np.array([a.max() - a.min() for a in arrays])
    offset_step = float(np.percentile(pp, 75) * 1.5) if pp.mean() > 0 else 100.0

    fig = go.Figure()
    tick_vals, tick_texts = [], []

    for i, (ch, arr) in enumerate(zip(ordered, arrays)):
        offset = i * offset_step
        fig.add_trace(go.Scatter(
            x=times,
            y=(arr + offset).tolist(),
            mode="lines",
            name=ch,
            line=dict(width=0.9, color=_WAVEFORM_COLORS[i % len(_WAVEFORM_COLORS)]),
            showlegend=False,
            hovertemplate=f"<b>{ch}</b><br>t=%{{x:.2f}}s  %{{customdata:.1f}} µV<extra></extra>",
            customdata=arr.tolist(),
        ))
        tick_vals.append(offset)
        tick_texts.append(ch)

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis_title="Time (s)",
        yaxis=dict(
            tickvals=tick_vals,
            ticktext=tick_texts,
            showgrid=False,
            zeroline=False,
        ),
        height=max(320, len(ordered) * 45),
        margin=dict(t=40, b=30, l=55, r=10),
    )
    return fig


def render_waveform(feat):
    """Side-by-side before/after waveform viewer."""
    pre = feat.waveform_pre
    post = feat.waveform_post

    if not pre or not post:
        st.info("Waveform snapshots not available for this recording.")
        return

    all_channels = pre.get("channels", [])
    default = [ch for ch in _POSTERIOR_DEFAULT if ch in all_channels] or all_channels[:6]

    selected = st.multiselect(
        "Channels",
        options=all_channels,
        default=default,
        help="Select channels to display. Both panels show the same channels.",
    )

    if not selected:
        st.caption("Select at least one channel above.")
        return

    c1, c2 = st.columns(2)
    with c1:
        fig = _waveform_fig(pre, selected, "Before preprocessing")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = _waveform_fig(post, selected, "After preprocessing")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"First {pre['times'][-1]:.1f} s shown · "
        f"sampled at {pre.get('sfreq', '?'):.0f} Hz · "
        "Left: post channel-validation, pre-filter · "
        "Right: post bandpass (0.5–45 Hz) + average reference"
    )


def render_single_report(report, feat, label: str):
    """5-sub-tab report for the selected file."""
    st.caption(f"Viewing: **{label}**")
    if report.reasoning_error:
        st.error(f"Reasoning agent error: {report.reasoning_error}")

    sub1, sub2, sub3, sub4, sub5 = st.tabs([
        "Objective Findings",
        "Differential Diagnosis",
        "Uncertainty & Confidence",
        "Next Steps",
        "Waveform Viewer",
    ])
    with sub1:
        render_objective_findings(report)
    with sub2:
        render_differential(report)
    with sub3:
        render_uncertainty(report)
    with sub4:
        render_next_steps(report)
    with sub5:
        render_waveform(feat)

    st.divider()
    dc1, dc2 = st.columns(2)
    safe = label.replace(" ", "_").replace("/", "-")
    with dc1:
        st.download_button(
            "⬇️ Download JSON report",
            data=report.to_json(),
            file_name=f"eeg_report_{safe}.json",
            mime="application/json",
            key=f"dl_json_{safe}",
        )
    with dc2:
        st.download_button(
            "⬇️ Download text report",
            data=report.to_text(),
            file_name=f"eeg_report_{safe}.txt",
            mime="text/plain",
            key=f"dl_txt_{safe}",
        )


def render_compare(results: dict):
    """Side-by-side comparison of up to 2 selected reports."""
    labels = list(results.keys())
    reports = [results[l]["report"] for l in labels]

    if len(labels) == 0:
        st.info("Select up to 2 files from the **History** panel on the left to compare.")
        return
    if len(labels) == 1:
        st.info(f"**{labels[0]}** selected — pick one more file in the History panel to compare.")
        return

    n = len(reports)
    _COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

    # ── Summary row ──────────────────────────────────────────────────────────
    st.subheader("Summary")
    cols = st.columns(n)
    for col, label, report in zip(cols, labels, reports):
        f = report.objective_findings
        u = report.uncertainty
        with col:
            st.markdown(f"**{label}**")
            st.write(_quality_badge(f.data_quality))
            st.metric("Confidence", f"{u.overall_confidence:.0%}",
                      delta=u.overall_confidence_label.upper())
            adp = f.lead_ad_probability
            st.metric("AD Probability", f"{adp:.3f}" if adp is not None else "N/A")
            if report.differential_diagnosis.hypotheses:
                top = report.differential_diagnosis.hypotheses[0]
                st.caption(f"Top: {top.name} ({top.confidence:.0%})")
            if report.reasoning_error:
                st.error("Reasoning failed")

    st.divider()

    # ── Biomarkers table ─────────────────────────────────────────────────────
    st.subheader("Key AD Biomarkers")
    all_names = list(dict.fromkeys(
        b.name for r in reports for b in r.objective_findings.biomarkers
    ))
    header_cols = st.columns([3] + [2] * n)
    header_cols[0].markdown("**Biomarker**")
    for col, label in zip(header_cols[1:], labels):
        col.markdown(f"**{label}**")
    st.divider()
    for name in all_names:
        row = st.columns([3] + [2] * n)
        row[0].write(name)
        for col, report in zip(row[1:], reports):
            bmap = {b.name: b for b in report.objective_findings.biomarkers}
            b = bmap.get(name)
            if b is None or b.value is None:
                col.write("—")
            elif b.abnormal:
                col.markdown(f"🔴 **{b.value:.3f}** {b.unit}")
            else:
                col.markdown(f"🟢 {b.value:.3f} {b.unit}")

    st.divider()

    # ── Band power chart ─────────────────────────────────────────────────────
    if all(r.objective_findings.band_power_relative for r in reports):
        st.subheader("Relative Band Power")
        bands = list(reports[0].objective_findings.band_power_relative.keys())
        fig = go.Figure()
        for i, (label, report) in enumerate(zip(labels, reports)):
            bp = report.objective_findings.band_power_relative
            values = [
                float(np.mean(list(bp[band].values()))) if bp.get(band) else 0.0
                for band in bands
            ]
            fig.add_trace(go.Bar(
                name=label, x=bands, y=values,
                marker_color=_COLORS[i % len(_COLORS)],
            ))
        fig.update_layout(
            barmode="group",
            xaxis_title="Band",
            yaxis_title="Mean Relative Power (across regions)",
            legend_title="File",
            height=350,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

    # ── LEAD embedding stats ─────────────────────────────────────────────────
    if any(r.objective_findings.lead_embeddings_available for r in reports):
        st.subheader("LEAD Embedding Stats")
        cols = st.columns(n)
        for col, label, report in zip(cols, labels, reports):
            f = report.objective_findings
            with col:
                st.markdown(f"**{label}**")
                if not f.lead_embeddings_available:
                    st.caption("LEAD unavailable")
                    continue
                es = f.lead_embedding_stats
                st.metric("Windows", f.n_lead_windows)
                st.metric("Inter-window Var",
                          f"{es.get('inter_window_variance', 0):.4f}",
                          help="Higher = more instability across windows")
                st.metric("Embedding Norm", f"{es.get('embedding_norm', 0):.2f}")
                pca = es.get("pca_top1_variance_ratio")
                st.metric("PCA Top-1 Var", f"{pca:.3f}" if pca else "N/A")
        st.divider()

    # ── Clinical impressions ─────────────────────────────────────────────────
    st.subheader("Clinical Impressions")
    cols = st.columns(n)
    for col, label, report in zip(cols, labels, reports):
        dd = report.differential_diagnosis
        with col:
            st.markdown(f"**{label}**")
            if dd.clinical_impression:
                st.info(dd.clinical_impression)
            for flag in dd.clinical_flags:
                st.warning(f"⚠️ {flag}")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

st.title("EEG Interpreter")
st.caption("Alzheimer's-focused quantitative EEG reasoning pipeline · Powered by LEAD + Qwen3:8b")

_EEG_EXTS = {".fif", ".edf", ".bdf", ".set", ".gdf", ".cnt", ".vhdr",
             ".csv", ".tsv", ".npz", ".npy", ".mat"}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Input")
    uploaded_files = st.file_uploader(
        "Upload EEG files",
        type=["edf", "bdf", "fif", "set", "gdf", "cnt", "vhdr", "csv", "tsv", "npz", "npy", "mat"],
        accept_multiple_files=True,
        help="Supported: EDF, BDF, FIF, SET, GDF, CNT, VHDR, CSV, NPZ, MAT",
    )

    st.divider()
    run_btn = st.button("Analyze EEG", type="primary", use_container_width=True)
    if st.button("Clear results", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("compare_"):
                del st.session_state[k]
        for key in ("results", "selected_file", "view_mode"):
            st.session_state.pop(key, None)
        st.rerun()

    # ── History ───────────────────────────────────────────────────────────────
    _sidebar_results: dict = st.session_state.get("results", {})
    if _sidebar_results:
        st.divider()
        st.markdown("**History**")

        _n_compare = sum(
            1 for l in _sidebar_results
            if st.session_state.get(f"compare_{l}", False)
        )
        _viewing = st.session_state.get("selected_file")

        for _label in _sidebar_results:
            _is_active = _label == _viewing and st.session_state.get("view_mode") == "file"
            _cb_checked = st.session_state.get(f"compare_{_label}", False)
            _cb_disabled = _n_compare >= 2 and not _cb_checked

            _c1, _c2 = st.columns([1, 6])
            with _c1:
                st.checkbox(
                    "",
                    key=f"compare_{_label}",
                    disabled=_cb_disabled,
                    label_visibility="collapsed",
                    help="Deselect another file first" if _cb_disabled else "Select for comparison",
                )
            with _c2:
                if st.button(
                    _label,
                    key=f"view_{_label}",
                    type="primary" if _is_active else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["selected_file"] = _label
                    st.session_state["view_mode"] = "file"
                    st.rerun()

        _compare_selection = [
            l for l in _sidebar_results
            if st.session_state.get(f"compare_{l}", False)
        ]
        if len(_compare_selection) == 2:
            st.divider()
            if st.button("Compare selected", use_container_width=True):
                st.session_state["view_mode"] = "compare"
                st.rerun()

# ---------------------------------------------------------------------------
# Collect all input paths with display labels
# ---------------------------------------------------------------------------
input_items: list[tuple[str, str]] = []  # (label, filepath)

for uf in (uploaded_files or []):
    suffix = Path(uf.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uf.read())
    tmp.flush()
    input_items.append((uf.name, tmp.name))

# ---------------------------------------------------------------------------
# Run pipeline for new inputs
# ---------------------------------------------------------------------------
if run_btn:
    if not input_items:
        st.error("Please upload or select at least one file.")
        st.stop()

    if "results" not in st.session_state:
        st.session_state["results"] = {}

    for label, filepath in input_items:
        if label in st.session_state["results"]:
            continue  # already processed — skip
        try:
            feat, reasoning, report = run_pipeline(filepath, label=label)
            st.session_state["results"][label] = {
                "feat": feat,
                "reasoning": reasoning,
                "report": report,
            }
            # Auto-select the most recently processed file
            st.session_state["selected_file"] = label
            st.session_state["view_mode"] = "file"
        except Exception as e:
            st.error(f"Pipeline failed for **{label}**: {e}")

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
results: dict = st.session_state.get("results", {})
_view_mode = st.session_state.get("view_mode", "file")
_selected_file = st.session_state.get("selected_file")

if not results:
    st.info("Upload EEG files and click **▶ Run pipeline** to begin.")
elif _view_mode == "compare":
    _compare_labels = [l for l in results if st.session_state.get(f"compare_{l}", False)]
    render_compare({l: results[l] for l in _compare_labels})
elif _selected_file and _selected_file in results:
    render_single_report(results[_selected_file]["report"], results[_selected_file]["feat"], _selected_file)
else:
    st.info("Select a file from the History panel to view its report.")
