# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re

# =====================================================
# Page configuration + light UI polish
# =====================================================
st.set_page_config(page_title="CRISPR Guide Design Mini-Tool", layout="wide")

CUSTOM_CSS = """
<style>
/* soften page */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
div[data-testid="stMetric"] { background: rgba(0,0,0,0.03); padding: 12px 14px; border-radius: 14px; }
.badge {
  display:inline-block; padding:4px 10px; border-radius:999px;
  background: rgba(0, 180, 120, 0.12); border: 1px solid rgba(0, 180, 120, 0.30);
  font-size: 12px; font-weight: 600;
}
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}
.card {
  background: rgba(0,0,0,0.02);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 16px;
  padding: 14px 16px;
}
.small { font-size: 13px; opacity: 0.9; }
hr { margin: 0.8rem 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =====================================================
# CRISPR Cas systems and PAMs (literature-based, educational)
# =====================================================
CAS_PAM_TABLE = {
    "SpCas9 (S. pyogenes, Type II)": "NGG",
    "SaCas9 (S. aureus, Type II-A)": "NNGRRT",
    "StCas9 (S. thermophilus, Type II-A)": "NNAGAA",
    "S. solfataricus (Type I-A1)": "CCN",
    "S. solfataricus (Type I-A2)": "TCN",
    "H. walsbyi (Type I-B)": "TTC",
    "E. coli (Type I-E)": "AWG",
    "E. coli (Type I-F)": "CC",
    "P. aeruginosa (Type I-F)": "CC",
    "FnCas12a (F. novicida, Type V-A)": "TTTN",
    "AsCas12a (Acidaminococcus, Type V-A)": "TTTN",
}

# =====================================================
# IUPAC ambiguity codes
# =====================================================
IUPAC_CODES = {
    "A": ["A"], "T": ["T"], "G": ["G"], "C": ["C"],
    "N": ["A", "T", "G", "C"],
    "R": ["A", "G"],
    "Y": ["C", "T"],
    "W": ["A", "T"],
    "V": ["A", "C", "G"],
}

def pam_matches(seq_fragment: str, pam_pattern: str) -> bool:
    if len(seq_fragment) != len(pam_pattern):
        return False
    for b, p in zip(seq_fragment, pam_pattern):
        if b not in IUPAC_CODES.get(p, []):
            return False
    return True

# =====================================================
# Helper functions
# =====================================================
def clean_sequence(seq: str) -> str:
    seq = seq.upper()
    return "".join(b for b in seq if b in {"A", "T", "G", "C"})

def parse_fasta(text: str) -> str:
    return "".join(l.strip() for l in text.splitlines() if l.strip() and not l.startswith(">"))

def gc_content(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq) * 100 if seq else 0.0

def complement(b: str) -> str:
    return {"A": "T", "T": "A", "G": "C", "C": "G"}.get(b, "N")

def rev_complement(seq: str) -> str:
    return "".join(complement(b) for b in seq[::-1])

def self_complementarity_score(seq: str, window: int = 4) -> int:
    # simple heuristic: count short reverse-complement windows that appear within the guide
    score = 0
    if len(seq) < window:
        return 0
    for i in range(len(seq) - window + 1):
        w = seq[i:i + window]
        if rev_complement(w) in seq:
            score += 1
    return score

def off_target_score(seq: str, guide: str, start_idx: int, max_mismatches: int = 5) -> int:
    # educational-only heuristic: scan same input sequence for similar matches
    score = 0
    L = len(guide)
    for i in range(len(seq) - L + 1):
        if i == start_idx:
            continue
        mismatches = sum(a != b for a, b in zip(seq[i:i + L], guide))
        if mismatches <= max_mismatches:
            score += 1
    return score

def find_guides_forward(seq: str, guide_len: int, pam: str):
    guides = []
    pam_len = len(pam)
    for i in range(len(seq) - guide_len - pam_len + 1):
        pam_seq = seq[i + guide_len:i + guide_len + pam_len]
        if pam_matches(pam_seq, pam):
            guides.append({
                "guide_start": i,
                "guide_end": i + guide_len,                 # 0-based, exclusive
                "pam_start": i + guide_len,
                "pam_end": i + guide_len + pam_len,         # exclusive
                "guide_seq": seq[i:i + guide_len],
                "pam_seq": pam_seq
            })
    return guides

def compute_guide_table(seq: str, guide_len: int, pam: str, advanced: bool = True) -> pd.DataFrame:
    rows = []
    for g in find_guides_forward(seq, guide_len, pam):
        gc = gc_content(g["guide_seq"])
        self_c = self_complementarity_score(g["guide_seq"]) if advanced else np.nan
        off_t = off_target_score(seq, g["guide_seq"], g["guide_start"]) if advanced else np.nan

        # total heuristic: keep it simple, explainable
        total = (abs(gc - 50) / 5) + (self_c if not np.isnan(self_c) else 0) + ((off_t * 2) if not np.isnan(off_t) else 0)

        rows.append({
            "Rank": None,  # filled later
            "Guide Start (0-based)": g["guide_start"],
            "Guide End (0-based, excl)": g["guide_end"],
            "Guide Sequence (5'→3')": g["guide_seq"],
            "PAM Pattern": pam,
            "Matched PAM (instance)": g["pam_seq"],
            "GC %": round(gc, 2),
            "Self-Complementarity": self_c,
            "Off-target-like Matches": off_t,
            "Total Score (lower is better)": round(total, 2),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("Total Score (lower is better)", ascending=True).reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1)
    return df

def highlight_sequence_html(seq: str, guides_meta, width: int = 70) -> str:
    # mark each base: normal / guide / pam
    pos = ["normal"] * len(seq)
    for g in guides_meta:
        for i in range(max(0, g["guide_start"]), min(len(seq), g["guide_end"])):
            pos[i] = "guide"
        for i in range(max(0, g["pam_start"]), min(len(seq), g["pam_end"])):
            pos[i] = "pam"

    lines = []
    for i in range(0, len(seq), width):
        block = []
        for j in range(i, min(i + width, len(seq))):
            if pos[j] == "guide":
                block.append(f"<span style='color:#0A8F5B;font-weight:700'>{seq[j]}</span>")
            elif pos[j] == "pam":
                block.append(f"<span style='color:#D83A3A;font-weight:800'>{seq[j]}</span>")
            else:
                block.append(seq[j])
        lines.append(f"<span style='color:gray'>[{i:04d}] </span>" + "".join(block))
    return "<br>".join(lines)

def slice_context(seq: str, guide_start: int, guide_end: int, pam_end: int, flank: int = 12) -> str:
    left = max(0, guide_start - flank)
    right = min(len(seq), pam_end + flank)
    return seq[left:right], left, right

# =====================================================
# Sidebar (clean + informative)
# =====================================================
st.sidebar.markdown("## ⚙️ Settings")

cas_choice = st.sidebar.selectbox("CRISPR system / organism", list(CAS_PAM_TABLE.keys()))
pam_choice = CAS_PAM_TABLE[cas_choice]

st.sidebar.markdown("### 🧬 PAM in use")
st.sidebar.code(pam_choice)

st.sidebar.markdown(f"**PAM length:** `{len(pam_choice)}` nt")

with st.sidebar.expander("IUPAC legend", expanded=False):
    st.markdown(
        """
- **N** = A / T / G / C  
- **R** = A or G  
- **Y** = C or T  
- **W** = A or T  
- **V** = A / C / G
"""
    )

guide_len = st.sidebar.number_input("Guide length (nt)", min_value=18, max_value=24, value=20, step=1)
advanced_scores = st.sidebar.checkbox("Compute advanced scores", value=True)
gc_min, gc_max = st.sidebar.slider("Filter by GC %", 0, 100, (30, 80))

st.sidebar.markdown("---")
st.sidebar.warning("Educational only — **not** for lab / clinical design.", icon="⚠️")

# =====================================================
# Header
# =====================================================
st.markdown(
    """
<div class="card">
  <div style="display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap;">
    <div>
      <h2 style="margin:0;">🧬 CRISPR Guide Design Mini-Tool</h2>
      <div class="small">Scan DNA for PAMs → extract guides → score & visualize (teaching-only).</div>
    </div>
    <div style="display:flex; align-items:center; gap:10px;">
      <span class="badge">PAM: <span class="mono">""" + pam_choice + """</span></span>
      <span class="badge">Guide: <span class="mono">""" + str(guide_len) + """ nt</span></span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tabs = st.tabs(["1️⃣ Input", "2️⃣ Guides", "3️⃣ Sequence Map", "4️⃣ Plots + Interpretation"])

# =====================================================
# 1) Input
# =====================================================
with tabs[0]:
    st.subheader("1️⃣ Input DNA sequence")
    cA, cB = st.columns([2, 1], gap="large")

    with cA:
        seq_text = st.text_area(
            "Paste DNA or FASTA (A/T/G/C only will be kept):",
            height=220,
            placeholder="Example:\n>my_seq\nATGCGT... (you can paste FASTA too)"
        )

    with cB:
        fasta = st.file_uploader("Or upload FASTA / TXT", ["fa", "fasta", "txt"])
        st.caption("Tip: If you upload a file, it overrides the text box.")

    raw_seq = ""
    if fasta is not None:
        raw_seq = fasta.read().decode(errors="ignore")
        raw_seq = parse_fasta(raw_seq)
    else:
        # if user pasted FASTA-like text, parse it too
        raw_seq = parse_fasta(seq_text) if ">" in (seq_text or "") else (seq_text or "")

    clean_seq = clean_sequence(raw_seq)

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1:
        st.metric("Sequence length", f"{len(clean_seq)} bp")
    with c2:
        st.metric("GC % (whole input)", f"{gc_content(clean_seq):.2f}" if clean_seq else "—")
    with c3:
        st.metric("Selected system", cas_choice.split(" (")[0])
    with c4:
        st.metric("PAM pattern", pam_choice)

    if not clean_seq:
        st.info("Provide a DNA sequence to begin.")
        st.stop()

    if len(clean_seq) < (guide_len + len(pam_choice)):
        st.error("Sequence is too short for the selected guide length + PAM.")
        st.stop()

# =====================================================
# 2) Guides
# =====================================================
with tabs[1]:
    st.subheader("2️⃣ Detected guides")

    @st.cache_data(show_spinner=False)
    def _cached_guides(seq: str, gl: int, pam: str, adv: bool):
        return compute_guide_table(seq, gl, pam, adv)

    df = _cached_guides(clean_seq, guide_len, pam_choice, advanced_scores)

    if df.empty:
        st.warning("No guides found for this PAM on the forward strand with current settings.")
        st.stop()

    # Filter by GC
    df_filt = df[(df["GC %"] >= gc_min) & (df["GC %"] <= gc_max)].copy()

    top = st.columns([1, 1, 1, 1], gap="large")
    top[0].metric("Guides found (raw)", len(df))
    top[1].metric("Guides after GC filter", len(df_filt))
    top[2].metric("Best score (filtered)", f"{df_filt['Total Score (lower is better)'].min():.2f}" if len(df_filt) else "—")
    top[3].metric("Worst score (filtered)", f"{df_filt['Total Score (lower is better)'].max():.2f}" if len(df_filt) else "—")

    st.markdown("---")

    # A simple "pick a guide" control (reliable across Streamlit versions)
    if len(df_filt) == 0:
        st.error("No guides remain after GC filtering. Widen the GC% slider.")
        st.stop()

    pick_col, dl_col = st.columns([2, 1], gap="large")
    with pick_col:
        pick_rank = st.selectbox(
            "Pick a guide to preview (by Rank):",
            options=df_filt["Rank"].tolist(),
            index=0
        )
    with dl_col:
        st.download_button(
            "⬇️ Download guide table (CSV)",
            df_filt.to_csv(index=False),
            file_name="crispr_guides.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.dataframe(
        df_filt,
        use_container_width=True,
        height=420,
    )

    chosen = df_filt[df_filt["Rank"] == pick_rank].iloc[0]
    st.markdown("#### ✅ Selected guide (quick preview)")
    p1, p2, p3 = st.columns([1.4, 1.2, 1.4], gap="large")

    with p1:
        st.markdown(
            f"""
<div class="card">
  <div class="small">Guide (5'→3')</div>
  <div class="mono" style="font-size:18px; font-weight:800;">{chosen["Guide Sequence (5\'→3\')"]}
</div>
  <div class="small">Start: <span class="mono">{int(chosen['Guide Start (0-based)'])}</span> • End: <span class="mono">{int(chosen['Guide End (0-based, excl)'])}</span></div>
</div>
""",
            unsafe_allow_html=True,
        )

    with p2:
        st.markdown(
            f"""
<div class="card">
  <div class="small">PAM</div>
  <div class="mono" style="font-size:18px; font-weight:800;">{chosen['Matched PAM (instance)']}</div>
  <div class="small">Pattern: <span class="mono">{pam_choice}</span> • Length: <span class="mono">{len(pam_choice)}</span></div>
</div>
""",
            unsafe_allow_html=True,
        )

    with p3:
        st.markdown(
            f"""
<div class="card">
  <div class="small">Scores</div>
  <div class="small">GC%: <span class="mono">{chosen['GC %']}</span> • Self-comp: <span class="mono">{chosen['Self-Complementarity']}</span> • Off-target-like: <span class="mono">{chosen['Off-target-like Matches']}</span></div>
  <div style="margin-top:6px; font-weight:900; font-size:18px;">Total: <span class="mono">{chosen['Total Score (lower is better)']}</span></div>
</div>
""",
            unsafe_allow_html=True,
        )

    # Context snippet (shows the exact chosen PAM sequence too)
    ctx, left, right = slice_context(
        clean_seq,
        int(chosen["Guide Start (0-based)"]),
        int(chosen["Guide End (0-based, excl)"]),
        int(chosen["Guide End (0-based, excl)"]) + len(pam_choice),
        flank=14,
    )
    g0 = int(chosen["Guide Start (0-based)"]) - left
    g1 = int(chosen["Guide End (0-based, excl)"]) - left
    p0 = g1
    p1_ = p0 + len(pam_choice)

    # highlight context with simple HTML
    ctx_html = (
        ctx[:g0]
        + f"<span style='color:#0A8F5B;font-weight:800'>{ctx[g0:g1]}</span>"
        + f"<span style='color:#D83A3A;font-weight:900'>{ctx[p0:p1_]}</span>"
        + ctx[p1_:]
    )
    st.markdown("**Local context (guide = green, PAM instance = red):**")
    st.markdown(
        f"<div class='mono' style='padding:10px 12px; border:1px solid rgba(0,0,0,0.08); border-radius:14px;'>"
        f"<span style='color:gray'>[{left:04d}]</span> {ctx_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

# =====================================================
# 3) Sequence Map (visualize top N guides for clarity)
# =====================================================
with tabs[2]:
    st.subheader("3️⃣ Sequence map (highlighted guides + PAMs)")

    st.markdown(
        "<div class='card small'>For readability, visualize the <b>Top N</b> guides (ranked by Total Score). "
        "Guides are <span style='color:#0A8F5B;font-weight:800'>green</span> and PAMs are "
        "<span style='color:#D83A3A;font-weight:900'>red</span>.</div>",
        unsafe_allow_html=True,
    )

    v1, v2 = st.columns([1, 1], gap="large")
    with v1:
        top_n = st.slider("Top N guides to display", min_value=1, max_value=min(50, len(df_filt)), value=min(10, len(df_filt)))
    with v2:
        width = st.slider("Characters per line", 50, 120, 70, step=5)

    show_df = df_filt.sort_values("Rank").head(top_n)

    guides_meta = []
    pam_len = len(pam_choice)
    for _, r in show_df.iterrows():
        ge = int(r["Guide End (0-based, excl)"])
        guides_meta.append({
            "guide_start": int(r["Guide Start (0-based)"]),
            "guide_end": ge,
            "pam_start": ge,
            "pam_end": ge + pam_len,
        })

    seq_html = highlight_sequence_html(clean_seq, guides_meta, width=width)
    st.markdown(
        f"<div class='mono' style='background:rgba(0,0,0,0.02); border:1px solid rgba(0,0,0,0.06); "
        f"border-radius:16px; padding:12px 14px; overflow:auto; max-height:520px;'>"
        f"{seq_html}</div>",
        unsafe_allow_html=True,
    )

    st.caption(f"Showing Top {top_n} guides. PAM pattern is **{pam_choice}** (instances highlighted in red).")

# =====================================================
# 4) Plots + Interpretation
# =====================================================
with tabs[3]:
    st.subheader("4️⃣ Plots + interpretation (simple heuristics)")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**GC% distribution (filtered guides)**")
        st.bar_chart(df_filt["GC %"], height=260)

    with c2:
        st.markdown("**Total score distribution (filtered guides)**")
        st.bar_chart(df_filt["Total Score (lower is better)"], height=260)

    st.markdown("---")
    st.markdown(
        """
<div class="card">
  <h4 style="margin-top:0;">📘 How to interpret the scores</h4>
  <ul class="small">
    <li><b>GC% ~ 40–60%</b> is often a comfortable range for many systems (rule-of-thumb).</li>
    <li><b>Self-complementarity</b> is a small heuristic for potential hairpins (higher → more risk).</li>
    <li><b>Off-target-like matches</b> here are checked only against the same input sequence (educational).</li>
    <li><b>Total Score</b> combines these metrics: lower ≈ better under this simplified model.</li>
  </ul>
  <div class="small">⚠️ This tool is for learning and visualization only, not for experimental/clinical guide design.</div>
</div>
""",
        unsafe_allow_html=True,
    )
