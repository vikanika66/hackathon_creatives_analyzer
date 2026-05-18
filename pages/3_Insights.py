"""
Insights page — ready-made analytical views per brand.
For marketing managers, brand clients, and designers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import base64
from pathlib import Path
from urllib.parse import quote

from utils import (
    COMMON_CSS,
    load_data,
    BINARY_FEATURES,
    CATEGORICAL_TAGS,
)

st.set_page_config(
    page_title="Insights — Creative Analyzer",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# Styles (same as Upload)
# ============================================================
PAGE_CSS = """
<style>
    .stApp { background: #ffffff; }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1200px;
    }

    .section-divider {
        height: 1px;
        background: #eeeeee;
        margin: 48px 0 36px 0;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
        letter-spacing: -0.6px;
        margin: 0 0 8px 0;
        color: #080808;
    }

    .section-kicker {
        font-size: 12px;
        color: #0009dc;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .section-subtitle {
        font-size: 15px;
        color: rgba(8,8,8,0.62);
        line-height: 1.55;
        max-width: 860px;
        margin-bottom: 24px;
    }

    .ca-chip {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 12px;
        margin-right: 6px;
        margin-top: 8px;
    }

    .ca-card {
        background: linear-gradient(180deg, #ffffff 0%, #f9f9f9 100%);
        border: 1px solid #eeeeee;
        border-radius: 22px;
        position: relative;
        overflow: hidden;
    }

    .stat-box {
        background: linear-gradient(180deg, #ffffff 0%, #f9f9f9 100%);
        border: 1px solid #eeeeee;
        border-radius: 18px;
        padding: 20px 18px;
        position: relative;
        overflow: hidden;
    }

    .stat-label {
        font-size: 11px;
        color: rgba(8,8,8,0.46);
        margin-bottom: 6px;
    }

    .stat-value {
        font-size: 28px;
        font-weight: 700;
        color: #080808;
        line-height: 1;
    }

    .stat-comparison {
        font-size: 12px;
        margin-top: 8px;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #eeeeee !important;
        border-radius: 18px !important;
        overflow: hidden !important;
        background: #ffffff !important;
    }

    div[data-testid="stExpander"] details summary {
        background:
            radial-gradient(circle at 8% 50%, rgba(174,243,62,0.20), transparent 24%),
            radial-gradient(circle at 92% 50%, rgba(0,9,220,0.08), transparent 26%),
            linear-gradient(135deg, #ffffff 0%, #f9f9f9 100%) !important;
        color: #080808 !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #eeeeee !important;
        padding: 13px 16px !important;
    }

    div[data-testid="stExpander"] details summary:hover {
        background:
            radial-gradient(circle at 8% 50%, rgba(174,243,62,0.28), transparent 24%),
            radial-gradient(circle at 92% 50%, rgba(0,9,220,0.12), transparent 26%),
            linear-gradient(135deg, #ffffff 0%, #f9f9f9 100%) !important;
    }

    div[data-testid="stExpander"] details summary p {
        font-size: 13px !important;
        color: #080808 !important;
    }

    div[data-testid="stExpander"] details div[data-testid="stExpanderDetails"] {
        background: #ffffff !important;
        padding: 18px 20px 20px 20px !important;
    }

    div[data-testid="stExpander"] details div[data-testid="stMarkdownContainer"] {
        color: rgba(8,8,8,0.68) !important;
        font-size: 14px !important;
        line-height: 1.65 !important;
    }

    div[data-testid="stExpander"] details div[data-testid="stMarkdownContainer"] strong {
        color: #080808 !important;
        font-weight: 650 !important;
    }
</style>
"""

st.markdown(COMMON_CSS, unsafe_allow_html=True)
st.markdown(PAGE_CSS, unsafe_allow_html=True)

def spacer(h=16):
    st.markdown(f'<div style="height:{h}px;"></div>', unsafe_allow_html=True)

# ============================================================
# Data
# ============================================================
df, shap_df, feature_cols, interactions_df = load_data()


# Human-readable tag names
TAG_DISPLAY_NAMES = {
    "has_person": "person",
    "close_up": "close-up",
    "top_down": "top-down",
    "warm_tones": "warm tones",
    "clean_background": "clean bg",
    "packaged": "packaged",
    "text_overlay": "text",
    "price_discount": "discount",
    "brand_logo": "logo",
    "cta": "CTA",
    "multiple_items": "multiple items",
}

def categorize_strength(value, strong_threshold, weak_threshold):
    """Categorize correlation: (direction, strength)."""
    abs_v = abs(value)
    if abs_v < weak_threshold:
        return "neutral", "none"
    direction = "up" if value > 0 else "down"
    strength = "strong" if abs_v >= strong_threshold else "weak"
    return direction, strength

def display_name(tag):
    if tag in TAG_DISPLAY_NAMES:
        return TAG_DISPLAY_NAMES[tag]
    clean = tag.replace("_", " ")
    for prefix in ["food type ", "drink type ", "main object ", "person who ", "person emotion ", "person action "]:
        clean = clean.replace(prefix, "")
    return clean


def divider():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ============================================================
# UI: Title and filters
# ============================================================
# ============================================================
# UI: Page hero and filters
# ============================================================

st.markdown(f"""
<div style="margin:24px 0 28px 0;padding:28px 30px;border-radius:26px;background:radial-gradient(circle at 8% 18%, rgba(174,243,62,0.24), transparent 28%),radial-gradient(circle at 92% 12%, rgba(255,124,245,0.12), transparent 28%),linear-gradient(135deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;">
<div style="display:grid;grid-template-columns:1.25fr 0.75fr;gap:28px;align-items:center;">

<div>
<div style="display:inline-flex;gap:8px;align-items:center;padding:7px 12px;border-radius:999px;background:#ffffff;border:1px solid #eeeeee;font-size:12px;color:rgba(8,8,8,0.58);margin-bottom:16px;">
<span style="width:7px;height:7px;border-radius:50%;background:#aef33e;display:inline-block;"></span>
Insights workspace
</div>

<div style="font-size:12px;color:#0009dc;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:10px;">
Brand-level analytics
</div>

<h1 style="font-size:40px;margin:0 0 12px 0;font-weight:650;letter-spacing:-1.2px;color:#080808;">
Insights
</h1>

<p style="font-size:17px;color:rgba(8,8,8,0.66);max-width:720px;margin:0;line-height:1.55;">
Explore ready-made analytical views for a selected brand: overall CTR, tag strengths, interactions, best and worst creatives, anomalies, and year-over-year movement.
</p>

<div style="margin-top:18px;">
<span class="ca-chip" style="background:#0009dc;color:#ffffff;">brand view</span>
<span class="ca-chip" style="background:#aef33e;color:#080808;">tag patterns</span>
<span class="ca-chip" style="background:#080808;color:#ffffff;">creative ranking</span>
<span class="ca-chip" style="background:#ff7cf5;color:#080808;">YoY trend</span>
</div>
</div>

<div class="ca-card" style="padding:22px 22px;min-height:190px;background:linear-gradient(180deg, rgba(0,9,220,0.06) 0%, rgba(65,105,225,0.10) 100%);border:1px solid rgba(0,9,220,0.12);">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#0009dc;"></div>
<div style="position:absolute;top:-42px;right:-42px;width:130px;height:130px;background:#0009dc;border-radius:50%;opacity:0.14;"></div>
<div style="position:absolute;bottom:-48px;left:-48px;width:150px;height:150px;background:#aef33e;border-radius:50%;opacity:0.16;"></div>

<div style="position:relative;z-index:1;">
<div style="font-size:12px;color:#0009dc;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:14px;font-weight:600;">
Insights workflow
</div>

<div style="font-size:14px;color:rgba(8,8,8,0.78);line-height:1.9;">
<span style="color:#0009dc;font-weight:650;">01</span> · Choose brand and year<br>
<span style="color:#0009dc;font-weight:650;">02</span> · Review overall performance<br>
<span style="color:#0009dc;font-weight:650;">03</span> · Find strong and weak tags<br>
<span style="color:#0009dc;font-weight:650;">04</span> · Open creatives for deep dive
</div>

<div style="height:1px;background:rgba(0,9,220,0.10);margin:16px 0 13px 0;"></div>

<div style="font-size:12px;color:rgba(8,8,8,0.56);line-height:1.5;">
Designed for strategy review, client reporting, and creative planning.
</div>
</div>
</div>

</div>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div style="margin:0 0 16px 0;">
<div style="font-size:12px;color:#0009dc;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:8px;">
Select analysis scope
</div>
</div>
""", unsafe_allow_html=True)

col_brand, col_year = st.columns([1, 1], gap="large")

with col_brand:
    all_brands = sorted(df["brand"].unique().tolist())
    selected_brand = st.selectbox(
        "Brand",
        options=all_brands,
        help="Choose a brand to analyze"
    )

with col_year:
    year_options = ["All years"] + sorted(df["year"].unique().tolist())
    selected_year = st.selectbox(
        "Year",
        options=year_options,
        help="Optional — filter by year"
    )

# Filter data
df_brand = df[df["brand"] == selected_brand].copy()
if selected_year != "All years":
    df_brand = df_brand[df_brand["year"] == int(selected_year)]

# Corresponding SHAP data
filenames = df_brand["filename"].tolist()
shap_brand = shap_df[shap_df["filename"].isin(filenames)].copy()

# Baseline data for comparison (whole database)
df_all = df.copy()
if selected_year != "All years":
    df_all = df_all[df_all["year"] == int(selected_year)]

if len(df_brand) < 10:
    st.warning(f"⚠️ Only {len(df_brand)} creatives in the selected segment. Analytics may be unreliable.")

st.caption(f"Analysis based on **{len(df_brand)}** creatives of brand **{selected_brand}**" + 
           (f" in **{selected_year}**" if selected_year != "All years" else ""))


# ============================================================
# BLOCK 1: Overall brand statistics
# ============================================================
divider()

st.markdown("""
<div style="margin:8px 0 28px 0;">
<div class="section-kicker">Performance snapshot</div>
<div class="section-title">Overall statistics</div>
<div class="section-subtitle">
How the selected brand performs compared with the full creative database.
</div>
</div>
""", unsafe_allow_html=True)

# Compute metrics
brand_avg_ctr = df_brand["ctr"].mean()
brand_median_ctr = df_brand["ctr"].median()
brand_max_ctr = df_brand["ctr"].max()
brand_min_ctr = df_brand["ctr"].min()
brand_count = len(df_brand)

industry_avg_ctr = df_all["ctr"].mean()
ctr_diff = round(brand_avg_ctr, 2) - round(industry_avg_ctr, 2)

# Comparison with average
diff_color = "#1D9E75" if ctr_diff > 0 else "#E24B4A"
diff_sign = "+" if ctr_diff > 0 else ""
diff_text = f"{diff_sign}{ctr_diff:.2f}% vs database avg ({industry_avg_ctr:.2f}%)"

st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(190px, 1fr));gap:12px;">

<div class="stat-box">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#0009dc;"></div>
<div class="stat-label">Creatives</div>
<div class="stat-value">{brand_count}</div>
<div style="font-size:12px;color:rgba(8,8,8,0.48);margin-top:8px;">
selected segment
</div>
</div>

<div class="stat-box">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:{diff_color};"></div>
<div class="stat-label">Average CTR</div>
<div class="stat-value" style="color:{diff_color};">{brand_avg_ctr:.2f}%</div>
<div class="stat-comparison" style="color:{diff_color};font-weight:650;">
{diff_text}
</div>
</div>

<div class="stat-box">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#aef33e;"></div>
<div class="stat-label">Median CTR</div>
<div class="stat-value">{brand_median_ctr:.2f}%</div>
<div style="font-size:12px;color:rgba(8,8,8,0.48);margin-top:8px;">
middle creative
</div>
</div>

<div class="stat-box">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#ff7cf5;"></div>
<div class="stat-label">CTR range</div>
<div class="stat-value" style="font-size:22px;">{brand_min_ctr:.2f}% — {brand_max_ctr:.2f}%</div>
<div style="font-size:12px;color:rgba(8,8,8,0.48);margin-top:8px;">
min to max
</div>
</div>

</div>
""", unsafe_allow_html=True)

spacer()


with st.expander("ℹ️ How this is calculated"):
    st.markdown("""
    **Brand creatives** — total count of creatives in the selected brand and year filter.
    
    **Average / Median CTR** — calculated across all brand creatives in the filter.
    
    **CTR range** — minimum and maximum CTR among brand creatives.
    
    **Comparison with industry** — difference between brand's average CTR and the
    average CTR across all creatives in the entire database (within the same year if year filter is applied).
    Green means the brand performs above industry average, red means below.
    """)

# ============================================================
# BLOCK 2: Top-5 strong and weak tags
# ============================================================
divider()

st.markdown("""
<div style="margin:8px 0 28px 0;">
<div class="section-kicker">Tag signals</div>
<div class="section-title">What works for the brand</div>
<div class="section-subtitle">
Which visual tags are more often connected with high or low CTR for the selected brand.
This is a pattern in the data, not a guaranteed effect.
</div>
</div>
""", unsafe_allow_html=True)

# Thresholds
TAG_STRONG = 0.30
TAG_WEAK = 0.01

# Average SHAP for each tag when active
tag_effects = []
for feat in BINARY_FEATURES:
    if feat in shap_brand.columns:
        mask = df_brand[feat] == True
        if mask.sum() >= 3:
            avg_shap = shap_brand.loc[mask.values, feat].mean()
            tag_effects.append({
                "tag": feat,
                "avg_shap": avg_shap,
                "count": int(mask.sum()),
            })

tag_effects_df = pd.DataFrame(tag_effects).sort_values("avg_shap", ascending=False)


def render_tag_row(row, kind):
    """Render one tag row for strong / weak tag cards."""
    val = row["avg_shap"]
    abs_v = abs(val)
    
    if abs_v < TAG_WEAK:
        arrow = '<span style="color:rgba(8,8,8,0.34);font-size:20px;font-weight:650;">—</span>'
    elif val > 0:
        arrows = "↑↑" if abs_v >= TAG_STRONG else "↑"
        arrow = f'<span style="color:#1D9E75;font-weight:750;font-size:22px;letter-spacing:-3px;">{arrows}</span>'
    else:
        arrows = "↓↓" if abs_v >= TAG_STRONG else "↓"
        arrow = f'<span style="color:#E24B4A;font-weight:750;font-size:22px;letter-spacing:-3px;">{arrows}</span>'
    
    return f"""
<div style="padding:15px 0;border-bottom:1px solid #eeeeee;display:grid;grid-template-columns:64px 1fr auto;gap:12px;align-items:center;">
<div style="text-align:center;">{arrow}</div>
<div style="font-size:14px;color:rgba(8,8,8,0.74);font-weight:500;">
{display_name(row['tag'])}
</div>

</div>
"""


TOP_N = 5

# Candidates for each column — strictly by sign, not below WEAK threshold
strong_candidates = tag_effects_df[
    tag_effects_df["avg_shap"] >= TAG_WEAK
].head(TOP_N)

weak_candidates = tag_effects_df[
    tag_effects_df["avg_shap"] <= -TAG_WEAK
].tail(TOP_N).iloc[::-1]


col_strong, col_weak = st.columns(2, gap="large")

with col_strong:
    if len(strong_candidates) == 0:
        strong_body = """
<div style="font-size:14px;color:rgba(8,8,8,0.58);padding:16px 0;">
No tags clearly correlate with high CTR for this brand.
</div>
"""
    else:
        strong_body = "".join(
            render_tag_row(row, "strong")
            for _, row in strong_candidates.iterrows()
        )

    st.markdown(f"""
<div class="ca-card" style="padding:22px 24px;">
<div style="position:absolute;top:0;left:0;right:0;height:4px;background:#1D9E75;"></div>

<div style="display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px;">
<div>
<div style="font-size:12px;color:#1D9E75;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:6px;font-weight:650;">
Positive signals
</div>
<div style="font-size:22px;font-weight:650;color:#080808;letter-spacing:-0.4px;">
Strong tags
</div>
</div>

<div style="background:rgba(29,158,117,0.10);color:#1D9E75;border-radius:999px;padding:7px 11px;font-size:12px;font-weight:650;">
CTR ↑
</div>
</div>

{strong_body}
</div>
""", unsafe_allow_html=True)

with col_weak:
    if len(weak_candidates) == 0:
        weak_body = """
<div style="font-size:14px;color:rgba(8,8,8,0.58);padding:16px 0;">
No tags clearly correlate with low CTR for this brand.
</div>
"""
    else:
        weak_body = "".join(
            render_tag_row(row, "weak")
            for _, row in weak_candidates.iterrows()
        )

    st.markdown(f"""
<div class="ca-card" style="padding:22px 24px;">
<div style="position:absolute;top:0;left:0;right:0;height:4px;background:#E24B4A;"></div>

<div style="display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px;">
<div>
<div style="font-size:12px;color:#E24B4A;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:6px;font-weight:650;">
Negative signals
</div>
<div style="font-size:22px;font-weight:650;color:#080808;letter-spacing:-0.4px;">
Weak tags
</div>
</div>

<div style="background:rgba(226,75,74,0.10);color:#E24B4A;border-radius:999px;padding:7px 11px;font-size:12px;font-weight:650;">
CTR ↓
</div>
</div>

{weak_body}
</div>
""", unsafe_allow_html=True)

spacer()

with st.expander("ℹ️ How this is calculated"):
    st.markdown("""
    For each visual tag (e.g. *has person*, *warm tones*, *close-up*), we calculate
    the average SHAP value across all brand creatives where that tag is active.
    SHAP shows how much that specific tag pushes CTR up or down compared to the brand's baseline.
    
    Tags are ranked by their average SHAP and split into two columns:
    
    - **Strong tags** — average SHAP is positive, meaning creatives with this tag tend to have higher CTR
    - **Weak tags** — average SHAP is negative, meaning creatives with this tag tend to have lower CTR
    
    Arrow indicators show the strength of the correlation:
    
    - ↑↑ / ↓↓ — strong (≥ 0.30 CTR percentage points)
    - ↑ / ↓ — moderate (≥ 0.01)
    - — — neutral (below 0.01)
    
    A tag must appear in at least 3 brand creatives to be shown. This is a pattern in the data,
    not a guaranteed effect — actual CTR depends on targeting and audience too.
    """)

# ============================================================
# BLOCK 3: Tag interactions heatmap
# ============================================================
divider()

st.markdown("""
<div style="margin:8px 0 28px 0;">
<div class="section-kicker">Interaction map</div>
<div class="section-title">Tag interactions map</div>
<div class="section-subtitle">
Which tag pairs tend to work better or worse together. Green cells indicate positive interaction, red cells indicate negative interaction, and empty cells are neutral or rare.
</div>
</div>
""", unsafe_allow_html=True)

INT_STRONG = 0.05
INT_WEAK = 0.01

all_binary = BINARY_FEATURES
n = len(all_binary)

# Build value matrix
matrix = np.zeros((n, n))
seg_filter = interactions_df["segment_type"] == "all"
seg_data = interactions_df[seg_filter]

for _, row in seg_data.iterrows():
    if row["feat1"] in all_binary and row["feat2"] in all_binary:
        i = all_binary.index(row["feat1"])
        j = all_binary.index(row["feat2"])
        matrix[i][j] = row["interaction"]
        matrix[j][i] = row["interaction"]

# Convert to categories: -2, -1, 0, 1, 2
def categorize_cell(val):
    abs_v = abs(val)
    if abs_v < INT_WEAK:
        return 0
    if val > 0:
        return 2 if abs_v >= INT_STRONG else 1
    return -2 if abs_v >= INT_STRONG else -1

cat_matrix = np.vectorize(categorize_cell)(matrix)

# Display symbols
SYMBOLS = {-2: "↓↓", -1: "↓", 0: "", 1: "↑", 2: "↑↑"}
text_matrix = np.vectorize(lambda x: SYMBOLS[int(x)])(cat_matrix)

# Clear diagonal — a tag doesn't compare with itself
for i in range(n):
    text_matrix[i][i] = ""
    cat_matrix[i][i] = 0  # neutral white color

labels = [display_name(t) for t in all_binary]

# Discrete color scale
fig = go.Figure(data=go.Heatmap(
    z=cat_matrix,
    x=labels,
    y=labels,
    text=text_matrix,
    texttemplate="%{text}",
    textfont={"size": 16, "color": "#1a1a1a"},
    colorscale=[
        [0.0,  "#e8506e"],   # -2 strong negative
        [0.25, "#f5b0bc"],   # -1 weak negative
        [0.5,  "#ffffff"],   # 0 neutral
        [0.75, "#d5f0b0"],   # +1 weak positive
        [1.0,  "#aef33e"],   # +2 strong positive
    ],
    zmin=-2,
    zmax=2,
    hovertemplate="%{y} × %{x}<extra></extra>",
    showscale=False,
))

fig.update_layout(
    height=560,
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    font=dict(color="#080808"),
    xaxis=dict(
        side="bottom",
        tickfont=dict(size=13, color="rgba(8,8,8,0.72)"),
        showgrid=False,
        zeroline=False,
    ),
    yaxis=dict(
        autorange="reversed",
        tickfont=dict(size=13, color="rgba(8,8,8,0.72)"),
        showgrid=False,
        zeroline=False,
    ),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown("""
<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:16px 18px;margin-top:12px;">
<div style="font-size:12px;color:#0009dc;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:8px;">
How to read
</div>
<div style="font-size:14px;color:rgba(8,8,8,0.66);line-height:1.65;">
<b style="color:#1D9E75;">Green cells</b> show tag pairs that tend to work better together.
<b style="color:#E24B4A;">Red cells</b> show pairs that tend to work worse together.
Empty cells are neutral or too rare to read confidently.
</div>
</div>
""", unsafe_allow_html=True)

spacer()

with st.expander("ℹ️ How this is calculated"):
    st.markdown("""
    For each pair of tags (A, B), we measure how much **adding the second tag** to a creative
    that already has the first tag changes CTR — beyond what each tag would contribute alone.
    This isolates the **interaction effect** of the pair.
    
    Calculation: average CTR of creatives with both A and B, minus the sum of individual
    contributions of A and B taken separately.
    
    Cell legend:
    
    - ↑↑ (bright green) — strong positive interaction: the pair works much better together than apart
    - ↑ (light green) — moderate positive interaction
    - empty (white) — neutral: the pair behaves like the sum of its parts
    - ↓ (light red) — moderate negative interaction
    - ↓↓ (bright red) — strong negative interaction: the pair works worse together
    
    The diagonal is empty because a tag can't interact with itself.
    
    Calculations use the same SHAP-based methodology as the single-tag analysis,
    aggregated across the full database.
    """)

# ============================================================
# BLOCK 4: Best and worst brand creatives
# ============================================================
divider()

st.markdown("""
<div style="margin:8px 0 28px 0;">
<div class="section-kicker">Creative ranking</div>
<div class="section-title">Best and worst creatives</div>
<div class="section-subtitle">
Top performers and bottom performers for the selected brand. Use them as concrete cases for what to repeat, avoid, or inspect deeper.
</div>
</div>
""", unsafe_allow_html=True)

# Image path — change if you use a different folder
IMAGES_DIR = "images"

import os

def encode_image(path):
    """Convert image to data URL for inline HTML."""
    try:
        ext = Path(path).suffix.lstrip(".").lower()
        if ext == "jpg":
            ext = "jpeg"
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:image/{ext};base64,{data}"
    except Exception:
        return None


def render_creative_card(row, rank, tone):
    color = "#1D9E75" if tone == "good" else "#E24B4A"
    label = "Top performer" if tone == "good" else "Bottom performer"
    
    active_tags = []
    for feat in BINARY_FEATURES:
        if feat in row.index and row[feat] == True:
            active_tags.append(display_name(feat))
        if len(active_tags) >= 3:
            break
    
    tag_str = ", ".join(active_tags) if active_tags else "—"
    
    image_path = os.path.join(IMAGES_DIR, row["filename"])
    data_url = encode_image(image_path)
    
    if data_url:
        st.markdown(
            f'<div style="aspect-ratio:1;border-radius:18px 18px 0 0;overflow:hidden;background:#ffffff;'
            f'border:1px solid #eeeeee;border-bottom:none;display:flex;align-items:center;justify-content:center;">'
            f'<img src="{data_url}" style="max-width:100%;max-height:100%;object-fit:contain;display:block;"/>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="aspect-ratio:1;border-radius:18px 18px 0 0;background:#ffffff;'
            'border:1px solid #eeeeee;border-bottom:none;display:flex;align-items:center;justify-content:center;'
            'color:rgba(8,8,8,0.38);font-size:12px;">no image</div>',
            unsafe_allow_html=True,
        )
    
    st.markdown(
        f'<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);'
        f'border:1px solid #eeeeee;border-top:3px solid {color};'
        f'border-radius:0 0 18px 18px;padding:13px 13px 14px 13px;margin-bottom:10px;">'
        f'<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:8px;">'
        f'<div>'
        f'<div style="font-size:22px;font-weight:750;color:{color};line-height:1;">{row["ctr"]:.2f}%</div>'
        f'<div style="font-size:11px;color:rgba(8,8,8,0.46);margin-top:4px;">CTR · #{rank}</div>'
        f'</div>'
        f'<div style="background:rgba(8,8,8,0.04);border-radius:999px;padding:6px 9px;font-size:11px;color:rgba(8,8,8,0.62);white-space:nowrap;">'
        f'{label}'
        f'</div>'
        f'</div>'
        f'<div style="height:1px;background:#eeeeee;margin:12px 0 10px 0;"></div>'
        f'<div style="font-size:12px;color:rgba(8,8,8,0.62);line-height:1.45;min-height:52px;">'
        f'<b style="color:#080808;">{row.get("brand", "—")}</b> · {row.get("year", "—")}<br>'
        f'{tag_str}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    
    st.link_button(
        "Open in Library →",
        f"/Library?creative={quote(str(row['filename']))}",
        use_container_width=True,
    )


df_sorted = df_brand.sort_values("ctr", ascending=False).reset_index(drop=True)
top_5 = df_sorted.head(5)
bottom_5 = df_sorted.tail(5).iloc[::-1].reset_index(drop=True)

# Top-5
st.markdown("""
<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(29,158,117,0.10);color:#1D9E75;border-radius:999px;padding:8px 13px;font-size:13px;font-weight:650;margin:4px 0 14px 0;">
<span style="width:7px;height:7px;border-radius:50%;background:#1D9E75;display:inline-block;"></span>
Top 5 performers
</div>
""", unsafe_allow_html=True)
cols = st.columns(5, gap="small")
for i, col in enumerate(cols):
    if i < len(top_5):
        with col:
            render_creative_card(top_5.iloc[i], i + 1, "good")

# Bottom-5
st.markdown("""
<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(226,75,74,0.10);color:#E24B4A;border-radius:999px;padding:8px 13px;font-size:13px;font-weight:650;margin:28px 0 14px 0;">
<span style="width:7px;height:7px;border-radius:50%;background:#E24B4A;display:inline-block;"></span>
Bottom 5 performers
</div>
""", unsafe_allow_html=True)
cols = st.columns(5, gap="small")
for i, col in enumerate(cols):
    if i < len(bottom_5):
        with col:
            render_creative_card(bottom_5.iloc[i], i + 1, "bad")

spacer()

with st.expander("ℹ️ How this is calculated"):
    st.markdown("""
    Creatives are ranked by their **actual CTR** (not predicted).
    
    - **Top 5** — five creatives of the brand with the highest CTR
    - **Bottom 5** — five creatives with the lowest CTR
    
    Click "Open in Library" on any card to see the full breakdown of that creative —
    its tags, how each tag correlates with CTR in your database, and which tag combinations
    work for or against it.
    """)

# ============================================================
# BLOCK 5: Anomalies — creatives falling outside the tag pattern
# ============================================================
divider()

st.markdown("""
<div style="margin:8px 0 28px 0;">
<div class="section-kicker">Creative anomalies</div>
<div class="section-title">Creatives outside the tag pattern</div>
<div class="section-subtitle">
Creatives whose actual CTR differs from what their visual tag combination would suggest.
These are useful for spotting execution quality, copy strength, layout issues, or creative magic beyond tags.
</div>
</div>
""", unsafe_allow_html=True)

# Merge predicted_ctr into df_brand
brand_with_pred = df_brand.merge(
    shap_brand[["filename", "predicted_ctr"]],
    on="filename",
    how="left"
).dropna(subset=["predicted_ctr"])

brand_with_pred["residual"] = brand_with_pred["ctr"] - brand_with_pred["predicted_ctr"]

hidden_gems = brand_with_pred.nlargest(3, "residual")
underperformers = brand_with_pred.nsmallest(3, "residual")


def render_anomaly_card(row, kind):
    if kind == "gem":
        color = "#1D9E75"
        label = "Hidden gem"
        comparison_label = "above expected"
    else:
        color = "#E24B4A"
        label = "Underperformer"
        comparison_label = "below expected"

    active_tags = []
    for feat in BINARY_FEATURES:
        if feat in row.index and row[feat] == True:
            active_tags.append(display_name(feat))
        if len(active_tags) >= 3:
            break

    tag_str = ", ".join(active_tags) if active_tags else "—"

    image_path = os.path.join(IMAGES_DIR, row["filename"])
    data_url = encode_image(image_path)

    residual = row["residual"]
    abs_r = abs(residual)

    if abs_r < 0.10:
        arrow_inline = '<span style="color:rgba(8,8,8,0.34);font-size:18px;font-weight:650;">—</span>'
    elif residual > 0:
        arrows = "↑↑" if abs_r >= 0.30 else "↑"
        arrow_inline = f'<span style="color:#1D9E75;font-weight:750;font-size:20px;letter-spacing:-3px;">{arrows}</span>'
    else:
        arrows = "↓↓" if abs_r >= 0.30 else "↓"
        arrow_inline = f'<span style="color:#E24B4A;font-weight:750;font-size:20px;letter-spacing:-3px;">{arrows}</span>'

    if data_url:
        st.markdown(
            f'<div style="aspect-ratio:1;border-radius:18px 18px 0 0;overflow:hidden;background:#ffffff;'
            f'border:1px solid #eeeeee;border-bottom:none;display:flex;align-items:center;justify-content:center;">'
            f'<img src="{data_url}" style="max-width:100%;max-height:100%;object-fit:contain;display:block;"/>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="aspect-ratio:1;border-radius:18px 18px 0 0;background:#ffffff;'
            'border:1px solid #eeeeee;border-bottom:none;display:flex;align-items:center;justify-content:center;'
            'color:rgba(8,8,8,0.38);font-size:12px;">no image</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);'
        f'border:1px solid #eeeeee;border-top:3px solid {color};'
        f'border-radius:0 0 18px 18px;padding:13px 13px 14px 13px;margin-bottom:10px;">'

        f'<div style="display:flex;align-items:flex-start;justify-content:space-between;gap:8px;">'
        f'<div>'
        f'<div style="font-size:22px;font-weight:750;color:{color};line-height:1;">{row["ctr"]:.2f}%</div>'
        f'<div style="font-size:11px;color:rgba(8,8,8,0.46);margin-top:4px;">Actual CTR</div>'
        f'</div>'
        f'<div style="background:rgba(8,8,8,0.04);border-radius:999px;padding:6px 9px;font-size:11px;color:{color};font-weight:650;white-space:nowrap;">'
        f'{label}'
        f'</div>'
        f'</div>'

        f'<div style="height:1px;background:#eeeeee;margin:12px 0 10px 0;"></div>'

        f'<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:10px;">'
        f'<div style="font-size:12px;color:rgba(8,8,8,0.56);">{comparison_label}</div>'
        f'<div>{arrow_inline}</div>'
        f'</div>'

        f'<div style="font-size:12px;color:rgba(8,8,8,0.62);line-height:1.45;min-height:48px;">'
        f'<b style="color:#080808;">{row.get("brand", "—")}</b> · {row.get("year", "—")}<br>'
        f'{tag_str}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.link_button(
        "Open in Library →",
        f"/Library?creative={quote(str(row['filename']))}",
        use_container_width=True,
    )


# Hidden gems
st.markdown("""
<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(29,158,117,0.10);color:#1D9E75;border-radius:999px;padding:8px 13px;font-size:13px;font-weight:650;margin:4px 0 14px 0;">
<span style="width:7px;height:7px;border-radius:50%;background:#1D9E75;display:inline-block;"></span>
Hidden gems · CTR above similar creatives
</div>
""", unsafe_allow_html=True)

if len(hidden_gems) == 0:
    st.info("Not enough data to detect hidden gems in this segment")
else:
    cols = st.columns(3, gap="small")
    for i, col in enumerate(cols):
        if i < len(hidden_gems):
            with col:
                render_anomaly_card(hidden_gems.iloc[i], "gem")
    st.markdown("""
<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:16px 18px;margin:14px 0 26px 0;">
<div style="font-size:12px;color:#1D9E75;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:8px;font-weight:650;">
Why review them
</div>
<div style="font-size:14px;color:rgba(8,8,8,0.66);line-height:1.65;">
These creatives perform better than similar tag combinations would suggest. Look for what tags do not capture: copy, emotion, composition, offer clarity, or execution quality.
</div>
</div>
""", unsafe_allow_html=True)

# Underperformers
st.markdown("""
<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(226,75,74,0.10);color:#E24B4A;border-radius:999px;padding:8px 13px;font-size:13px;font-weight:650;margin:8px 0 14px 0;">
<span style="width:7px;height:7px;border-radius:50%;background:#E24B4A;display:inline-block;"></span>
Underperformers · CTR below similar creatives
</div>
""", unsafe_allow_html=True)

if len(underperformers) == 0:
    st.info("Not enough data to detect underperformers in this segment")
else:
    cols = st.columns(3, gap="small")
    for i, col in enumerate(cols):
        if i < len(underperformers):
            with col:
                render_anomaly_card(underperformers.iloc[i], "underperformer")
    st.markdown("""
<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:16px 18px;margin-top:14px;">
<div style="font-size:12px;color:#E24B4A;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:8px;font-weight:650;">
What to inspect
</div>
<div style="font-size:14px;color:rgba(8,8,8,0.66);line-height:1.65;">
These creatives perform worse than similar tag combinations would suggest. Check execution issues: readability, offer strength, visual hierarchy, photo quality, CTA clarity, or layout friction.
</div>
</div>
""", unsafe_allow_html=True)

spacer()

with st.expander("ℹ️ How this is calculated"):
    st.markdown("""
    For each creative we compare its **actual CTR** with the **expected CTR** for creatives
    with a similar tag combination. The expected CTR comes from the LightGBM model trained
    on the full database.
    
    When a creative performs noticeably **better** than other creatives with similar visual
    elements — something is working **beyond** the tags we can see (copy, emotion, composition,
    offer clarity, execution quality).
    
    When a creative performs noticeably **worse** than similar ones — likely something in
    execution drags it down (photo quality, layout, readability, CTA friction).
    
    - **💎 Hidden gems** — creatives consistently performing above what their tags would suggest
    - **📉 Underperformers** — creatives consistently performing below what their tags would suggest
    
    Arrow indicators show direction and strength of the deviation:
    
    - ↑↑ / ↓↓ — strong deviation from expected
    - ↑ / ↓ — moderate deviation
    - — — close to expected
    
    These are signals to investigate manually — what makes the gems special that we can't
    capture with tags, and what's broken in the underperformers?
    """)

# ============================================================
# BLOCK 6: Year-over-year trend (ignores year filter)
# ============================================================
divider()

st.markdown("""
<div style="margin:8px 0 28px 0;">
<div class="section-kicker">Timeline view</div>
<div class="section-title">Year-over-year trend</div>
<div class="section-subtitle">
How the brand position changed over time and which visual tags gained or lost weight.
<b>The year filter is not applied here</b> because trend analysis needs the full available timeline.
</div>
</div>
""", unsafe_allow_html=True)

# Ignore year filter — calculate trend across all brand data
df_brand_all_years = df[df["brand"] == selected_brand].copy()
filenames_all = df_brand_all_years["filename"].tolist()
shap_brand_all = shap_df[shap_df["filename"].isin(filenames_all)].copy()

years_available = sorted(df_brand_all_years["year"].unique().tolist())

if len(years_available) < 2:
    st.info(
        f"Brand {selected_brand} has creatives only for one year "
        f"({years_available[0] if years_available else '—'}). Cannot build a trend."
    )
else:
    # === Part 1: Brand position in industry ===
    st.markdown("""
<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(0,9,220,0.08);color:#0009dc;border-radius:999px;padding:8px 13px;font-size:13px;font-weight:650;margin:4px 0 16px 0;">
<span style="width:7px;height:7px;border-radius:50%;background:#0009dc;display:inline-block;"></span>
Brand position in industry
</div>
""", unsafe_allow_html=True)
    
    # Calculate brand position (percentile) for each year
    position_data = []
    for year in years_available:
        df_year_all = df[df["year"] == year]
        df_year_brand = df_brand_all_years[df_brand_all_years["year"] == year]
        if len(df_year_brand) == 0 or len(df_year_all) == 0:
            continue
        brand_mean = df_year_brand["ctr"].mean()
        percentile = (df_year_all["ctr"] < brand_mean).mean() * 100
        position_data.append({
            "year": year,
            "percentile": percentile,
        })
    
    position_df = pd.DataFrame(position_data)
    
    def position_label(p):
        if p >= 75: return ("Top of industry", "🏆", "#1D9E75")
        if p >= 50: return ("Above average", "📈", "#7ABE5C")
        if p >= 25: return ("Below average", "📊", "#E8A24A")
        return ("Bottom segment", "⚠️", "#E24B4A")
    
    # Take second-to-last and last year — compare actual recent dynamics
    first_row = position_df.iloc[-2]
    last_row = position_df.iloc[-1]
    
    first_label, first_icon, first_color = position_label(first_row["percentile"])
    last_label, last_icon, last_color = position_label(last_row["percentile"])
    
    # Brand and industry CTR for each year
    first_year_int = int(first_row["year"])
    last_year_int = int(last_row["year"])
    
    first_brand_ctr = df_brand_all_years[df_brand_all_years["year"] == first_year_int]["ctr"].mean()
    last_brand_ctr = df_brand_all_years[df_brand_all_years["year"] == last_year_int]["ctr"].mean()
    first_ind_ctr = df[df["year"] == first_year_int]["ctr"].mean()
    last_ind_ctr = df[df["year"] == last_year_int]["ctr"].mean()
    
    first_diff = round(first_brand_ctr, 2) - round(first_ind_ctr, 2)
    last_diff = round(last_brand_ctr, 2) - round(last_ind_ctr, 2)
    
    def diff_html(diff):
        if diff > 0:
            return f'<span style="color:#1D9E75;font-weight:600;">+{diff:.2f}%</span> above industry'
        elif diff < 0:
            return f'<span style="color:#E24B4A;font-weight:600;">{diff:.2f}%</span> below industry'
        else:
            return '<span style="color:#888;font-weight:600;">at industry level</span>'
    
    # Direction of change
    diff_p = last_row["percentile"] - first_row["percentile"]
    if diff_p > 10:
        trend_arrow = '<span style="color:#1D9E75;font-size:42px;font-weight:600;">↑</span>'
        trend_text = "gained position"
        trend_color = "#1D9E75"
    elif diff_p < -10:
        trend_arrow = '<span style="color:#E24B4A;font-size:42px;font-weight:600;">↓</span>'
        trend_text = "lost position"
        trend_color = "#E24B4A"
    else:
        trend_arrow = '<span style="color:#888;font-size:42px;">→</span>'
        trend_text = "position stable"
        trend_color = "#888"
    
    # Plates: before → arrow → after
    st.markdown(f"""
<div style="display:grid;grid-template-columns:1fr 90px 1fr;gap:14px;align-items:stretch;margin-bottom:18px;">

<div class="ca-card" style="padding:26px 24px;text-align:center;">
<div style="position:absolute;top:0;left:0;right:0;height:4px;background:{first_color};"></div>

<div style="font-size:12px;color:rgba(8,8,8,0.46);letter-spacing:1.1px;text-transform:uppercase;margin-bottom:10px;">
{first_year_int}
</div>

<div style="font-size:42px;margin-bottom:8px;">{first_icon}</div>

<div style="font-size:21px;font-weight:650;color:{first_color};margin-bottom:16px;letter-spacing:-0.4px;">
{first_label}
</div>

<div style="height:1px;background:#eeeeee;margin:0 0 16px 0;"></div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;">
<div>
<div style="font-size:11px;color:rgba(8,8,8,0.44);margin-bottom:4px;">Brand CTR</div>
<div style="font-size:22px;font-weight:700;color:#080808;">{first_brand_ctr:.2f}%</div>
</div>

<div>
<div style="font-size:11px;color:rgba(8,8,8,0.44);margin-bottom:4px;">Industry CTR</div>
<div style="font-size:22px;font-weight:700;color:#080808;">{first_ind_ctr:.2f}%</div>
</div>
</div>

<div style="font-size:13px;color:rgba(8,8,8,0.64);line-height:1.5;">
{diff_html(first_diff)}
</div>
</div>

<div style="display:flex;align-items:center;justify-content:center;">
<div style="width:72px;height:72px;border-radius:999px;background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;display:flex;align-items:center;justify-content:center;">
{trend_arrow}
</div>
</div>

<div class="ca-card" style="padding:26px 24px;text-align:center;">
<div style="position:absolute;top:0;left:0;right:0;height:4px;background:{last_color};"></div>

<div style="font-size:12px;color:rgba(8,8,8,0.46);letter-spacing:1.1px;text-transform:uppercase;margin-bottom:10px;">
{last_year_int}
</div>

<div style="font-size:42px;margin-bottom:8px;">{last_icon}</div>

<div style="font-size:21px;font-weight:650;color:{last_color};margin-bottom:16px;letter-spacing:-0.4px;">
{last_label}
</div>

<div style="height:1px;background:#eeeeee;margin:0 0 16px 0;"></div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;">
<div>
<div style="font-size:11px;color:rgba(8,8,8,0.44);margin-bottom:4px;">Brand CTR</div>
<div style="font-size:22px;font-weight:700;color:#080808;">{last_brand_ctr:.2f}%</div>
</div>

<div>
<div style="font-size:11px;color:rgba(8,8,8,0.44);margin-bottom:4px;">Industry CTR</div>
<div style="font-size:22px;font-weight:700;color:#080808;">{last_ind_ctr:.2f}%</div>
</div>
</div>

<div style="font-size:13px;color:rgba(8,8,8,0.64);line-height:1.5;">
{diff_html(last_diff)}
</div>
</div>

</div>
""", unsafe_allow_html=True)
    
    # Summary phrase
    st.markdown(f"""
<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:16px 18px;margin:8px 0 28px 0;">
<div style="font-size:12px;color:#0009dc;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:8px;">
Trend summary
</div>
<div style="font-size:14px;color:rgba(8,8,8,0.68);line-height:1.65;">
From <b style="color:#080808;">{first_year_int}</b> to <b style="color:#080808;">{last_year_int}</b>, the brand
<b style="color:{trend_color};">{trend_text}</b> in the industry.
</div>
</div>
""", unsafe_allow_html=True)
    
    # === Part 2: What changed in how tags work ===
    first_year = years_available[-2]
    last_year = years_available[-1]
    
    st.markdown(f"""
<div style="margin:28px 0 18px 0;">
<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(255,124,245,0.14);color:#080808;border-radius:999px;padding:8px 13px;font-size:13px;font-weight:650;margin-bottom:12px;">
<span style="width:7px;height:7px;border-radius:50%;background:#ff7cf5;display:inline-block;"></span>
What changed · {first_year} → {last_year}
</div>
<div style="font-size:14px;color:rgba(8,8,8,0.62);line-height:1.55;">
Which tags became more or less correlated with CTR. Arrows show the direction of change.
</div>
</div>
""", unsafe_allow_html=True)
    
    
    shap_with_year = shap_brand_all.merge(
        df_brand_all_years[["filename", "year"]], on="filename", how="left"
    )
    
    deltas = []
    for feat in BINARY_FEATURES:
        if feat not in shap_with_year.columns:
            continue
        
        first_active_files = df_brand_all_years[
            (df_brand_all_years["year"] == first_year) & (df_brand_all_years[feat] == True)
        ]["filename"].tolist()
        first_shap = shap_with_year[
            (shap_with_year["year"] == first_year) &
            (shap_with_year["filename"].isin(first_active_files))
        ][feat]
        
        last_active_files = df_brand_all_years[
            (df_brand_all_years["year"] == last_year) & (df_brand_all_years[feat] == True)
        ]["filename"].tolist()
        last_shap = shap_with_year[
            (shap_with_year["year"] == last_year) &
            (shap_with_year["filename"].isin(last_active_files))
        ][feat]
        
        if len(first_shap) >= 2 and len(last_shap) >= 2:
            deltas.append({
                "tag": feat,
                "delta": last_shap.mean() - first_shap.mean(),
            })
    
    if len(deltas) == 0:
        st.info(
            f"Not enough data to compare {first_year} and {last_year}. "
            "At least 2 creatives with each tag are needed in each year."
        )
    else:
        DELTA_STRONG = 0.10
        DELTA_WEAK = 0.01
        
        deltas_df = pd.DataFrame(deltas)
        deltas_df["abs_delta"] = deltas_df["delta"].abs()
        deltas_df = deltas_df.nlargest(10, "abs_delta").sort_values("delta", ascending=False)
        
        def render_delta_row(row):
            val = row["delta"]
            abs_v = abs(val)
            
            if abs_v < DELTA_WEAK:
                arrow = '<span style="color:rgba(8,8,8,0.34);font-size:20px;font-weight:650;">—</span>'
                row_color = "rgba(8,8,8,0.42)"
                row_bg = "rgba(8,8,8,0.04)"
                label = "stable"
            elif val > 0:
                arrows = "↑↑" if abs_v >= DELTA_STRONG else "↑"
                arrow = f'<span style="color:#1D9E75;font-weight:750;font-size:22px;letter-spacing:-3px;">{arrows}</span>'
                row_color = "#1D9E75"
                row_bg = "rgba(29,158,117,0.08)"
                label = "gained weight"
            else:
                arrows = "↓↓" if abs_v >= DELTA_STRONG else "↓"
                arrow = f'<span style="color:#E24B4A;font-weight:750;font-size:22px;letter-spacing:-3px;">{arrows}</span>'
                row_color = "#E24B4A"
                row_bg = "rgba(226,75,74,0.08)"
                label = "lost weight"
            
            return f"""
        <div style="padding:15px 0;border-bottom:1px solid #eeeeee;display:grid;grid-template-columns:72px 1fr 140px;gap:14px;align-items:center;">
        <div style="text-align:center;background:{row_bg};border-radius:14px;padding:8px 0;">
        {arrow}
        </div>

        <div style="font-size:14px;color:rgba(8,8,8,0.74);font-weight:500;">
        {display_name(row['tag'])}
        </div>

        <div style="text-align:right;">
        <span style="display:inline-flex;align-items:center;gap:6px;background:{row_bg};color:{row_color};border-radius:999px;padding:6px 10px;font-size:11px;font-weight:650;">
        <span style="width:6px;height:6px;border-radius:50%;background:{row_color};display:inline-block;"></span>
        {label}
        </span>
        </div>
        </div>
        """
        
        rows_html = "".join(render_delta_row(row) for _, row in deltas_df.iterrows())

        st.markdown(f"""
        <div class="ca-card" style="padding:22px 26px;margin-top:10px;">
        <div style="position:absolute;top:0;left:0;right:0;height:4px;background:#ff7cf5;"></div>
        {rows_html}
        </div>
        """, unsafe_allow_html=True)
        
        # Brief summary
        growth = deltas_df[deltas_df["delta"] >= DELTA_WEAK]
        decline = deltas_df[deltas_df["delta"] <= -DELTA_WEAK]
        
        summary_cards = ""

        if len(growth) > 0:
            growth_chips = "".join(
                f'<span style="display:inline-block;background:rgba(29,158,117,0.10);color:#1D9E75;border-radius:999px;padding:6px 10px;font-size:12px;font-weight:650;margin:4px 4px 0 0;">{display_name(t)}</span>'
                for t in growth.head(3)["tag"]
            )

            summary_cards += f"""
        <div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:16px 18px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:#1D9E75;"></div>
        <div style="font-size:12px;color:#1D9E75;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:8px;font-weight:650;">
        Gained weight
        </div>
        <div style="font-size:14px;color:rgba(8,8,8,0.66);line-height:1.55;margin-bottom:8px;">
        These tags became more positively associated with CTR.
        </div>
        <div>
        {growth_chips}
        </div>
        </div>
        """

        if len(decline) > 0:
            decline_chips = "".join(
                f'<span style="display:inline-block;background:rgba(226,75,74,0.10);color:#E24B4A;border-radius:999px;padding:6px 10px;font-size:12px;font-weight:650;margin:4px 4px 0 0;">{display_name(t)}</span>'
                for t in decline.tail(3)["tag"]
            )

            summary_cards += f"""
        <div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:16px 18px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:#E24B4A;"></div>
        <div style="font-size:12px;color:#E24B4A;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:8px;font-weight:650;">
        Lost weight
        </div>
        <div style="font-size:14px;color:rgba(8,8,8,0.66);line-height:1.55;margin-bottom:8px;">
        These tags became less positively associated with CTR.
        </div>
        <div>
        {decline_chips}
        </div>
        </div>
        """

        if summary_cards:
            st.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(260px, 1fr));gap:12px;margin-top:16px;">
        {summary_cards}
        </div>
        """, unsafe_allow_html=True)

spacer()

with st.expander("ℹ️ How this is calculated"):
    st.markdown("""
    **Brand position in industry.** For each year we calculate what percentile of industry CTR
    the brand's average CTR falls into:
    
    - 75–100th percentile → 🏆 Top of industry
    - 50–75th → 📈 Above average
    - 25–50th → 📊 Below average
    - 0–25th → ⚠️ Bottom segment
    
    We compare the **last two available years** to show where the brand moved.
    The year filter at the top of the page is intentionally ignored here — the trend
    needs the full timeline to be meaningful.
    
    **Tag changes.** For each tag we calculate the average SHAP across active creatives
    in the first year and the second year, then show the difference. Tags ranked by absolute
    delta — biggest changes (positive or negative) come first.
    
    A tag needs at least 2 active creatives in each year to be included.
    """)

st.markdown("""
<div style="text-align:center;font-size:13px;color:#bbb;margin:60px 0 20px 0;padding-top:24px;border-top:1px solid #eee;">
    Creative Analyzer · Hackathon MVP · Built in Streamlit · by Viktoriia Iachmeneva · 2026
</div>
""", unsafe_allow_html=True)