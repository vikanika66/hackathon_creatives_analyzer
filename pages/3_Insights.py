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
    .stApp { background: white; }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1200px;
    }
    .section-divider {
        height: 1px;
        background: #e8e6df;
        margin: 48px 0 36px 0;
    }
    .section-title {
        font-size: 22px;
        font-weight: 500;
        margin: 0 0 8px 0;
        color: #1a1a1a;
    }
    .section-subtitle {
        font-size: 14px;
        color: #888;
        margin-bottom: 24px;
    }
    .stat-box {
        background: #faf9f5;
        border-radius: 12px;
        padding: 18px 22px;
    }
    .stat-label {
        font-size: 12px;
        color: #888;
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 26px;
        font-weight: 600;
        color: #1a1a1a;
    }
    .stat-comparison {
        font-size: 12px;
        margin-top: 4px;
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
st.title("📊 Insights")
st.caption("Ready-made analytical views per brand — for marketing managers, clients, and designers")

# Filters
col_brand, col_year = st.columns([1, 1])

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

st.markdown('<div class="section-title">Overall statistics</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">How the brand looks compared to the entire database</div>', unsafe_allow_html=True)

# Compute metrics
brand_avg_ctr = df_brand["ctr"].mean()
brand_median_ctr = df_brand["ctr"].median()
brand_max_ctr = df_brand["ctr"].max()
brand_min_ctr = df_brand["ctr"].min()
brand_count = len(df_brand)

industry_avg_ctr = df_all["ctr"].mean()
ctr_diff = brand_avg_ctr - industry_avg_ctr

# Comparison with average
diff_color = "#1D9E75" if ctr_diff > 0 else "#E24B4A"
diff_sign = "+" if ctr_diff > 0 else ""
diff_text = f"{diff_sign}{ctr_diff:.2f}% vs all CTR ({industry_avg_ctr:.2f}%)"

st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:12px;">
    <div class="stat-box">
        <div class="stat-label">Creatives</div>
        <div class="stat-value">{brand_count}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Average CTR</div>
        <div class="stat-value">{brand_avg_ctr:.2f}%</div>
        <div class="stat-comparison" style="color:{diff_color};">{diff_text}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Median CTR</div>
        <div class="stat-value">{brand_median_ctr:.2f}%</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Range</div>
        <div class="stat-value" style="font-size:18px;">{brand_min_ctr:.2f}% — {brand_max_ctr:.2f}%</div>
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

st.markdown('<div class="section-title">What works for the brand</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">How often each tag appears in creatives with high or low CTR. '
    'This is a pattern in the data, not a guaranteed effect.</div>',
    unsafe_allow_html=True
)

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
    """kind: 'strong' or 'weak'"""
    val = row["avg_shap"]
    abs_v = abs(val)
    
    if abs_v < TAG_WEAK:
        arrow = '<span style="color:#b5b3a8;font-size:18px;">—</span>'
    elif val > 0:
        arrows = "↑↑" if abs_v >= TAG_STRONG else "↑"
        arrow = f'<span style="color:#1D9E75;font-weight:600;font-size:20px;letter-spacing:-3px;">{arrows}</span>'
    else:
        arrows = "↓↓" if abs_v >= TAG_STRONG else "↓"
        arrow = f'<span style="color:#E24B4A;font-weight:600;font-size:20px;letter-spacing:-3px;">{arrows}</span>'
    
    return f"""
    <div style="padding:10px 0;border-bottom:1px solid #f5f3ed;display:flex;align-items:center;gap:12px;">
        <span style="min-width:50px;text-align:center;">{arrow}</span>
        <span style="flex:1;font-size:14px;">{display_name(row['tag'])}</span>
        <span style="font-size:11px;color:#888;">{row['count']} creatives</span>
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
    st.markdown(
        "<div style='font-size:16px;font-weight:600;margin-bottom:12px;color:#1D9E75;'>"
        "✓ Strong tags</div>",
        unsafe_allow_html=True,
    )
    if len(strong_candidates) == 0:
        st.markdown(
            "<div style='color:#888;font-size:14px;padding:10px 0;'>"
            "No tags clearly correlate with high CTR for this brand</div>",
            unsafe_allow_html=True,
        )
    else:
        rows_html = "".join(render_tag_row(row, "strong") for _, row in strong_candidates.iterrows())
        st.markdown(rows_html, unsafe_allow_html=True)

with col_weak:
    st.markdown(
        "<div style='font-size:16px;font-weight:600;margin-bottom:12px;color:#E24B4A;'>"
        "✗ Weak tags</div>",
        unsafe_allow_html=True,
    )
    if len(weak_candidates) == 0:
        st.markdown(
            "<div style='color:#888;font-size:14px;padding:10px 0;'>"
            "No tags clearly correlate with low CTR for this brand</div>",
            unsafe_allow_html=True,
        )
    else:
        rows_html = "".join(render_tag_row(row, "weak") for _, row in weak_candidates.iterrows())
        st.markdown(rows_html, unsafe_allow_html=True)

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

st.markdown('<div class="section-title">Tag interactions map</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Which tag pairs more often appear together in successful or weak creatives. '
    '<b>↑↑ / ↓↓</b> — strong correlation, <b>↑ / ↓</b> — moderate, empty — neutral.</div>',
    unsafe_allow_html=True
)

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
    height=550,
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="white",
    xaxis=dict(side="bottom", tickfont=dict(size=13)),
    yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
)

st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown("""
<div style="font-size:13px;color:#888;margin-top:8px;">
💡 <b>How to read:</b> Green cells with up arrows — tag pairs that appear more often in high-CTR creatives. 
Red cells with down arrows — pairs more often in low-CTR creatives. 
Empty cells — neutral or rare pairings.
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

st.markdown('<div class="section-title">Best and worst creatives</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Top-5 performers and bottom-5 underachievers — learn from your own cases</div>', unsafe_allow_html=True)

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
    sign = "🏆" if tone == "good" else "⚠️"
    
    # Active tags (max 4)
    active_tags = []
    for feat in BINARY_FEATURES:
        if feat in row.index and row[feat] == True:
            active_tags.append(display_name(feat))
        if len(active_tags) >= 4:
            break
    
    tags_html = " · ".join(f"<span style='color:#666;'>{t}</span>" for t in active_tags) or "<span style='color:#bbb;'>—</span>"
    
    # Image as base64 — contain, to show full
    image_path = os.path.join(IMAGES_DIR, row["filename"])
    data_url = encode_image(image_path)
    
    if data_url:
        image_html = (
            '<div style="aspect-ratio:1;border-radius:8px;overflow:hidden;background:#fff;'
            'display:flex;align-items:center;justify-content:center;">'
            f'<img src="{data_url}" style="max-width:100%;max-height:100%;object-fit:contain;display:block;"/>'
            '</div>'
        )
    else:
        image_html = (
            '<div style="aspect-ratio:1;border-radius:8px;background:#fff;'
            'display:flex;align-items:center;justify-content:center;color:#bbb;font-size:12px;">'
            'no image</div>'
        )
    
    # Card as one block, no <a> wrap
    card_html = (
        '<div style="background:#faf9f5;border-radius:12px;padding:14px;">'
            '<div style="height:64px;margin-bottom:10px;overflow:hidden;">'
                '<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">'
                    f'<span style="font-size:14px;">{sign}</span>'
                    f'<span style="color:{color};font-weight:600;font-size:16px;">{row["ctr"]:.2f}%</span>'
                    f'<span style="font-size:11px;color:#888;">#{rank}</span>'
                '</div>'
                f'<div style="font-size:11px;line-height:1.5;">{tags_html}</div>'
            '</div>'
            f'{image_html}'
        '</div>'
    )
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Link button below card
    st.link_button(
        "Open in Library →",
        f"/Library?creative={quote(str(row['filename']))}",
        use_container_width=True,
    )


df_sorted = df_brand.sort_values("ctr", ascending=False).reset_index(drop=True)
top_5 = df_sorted.head(5)
bottom_5 = df_sorted.tail(5).iloc[::-1].reset_index(drop=True)

# Top-5
st.markdown("<div style='font-size:15px;font-weight:600;margin:8px 0 12px 0;color:#1D9E75;'>Top 5</div>", unsafe_allow_html=True)
cols = st.columns(5, gap="small")
for i, col in enumerate(cols):
    if i < len(top_5):
        with col:
            render_creative_card(top_5.iloc[i], i + 1, "good")

# Bottom-5
st.markdown("<div style='font-size:15px;font-weight:600;margin:24px 0 12px 0;color:#E24B4A;'>Bottom 5</div>", unsafe_allow_html=True)
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

st.markdown('<div class="section-title">Anomalies: creatives outside the tag pattern</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Creatives whose CTR differs significantly from others '
    'with a similar tag combination. They suggest that something <b>beyond</b> the visible elements is at play.</div>',
    unsafe_allow_html=True
)

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
        icon = "💎"
        label = "Hidden gem"
        comparison_label = "CTR above similar"
    else:
        color = "#E24B4A"
        icon = "📉"
        label = "Underperformer"
        comparison_label = "CTR below similar"

    # Active tags (max 3)
    active_tags = []
    for feat in BINARY_FEATURES:
        if feat in row.index and row[feat] == True:
            active_tags.append(display_name(feat))
        if len(active_tags) >= 3:
            break

    tags_html = " · ".join(f"<span style='color:#666;'>{t}</span>" for t in active_tags) or "<span style='color:#bbb;'>—</span>"

    # Image
    image_path = os.path.join(IMAGES_DIR, row["filename"])
    data_url = encode_image(image_path)

    if data_url:
        image_html = (
            '<div style="aspect-ratio:1;border-radius:8px;overflow:hidden;background:#fff;'
            'display:flex;align-items:center;justify-content:center;">'
            f'<img src="{data_url}" style="max-width:100%;max-height:100%;object-fit:contain;display:block;"/>'
            '</div>'
        )
    else:
        image_html = (
            '<div style="aspect-ratio:1;border-radius:8px;background:#fff;'
            'display:flex;align-items:center;justify-content:center;color:#bbb;font-size:12px;">'
            'no image</div>'
        )

    # Arrow by anomaly strength
    residual = row["residual"]
    abs_r = abs(residual)
    if abs_r < 0.10:
        arrow_inline = '<span style="color:#b5b3a8;font-size:18px;">—</span>'
    elif residual > 0:
        arrows = "↑↑" if abs_r >= 0.30 else "↑"
        arrow_inline = f'<span style="color:#1D9E75;font-weight:600;font-size:18px;letter-spacing:-3px;">{arrows}</span>'
    else:
        arrows = "↓↓" if abs_r >= 0.30 else "↓"
        arrow_inline = f'<span style="color:#E24B4A;font-weight:600;font-size:18px;letter-spacing:-3px;">{arrows}</span>'

    card_html = (
        '<div style="background:#faf9f5;border-radius:12px;padding:14px;">'
            '<div style="height:96px;margin-bottom:10px;overflow:hidden;">'
                '<div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">'
                    f'<span style="font-size:14px;">{icon}</span>'
                    f'<span style="color:{color};font-weight:600;font-size:13px;">{label}</span>'
                '</div>'
                '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:12px;color:#666;flex-wrap:wrap;">'
                    f'<span>CTR: <b style="color:#1a1a1a;">{row["ctr"]:.2f}%</b></span>'
                    f'<span style="color:#bbb;">|</span>'
                    f'<span>{comparison_label} {arrow_inline}</span>'
                '</div>'
                f'<div style="font-size:11px;line-height:1.5;">{tags_html}</div>'
            '</div>'
            f'{image_html}'
        '</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)

    st.link_button(
        "Open in Library →",
        f"/Library?creative={quote(str(row['filename']))}",
        use_container_width=True,
    )


# Hidden gems
st.markdown(
    "<div style='font-size:15px;font-weight:600;margin:8px 0 12px 0;color:#1D9E75;'>"
    "💎 Hidden gems — CTR above creatives with similar tags</div>",
    unsafe_allow_html=True,
)
if len(hidden_gems) == 0:
    st.info("Not enough data to detect hidden gems in this segment")
else:
    cols = st.columns(3, gap="small")
    for i, col in enumerate(cols):
        if i < len(hidden_gems):
            with col:
                render_anomaly_card(hidden_gems.iloc[i], "gem")
    st.markdown(
        "<div style='font-size:13px;color:#666;margin:12px 0 24px 0;'>"
        "💡 These creatives have notably higher CTR than others with similar tag combinations. "
        "Something is working <b>beyond</b> the visible tags — copy, emotion, execution. "
        "Worth a manual review: what makes them special that can't be captured by a tag?"
        "</div>",
        unsafe_allow_html=True
    )

# Underperformers
st.markdown(
    "<div style='font-size:15px;font-weight:600;margin:8px 0 12px 0;color:#E24B4A;'>"
    "📉 Underperformers — CTR below creatives with similar tags</div>",
    unsafe_allow_html=True,
)
if len(underperformers) == 0:
    st.info("Not enough data to detect underperformers in this segment")
else:
    cols = st.columns(3, gap="small")
    for i, col in enumerate(cols):
        if i < len(underperformers):
            with col:
                render_anomaly_card(underperformers.iloc[i], "underperformer")
    st.markdown(
        "<div style='font-size:13px;color:#666;margin-top:12px;'>"
        "💡 These creatives have notably lower CTR than others with similar tag combinations. "
        "The concept (tag combination) is close to strong creatives, but the <b>execution</b> falls short — "
        "photo quality, unreadable text, weak CTA, poor composition."
        "</div>",
        unsafe_allow_html=True
    )

spacer()

with st.expander("ℹ️ How this is calculated"):
    st.markdown("""
    For each creative we calculate the **residual** = actual CTR − expected CTR for creatives
    with the same tag combination. The expected CTR comes from the LightGBM model trained on
    the full database.
    
    A large positive residual means: this creative performs better than other creatives with
    similar visual elements — something is working **beyond** the tags we can see.
    A large negative residual means: it performs worse than similar creatives — likely something
    in execution (photo quality, copy, layout) drags it down.
    
    - **💎 Hidden gems** — top 3 by residual (CTR above similar creatives)
    - **📉 Underperformers** — bottom 3 by residual (CTR below similar creatives)
    
    These are signals to investigate manually — what makes the gems special that we can't
    capture with tags? What's broken in the underperformers?
    """)

# ============================================================
# BLOCK 6: Year-over-year trend (ignores year filter)
# ============================================================
divider()

st.markdown('<div class="section-title">Year-over-year trend</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">How the brand position in the industry evolved and which tags gained or lost weight. '
    '<b>The year filter is not applied here</b> — the trend is always calculated across all available brand years.</div>',
    unsafe_allow_html=True,
)

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
    st.markdown(
        "<div style='font-size:15px;font-weight:600;margin:8px 0 12px 0;'>"
        "Brand position in industry</div>",
        unsafe_allow_html=True,
    )
    
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
    
    first_diff = first_brand_ctr - first_ind_ctr
    last_diff = last_brand_ctr - last_ind_ctr
    
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
    <div style="display:grid;grid-template-columns:1fr 80px 1fr;gap:12px;align-items:center;margin-bottom:16px;">
        <div style="background:#faf9f5;border-radius:12px;padding:24px;text-align:center;">
            <div style="font-size:12px;color:#888;margin-bottom:8px;">{first_year_int}</div>
            <div style="font-size:42px;margin-bottom:8px;">{first_icon}</div>
            <div style="font-size:18px;font-weight:600;color:{first_color};margin-bottom:14px;">{first_label}</div>
            <div style="font-size:13px;color:#666;line-height:1.6;padding-top:12px;border-top:1px solid #eae8df;">
                <div style="margin-bottom:4px;">
                    Brand CTR: <b style="color:#1a1a1a;">{first_brand_ctr:.2f}%</b>
                </div>
                <div style="margin-bottom:4px;color:#888;">
                    Industry: {first_ind_ctr:.2f}%
                </div>
                <div style="margin-top:6px;font-size:13px;">
                    {diff_html(first_diff)}
                </div>
            </div>
        </div>
        <div style="text-align:center;">{trend_arrow}</div>
        <div style="background:#faf9f5;border-radius:12px;padding:24px;text-align:center;">
            <div style="font-size:12px;color:#888;margin-bottom:8px;">{last_year_int}</div>
            <div style="font-size:42px;margin-bottom:8px;">{last_icon}</div>
            <div style="font-size:18px;font-weight:600;color:{last_color};margin-bottom:14px;">{last_label}</div>
            <div style="font-size:13px;color:#666;line-height:1.6;padding-top:12px;border-top:1px solid #eae8df;">
                <div style="margin-bottom:4px;">
                    Brand CTR: <b style="color:#1a1a1a;">{last_brand_ctr:.2f}%</b>
                </div>
                <div style="margin-bottom:4px;color:#888;">
                    Industry: {last_ind_ctr:.2f}%
                </div>
                <div style="margin-top:6px;font-size:13px;">
                    {diff_html(last_diff)}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary phrase
    st.markdown(
        f"<div style='font-size:14px;color:#1a1a1a;margin:8px 0 24px 0;'>"
        f"From {first_year_int} to {last_year_int}, the brand "
        f"<b style='color:{trend_color};'>{trend_text}</b> in the industry."
        f"</div>",
        unsafe_allow_html=True,
    )
    
    # === Part 2: What changed in how tags work ===
    first_year = years_available[-2]
    last_year = years_available[-1]
    
    st.markdown(
        f"<div style='font-size:15px;font-weight:600;margin:24px 0 4px 0;'>"
        f"What changed: {first_year} → {last_year}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:13px;color:#888;margin-bottom:12px;'>"
        "Which tags became more or less correlated with CTR. Arrows show the direction of change.</div>",
        unsafe_allow_html=True,
    )
    
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
                arrow = '<span style="color:#b5b3a8;font-size:18px;">—</span>'
            elif val > 0:
                arrows = "↑↑" if abs_v >= DELTA_STRONG else "↑"
                arrow = f'<span style="color:#1D9E75;font-weight:600;font-size:20px;letter-spacing:-3px;">{arrows}</span>'
            else:
                arrows = "↓↓" if abs_v >= DELTA_STRONG else "↓"
                arrow = f'<span style="color:#E24B4A;font-weight:600;font-size:20px;letter-spacing:-3px;">{arrows}</span>'
            
            return f"""
            <div style="padding:10px 0;border-bottom:1px solid #f5f3ed;display:flex;align-items:center;gap:12px;">
                <span style="min-width:50px;text-align:center;">{arrow}</span>
                <span style="flex:1;font-size:14px;">{display_name(row['tag'])}</span>
            </div>
            """
        
        rows_html = "".join(render_delta_row(row) for _, row in deltas_df.iterrows())
        st.markdown(rows_html, unsafe_allow_html=True)
        
        # Brief summary
        growth = deltas_df[deltas_df["delta"] >= DELTA_WEAK]
        decline = deltas_df[deltas_df["delta"] <= -DELTA_WEAK]
        
        summary_html = "<div style='font-size:13px;color:#666;margin-top:16px;line-height:1.7;'>"
        if len(growth) > 0:
            top_growth_tags = ", ".join(f"<b>{display_name(t)}</b>" for t in growth.head(3)["tag"])
            summary_html += f"📈 <b style='color:#1D9E75;'>Gained weight:</b> {top_growth_tags}<br>"
        if len(decline) > 0:
            top_decline_tags = ", ".join(f"<b>{display_name(t)}</b>" for t in decline.tail(3)["tag"])
            summary_html += f"📉 <b style='color:#E24B4A;'>Lost weight:</b> {top_decline_tags}"
        summary_html += "</div>"
        st.markdown(summary_html, unsafe_allow_html=True)

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