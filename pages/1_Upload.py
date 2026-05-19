"""
Upload and analyze page — version 3.
Clean typography, no white cards.
"""

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import io
import os
from PIL import Image

from utils import (
    COMMON_CSS,
    load_data,
    load_model,
    get_explainer,
    tag_image_cached,
    tags_to_features,
    BINARY_FEATURES,
    CATEGORICAL_TAGS,
)

import base64 as _base64_for_openai
from openai import OpenAI


@st.cache_resource
def get_openai_client():
    """Cached OpenAI client."""
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def generate_image_via_openai(prompt, source_image_bytes, quality="medium"):
    """Image-to-image. Size auto-picked based on source aspect."""
    client = get_openai_client()
    if client is None:
        return None, "OPENAI_API_KEY is not set in .streamlit/secrets.toml"
    
    try:
        from io import BytesIO
        
        img = Image.open(BytesIO(source_image_bytes))
        
        # Pick closest supported size by aspect ratio
        aspect = img.width / img.height
        if aspect > 1.2:
            size = "1536x1024"   # landscape
        elif aspect < 0.85:
            size = "1024x1536"   # portrait
        else:
            size = "1024x1024"   # square
        
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        
        png_buffer = BytesIO()
        img.save(png_buffer, format="PNG")
        png_buffer.seek(0)
        png_buffer.name = "source.png"
        
        result = client.images.edit(
            model="gpt-image-1.5",
            image=png_buffer,
            prompt=prompt,
            n=1,
            size=size,
            quality=quality,
        )
        image_b64 = result.data[0].b64_json
        return _base64_for_openai.b64decode(image_b64), None
    except Exception as e:
        return None, str(e)


st.set_page_config(
    page_title="Analyze — Creative Analyzer",
    page_icon="🎨",
    layout="wide",
)

# ============================================================
# Styles — clean typography
# ============================================================
PAGE_CSS = """
<style>
    :root {
        --ca-blue: #0009dc;
        --ca-white: #ffffff;
        --ca-black: #080808;
        --ca-soft: #f9f9f9;
        --ca-green: #aef33e;
        --ca-pink: #ff7cf5;
        --ca-royal: #4169e1;
        --ca-border: #eeeeee;
        --ca-muted: rgba(8,8,8,0.62);
    }

    .stApp { background: var(--ca-white); }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1200px;
    }

    .section-divider {
        height: 1px;
        background: #e9e9e9;
        margin: 52px 0 36px 0;
    }

    .ca-kicker {
        font-size: 12px;
        color: var(--ca-blue);
        letter-spacing: 1.35px;
        text-transform: uppercase;
        margin-bottom: 7px;
        font-weight: 500;
    }

    .section-title {
        font-size: 26px;
        font-weight: 650;
        margin: 0 0 8px 0;
        color: var(--ca-black);
        letter-spacing: -0.5px;
    }

    .section-subtitle {
        font-size: 15px;
        color: var(--ca-muted);
        margin-bottom: 24px;
        line-height: 1.55;
        max-width: 820px;
    }

    .ca-card {
        background: linear-gradient(180deg, #ffffff 0%, #f9f9f9 100%);
        border: 1px solid var(--ca-border);
        border-radius: 22px;
        padding: 24px 24px;
        position: relative;
        overflow: hidden;
    }

    .ca-card-blue::before,
    .ca-card-green::before,
    .ca-card-pink::before,
    .ca-card-black::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }
    .ca-card-blue::before { background: var(--ca-blue); }
    .ca-card-green::before { background: var(--ca-green); }
    .ca-card-pink::before { background: var(--ca-pink); }
    .ca-card-black::before { background: var(--ca-black); }

    .metric-mini {
        background: linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);
        border: 1px solid var(--ca-border);
        border-radius: 18px;
        padding: 18px 16px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .metric-mini::before {
        content:"";
        position:absolute;
        top:0;
        left:0;
        right:0;
        height:3px;
        background: var(--ca-blue);
    }
    .metric-mini-label {
        font-size: 11px;
        color: rgba(8,8,8,0.55);
        margin-bottom: 6px;
        letter-spacing: 0.2px;
    }
    .metric-mini-value {
        font-size: 26px;
        line-height: 1;
        font-weight: 650;
        color: var(--ca-black);
    }

    .tag-chip {
        display: inline-block;
        padding: 6px 11px;
        margin: 4px 4px 4px 0;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 500;
        border: 1px solid transparent;
    }
    .tag-binary-active {
        background: var(--ca-blue);
        color: var(--ca-white);
    }
    .tag-categorical-active {
        background: var(--ca-green);
        color: var(--ca-black);
    }
    .tag-absent {
        background: var(--ca-white);
        color: rgba(8,8,8,0.35);
        border-color: #dddddd;
        text-decoration: line-through;
    }

    .effect-row {
        display: flex;
        align-items: center;
        padding: 13px 0;
        border-bottom: 1px solid #eeeeee;
    }
    .effect-row:last-child { border-bottom: none; }
    .effect-name {
        flex: 1;
        font-size: 14px;
        color: rgba(8,8,8,0.74);
        margin-left: 12px;
    }

    .rec-row {
        padding: 14px 0;
        border-bottom: 1px solid #eeeeee;
        font-size: 14px;
        color: rgba(8,8,8,0.68);
        line-height: 1.55;
    }
    .rec-row:last-child { border-bottom: none; }
    .rec-add { color: #0009dc; font-weight: 650; }
    .rec-remove { color: #080808; font-weight: 650; }

    .similar-open-btn {
        display: inline-block;
        margin-top: 8px;
        padding: 7px 13px;
        background: var(--ca-blue);
        color: white !important;
        text-decoration: none !important;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 500;
    }
    .similar-open-btn:hover { background: #0007b8; }

    .stButton > button[kind="primary"],
    .stButton > button {
        border-radius: 999px !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"] {
        background-color: var(--ca-blue) !important;
        border-color: var(--ca-blue) !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #0007b8 !important;
        border-color: #0007b8 !important;
        color: white !important;
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
    }
    div[data-testid="stExpander"] details summary:hover {
        background:
            radial-gradient(circle at 8% 50%, rgba(174,243,62,0.28), transparent 24%),
            radial-gradient(circle at 92% 50%, rgba(0,9,220,0.12), transparent 26%),
            linear-gradient(135deg, #ffffff 0%, #f9f9f9 100%) !important;
    }

    section[data-testid="stSidebar"] {
        background: #f9f9f9;
        border-right: 1px solid #eeeeee;
    }

    div[data-testid="stFileUploader"] section {
        background:
            radial-gradient(circle at 16% 20%, rgba(174,243,62,0.24), transparent 28%),
            radial-gradient(circle at 86% 16%, rgba(255,124,245,0.14), transparent 30%),
            linear-gradient(135deg, #f9f9f9 0%, #ffffff 100%);
        border: 1px dashed rgba(0,9,220,0.35);
        border-radius: 24px;
        padding: 26px;
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
model = load_model()
explainer = get_explainer(model)

IMAGES_DIR = "images"

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

def display_name(tag):
    if tag in TAG_DISPLAY_NAMES:
        return TAG_DISPLAY_NAMES[tag]
    clean = tag.replace("_", " ")
    for prefix in ["food type ", "drink type ", "main object ", "person who ", "person emotion ", "person action "]:
        clean = clean.replace(prefix, "")
    return clean

def categorize_strength(value, strong_threshold, weak_threshold):
    """Categorize correlation: (direction, strength)."""
    abs_v = abs(value)
    if abs_v < weak_threshold:
        return "neutral", "none"
    direction = "up" if value > 0 else "down"
    strength = "strong" if abs_v >= strong_threshold else "weak"
    return direction, strength


def arrow_symbol(direction, strength):
    """HTML arrows: green for positive, red for negative, gray for neutral."""
    if direction == "neutral":
        return '<span style="color:rgba(8,8,8,0.34);font-size:20px;font-weight:650;">—</span>'

    if direction == "up":
        arrows = "↑↑" if strength == "strong" else "↑"
        color = "#1D9E75"
    else:
        arrows = "↓↓" if strength == "strong" else "↓"
        color = "#E24B4A"

    return f'<span style="color:{color};font-weight:750;font-size:22px;letter-spacing:-3px;">{arrows}</span>'

# ============================================================
# History
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "selected_history" not in st.session_state:
    st.session_state.selected_history = None


# ============================================================
# Analysis functions
# ============================================================

def get_active_tags(tags):
    active = []
    for binary in BINARY_FEATURES:
        if tags.get(binary, False):
            active.append(binary)
    for cat in CATEGORICAL_TAGS:
        value = tags.get(cat, "none")
        if value != "none":
            col_name = f"{cat}_{value}"
            if col_name in feature_cols:
                active.append(col_name)
    return active


def calculate_tag_effects(active_tags):
    """Only binary tags — categories don't go here."""
    effects = []
    for tag in active_tags:
        if tag not in BINARY_FEATURES:
            continue
        if tag in shap_df.columns:
            mask = df[tag] == True
            if mask.sum() > 0:
                avg_shap = shap_df.loc[mask.values, tag].mean()
                effects.append((tag, avg_shap))
    effects.sort(key=lambda x: abs(x[1]), reverse=True)
    return effects


import plotly.graph_objects as go

def build_tag_effects_waterfall(effects, base_ctr):
    """Compact waterfall for tag effects."""
    if not effects:
        return None
    
    labels = ["average CTR"]
    values = [base_ctr]
    measure = ["absolute"]
    
    running_total = base_ctr
    for tag, val in effects:
        labels.append(display_name(tag))
        values.append(val)
        measure.append("relative")
        running_total += val
    
    labels.append("expected CTR")
    values.append(0)
    measure.append("total")
    
    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=measure,
        y=labels,
        x=values,
        textposition="outside",
        text=[f"{base_ctr:.2f}%"] + [f"{'+' if v > 0 else ''}{v:.2f}%" for _, v in effects] + [f"{running_total:.2f}%"],
        connector={"line": {"color": "#e0ddd5", "width": 1, "dash": "dot"}},
        increasing={"marker": {"color": "#aef33e"}},
        decreasing={"marker": {"color": "#e8506e"}},
        totals={"marker": {"color": "#0009dc"}},
    ))
    
    fig.update_layout(
        showlegend=False,
        height=max(280, 50 + 42 * len(labels)),
        margin=dict(l=0, r=60, t=10, b=10),
        plot_bgcolor="white",
        yaxis=dict(autorange="reversed", showgrid=False, tickfont=dict(size=13)),
        xaxis=dict(showgrid=True, gridcolor="#f5f3ed", zeroline=False, showticklabels=False),
        font=dict(size=13, color="#1a1a1a"),
        bargap=0.5,
    )
    return fig


def find_similar_creatives(active_tags, food_type=None, drink_type=None, top_n=20):
    """
    Category-priority cascade for similar creatives:
      1. Same brand + same category — ideal
      2. If brand has < 5 in this category — expand to OTHER brands in same category
      3. If still < 5 — fall back to whole brand portfolio
    
    Returns (similarities, scope):
      similarities: list of (idx, score) where idx refers to df_full
      scope: "brand-category" | "category-expanded" | "brand-portfolio" | "brand-only"
    """
    # Feature importance — global signal across the whole database
    feature_importance = {
        c: np.abs(shap_df_full[c]).mean() for c in feature_cols if c in shap_df_full.columns
    }

    has_category = (food_type and food_type != "none") or (drink_type and drink_type != "none")

    # Step 1: same brand + same category
    candidates = df.copy()  # df = brand-filtered here
    if food_type and food_type != "none":
        candidates = candidates[candidates["food_type"] == food_type]
    elif drink_type and drink_type != "none":
        candidates = candidates[candidates["drink_type"] == drink_type]

    scope = "brand-category" if has_category else "brand-only"

    # Step 2: expand to other brands in the same category if too few
    if len(candidates) < 5 and has_category:
        fallback = df_full.copy()
        if food_type and food_type != "none":
            fallback = fallback[fallback["food_type"] == food_type]
        elif drink_type and drink_type != "none":
            fallback = fallback[fallback["drink_type"] == drink_type]
        candidates = pd.concat([candidates, fallback]).drop_duplicates(subset=["filename"])
        scope = "category-expanded"

    # Step 3: last fallback — whole brand portfolio
    if len(candidates) < 5:
        candidates = df.copy()
        scope = "brand-portfolio"

    similarities = []
    for idx, row in candidates.iterrows():
        creative_active = []
        for binary in BINARY_FEATURES:
            if binary in row.index and row[binary] == True:
                creative_active.append(binary)
        for cat in CATEGORICAL_TAGS:
            value = row.get(cat, "none")
            if value != "none":
                col_name = f"{cat}_{value}"
                if col_name in feature_cols:
                    creative_active.append(col_name)
        common = set(active_tags) & set(creative_active)
        sim = sum(feature_importance.get(t, 0) for t in common)
        similarities.append((idx, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n], scope


def get_segment_context(tags):
    """Returns insights about creative's category and base CTR.
    Each insight has status='ok' (>=5 creatives) or status='insufficient' (<5)."""
    base_ctr = df["ctr"].mean()
    insights = []

    food = tags.get("food_type", "none")
    if food != "none":
        segment_df = df[df["food_type"] == food]
        count = len(segment_df)
        if count >= 5:
            seg_ctr = segment_df["ctr"].mean()
            insights.append({
                "label": food.replace("_", " "),
                "category": "food type",
                "ctr": seg_ctr,
                "diff": seg_ctr - base_ctr,
                "count": count,
                "status": "ok",
            })
        else:
            insights.append({
                "label": food.replace("_", " "),
                "category": "food type",
                "count": count,
                "status": "insufficient",
            })

    drink = tags.get("drink_type", "none")
    if drink != "none":
        segment_df = df[df["drink_type"] == drink]
        count = len(segment_df)
        if count >= 5:
            seg_ctr = segment_df["ctr"].mean()
            insights.append({
                "label": drink.replace("_", " "),
                "category": "drink type",
                "ctr": seg_ctr,
                "diff": seg_ctr - base_ctr,
                "count": count,
                "status": "ok",
            })
        else:
            insights.append({
                "label": drink.replace("_", " "),
                "category": "drink type",
                "count": count,
                "status": "insufficient",
            })

    return insights, base_ctr


def calculate_active_combinations(active_tags, food_type=None, drink_type=None, top_n_each=3):
    """Find strongest positive and negative combinations considering the segment."""
    
    if food_type and food_type != "none":
        segment_filter = (
            (interactions_df["segment_type"] == "food") &
            (interactions_df["segment"] == food_type)
        )
    elif drink_type and drink_type != "none":
        segment_filter = (
            (interactions_df["segment_type"] == "drink") &
            (interactions_df["segment"] == drink_type)
        )
    else:
        segment_filter = interactions_df["segment_type"] == "all"
    
    segment_df = interactions_df[segment_filter]
    
    if len(segment_df) < 5:
        segment_df = interactions_df[interactions_df["segment_type"] == "all"]
    
    active_set = set(active_tags)

    active_pairs = segment_df[
        segment_df["feat1"].isin(active_set) &
        segment_df["feat2"].isin(active_set)
    ].copy()

    if len(active_pairs) == 0:
        return [], []

    positive = (
        active_pairs[active_pairs["interaction"] > 0]
        .nlargest(top_n_each, "interaction")
    )
    negative = (
        active_pairs[active_pairs["interaction"] < 0]
        .nsmallest(top_n_each, "interaction")
    )

    return positive.to_dict("records"), negative.to_dict("records")


def get_recommendations_full(active_tags, X_new):
    recs_add = []
    recs_remove = []
    for feat in BINARY_FEATURES:
        if feat not in feature_cols or feat not in shap_df.columns:
            continue
        mask = df[feat] == True
        if mask.sum() < 10:
            continue
        avg_shap = shap_df.loc[mask.values, feat].mean()
        if X_new[feat].iloc[0] == 0 and avg_shap > 0.05:
            recs_add.append((feat, avg_shap))
        elif X_new[feat].iloc[0] == 1 and avg_shap < -0.05:
            recs_remove.append((feat, avg_shap))
    recs_add.sort(key=lambda x: x[1], reverse=True)
    recs_remove.sort(key=lambda x: x[1])
    return recs_add[:5], recs_remove[:5]


def divider():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def is_food_drink_creative(entry):
    tags = entry["tags"]

    food = tags.get("food_type", "none")
    drink = tags.get("drink_type", "none")
    main_object = tags.get("main_object", "none")

    return (
        food != "none"
        or drink != "none"
        or main_object in ("main_dish", "snack", "dessert", "drink", "fruit", "healthy")
    )

def render_non_fnb_alert(entry):
    col_img, col_msg = st.columns([1, 1.5], gap="large")

    with col_img:
        image = Image.open(io.BytesIO(entry["image_bytes"]))
        max_height = 360
        if image.height > max_height:
            ratio = max_height / image.height
            image = image.resize((int(image.width * ratio), max_height), Image.LANCZOS)
        st.image(image)

    with col_msg:
        st.markdown("""
<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:22px;padding:24px 24px;margin-top:8px;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#ff7cf5;"></div>
<div style="font-size:12px;color:#0009dc;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:10px;">Category mismatch</div>
<div style="font-size:20px;font-weight:650;margin-bottom:10px;color:#080808;">This does not look like a Food &amp; Drinks creative</div>
<div style="font-size:14px;color:rgba(8,8,8,0.68);line-height:1.7;">
The AI did not detect any food or drink elements in this image.
Analysis is not shown because the model is trained only on Food &amp; Drinks creatives, so the results would not be meaningful for this category.
<br><br>
<b style="color:#080808;">Try uploading a creative featuring:</b>
<ul style="margin:8px 0 0 0;padding-left:20px;">
<li>Food: burgers, pizza, snacks, desserts, healthy meals</li>
<li>Drinks: soda, juice, coffee, alcohol</li>
<li>Restaurant, delivery, or packaged food visuals</li>
</ul>
</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Block renders
# ============================================================

def render_block_creative_card(entry, selected_brand):
    tags = entry["tags"]
    active_tags = get_active_tags(tags)

    similar, scope = find_similar_creatives(
        active_tags,
        food_type=tags.get("food_type"),
        drink_type=tags.get("drink_type"),
        top_n=30,
    )

    similar_indices = [idx for idx, _ in similar]
    similar_ctrs = df_full.loc[similar_indices, "ctr"]
    ctr_min = similar_ctrs.min()
    ctr_max = similar_ctrs.max()
    ctr_median = similar_ctrs.median()

    st.markdown("""
<div style="margin:8px 0 28px 0;">
<div style="font-size:12px;color:#0009dc;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:8px;">
Uploaded creative
</div>
<div style="font-size:28px;font-weight:650;color:#080808;margin-bottom:8px;letter-spacing:-0.6px;">
Creative breakdown
</div>
<div style="font-size:15px;color:rgba(8,8,8,0.62);max-width:820px;line-height:1.55;">
The uploaded creative is tagged, compared with similar brand creatives, and translated into CTR-related patterns.
</div>
</div>
""", unsafe_allow_html=True)

    col_img, col_info = st.columns([0.85, 1.55], gap="large")

    with col_img:
        image = Image.open(io.BytesIO(entry["image_bytes"]))
        max_height = 460

        if image.height > max_height:
            ratio = max_height / image.height
            image = image.resize((int(image.width * ratio), max_height), Image.LANCZOS)

        st.image(image)

    with col_info:
        st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:12px;margin-bottom:12px;">

<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:20px 14px;text-align:center;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#0009dc;"></div>
<div style="font-size:11px;color:rgba(8,8,8,0.48);margin-bottom:7px;">Similar CTR min</div>
<div style="font-size:25px;font-weight:650;color:#080808;line-height:1;">{ctr_min:.2f}%</div>
</div>

<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:20px 14px;text-align:center;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#0009dc;"></div>
<div style="font-size:11px;color:rgba(8,8,8,0.48);margin-bottom:7px;">Median</div>
<div style="font-size:25px;font-weight:650;color:#080808;line-height:1;">{ctr_median:.2f}%</div>
</div>

<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:20px 14px;text-align:center;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#0009dc;"></div>
<div style="font-size:11px;color:rgba(8,8,8,0.48);margin-bottom:7px;">Similar CTR max</div>
<div style="font-size:25px;font-weight:650;color:#080808;line-height:1;">{ctr_max:.2f}%</div>
</div>

</div>
""", unsafe_allow_html=True)

        food = tags.get("food_type", "none")
        drink = tags.get("drink_type", "none")
        category_label = food.replace("_", " ") if food != "none" else (
            drink.replace("_", " ") if drink != "none" else ""
        )

        if scope == "brand-category":
            caption_text = (
                f"CTR range across similar creatives of <b>{selected_brand}</b> "
                f"in the <b>{category_label}</b> category"
            )
        elif scope == "category-expanded":
            caption_text = (
                f"CTR range across similar creatives in the <b>{category_label}</b> category "
                f"(expanded to other brands — {selected_brand} has few examples here)"
            )
        elif scope == "brand-portfolio":
            caption_text = (
                f"CTR range across all creatives of <b>{selected_brand}</b> "
                f"(category had too few examples to filter)"
            )
        else:
            caption_text = f"CTR range across creatives of <b>{selected_brand}</b>"

        st.markdown(f"""
<div style="font-size:12px;color:rgba(8,8,8,0.48);line-height:1.5;margin:4px 0 26px 0;">
{caption_text}
</div>
""", unsafe_allow_html=True)

        cat_chips = []
        for cat in CATEGORICAL_TAGS:
            value = tags.get(cat, "none")
            if value != "none":
                cat_chips.append(
                    f'<span class="tag-chip tag-categorical-active">{cat.replace("_"," ")}: {value}</span>'
                )

        if cat_chips:
            st.markdown("""
<div style="font-size:12px;color:#0009dc;letter-spacing:1.3px;text-transform:uppercase;margin-bottom:10px;">
Categories
</div>
""", unsafe_allow_html=True)
            st.markdown("".join(cat_chips), unsafe_allow_html=True)
            st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)

        tag_chips = []

        for binary in BINARY_FEATURES:
            if tags.get(binary, False):
                tag_chips.append(
                    f'<span class="tag-chip tag-binary-active">{display_name(binary)}</span>'
                )

        for binary in BINARY_FEATURES:
            if not tags.get(binary, False):
                tag_chips.append(
                    f'<span class="tag-chip tag-absent">{display_name(binary)}</span>'
                )

        st.markdown("""
<div style="font-size:12px;color:#0009dc;letter-spacing:1.3px;text-transform:uppercase;margin-bottom:10px;">
Visual tags
</div>
""", unsafe_allow_html=True)

        st.markdown("".join(tag_chips), unsafe_allow_html=True)

    return active_tags, similar

def render_block_tag_effects(active_tags):
    """Tag correlation with CTR — no numbers or chart, only direction and strength."""
    effects = calculate_tag_effects(active_tags)

    TAG_STRONG = 0.30
    TAG_WEAK = 0.00001

    categorized = []
    for tag, val in effects:
        direction, strength = categorize_strength(val, TAG_STRONG, TAG_WEAK)
        categorized.append((tag, direction, strength))

    st.markdown("""
<div class="ca-kicker">Pattern strength</div>
<div class="section-title">Tag correlation with CTR</div>
<div class="section-subtitle">Which detected tags appear more often in creatives with high or low CTR. <b>↑↑ / ↓↓</b> means strong correlation, <b>↑ / ↓</b> means moderate. This is a pattern in the data, not a guaranteed effect.</div>
""", unsafe_allow_html=True)

    if not categorized:
        st.markdown('<div class="ca-card ca-card-green" style="color:rgba(8,8,8,0.62);font-size:14px;">No significant correlation with CTR for the detected tags.</div>', unsafe_allow_html=True)
        return

    rows_html = '<div class="ca-card ca-card-blue">'
    for tag, direction, strength in categorized:
        rows_html += f'''
            <div class="effect-row">
                <span style="width:72px;text-align:center;">{arrow_symbol(direction, strength)}</span>
                <span class="effect-name">{display_name(tag)}</span>
            </div>'''
    rows_html += '</div>'
    st.markdown(rows_html, unsafe_allow_html=True)

def render_block_combinations(active_tags):
    """Tag pair correlation with CTR — no numbers."""
    positive, negative = calculate_active_combinations(active_tags)

    COMBO_STRONG = 0.02
    COMBO_WEAK = 0.001
    all_combos = []
    for combo in positive + negative:
        direction, strength = categorize_strength(combo["interaction"], COMBO_STRONG, COMBO_WEAK)
        all_combos.append((combo, direction, strength))

    st.markdown("""
<div class="ca-kicker">Combinations</div>
<div class="section-title">Tag pair correlation with CTR</div>
<div class="section-subtitle">Which tag combinations within this brand correlate with CTR more or less than the tags taken separately.</div>
""", unsafe_allow_html=True)

    if not all_combos:
        st.markdown('<div class="ca-card ca-card-green" style="color:rgba(8,8,8,0.62);font-size:14px;">No significant pair correlations found among the active tags.</div>', unsafe_allow_html=True)
        return

    rows_html = '<div class="ca-card ca-card-pink">'
    for combo, direction, strength in all_combos:
        rows_html += f'<div class="effect-row"><span style="width:70px;text-align:center;">{arrow_symbol(direction, strength)}</span><span class="effect-name"><b>{display_name(combo["feat1"])}</b> × <b>{display_name(combo["feat2"])}</b></span></div>'
    rows_html += '</div>'
    st.markdown(rows_html, unsafe_allow_html=True)

def render_block_recommendations(tags):
    X_new = tags_to_features(tags, feature_cols)
    active_tags = get_active_tags(tags)
    recs_add, recs_remove = get_recommendations_full(active_tags, X_new)

    st.markdown("""
<div class="ca-kicker">Next steps</div>
<div class="section-title">Recommendations</div>
<div class="section-subtitle">What you could change to potentially improve CTR. You can check the exact effect for this creative in A/B scenarios below.</div>
""", unsafe_allow_html=True)

    if not recs_add and not recs_remove:
        st.markdown('<div class="ca-card ca-card-green" style="color:rgba(8,8,8,0.68);font-size:14px;">The tag set looks strong. No obvious changes suggested by the current brand patterns.</div>', unsafe_allow_html=True)
    else:
        rows_html = '<div class="ca-card ca-card-green">'
        for feat, _ in recs_add:
            rows_html += f'<div class="rec-row"><span class="rec-add">Add</span>&nbsp;&nbsp;<b>{display_name(feat)}</b> — creatives with this element tend to perform better on average.</div>'
        for feat, _ in recs_remove:
            rows_html += f'<div class="rec-row"><span class="rec-remove">Remove</span>&nbsp;&nbsp;<b>{display_name(feat)}</b> — creatives without it tend to perform better on average.</div>'
        rows_html += '</div>'
        st.markdown(rows_html, unsafe_allow_html=True)

    return recs_add, recs_remove

def build_scenario_prompt(changes):
    """Prompt for generation by A/B scenario — takes explicit change list."""
    to_add = [display_name(f) for f, _, v in changes if v]
    to_remove = [display_name(f) for f, _, v in changes if not v]
    
    instructions = []
    if to_remove:
        instructions.append(f"reduce or remove these elements: {', '.join(to_remove)}")
    if to_add:
        instructions.append(f"add or emphasize these elements: {', '.join(to_add)}")
    
    prompt = (
        "Modify this advertisement creative according to the user's scenario. "
        f"Required changes: {'. '.join(instructions)}. "
        "Keep the same product, brand identity, typography style, and overall composition. "
        "Maintain commercial photography quality."
    )
    return prompt

def render_block_ab_scenarios(tags, entry_hash, source_image_bytes):
    """Combined block: A/B scenarios + AI reference generation."""

    st.markdown("""
<div style="margin:8px 0 28px 0;">
<div style="font-size:12px;color:#0009dc;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:8px;">
Scenario testing
</div>
<div style="font-size:28px;font-weight:650;color:#080808;margin-bottom:8px;letter-spacing:-0.6px;">
A/B scenarios + AI reference
</div>
<div style="font-size:15px;color:rgba(8,8,8,0.62);max-width:820px;line-height:1.55;">
Toggle visual tags, compare the direction of the predicted CTR shift, and generate a visual reference for the selected scenario.
</div>
</div>
""", unsafe_allow_html=True)

    state_key = f"ab_state_{entry_hash}"

    if state_key not in st.session_state:
        st.session_state[state_key] = {
            feat: bool(tags.get(feat, False)) for feat in BINARY_FEATURES
        }

        # Reset button
    if st.button("↻ Reset scenario", key=f"reset_{entry_hash}"):
        st.session_state[state_key] = {
            feat: bool(tags.get(feat, False)) for feat in BINARY_FEATURES
        }

        for feat in BINARY_FEATURES:
            toggle_key = f"toggle_{entry_hash}_{feat}"
            if toggle_key in st.session_state:
                st.session_state[toggle_key] = bool(tags.get(feat, False))

        for key in list(st.session_state.keys()):
            if key.startswith(f"ab_generated_{entry_hash}"):
                del st.session_state[key]

        st.rerun()

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

    # Toggles
    cols = st.columns(3, gap="large")

    for i, feat in enumerate(BINARY_FEATURES):
        toggle_key = f"toggle_{entry_hash}_{feat}"

        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = st.session_state[state_key][feat]

        with cols[i % 3]:
            new_value = st.toggle(
                display_name(feat),
                value=st.session_state[toggle_key],
                key=toggle_key,
            )
            st.session_state[state_key][feat] = new_value

    # Compute original and modified predictions
    original_X = tags_to_features(tags, feature_cols)
    original_ctr = float(model.predict(original_X)[0])

    modified_tags = dict(tags)
    for feat in BINARY_FEATURES:
        modified_tags[feat] = st.session_state[state_key][feat]

    modified_X = tags_to_features(modified_tags, feature_cols)
    modified_ctr = float(model.predict(modified_X)[0])
    diff = modified_ctr - original_ctr

    # Changes list
    changes = []
    for feat in BINARY_FEATURES:
        original_val = bool(tags.get(feat, False))
        new_val = bool(st.session_state[state_key][feat])

        if original_val != new_val:
            action = "turn on" if new_val else "turn off"
            changes.append((feat, action, new_val))

    # Direction categorization
    AB_STRONG = 0.30
    AB_WEAK = 0.10
    direction, strength = categorize_strength(diff, AB_STRONG, AB_WEAK)

    # Result card
    if not changes:
        accent_color = "rgba(8,8,8,0.34)"
        accent_bg = "rgba(8,8,8,0.04)"
        arrow_html = '<span style="color:rgba(8,8,8,0.34);font-size:40px;font-weight:700;">—</span>'
        label = "Nothing changed yet"
        details = "Toggle one or more visual tags to test an alternative creative scenario."

    else:
        changes_text = ", ".join(
            f"<b>{action} {display_name(feat)}</b>" for feat, action, _ in changes
        )

        if direction == "up":
            accent_color = "#1D9E75"
            accent_bg = "rgba(29,158,117,0.10)"
            arrows = "↑↑" if strength == "strong" else "↑"
            arrow_html = f'<span style="color:{accent_color};font-size:42px;font-weight:750;letter-spacing:-6px;">{arrows}</span>'
            label = "Predicted CTR signal improves" if strength == "strong" else "Predicted CTR signal slightly improves"
            details = changes_text

        elif direction == "down":
            accent_color = "#E24B4A"
            accent_bg = "rgba(226,75,74,0.10)"
            arrows = "↓↓" if strength == "strong" else "↓"
            arrow_html = f'<span style="color:{accent_color};font-size:42px;font-weight:750;letter-spacing:-6px;">{arrows}</span>'
            label = "Predicted CTR signal weakens" if strength == "strong" else "Predicted CTR signal slightly weakens"
            details = changes_text

        else:
            accent_color = "rgba(8,8,8,0.42)"
            accent_bg = "rgba(8,8,8,0.04)"
            arrow_html = '<span style="color:rgba(8,8,8,0.34);font-size:40px;font-weight:700;">—</span>'
            label = "Predicted CTR signal stays neutral"
            details = changes_text

    st.markdown(f"""
<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:22px;padding:22px 24px;margin-top:26px;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:4px;background:{accent_color};"></div>

<div style="display:grid;grid-template-columns:90px 1fr;gap:20px;align-items:center;">
<div style="display:flex;align-items:center;justify-content:center;background:{accent_bg};border-radius:18px;min-height:78px;">
{arrow_html}
</div>

<div>
<div style="font-size:16px;font-weight:650;color:{accent_color};margin-bottom:6px;">
{label}
</div>
<div style="font-size:13px;color:rgba(8,8,8,0.62);line-height:1.6;">
{details}
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    # Reference generation
    if not changes:
        st.markdown(
            "<div style='font-size:13px;color:rgba(8,8,8,0.48);margin-top:14px;'>"
            "Toggle some tags above to generate a visual reference for the scenario."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    scenario_signature = "|".join(
        f"{feat}={int(val)}" for feat, _, val in sorted(changes, key=lambda x: x[0])
    )
    scenario_hash = hashlib.md5(scenario_signature.encode()).hexdigest()[:8]
    cache_key = f"ab_generated_{entry_hash}_{scenario_hash}"

    st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)

    if cache_key not in st.session_state:
        if st.button(
            "✨ Generate a visual reference for this scenario",
            use_container_width=True,
            key=f"gen_btn_{entry_hash}_{scenario_hash}",
        ):
            with st.spinner("Generating (~15-30 seconds)..."):
                prompt = build_scenario_prompt(changes)
                image_bytes, error = generate_image_via_openai(prompt, source_image_bytes)

                if error:
                    st.error(f"Generation error: {error}")
                else:
                    st.session_state[cache_key] = {
                        "image": image_bytes,
                        "prompt": prompt,
                        "scenario": changes,
                    }
                    st.rerun()

    else:
        result = st.session_state[cache_key]

        col_orig, col_new = st.columns(2, gap="large")

        with col_orig:
            st.markdown(
                "<div style='font-size:12px;color:#0009dc;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:8px;'>Original</div>",
                unsafe_allow_html=True,
            )
            st.image(source_image_bytes, use_container_width=True)

        with col_new:
            st.markdown(
                "<div style='font-size:12px;color:#0009dc;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:8px;'>AI reference for the scenario</div>",
                unsafe_allow_html=True,
            )
            st.image(result["image"], use_container_width=True)

        st.markdown(
            "<div style='font-size:12px;color:rgba(8,8,8,0.48);margin-top:12px;line-height:1.6;'>"
            "This reference is a starting point for the designer. Logo, typography, and final execution are up to the designer."
            "</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Full prompt"):
            st.code(result["prompt"], language=None)

        if st.button("🔄 Regenerate", key=f"regen_{entry_hash}_{scenario_hash}"):
            del st.session_state[cache_key]
            st.rerun()

def render_block_similar(similar):
    st.markdown("""
<div class="ca-kicker">Reference set</div>
<div class="section-title">Similar creatives from the database</div>
<div class="section-subtitle">Creatives with a similar tag set, sorted by actual CTR.</div>
""", unsafe_allow_html=True)

    top_similar = similar[:10]
    creatives = []
    for idx, sim_score in top_similar:
        row = df_full.loc[idx]
        creatives.append({"filename": row["filename"], "ctr": row["ctr"], "row": row})
    creatives.sort(key=lambda x: x["ctr"], reverse=True)
    median_ctr = df["ctr"].median()

    for row_start in [0, 5]:
        if row_start == 5:
            st.markdown('<div style="height:22px;"></div>', unsafe_allow_html=True)
        cols = st.columns(5, gap="small")
        for i, c in enumerate(creatives[row_start:row_start+5]):
            with cols[i]:
                image_path = os.path.join(IMAGES_DIR, c["filename"])
                if os.path.exists(image_path):
                    st.markdown(f"""
<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:8px;overflow:hidden;">
<div style="height:200px;background:#f9f9f9;border-radius:14px;overflow:hidden;display:flex;align-items:center;justify-content:center;">
<img src="data:image/png;base64,{__import__('base64').b64encode(open(image_path, 'rb').read()).decode()}" style="width:100%;height:100%;object-fit:cover;"/>
</div>
""", unsafe_allow_html=True)
                else:
                    st.markdown('<div style="height:200px;background:#f9f9f9;border-radius:14px;border:1px solid #eeeeee;"></div>', unsafe_allow_html=True)

                ctr_color = "#0009dc" if c["ctr"] >= median_ctr else "#080808"
                active = []
                for binary in BINARY_FEATURES:
                    if binary in c["row"].index and c["row"][binary] == True:
                        active.append(binary)
                feat_imp = {b: np.abs(shap_df_full[b]).mean() for b in active if b in shap_df_full.columns}
                top_tags = sorted(active, key=lambda x: feat_imp.get(x, 0), reverse=True)[:3]
                tag_str = ", ".join(display_name(t) for t in top_tags)

                st.markdown(f"""
<div style="font-size:22px;font-weight:650;color:{ctr_color};margin-top:10px;">{c['ctr']:.2f}%</div>
<div style="font-size:11px;color:rgba(8,8,8,0.52);min-height:34px;line-height:1.45;">{tag_str if tag_str else '—'}</div>
<a href="/Library?creative={c['filename']}" target="_blank" class="similar-open-btn">Open →</a>
</div>
""", unsafe_allow_html=True)

def render_block_segment_context(tags):
    """Insight block about the segment — how strong this category is for the brand.
    Shows either a full comparison card (>=5 creatives) or a disclaimer (<5)."""
    insights, base_ctr = get_segment_context(tags)

    if not insights:
        return

    st.markdown("""
<div style="margin:8px 0 28px 0;">
<div style="font-size:12px;color:#0009dc;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:8px;">
Category context
</div>
<div style="font-size:28px;font-weight:650;color:#080808;margin-bottom:8px;letter-spacing:-0.6px;">
Creative category
</div>
<div style="font-size:15px;color:rgba(8,8,8,0.62);max-width:820px;line-height:1.55;">
How this food or drink category performs compared with the brand average.
</div>
</div>
""", unsafe_allow_html=True)

    rows_html = ""
    NEUTRAL_THRESHOLD = 0.09

    for ins in insights:

        # ============================================================
        # Insufficient data — show disclaimer card
        # ============================================================
        if ins["status"] == "insufficient":
            count = ins["count"]
            if count == 0:
                verdict = (
                    f"<b style='color:#080808;'>{selected_brand}</b> has no creatives "
                    f"in the <b style='color:#080808;'>{ins['label']}</b> category, "
                    f"so a comparison can't be calculated. The detected category "
                    f"falls outside this brand's portfolio."
                )
            else:
                creative_word = "creative" if count == 1 else "creatives"
                verdict = (
                    f"<b style='color:#080808;'>{selected_brand}</b> has only "
                    f"<b style='color:#080808;'>{count}</b> {creative_word} in the "
                    f"<b style='color:#080808;'>{ins['label']}</b> category. "
                    f"A minimum of <b style='color:#080808;'>5</b> is needed for a reliable "
                    f"comparison, so the average is not shown."
                )

            rows_html += f"""
<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:22px;padding:26px 26px;margin-bottom:14px;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:4px;background:rgba(8,8,8,0.34);"></div>

<div style="font-size:22px;font-weight:650;color:#080808;text-transform:capitalize;margin-bottom:8px;">
{ins['label']}
</div>

<div style="display:inline-flex;align-items:center;gap:8px;background:rgba(8,8,8,0.06);color:rgba(8,8,8,0.58);border-radius:999px;padding:7px 12px;font-size:12px;font-weight:650;margin-bottom:14px;">
<span style="width:7px;height:7px;border-radius:50%;background:rgba(8,8,8,0.42);display:inline-block;"></span>
Not enough data
</div>

<div style="font-size:14px;color:rgba(8,8,8,0.66);line-height:1.65;max-width:760px;">
{verdict}
</div>
</div>
"""
            continue

        # ============================================================
        # OK — full comparison card (original logic)
        # ============================================================
        diff = ins["diff"]
        abs_diff = abs(diff)

        if diff > 0:
            accent_color = "#1D9E75"
            accent_bg = "rgba(29,158,117,0.10)"
            diff_text = f"+{diff:.2f}%"

            if abs_diff < NEUTRAL_THRESHOLD:
                accent_label = "Slightly above average"
                verdict = (
                    "This category is close to the brand average, with a small positive difference. "
                    "Treat it as a neutral-to-positive signal, not a strong outperformance."
                )
            else:
                accent_label = "Above brand average"
                verdict = "Creatives in this category tend to perform better than the brand average."

        elif diff < 0:
            accent_color = "#E24B4A"
            accent_bg = "rgba(226,75,74,0.10)"
            diff_text = f"{diff:.2f}%"

            if abs_diff < NEUTRAL_THRESHOLD:
                accent_label = "Slightly below average"
                verdict = (
                    "This category is close to the brand average, with a small negative difference. "
                    "Treat it as a neutral-to-negative signal, not a strong underperformance."
                )
            else:
                accent_label = "Below brand average"
                verdict = "Creatives in this category tend to underperform compared with the brand average."

        else:
            accent_color = "#888888"
            accent_bg = "rgba(8,8,8,0.04)"
            diff_text = "0.00%"
            accent_label = "Neutral"
            verdict = "This category performs at the same level as the brand average."

        rows_html += f"""
<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:22px;padding:26px 26px;margin-bottom:14px;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:4px;background:{accent_color};"></div>

<div style="display:grid;grid-template-columns:1.2fr 0.8fr;gap:28px;align-items:center;">

<div>
<div style="font-size:22px;font-weight:650;color:#080808;text-transform:capitalize;margin-bottom:8px;">
{ins['label']}
</div>

<div style="display:inline-flex;align-items:center;gap:8px;background:{accent_bg};color:{accent_color};border-radius:999px;padding:7px 12px;font-size:12px;font-weight:650;margin-bottom:14px;">
<span style="width:7px;height:7px;border-radius:50%;background:{accent_color};display:inline-block;"></span>
{accent_label}
</div>

<div style="font-size:14px;color:rgba(8,8,8,0.66);line-height:1.65;max-width:620px;">
{verdict}
</div>

<div style="font-size:12px;color:rgba(8,8,8,0.46);margin-top:12px;">
Based on {ins['count']} creatives
</div>
</div>

<div style="text-align:right;">
<div style="font-size:12px;color:rgba(8,8,8,0.46);letter-spacing:1.1px;text-transform:uppercase;margin-bottom:6px;">
Difference
</div>

<div style="font-size:46px;font-weight:750;color:{accent_color};line-height:1;letter-spacing:-1.4px;margin-bottom:14px;">
{diff_text}
</div>

<div style="display:flex;justify-content:flex-end;gap:18px;flex-wrap:wrap;">
<div>
<div style="font-size:11px;color:rgba(8,8,8,0.44);margin-bottom:3px;">Category CTR</div>
<div style="font-size:18px;font-weight:650;color:#080808;">{ins['ctr']:.2f}%</div>
</div>

<div>
<div style="font-size:11px;color:rgba(8,8,8,0.44);margin-bottom:3px;">Brand avg CTR</div>
<div style="font-size:18px;font-weight:650;color:#080808;">{base_ctr:.2f}%</div>
</div>
</div>
</div>

</div>
</div>
"""

    st.markdown(rows_html, unsafe_allow_html=True)

def render_analysis(entry, selected_brand):
    active_tags, similar = render_block_creative_card(entry, selected_brand)
    
    divider()
    render_block_tag_effects(active_tags)
    spacer()
    with st.expander("ℹ️ How this is calculated"):
        st.markdown("""
        For each tag detected in the uploaded creative, we look at all creatives of the
        selected brand where this tag is active and calculate the average SHAP value —
        how much this tag tends to push CTR up or down compared to the brand's baseline.
        
        Arrow indicators:
        
        - ↑↑ / ↓↓ — strong correlation (≥ 0.30 CTR percentage points)
        - ↑ / ↓ — moderate correlation
        - — — neutral
        
        This shows correlation in the brand's historical data, not a guaranteed effect
        for this specific creative.
        """)
    
    divider()
    render_block_segment_context(entry["tags"])
    spacer()
    with st.expander("ℹ️ How this is calculated"):
        st.markdown("""
        Compares the average CTR of brand creatives in the same food/drink category as
        the uploaded creative with the average CTR across all brand creatives.
        
        Positive difference (green) means creatives of this category tend to perform better
        than the brand average. Negative (red) means they tend to underperform.
        
        A minimum of **5 creatives** in the category is required to calculate a reliable average.
        If the brand has fewer than 5, a disclaimer card is shown instead of the comparison —
        this avoids drawing conclusions from too small a sample.
        """)
    
    divider()
    render_block_combinations(active_tags)
    spacer()
    with st.expander("ℹ️ How this is calculated"):
        st.markdown("""
        For each pair of active tags in the uploaded creative, we measure how much
        **adding the second tag** changes CTR beyond what each tag would contribute alone.
        
        We show the top 3 positive pairs (combinations that work well together) and the
        top 3 negative pairs (combinations that work against each other) within the brand's data.
        
        If no significant pairs are found, the tag combination of this creative doesn't have
        strong synergies or conflicts in the data.
        """)
    
    divider()
    recs_add, recs_remove = render_block_recommendations(entry["tags"])
    spacer()
    with st.expander("ℹ️ How this is calculated"):
        st.markdown("""
        For each binary tag we calculate the average SHAP across creatives where this tag is
        active in the brand's data. Then:
        
        - **Add** — tags that are currently OFF in the uploaded creative but have a positive
          average SHAP (> 0.05) — creatives with this tag tend to perform better
        - **Remove** — tags that are currently ON in the uploaded creative but have a negative
          average SHAP (< -0.05) — creatives without this tag tend to perform better
        
        At most 5 add and 5 remove recommendations are shown, ranked by magnitude.
        
        These are data-driven suggestions, not guarantees. Test specific scenarios using A/B
        scenarios below to see the predicted impact for your exact creative.
        """)
    
    divider()
    render_block_ab_scenarios(entry["tags"], entry["hash"], entry["image_bytes"])
    spacer()
    with st.expander("ℹ️ How this is calculated"):
        st.markdown("""
        **A/B scenarios.** Toggling tags re-runs the LightGBM model with the modified tag
        combination. The arrow shows the direction and strength of the predicted CTR shift
        compared to the original tag set:
        
        - ↑↑ / ↓↓ — strong shift (≥ 0.30 CTR percentage points)
        - ↑ / ↓ — moderate shift (≥ 0.10)
        - — — neutral
        
        **AI reference generation.** Once you've changed at least one tag, you can generate
        a visual reference. The system builds a prompt from your changes and uses OpenAI's
        gpt-image-1.5 (image-to-image) to modify the original creative — keeping the brand
        identity but adjusting the visual elements according to your scenario.
        
        The result is a **reference for the designer**, not a final creative — expect visual
        artifacts. Final logo, typography, and execution are up to the designer.
        """)
    
    divider()
    render_block_similar(similar)
    spacer()
    with st.expander("ℹ️ How this is calculated"):
        st.markdown("""
        Similar creatives use a **category-priority cascade**, so the reference set stays visually relevant:
        
        1. **Same brand + same category** — the ideal case (same brand, same food/drink type)
        2. **Other brands in the same category** — added if the brand has fewer than 5 examples in this category, because visual patterns transfer across brands within a category
        3. **Whole brand portfolio** — last fallback if the category is too rare overall
        
        Similarity is computed as the overlap of tag sets, weighted by each tag's **global** importance (average absolute SHAP across the full database — not per-brand, because we want general visual signals).
        
        The top 10 most similar creatives are shown, sorted by their actual CTR (highest first). Click "Open →" on any card to see its full breakdown in the Library.
        """)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### History")
    st.caption(f"Analyzed: {len(st.session_state.history)}")

    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history)):
            real_idx = len(st.session_state.history) - 1 - i
            image = Image.open(io.BytesIO(entry["image_bytes"]))
            image.thumbnail((200, 200))
            col_thumb, col_info = st.columns([1, 1.5])
            with col_thumb:
                st.image(image)
            with col_info:
                st.caption(entry["name"][:18])
                if st.button("Open", key=f"open_{real_idx}", width='stretch'):
                    st.session_state.selected_history = real_idx
                    st.rerun()
        st.divider()
        if st.button("🗑️ Clear", width='stretch'):
            st.session_state.history = []
            st.session_state.selected_history = None
            st.rerun()
    else:
        st.caption("Upload a creative to start")


# ============================================================
# MAIN
# ============================================================

# ============================================================
# PAGE HERO
# ============================================================
st.markdown("""
<div style="margin:24px 0 26px 0;padding:28px 30px;border-radius:26px;background:radial-gradient(circle at 8% 18%, rgba(174,243,62,0.24), transparent 28%),radial-gradient(circle at 92% 12%, rgba(255,124,245,0.12), transparent 28%),linear-gradient(135deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;">
<div style="display:grid;grid-template-columns:1.25fr 0.75fr;gap:28px;align-items:center;">

<div>
<div style="display:inline-flex;gap:8px;align-items:center;padding:7px 12px;border-radius:999px;background:#ffffff;border:1px solid #eeeeee;font-size:12px;color:rgba(8,8,8,0.58);margin-bottom:16px;">
<span style="width:7px;height:7px;border-radius:50%;background:#aef33e;display:inline-block;"></span>
Upload workspace
</div>

<div style="font-size:12px;color:#0009dc;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:10px;">
Analyze a new creative
</div>

<h1 style="font-size:40px;margin:0 0 12px 0;font-weight:650;letter-spacing:-1.2px;color:#080808;">
Upload &amp; Analyze
</h1>

<p style="font-size:17px;color:rgba(8,8,8,0.66);max-width:680px;margin:0;line-height:1.55;">
Drop a Food &amp; Drinks ad creative, detect its visual tags, and see how similar patterns correlate with CTR in the selected brand dataset.
</p>

<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:20px;">
<span style="background:#0009dc;color:#ffffff;padding:7px 12px;border-radius:999px;font-size:12px;">upload</span>
<span style="background:#aef33e;color:#080808;padding:7px 12px;border-radius:999px;font-size:12px;">AI tagging</span>
<span style="background:#080808;color:#ffffff;padding:7px 12px;border-radius:999px;font-size:12px;">CTR patterns</span>
<span style="background:#ff7cf5;color:#080808;padding:7px 12px;border-radius:999px;font-size:12px;">AI reference</span>
</div>
</div>

<div style="background:linear-gradient(180deg, rgba(0,9,220,0.06) 0%, rgba(65,105,225,0.10) 100%);border:1px solid rgba(0,9,220,0.12);border-radius:22px;padding:22px 22px;position:relative;overflow:hidden;min-height:190px;">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#0009dc;"></div>
<div style="position:absolute;top:-42px;right:-42px;width:130px;height:130px;background:#0009dc;border-radius:50%;opacity:0.14;"></div>
<div style="position:absolute;bottom:-48px;left:-48px;width:150px;height:150px;background:#aef33e;border-radius:50%;opacity:0.16;"></div>

<div style="position:relative;z-index:1;">
<div style="font-size:12px;color:#0009dc;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:14px;font-weight:600;">
Workflow
</div>

<div style="font-size:14px;color:rgba(8,8,8,0.78);line-height:1.9;">
<span style="color:#0009dc;font-weight:650;">01</span> · Choose brand dataset<br>
<span style="color:#0009dc;font-weight:650;">02</span> · Upload creative image<br>
<span style="color:#0009dc;font-weight:650;">03</span> · Detect visual elements<br>
<span style="color:#0009dc;font-weight:650;">04</span> · Explore recommendations
</div>

<div style="height:1px;background:rgba(0,9,220,0.10);margin:16px 0 13px 0;"></div>

<div style="font-size:12px;color:rgba(8,8,8,0.56);line-height:1.5;">
Designed for quick creative diagnostics before deeper analysis.
</div>
</div>
</div>
""", unsafe_allow_html=True)



# Info banner about MVP scope
st.markdown("""
<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:22px;padding:20px 22px;margin:12px 0 26px 0;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#aef33e;"></div>
<div style="font-size:12px;color:#0009dc;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:8px;">MVP scope</div>
<div style="font-size:14px;color:rgba(8,8,8,0.66);line-height:1.65;">
<b style="color:#080808;">Food &amp; Drinks only.</b> For best results, upload creatives featuring food, drinks, restaurant, delivery, or related visuals. Non-F&amp;B uploads will produce noisy analytics because the model is trained on F&amp;B data only.<br><br>
All CTR data is <b style="color:#080808;">synthetic</b>, generated based on the structure of real creatives from Meta Ad Library. Brand names are test placeholders. In production, this dropdown would be optional and connected to your own data.
</div>
</div>
""", unsafe_allow_html=True)

# Brand dropdown
all_brands = sorted(df["brand"].unique().tolist())
selected_brand = st.selectbox(
    "Brand for analysis",
    options=all_brands,
    help=(
        "Analysis will be built on creatives of the selected brand. "
        "These are test brands — in a real product this selector would be optional, "
        "and analysis could run across the entire brand portfolio or any custom segment."
    )
)

# Filter data
df_filtered = df[df["brand"] == selected_brand].copy()
filenames = df_filtered["filename"].tolist()
shap_df_filtered = shap_df[shap_df["filename"].isin(filenames)].copy()

# Warning for small segments
if len(df_filtered) < 30:
    st.warning(f"⚠️ Brand {selected_brand} has only {len(df_filtered)} creatives. Analysis may be unreliable.")

# Save full data for cross-brand similar lookup BEFORE override
df_full = df.copy()
shap_df_full = shap_df.copy()

# Override globals for analysis functions
df = df_filtered
shap_df = shap_df_filtered

# # Override globals for analysis functions
# df_all = df.copy()           
# shap_df_all = shap_df.copy() 
# df = df_filtered
# shap_df = shap_df_filtered

st.markdown(f'<div style="font-size:13px;color:rgba(8,8,8,0.55);margin:8px 0 22px 0;">Analysis based on <b>{len(df_filtered)}</b> creatives of <b>{selected_brand}</b></div>', unsafe_allow_html=True)

# ============================================================
# Page logic
# ============================================================

def is_food_drink_creative(entry):
    tags = entry["tags"]

    return (
        tags.get("food_type", "none") != "none"
        or tags.get("drink_type", "none") != "none"
        or tags.get("main_object", "none") in (
            "main_dish",
            "snack",
            "dessert",
            "drink",
            "fruit",
            "healthy",
        )
    )


def render_non_fnb_alert(entry):
    col_img, col_msg = st.columns([1, 1.5], gap="large")

    with col_img:
        image = Image.open(io.BytesIO(entry["image_bytes"]))
        max_height = 360

        if image.height > max_height:
            ratio = max_height / image.height
            image = image.resize((int(image.width * ratio), max_height), Image.LANCZOS)

        st.image(image)

    with col_msg:
        st.markdown("""
<div class="ca-card ca-card-pink" style="margin-top:8px;">
<div style="font-size:12px;color:#0009dc;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:10px;">
Category mismatch
</div>

<div style="font-size:20px;font-weight:650;margin-bottom:10px;color:#080808;">
This does not look like a Food &amp; Drinks creative
</div>

<div style="font-size:14px;color:rgba(8,8,8,0.66);line-height:1.7;">
The AI did not detect food or drink elements in this image. Analysis is not shown because the model is trained only on F&amp;B creatives, so the results would not be meaningful for this category.
<br><br>

<b style="color:#080808;">Try uploading a creative featuring:</b>

<ul style="margin:8px 0 0 0;padding-left:20px;">
<li>Food: burgers, pizza, snacks, desserts, healthy meals</li>
<li>Drinks: soda, juice, coffee, alcohol</li>
<li>Restaurant, delivery, or packaged product visuals</li>
</ul>
</div>
</div>
""", unsafe_allow_html=True)


if st.session_state.selected_history is not None:
    entry = st.session_state.history[st.session_state.selected_history]

    if st.button("← Back to upload"):
        st.session_state.selected_history = None
        st.rerun()

    if not is_food_drink_creative(entry):
        render_non_fnb_alert(entry)
    else:
        render_analysis(entry, selected_brand)

else:
    uploaded_file = st.file_uploader(
        "Drop an image or click to choose",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )
    st.markdown("""
<div style="display:inline-flex;align-items:center;gap:8px;background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:999px;padding:8px 12px;font-size:13px;color:rgba(8,8,8,0.58);margin:10px 0 18px 0;">
<span style="width:7px;height:7px;border-radius:50%;background:#aef33e;display:inline-block;"></span>
Need a quick test? Use sample creatives from the 
<a href="https://drive.google.com/drive/folders/1xWNcyCTl94PQMWHhaUhGzE9T7ZrUeYqY" target="_blank" style="color:#0009dc;text-decoration:none;border-bottom:1px solid rgba(0,9,220,0.28);font-weight:650;">Test creatives</a>
folder.
</div>
""", unsafe_allow_html=True)

    if uploaded_file is not None:
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()
        image_hash = hashlib.md5(image_bytes).hexdigest()
        mime = f"image/{uploaded_file.type.split('/')[-1]}"

        existing = next(
            (e for e in st.session_state.history if e["hash"] == image_hash),
            None
        )

        if existing:
            entry = existing
        else:
            with st.spinner("Analyzing the image..."):
                tags = tag_image_cached(image_hash, image_bytes, mime)

            entry = {
                "hash": image_hash,
                "name": uploaded_file.name,
                "image_bytes": image_bytes,
                "tags": tags,
            }

            st.session_state.history.append(entry)
            st.rerun()

        if not is_food_drink_creative(entry):
            render_non_fnb_alert(entry)
        else:
            render_analysis(entry, selected_brand)


st.markdown("""
<div style="height:1px;background:#e9e9e9;margin:60px 0 24px 0;"></div>
<div style="text-align:center;font-size:12px;color:rgba(8,8,8,0.42);margin:0 0 20px 0;">
Creative Analyzer · Hackathon MVP · Built in Streamlit · by Viktoriia Iachmeneva · 2026
</div>
""", unsafe_allow_html=True)