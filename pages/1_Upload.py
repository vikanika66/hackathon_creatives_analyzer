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
    page_icon="🚀",
    layout="wide",
)

# ============================================================
# Styles — clean typography
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

    .metric-mini {
        background: #faf9f5;
        border-radius: 10px;
        padding: 14px 18px;
    }
    .metric-mini-label {
        font-size: 11px;
        color: #888;
        margin-bottom: 4px;
    }
    .metric-mini-value {
        font-size: 22px;
        font-weight: 500;
    }

    .tag-chip {
        display: inline-block;
        padding: 5px 12px;
        margin: 3px;
        border-radius: 14px;
        font-size: 12px;
        font-weight: 500;
    }
    .tag-binary-active {
    background: #E6F1FB;
    color: #0C447C;
    }
    .tag-categorical-active {
        background: #E1F5EE;
        color: #085041;
    }
    .tag-absent {
        background: #f0efe9;
        color: #b5b3a8;
        text-decoration: line-through;
    }

    .effect-row {
        display: flex;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #f5f3ed;
    }
    .effect-row:last-child { border-bottom: none; }
    .effect-arrow {
        font-size: 18px;
        width: 30px;
        text-align: center;
    }
    .effect-arrow.up { color: #1D9E75; }
    .effect-arrow.down { color: #E24B4A; }
    .effect-value {
        font-weight: 600;
        font-size: 14px;
        width: 70px;
    }
    .effect-value.up { color: #1D9E75; }
    .effect-value.down { color: #E24B4A; }
    .effect-name {
        flex: 1;
        font-size: 14px;
        color: #1a1a1a;
        margin-left: 12px;
    }

    .rec-row {
        padding: 12px 0;
        border-bottom: 1px solid #f5f3ed;
        font-size: 14px;
    }
    .rec-row:last-child { border-bottom: none; }
    .rec-add { color: #1D9E75; font-weight: 600; }
    .rec-remove { color: #E24B4A; font-weight: 600; }

    .similar-open-btn {
        display: inline-block;
        margin-top: 8px;
        padding: 6px 14px;
        background: #0009dc;
        color: white !important;
        text-decoration: none !important;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 500;
    }
    .similar-open-btn:hover { background: #0007a8; }
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
    """HTML arrows: double/single colored, or gray dash for neutral."""
    if direction == "neutral":
        return '<span style="color:#b5b3a8;font-size:18px;">—</span>'
    if direction == "up":
        arrows = "↑↑" if strength == "strong" else "↑"
        color = "#1D9E75"
    else:
        arrows = "↓↓" if strength == "strong" else "↓"
        color = "#E24B4A"
    return f'<span style="color:{color};font-weight:600;font-size:20px;letter-spacing:-3px;">{arrows}</span>'

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
    """Find similar creatives, optionally filtered by category."""
    feature_importance = {
        c: np.abs(shap_df[c]).mean() for c in feature_cols if c in shap_df.columns
    }

    # Filter database by category
    candidates = df.copy()
    if food_type and food_type != "none":
        candidates = candidates[candidates["food_type"] == food_type]
    elif drink_type and drink_type != "none":
        candidates = candidates[candidates["drink_type"] == drink_type]

    # If too few creatives in category — expand to whole database
    if len(candidates) < 10:
        candidates = df.copy()

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
    return similarities[:top_n]


def get_segment_context(tags):
    """Returns insights about creative's category and base CTR."""
    base_ctr = df["ctr"].mean()
    insights = []

    food = tags.get("food_type", "none")
    if food != "none":
        segment_df = df[df["food_type"] == food]
        if len(segment_df) >= 5:
            seg_ctr = segment_df["ctr"].mean()
            diff = seg_ctr - base_ctr
            insights.append({
                "label": food.replace("_", " "),
                "category": "food type",
                "ctr": seg_ctr,
                "diff": diff,
                "count": len(segment_df),
            })

    drink = tags.get("drink_type", "none")
    if drink != "none":
        segment_df = df[df["drink_type"] == drink]
        if len(segment_df) >= 5:
            seg_ctr = segment_df["ctr"].mean()
            diff = seg_ctr - base_ctr
            insights.append({
                "label": drink.replace("_", " "),
                "category": "drink type",
                "ctr": seg_ctr,
                "diff": diff,
                "count": len(segment_df),
            })

    return insights, base_ctr


def calculate_active_combinations(active_tags, food_type=None, drink_type=None, top_n_each=3):
    """Find strongest combinations considering the segment."""
    
    if food_type and food_type != "none":
        segment_filter = (interactions_df["segment_type"] == "food") & \
                        (interactions_df["segment"] == food_type)
    elif drink_type and drink_type != "none":
        segment_filter = (interactions_df["segment_type"] == "drink") & \
                        (interactions_df["segment"] == drink_type)
    else:
        segment_filter = (interactions_df["segment_type"] == "all")
    
    segment_df = interactions_df[segment_filter]
    
    # If few pairs in segment — fallback to all
    if len(segment_df) < 5:
        segment_df = interactions_df[interactions_df["segment_type"] == "all"]
    
    active_set = set(active_tags)
    active_pairs = segment_df[
        segment_df["feat1"].isin(active_set) &
        segment_df["feat2"].isin(active_set)
    ].copy()

    if len(active_pairs) == 0:
        return [], []

    positive = active_pairs[active_pairs["interaction"] > 0.01]
    positive = positive.nlargest(top_n_each, "interaction")

    negative = active_pairs[active_pairs["interaction"] < -0.01]
    negative = negative.nsmallest(top_n_each, "interaction")

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


# ============================================================
# Block renders
# ============================================================

def render_block_creative_card(entry, selected_brand):
    tags = entry["tags"]
    active_tags = get_active_tags(tags)

    similar = find_similar_creatives(
        active_tags,
        food_type=tags.get("food_type"),
        drink_type=tags.get("drink_type"),
        top_n=30,
    )
    similar_indices = [idx for idx, _ in similar]
    similar_ctrs = df.loc[similar_indices, "ctr"]
    ctr_min = similar_ctrs.min()
    ctr_max = similar_ctrs.max()
    ctr_median = similar_ctrs.median()

    st.markdown('<div class="section-title">Creative</div>', unsafe_allow_html=True)

    col_img, col_info = st.columns([0.9, 1.4], gap="large")

    with col_img:
        image = Image.open(io.BytesIO(entry["image_bytes"]))
        max_height = 460
        if image.height > max_height:
            ratio = max_height / image.height
            image = image.resize((int(image.width * ratio), max_height), Image.LANCZOS)
        st.image(image)

    with col_info:
        st.markdown(f"""
        <div style="display:flex;gap:10px;margin-bottom:24px;">
            <div class="metric-mini" style="flex:1;">
                <div class="metric-mini-label">Similar CTR (min)</div>
                <div class="metric-mini-value">{ctr_min:.2f}%</div>
            </div>
            <div class="metric-mini" style="flex:1;">
                <div class="metric-mini-label">median</div>
                <div class="metric-mini-value">{ctr_median:.2f}%</div>
            </div>
            <div class="metric-mini" style="flex:1;">
                <div class="metric-mini-label">Similar CTR (max)</div>
                <div class="metric-mini-value">{ctr_max:.2f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Dynamic caption with brand and category
        food = tags.get("food_type", "none")
        drink = tags.get("drink_type", "none")
        category_label = food.replace("_", " ") if food != "none" else (drink.replace("_", " ") if drink != "none" else "")
        caption_text = f"CTR range across similar creatives of <b>{selected_brand}</b>"
        if category_label:
            caption_text += f" in the <b>{category_label}</b> category"

        st.markdown(f'<div style="font-size:12px;color:#888;margin-top:-12px;margin-bottom:18px;">{caption_text}</div>', unsafe_allow_html=True)

        # Categories — green row
        cat_chips = []
        for cat in CATEGORICAL_TAGS:
            value = tags.get(cat, "none")
            if value != "none":
                cat_chips.append(f'<span class="tag-chip tag-categorical-active">{cat.replace("_"," ")}: {value}</span>')

        if cat_chips:
            st.markdown("<div style='font-size:14px;font-weight:500;margin-bottom:8px;'>Categories</div>", unsafe_allow_html=True)
            st.markdown("".join(cat_chips), unsafe_allow_html=True)
            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        # Tags — blue row
        tag_chips = []
        for binary in BINARY_FEATURES:
            if tags.get(binary, False):
                tag_chips.append(f'<span class="tag-chip tag-binary-active">{display_name(binary)}</span>')
        for binary in BINARY_FEATURES:
            if not tags.get(binary, False):
                tag_chips.append(f'<span class="tag-chip tag-absent">{display_name(binary)}</span>')

        st.markdown("<div style='font-size:14px;font-weight:500;margin-bottom:8px;'>Tags</div>", unsafe_allow_html=True)
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
    
    st.markdown('<div class="section-title">Tag correlation with CTR</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Which tags appear more often in creatives with high or low CTR. '
        '<b>↑↑ / ↓↓</b> — strong correlation, <b>↑ / ↓</b> — moderate. '
        'This is a pattern in the data, not a guaranteed effect.</div>',
        unsafe_allow_html=True,
    )
    
    if not categorized:
        st.markdown(
            "<div style='color:#888;font-size:14px;'>No significant correlation with CTR for the detected tags</div>",
            unsafe_allow_html=True
        )
        return
    
    rows_html = ""
    for tag, direction, strength in categorized:
        rows_html += f"""
        <div class="effect-row">
            <span style="width:70px;text-align:center;">{arrow_symbol(direction, strength)}</span>
            <span class="effect-name">{display_name(tag)}</span>
        </div>
        """
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
    
    st.markdown('<div class="section-title">Tag pair correlation with CTR</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Which tag combinations within this brand correlate with CTR more or less '
        'than the tags taken separately</div>',
        unsafe_allow_html=True,
    )
    
    if not all_combos:
        st.markdown(
            "<div style='color:#888;font-size:14px;'>No significant pair correlations found among the active tags</div>",
            unsafe_allow_html=True
        )
        return
    
    rows_html = ""
    for combo, direction, strength in all_combos:
        rows_html += f"""
        <div class="effect-row">
            <span style="width:70px;text-align:center;">{arrow_symbol(direction, strength)}</span>
            <span class="effect-name"><b>{display_name(combo['feat1'])}</b> × <b>{display_name(combo['feat2'])}</b></span>
        </div>
        """
    st.markdown(rows_html, unsafe_allow_html=True)


def render_block_recommendations(tags):
    X_new = tags_to_features(tags, feature_cols)
    active_tags = get_active_tags(tags)
    recs_add, recs_remove = get_recommendations_full(active_tags, X_new)

    st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">What you could change to potentially improve CTR. '
        'You can check the exact effect for this creative in A/B scenarios below.</div>',
        unsafe_allow_html=True,
    )

    if not recs_add and not recs_remove:
        st.markdown(
            "<div style='color:#1D9E75;font-size:14px;'>The tag set looks strong! 💪</div>",
            unsafe_allow_html=True,
        )
    else:
        rows_html = ""
        for feat, _ in recs_add:
            rows_html += f"""
            <div class="rec-row">
                <span class="rec-add">➕ Add</span>
                &nbsp;&nbsp;<b>{display_name(feat)}</b> — creatives with this element tend to perform better on average
            </div>
            """
        for feat, _ in recs_remove:
            rows_html += f"""
            <div class="rec-row">
                <span class="rec-remove">➖ Remove</span>
                &nbsp;&nbsp;<b>{display_name(feat)}</b> — creatives without it tend to perform better on average
            </div>
            """
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
    
    st.markdown('<div class="section-title">A/B scenarios + AI reference</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Toggle tags — see how the correlation with CTR changes, '
        'and generate a visual reference based on your scenario</div>',
        unsafe_allow_html=True,
    )

    # Toggle state
    state_key = f"ab_state_{entry_hash}"
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            feat: bool(tags.get(feat, False)) for feat in BINARY_FEATURES
        }

    # Reset button
    col_reset, _ = st.columns([1, 4])
    with col_reset:
        if st.button("↻ Reset", key=f"reset_{entry_hash}"):
            st.session_state[state_key] = {
                feat: bool(tags.get(feat, False)) for feat in BINARY_FEATURES
            }
            for key in list(st.session_state.keys()):
                if key.startswith(f"ab_generated_{entry_hash}"):
                    del st.session_state[key]
            st.rerun()

    # Toggles
    cols = st.columns(3)
    for i, feat in enumerate(BINARY_FEATURES):
        with cols[i % 3]:
            current = st.session_state[state_key][feat]
            new_value = st.toggle(
                display_name(feat),
                value=current,
                key=f"toggle_{entry_hash}_{feat}",
            )
            st.session_state[state_key][feat] = new_value

    # Compute delta
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
        new_val = st.session_state[state_key][feat]
        if original_val != new_val:
            action = "turn on" if new_val else "turn off"
            changes.append((feat, action, new_val))

    # Direction categorization
    AB_STRONG = 0.30
    AB_WEAK = 0.10
    direction, strength = categorize_strength(diff, AB_STRONG, AB_WEAK)

    # Verdict text and arrow
    if not changes:
        verdict_html = '<span style="color:#888;font-size:15px;">Nothing changed</span>'
        arrow_html = '<span style="color:#b5b3a8;font-size:36px;">—</span>'
    else:
        changes_text = ", ".join(
            f"<b>{action} {display_name(feat)}</b>" for feat, action, _ in changes
        )
        
        if direction == "neutral":
            label = "Correlation barely changes"
            label_color = "#888"
        elif direction == "up":
            label = "Correlation with CTR strengthens" if strength == "strong" else "Correlation with CTR slightly strengthens"
            label_color = "#1D9E75"
        else:
            label = "Correlation with CTR weakens" if strength == "strong" else "Correlation with CTR slightly weakens"
            label_color = "#E24B4A"
        
        if direction == "neutral":
            arrow_html = '<span style="color:#b5b3a8;font-size:36px;">—</span>'
        elif direction == "up":
            arrows = "↑↑" if strength == "strong" else "↑"
            arrow_html = f'<span style="color:#1D9E75;font-size:36px;font-weight:600;letter-spacing:-6px;">{arrows}</span>'
        else:
            arrows = "↓↓" if strength == "strong" else "↓"
            arrow_html = f'<span style="color:#E24B4A;font-size:36px;font-weight:600;letter-spacing:-6px;">{arrows}</span>'
        
        verdict_html = (
            f'<div style="font-size:15px;color:{label_color};font-weight:500;margin-bottom:4px;">{label}</div>'
            f'<div style="font-size:13px;color:#666;">{changes_text}</div>'
        )

    # Result plate
    st.markdown(f"""
    <div style="margin-top:24px;padding:20px;background:#faf9f5;border-radius:12px;">
        <div style="display:flex;align-items:center;gap:24px;">
            <div style="min-width:80px;text-align:center;">{arrow_html}</div>
            <div style="flex:1;">{verdict_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # === Reference generation ===
    
    if not changes:
        st.markdown(
            "<div style='font-size:13px;color:#888;margin-top:16px;'>"
            "Toggle some tags above — and you'll be able to generate a visual reference."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    scenario_signature = "|".join(
        f"{feat}={int(val)}" for feat, _, val in sorted(changes, key=lambda x: x[0])
    )
    scenario_hash = hashlib.md5(scenario_signature.encode()).hexdigest()[:8]
    cache_key = f"ab_generated_{entry_hash}_{scenario_hash}"

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

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
            st.markdown("<div style='font-size:13px;color:#888;margin-bottom:8px;'>Original</div>", unsafe_allow_html=True)
            st.image(source_image_bytes, use_container_width=True)
        with col_new:
            st.markdown("<div style='font-size:13px;color:#888;margin-bottom:8px;'>AI reference for the scenario</div>", unsafe_allow_html=True)
            st.image(result["image"], use_container_width=True)
        
        st.markdown(
            "<div style='font-size:12px;color:#888;margin-top:12px;line-height:1.6;'>"
            "This reference is a starting point for the designer. "
            "Logo, typography, and final execution are up to the designer."
            "</div>",
            unsafe_allow_html=True,
        )
        
        with st.expander("Full prompt"):
            st.code(result["prompt"], language=None)
        
        if st.button("🔄 Regenerate", key=f"regen_{entry_hash}_{scenario_hash}"):
            del st.session_state[cache_key]
            st.rerun()

def render_block_similar(similar):
    st.markdown('<div class="section-title">Similar creatives from the database</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Creatives with a similar tag set, sorted by actual CTR</div>', unsafe_allow_html=True)

    top_similar = similar[:10]
    creatives = []
    for idx, sim_score in top_similar:
        row = df.loc[idx]
        creatives.append({"filename": row["filename"], "ctr": row["ctr"], "row": row})
    creatives.sort(key=lambda x: x["ctr"], reverse=True)
    median_ctr = df["ctr"].median()

    for row_start in [0, 5]:
        if row_start == 5:
            st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
        cols = st.columns(5, gap="small")
        for i, c in enumerate(creatives[row_start:row_start+5]):
            with cols[i]:
                image_path = os.path.join(IMAGES_DIR, c["filename"])
                if os.path.exists(image_path):
                    st.markdown(f"""
                    <div style="height:200px;background:#f5f3ed;border-radius:8px;overflow:hidden;display:flex;align-items:center;justify-content:center;">
                        <img src="data:image/png;base64,{__import__('base64').b64encode(open(image_path, 'rb').read()).decode()}"
                             style="width:100%;height:100%;object-fit:cover;"/>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown('<div style="height:200px;background:#f0efe9;border-radius:8px;"></div>', unsafe_allow_html=True)

                ctr_color = "#1D9E75" if c["ctr"] >= median_ctr else "#E24B4A"
                active = []
                for binary in BINARY_FEATURES:
                    if binary in c["row"].index and c["row"][binary] == True:
                        active.append(binary)
                feat_imp = {b: np.abs(shap_df[b]).mean() for b in active if b in shap_df.columns}
                top_tags = sorted(active, key=lambda x: feat_imp.get(x, 0), reverse=True)[:3]
                tag_str = ", ".join(display_name(t) for t in top_tags)

                st.markdown(f"""
                <div style="font-size:20px;font-weight:600;color:{ctr_color};margin-top:6px;">{c['ctr']:.2f}%</div>
                <div style="font-size:11px;color:#888;min-height:32px;">{tag_str if tag_str else '—'}</div>
                <a href="/Library?creative={c['filename']}" target="_blank" class="similar-open-btn">
                    Open →
                </a>
                """, unsafe_allow_html=True)


def render_block_segment_context(tags):
    """Insight block about the segment — how strong this category is for the brand."""
    insights, base_ctr = get_segment_context(tags)

    if not insights:
        return

    st.markdown('<div class="section-title">Creative category</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">How strong creatives of the same category are for this brand</div>',
        unsafe_allow_html=True,
    )

    rows_html = ""
    for ins in insights:
        diff = ins["diff"]
        is_better = diff > 0
        color = "#1D9E75" if is_better else "#E24B4A"
        sign = "+" if is_better else ""

        if abs(diff) < 0.1:
            verdict = "Category performs roughly the same as others for this brand"
            verdict_color = "#888"
        elif is_better:
            verdict = f"✓ Category outperforms others for this brand by {sign}{diff:.2f}%"
            verdict_color = "#1D9E75"
        else:
            verdict = f"✗ Category underperforms others for this brand by {diff:.2f}%"
            verdict_color = "#E24B4A"

        rows_html += f"""
        <div style="padding:18px 20px;margin-bottom:12px;background:#faf9f5;border-radius:12px;">
            <div style="font-size:18px;font-weight:600;margin-bottom:14px;color:#1a1a1a;text-transform:capitalize;">
                {ins['label']}
            </div>
            <div style="display:flex;gap:24px;margin-bottom:12px;">
                <div>
                    <div style="font-size:11px;color:#888;margin-bottom:2px;">Category CTR</div>
                    <div style="font-size:20px;font-weight:600;color:#1a1a1a;">{ins['ctr']:.2f}%</div>
                </div>
                <div>
                    <div style="font-size:11px;color:#888;margin-bottom:2px;">All brand creatives CTR</div>
                    <div style="font-size:20px;font-weight:600;color:#1a1a1a;">{base_ctr:.2f}%</div>
                </div>
                <div>
                    <div style="font-size:11px;color:#888;margin-bottom:2px;">Difference</div>
                    <div style="font-size:20px;font-weight:600;color:{color};">{sign}{diff:.2f}%</div>
                </div>
            </div>
            <div style="font-size:14px;color:{verdict_color};font-weight:500;">{verdict}</div>
            <div style="font-size:12px;color:#888;margin-top:6px;">Based on {ins['count']} creatives</div>
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
        
        Only categories with at least 5 creatives in the brand portfolio are shown.
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
        We compute the similarity of each brand creative to the uploaded one based on the
        overlap of their tag sets, weighted by tag importance (average absolute SHAP).
        
        If the uploaded creative has a food/drink category, we first look at creatives
        within the same category. If the category has fewer than 10 creatives, we expand
        to the whole brand portfolio.
        
        The top 10 most similar creatives are shown, sorted by their actual CTR (highest first).
        Click "Open →" on any card to see its full breakdown in the Library.
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

st.title("🚀 Creative analysis")
st.caption("Upload an ad creative — see how visual elements correlate with CTR")

# Info banner about MVP scope
st.markdown("""
<div style="background:#fff8e6;border-left:4px solid #e8a24a;border-radius:8px;
            padding:14px 18px;margin:16px 0 24px 0;font-size:13px;color:#5a4a1f;line-height:1.6;">
    <b>ℹ️ About this MVP:</b> This tool is focused on the <b>Food & Drinks</b> advertising category.
    For best results, upload creatives featuring food, drinks, or related visuals — non-F&B uploads
    will produce noisy analytics since the model is trained on F&B data only.
    All CTR data is <b>synthetic</b>, generated based on the structure of real creatives from
    the Meta Ad Library. Brand names (BurgerBee, CrunchBox, etc.) are test placeholders —
    in a production version this dropdown would be optional and connected to your own data.
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

# Override globals for analysis functions
df = df_filtered
shap_df = shap_df_filtered

st.caption(f"Analysis based on {len(df_filtered)} creatives of **{selected_brand}**")

# Page logic
if st.session_state.selected_history is not None:
    entry = st.session_state.history[st.session_state.selected_history]
    if st.button("← Back to upload"):
        st.session_state.selected_history = None
        st.rerun()
    render_analysis(entry, selected_brand)
else:
    uploaded_file = st.file_uploader(
        "Drop an image or click to choose",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )

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

        # Check if uploaded creative looks like F&B
        is_food_drink = (
            entry["tags"].get("food_type", "none") != "none"
            or entry["tags"].get("drink_type", "none") != "none"
            or entry["tags"].get("main_object", "none") in ("main_dish", "snack", "dessert", "drink", "fruit", "healthy")
        )
        
        if not is_food_drink:
            # Show uploaded image so user sees what they uploaded
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
                <div style="background:#fdecec;border-left:4px solid #E24B4A;border-radius:8px;
                            padding:20px 24px;margin-top:8px;font-size:14px;color:#7a2424;line-height:1.7;">
                    <div style="font-size:18px;font-weight:600;margin-bottom:10px;">
                        ⚠️ This doesn't look like a Food & Drinks creative
                    </div>
                    The AI didn't detect any food or drink elements in this image. 
                    Analysis won't be shown — the model is trained only on F&B creatives,
                    so the results wouldn't be meaningful for this category.
                    <br><br>
                    <b>Try uploading a creative featuring:</b>
                    <ul style="margin:8px 0 0 0;padding-left:20px;">
                        <li>Food (burgers, pizza, snacks, desserts, healthy meals)</li>
                        <li>Drinks (soda, juice, alcohol)</li>
                        <li>Restaurant or delivery visuals</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            render_analysis(entry, selected_brand)

st.markdown("""
<div style="text-align:center;font-size:13px;color:#bbb;margin:60px 0 20px 0;padding-top:24px;border-top:1px solid #eee;">
    Creative Analyzer · Hackathon MVP · Built in Streamlit · by Viktoriia Iachmeneva · 2026
</div>
""", unsafe_allow_html=True)