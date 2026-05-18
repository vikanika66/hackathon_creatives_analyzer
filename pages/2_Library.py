"""
Library page — gallery of creatives from the database with filters and tag breakdown.
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
from PIL import Image
import base64
from pathlib import Path
from openai import OpenAI
import io

from utils import (
    COMMON_CSS,
    load_data,
    load_model,
    get_explainer,
    tags_to_features,
    make_waterfall,
    render_tag_chips,
    BINARY_FEATURES,
)

from utils import CATEGORICAL_TAGS


st.set_page_config(
    page_title="Library — Creative Analyzer",
    page_icon="📚",
    layout="wide",
)

st.markdown(COMMON_CSS, unsafe_allow_html=True)

LIBRARY_CSS = """
<style>
    .stApp {
        background: #ffffff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1200px;
    }

    .ca-kicker {
        font-size: 12px;
        color: #0009dc;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .ca-title {
        font-size: 40px;
        font-weight: 650;
        letter-spacing: -1.2px;
        color: #080808;
        margin: 0 0 12px 0;
    }

    .ca-subtitle {
        font-size: 17px;
        color: rgba(8,8,8,0.66);
        line-height: 1.55;
        max-width: 720px;
        margin: 0;
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

    .ca-soft-blue {
        background: linear-gradient(180deg, rgba(0,9,220,0.06) 0%, rgba(65,105,225,0.10) 100%);
        border: 1px solid rgba(0,9,220,0.12);
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
        letter-spacing: 0.1px !important;
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

    div[data-testid="stExpander"] ul {
        margin-top: 8px !important;
        padding-left: 22px !important;
    }

    div[data-testid="stExpander"] li {
        margin-bottom: 6px !important;
    }
</style>
"""

st.markdown(LIBRARY_CSS, unsafe_allow_html=True)

# ============================================================
# Data and model
# ============================================================
df, shap_df, feature_cols, interactions_df = load_data()
model = load_model()
explainer = get_explainer(model)

IMAGES_DIR = "images"

# ============================================================
# OpenAI client + image generation (for A/B reference)
# ============================================================

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
        
        aspect = img.width / img.height
        if aspect > 1.2:
            size = "1536x1024"
        elif aspect < 0.85:
            size = "1024x1536"
        else:
            size = "1024x1024"
        
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
        return base64.b64decode(image_b64), None
    except Exception as e:
        return None, str(e)


def build_scenario_prompt(changes):
    """Prompt for A/B scenario generation."""
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


def read_image_bytes(filename):
    """Read image bytes from disk by filename."""
    image_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as f:
        return f.read()


def spacer(h=16):
    st.markdown(f'<div style="height:{h}px;"></div>', unsafe_allow_html=True)


# ============================================================
# Helpers for creative breakdown rendering
# ============================================================

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
# Calculation functions for creative breakdown
# ============================================================


def get_active_tags(tags):
    """Returns list of active binary tags + one-hot columns for categories."""
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
    """Average SHAP per active binary tag."""
    effects = []
    for tag in active_tags:
        if tag not in BINARY_FEATURES:
            continue
        if tag in shap_df.columns:
            mask = df[tag] == True
            if mask.sum() >= 5:
                avg_shap = shap_df.loc[mask.values, tag].mean()
                effects.append((tag, avg_shap))
    effects.sort(key=lambda x: abs(x[1]), reverse=True)
    return effects


def get_segment_context(tags):
    """Compare CTR of creative's category with the entire database.
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
                "diff": round(seg_ctr, 2) - round(base_ctr, 2),
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
                "diff": round(seg_ctr, 2) - round(base_ctr, 2),
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
    """Top positive and negative pairs among active tags."""
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


def find_similar_creatives(active_tags, current_filename, food_type=None, drink_type=None, top_n=10):
    """Category-priority cascade for similar creatives:
      1. Same brand + same category — ideal
      2. Other brands in same category — if brand has < 5 examples
      3. Same brand only — if category has < 5 anywhere
      4. Whole database — last fallback
    
    Returns (similarities, scope):
      similarities: list of (idx, score)
      scope: "brand-category" | "category-expanded" | "brand-only" | "database"
    """
    feature_importance = {
        c: np.abs(shap_df[c]).mean() for c in feature_cols if c in shap_df.columns
    }

    # Get current creative's brand
    current_row = df[df["filename"] == current_filename]
    if len(current_row) == 0:
        current_brand = None
    else:
        current_brand = current_row.iloc[0].get("brand", None)

    has_category = (food_type and food_type != "none") or (drink_type and drink_type != "none")

    # Helper: filter by category
    def filter_by_category(d):
        if food_type and food_type != "none":
            return d[d["food_type"] == food_type]
        elif drink_type and drink_type != "none":
            return d[d["drink_type"] == drink_type]
        return d

    # Step 1: same brand + same category
    if current_brand and has_category:
        candidates = filter_by_category(df[df["brand"] == current_brand])
        scope = "brand-category"
    elif current_brand:
        candidates = df[df["brand"] == current_brand]
        scope = "brand-only"
    else:
        candidates = filter_by_category(df) if has_category else df.copy()
        scope = "category-expanded" if has_category else "database"

    # Step 2: expand to other brands in the same category
    if len(candidates) < 5 and has_category and current_brand:
        candidates = filter_by_category(df)
        scope = "category-expanded"

    # Step 3: fallback to same brand only (any category)
    if len(candidates) < 5 and current_brand:
        candidates = df[df["brand"] == current_brand]
        scope = "brand-only"

    # Step 4: last resort — whole database
    if len(candidates) < 5:
        candidates = df.copy()
        scope = "database"

    # Exclude the current creative itself
    candidates = candidates[candidates["filename"] != current_filename]

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

# ============================================================
# Render functions for creative breakdown blocks
# ============================================================

def divider():
    st.markdown('<div style="height:1px;background:#e8e6df;margin:36px 0 28px 0;"></div>', unsafe_allow_html=True)


def render_block_tag_effects(active_tags):
    """Tag correlation with CTR - arrows instead of numbers."""
    effects = calculate_tag_effects(active_tags)
    
    TAG_STRONG = 0.1
    TAG_WEAK = 0.05
    
    categorized = []
    for tag, val in effects:
        direction, strength = categorize_strength(val, TAG_STRONG, TAG_WEAK)
        categorized.append((tag, direction, strength))
    
    st.markdown("""
<div style="margin:8px 0 28px 0;">
<div style="font-size:12px;color:#0009dc;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:8px;">
Pattern strength
</div>
<div style="font-size:28px;font-weight:650;color:#080808;margin-bottom:8px;letter-spacing:-0.6px;">
Tag correlation with CTR
</div>
<div style="font-size:15px;color:rgba(8,8,8,0.62);max-width:860px;line-height:1.55;">
Which detected tags appear more often in creatives with high or low CTR. <b>↑↑ / ↓↓</b> means strong correlation, <b>↑ / ↓</b> means moderate. This is a pattern in the data, not a guaranteed effect.
</div>
</div>
""", unsafe_allow_html=True)
    
    if not categorized:
        st.markdown("""
<div style="background:#ffffff;border:1px solid #eeeeee;border-radius:22px;padding:22px 24px;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#080808;"></div>
<div style="font-size:14px;color:rgba(8,8,8,0.58);">
No significant effects found for this creative.
</div>
</div>
""", unsafe_allow_html=True)
        return
    
    rows_html = ""

    for tag, direction, strength in categorized:
        if direction == "up":
            row_color = "#1D9E75"
            row_bg = "rgba(29,158,117,0.06)"
            label = "Positive signal"
        elif direction == "down":
            row_color = "#E24B4A"
            row_bg = "rgba(226,75,74,0.06)"
            label = "Negative signal"
        else:
            row_color = "rgba(8,8,8,0.38)"
            row_bg = "rgba(8,8,8,0.04)"
            label = "Neutral signal"

        rows_html += f"""
<div style="padding:16px 0;border-bottom:1px solid #eeeeee;display:grid;grid-template-columns:80px 1fr 140px;gap:14px;align-items:center;">
<div style="text-align:center;background:{row_bg};border-radius:14px;padding:8px 0;">
{arrow_symbol(direction, strength)}
</div>

<div style="font-size:14px;color:rgba(8,8,8,0.74);font-weight:500;">
<b style="color:#080808;">{display_name(tag)}</b>
</div>

<div style="text-align:right;">
<span style="display:inline-flex;align-items:center;gap:6px;background:{row_bg};color:{row_color};border-radius:999px;padding:6px 10px;font-size:11px;font-weight:650;">
<span style="width:6px;height:6px;border-radius:50%;background:{row_color};display:inline-block;"></span>
{label}
</span>
</div>
</div>
"""
    
    st.markdown(f"""
<div style="background:linear-gradient(180deg,#ffffff 0%,#fbfbfb 100%);border:1px solid #eeeeee;border-radius:22px;padding:22px 26px;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#0009dc;"></div>
{rows_html}
</div>
""", unsafe_allow_html=True)


def render_block_segment_context(tags):
    """Creative category — comparison with the entire database.
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
<div style="font-size:15px;color:rgba(8,8,8,0.62);max-width:860px;line-height:1.55;">
How this food or drink category performs compared with the full creative database.
</div>
</div>
""", unsafe_allow_html=True)

    rows_html = ""
    NEUTRAL_THRESHOLD = 0.10

    for ins in insights:

        # ============================================================
        # Insufficient data — show disclaimer card
        # ============================================================
        if ins["status"] == "insufficient":
            count = ins["count"]
            if count == 0:
                verdict = (
                    f"The database contains no creatives in the "
                    f"<b style='color:#080808;'>{ins['label']}</b> category, "
                    f"so a comparison can't be calculated."
                )
            else:
                creative_word = "creative" if count == 1 else "creatives"
                verdict = (
                    f"The database contains only <b style='color:#080808;'>{count}</b> "
                    f"{creative_word} in the <b style='color:#080808;'>{ins['label']}</b> "
                    f"category. A minimum of <b style='color:#080808;'>5</b> is needed for a "
                    f"reliable comparison, so the average is not shown."
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
                    "This category is close to the database average, with a small positive difference. "
                    "Treat it as a neutral-to-positive signal, not a strong outperformance."
                )
            else:
                accent_label = "Above database average"
                verdict = "Creatives in this category tend to perform better than the database average."

        elif diff < 0:
            accent_color = "#E24B4A"
            accent_bg = "rgba(226,75,74,0.10)"
            diff_text = f"{diff:.2f}%"

            if abs_diff < NEUTRAL_THRESHOLD:
                accent_label = "Slightly below average"
                verdict = (
                    "This category is close to the database average, with a small negative difference. "
                    "Treat it as a neutral-to-negative signal, not a strong underperformance."
                )
            else:
                accent_label = "Below database average"
                verdict = "Creatives in this category tend to underperform compared with the database average."

        else:
            accent_color = "rgba(8,8,8,0.42)"
            accent_bg = "rgba(8,8,8,0.04)"
            diff_text = "0.00%"
            accent_label = "Neutral"
            verdict = "This category performs at the same level as the database average."

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
<div style="font-size:11px;color:rgba(8,8,8,0.44);margin-bottom:3px;">Database avg CTR</div>
<div style="font-size:18px;font-weight:650;color:#080808;">{base_ctr:.2f}%</div>
</div>
</div>
</div>

</div>
</div>
"""

    st.markdown(rows_html, unsafe_allow_html=True)


def render_block_combinations(active_tags, tags):
    """Tag pair correlation with CTR — arrows instead of numbers."""
    positive, negative = calculate_active_combinations(
        active_tags,
        food_type=tags.get("food_type"),
        drink_type=tags.get("drink_type"),
    )
    
    COMBO_STRONG = 0.08
    COMBO_WEAK = 0.00001
    
    all_combos = []

    for combo in positive:
        direction, strength = categorize_strength(combo["interaction"], COMBO_STRONG, COMBO_WEAK)
        all_combos.append((combo, direction, strength))

    for combo in negative:
        direction, strength = categorize_strength(combo["interaction"], COMBO_STRONG, COMBO_WEAK)
        all_combos.append((combo, direction, strength))
    
    st.markdown("""
<div style="margin:8px 0 28px 0;">
<div style="font-size:12px;color:#0009dc;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:8px;">
Combinations
</div>
<div style="font-size:28px;font-weight:650;color:#080808;margin-bottom:8px;letter-spacing:-0.6px;">
Tag pair correlation with CTR
</div>
<div style="font-size:15px;color:rgba(8,8,8,0.62);max-width:860px;line-height:1.55;">
Which active tag combinations correlate with CTR more or less than the tags taken separately. Green pairs indicate positive interaction, red pairs indicate negative interaction.
</div>
</div>
""", unsafe_allow_html=True)
    
    if not all_combos:
        st.markdown("""
<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:22px;padding:22px 24px;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#080808;"></div>
<div style="font-size:14px;color:rgba(8,8,8,0.58);">
No meaningful tag pair correlations found for this creative.
</div>
</div>
""", unsafe_allow_html=True)
        return
    
    rows_html = ""

    for combo, direction, strength in all_combos:
        if direction == "up":
            row_color = "#1D9E75"
            row_bg = "rgba(29,158,117,0.06)"
            label = "Positive pair"
        elif direction == "down":
            row_color = "#E24B4A"
            row_bg = "rgba(226,75,74,0.06)"
            label = "Negative pair"
        else:
            row_color = "rgba(8,8,8,0.38)"
            row_bg = "rgba(8,8,8,0.04)"
            label = "Neutral pair"

        rows_html += f"""
<div style="padding:16px 0;border-bottom:1px solid #eeeeee;display:grid;grid-template-columns:80px 1fr 140px;gap:14px;align-items:center;">
<div style="text-align:center;background:{row_bg};border-radius:14px;padding:8px 0;">
{arrow_symbol(direction, strength)}
</div>

<div style="font-size:14px;color:rgba(8,8,8,0.74);font-weight:500;">
<b style="color:#080808;">{display_name(combo['feat1'])}</b>
<span style="color:rgba(8,8,8,0.38);"> × </span>
<b style="color:#080808;">{display_name(combo['feat2'])}</b>
</div>

<div style="text-align:right;">
<span style="display:inline-flex;align-items:center;gap:6px;background:{row_bg};color:{row_color};border-radius:999px;padding:6px 10px;font-size:11px;font-weight:650;">
<span style="width:6px;height:6px;border-radius:50%;background:{row_color};display:inline-block;"></span>
{label}
</span>
</div>
</div>
"""

    st.markdown(f"""
<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:22px;padding:22px 26px;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#ff7cf5;"></div>
{rows_html}
</div>
""", unsafe_allow_html=True)


def render_block_ab_scenarios_library(tags, filename, source_image_bytes):
    """A/B scenarios + AI reference for a library creative."""

    st.markdown("""
<div style="margin:8px 0 28px 0;">
<div style="font-size:12px;color:#0009dc;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:8px;">
Scenario testing
</div>
<div style="font-size:28px;font-weight:650;color:#080808;margin-bottom:8px;letter-spacing:-0.6px;">
A/B scenarios + AI reference
</div>
<div style="font-size:15px;color:rgba(8,8,8,0.62);max-width:860px;line-height:1.55;">
Toggle visual tags, compare the direction of the predicted CTR shift, and generate a visual reference for the selected scenario.
</div>
</div>
""", unsafe_allow_html=True)

    # Toggle state — uniqueness by filename
    state_key = f"lib_ab_state_{filename}"

    if state_key not in st.session_state:
        st.session_state[state_key] = {
            feat: bool(tags.get(feat, False)) for feat in BINARY_FEATURES
        }

    # Reset button
    if st.button("↻ Reset scenario", key=f"lib_reset_{filename}"):
        st.session_state[state_key] = {
            feat: bool(tags.get(feat, False)) for feat in BINARY_FEATURES
        }

        # Reset actual Streamlit toggle states too
        for feat in BINARY_FEATURES:
            toggle_key = f"lib_toggle_{filename}_{feat}"
            if toggle_key in st.session_state:
                st.session_state[toggle_key] = bool(tags.get(feat, False))

        # Remove generated references for this creative
        for key in list(st.session_state.keys()):
            if key.startswith(f"lib_ab_generated_{filename}"):
                del st.session_state[key]

        st.rerun()

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

    # Toggles
    cols = st.columns(3, gap="large")

    for i, feat in enumerate(BINARY_FEATURES):
        toggle_key = f"lib_toggle_{filename}_{feat}"

        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = st.session_state[state_key][feat]

        with cols[i % 3]:
            new_value = st.toggle(
                display_name(feat),
                value=st.session_state[toggle_key],
                key=toggle_key,
            )
            st.session_state[state_key][feat] = new_value

    # Compute delta via model
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

    AB_STRONG = 0.10
    AB_WEAK = 0.05
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

    # AI reference generation
    if not changes:
        st.markdown(
            "<div style='font-size:13px;color:rgba(8,8,8,0.48);margin-top:14px;'>"
            "Toggle some tags above to generate a visual reference for the scenario."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    import hashlib

    scenario_signature = "|".join(
        f"{feat}={int(val)}" for feat, _, val in sorted(changes, key=lambda x: x[0])
    )
    scenario_hash = hashlib.md5(scenario_signature.encode()).hexdigest()[:8]
    cache_key = f"lib_ab_generated_{filename}_{scenario_hash}"

    st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)

    if cache_key not in st.session_state:
        if st.button(
            "✨ Generate a visual reference for this scenario",
            use_container_width=True,
            key=f"lib_gen_btn_{filename}_{scenario_hash}",
        ):
            if source_image_bytes is None:
                st.error("Source image not found on disk")
            else:
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

        if st.button("🔄 Regenerate", key=f"lib_regen_{filename}_{scenario_hash}"):
            del st.session_state[cache_key]
            st.rerun()

def render_block_similar_library(similar, current_filename, scope="database", tags=None):
    """Show similar creatives from the database with links."""

    # Determine current creative's brand and category for the subtitle
    current_row = df[df["filename"] == current_filename]
    current_brand = current_row.iloc[0].get("brand", "this brand") if len(current_row) > 0 else "this brand"

    food = (tags or {}).get("food_type", "none")
    drink = (tags or {}).get("drink_type", "none")
    category_label = food.replace("_", " ") if food != "none" else (
        drink.replace("_", " ") if drink != "none" else ""
    )

    # Build subtitle by scope
    if scope == "brand-category" and category_label:
        subtitle = (
            f"Creatives of <b>{current_brand}</b> in the <b>{category_label}</b> category, "
            f"sorted by actual CTR. Use them as nearby references for what performed better or worse."
        )
    elif scope == "category-expanded" and category_label:
        subtitle = (
            f"Creatives in the <b>{category_label}</b> category (expanded to other brands — "
            f"<b>{current_brand}</b> has few examples here), sorted by actual CTR."
        )
    elif scope == "brand-only":
        subtitle = (
            f"Creatives of <b>{current_brand}</b> (category had too few examples to filter), "
            f"sorted by actual CTR."
        )
    else:
        subtitle = "Creatives with a similar tag set, sorted by actual CTR. Use them as nearby references for what performed better or worse."

    st.markdown(f"""
<div style="margin:8px 0 28px 0;">
<div style="font-size:12px;color:#0009dc;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:8px;">
Creative neighborhood
</div>
<div style="font-size:28px;font-weight:650;color:#080808;margin-bottom:8px;letter-spacing:-0.6px;">
Similar creatives from the database
</div>
<div style="font-size:15px;color:rgba(8,8,8,0.62);max-width:860px;line-height:1.55;">
{subtitle}
</div>
</div>
""", unsafe_allow_html=True)

    creatives = []
    for idx, sim_score in similar:
        row = df.loc[idx]
        creatives.append({
            "filename": row["filename"],
            "ctr": row["ctr"],
            "row": row,
        })

    creatives.sort(key=lambda x: x["ctr"], reverse=True)
    median_ctr = df["ctr"].median()

    for row_start in [0, 5]:
        if row_start == 5:
            st.markdown('<div style="height:22px;"></div>', unsafe_allow_html=True)

        cols = st.columns(5, gap="small")

        for i, c in enumerate(creatives[row_start:row_start + 5]):
            with cols[i]:
                image_path = os.path.join(IMAGES_DIR, c["filename"])

                ctr_color = "#1D9E75" if c["ctr"] >= median_ctr else "#E24B4A"
                ctr_label = "above median" if c["ctr"] >= median_ctr else "below median"

                active = []
                for binary in BINARY_FEATURES:
                    if binary in c["row"].index and c["row"][binary] == True:
                        active.append(binary)

                feat_imp = {
                    b: np.abs(shap_df[b]).mean()
                    for b in active
                    if b in shap_df.columns
                }

                top_tags = sorted(
                    active,
                    key=lambda x: feat_imp.get(x, 0),
                    reverse=True
                )[:3]

                tag_str = ", ".join(display_name(t) for t in top_tags)
                brand_label = c["row"].get("brand", "—")
                year_label = c["row"].get("year", "—")

                if os.path.exists(image_path):
                    img_data = base64.b64encode(open(image_path, "rb").read()).decode()
                    st.markdown(
                        f'<div style="aspect-ratio:1;border-radius:18px 18px 0 0;overflow:hidden;background:#ffffff;'
                        f'border:1px solid #eeeeee;border-bottom:none;display:flex;align-items:center;justify-content:center;">'
                        f'<img src="data:image/png;base64,{img_data}" style="max-width:100%;max-height:100%;object-fit:contain;display:block;"/>'
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
                    f'border:1px solid #eeeeee;border-top:3px solid {ctr_color};'
                    f'border-radius:0 0 18px 18px;padding:13px 13px 14px 13px;margin-bottom:10px;">'
                    f'<div style="font-size:22px;font-weight:700;color:{ctr_color};line-height:1;">{c["ctr"]:.2f}%</div>'
                    f'<div style="font-size:11px;color:rgba(8,8,8,0.46);margin-top:4px;">CTR · {ctr_label}</div>'
                    f'<div style="height:1px;background:#eeeeee;margin:12px 0 10px 0;"></div>'
                    f'<div style="font-size:12px;color:rgba(8,8,8,0.62);line-height:1.45;min-height:52px;">'
                    f'<b style="color:#080808;">{brand_label}</b> · {year_label}<br>'
                    f'{tag_str if tag_str else "—"}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                from urllib.parse import quote

                st.link_button(
                    "Open →",
                    f"/Library?creative={quote(str(c['filename']))}",
                    use_container_width=True,
                )

# ============================================================
# Read query param from Insights — if present, open that creative
# ============================================================

preselected = st.query_params.get("creative")
if preselected:
    matches = df.index[df["filename"] == preselected].tolist()
    if matches:
        st.session_state["selected_creative"] = int(matches[0])
    # clear query param so it doesn't stick on F5
    st.query_params.clear()

# ============================================================
# session_state — which creative is selected
# ============================================================
if "selected_creative" not in st.session_state:
    st.session_state.selected_creative = None

# ============================================================
# Function: properly render an image
# ============================================================

@st.cache_data
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


def render_library_creative(idx):
    """Show breakdown of a library creative (Upload-style, without predictions)."""
    row = df.iloc[idx]
    filename = row["filename"]
    image_path = os.path.join(IMAGES_DIR, filename)

    # === Determine creative status ===
    shap_row = shap_df[shap_df["filename"] == filename]
    badge_html = ""
    
    if len(shap_row) > 0:
        shap_row = shap_row.iloc[0]
        predicted_ctr = shap_row["predicted_ctr"]
        residual = row["ctr"] - predicted_ctr
        
        # Percentile of residual across the database to gauge anomaly rarity
        all_residuals = df.merge(
            shap_df[["filename", "predicted_ctr"]], on="filename", how="left"
        ).dropna(subset=["predicted_ctr"])
        all_residuals["residual"] = all_residuals["ctr"] - all_residuals["predicted_ctr"]
        
        # Top-10% positive residuals → hidden gem
        gem_threshold = all_residuals["residual"].quantile(0.90)
        # Bottom-10% → underperformer
        under_threshold = all_residuals["residual"].quantile(0.10)
        
        if residual >= gem_threshold and residual > 0.20:
            badge_html = """
            <div style="display:inline-flex;align-items:center;gap:8px;background:#e8f7f0;
                        color:#1D9E75;padding:8px 16px;border-radius:20px;font-size:13px;
                        font-weight:600;margin-bottom:16px;">
                💎 Hidden gem — CTR above similar creatives
            </div>
            """
        elif residual <= under_threshold and residual < -0.20:
            badge_html = """
            <div style="display:inline-flex;align-items:center;gap:8px;background:#fdecec;
                        color:#E24B4A;padding:8px 16px;border-radius:20px;font-size:13px;
                        font-weight:600;margin-bottom:16px;">
                📉 Underperformer — CTR below similar creatives
            </div>
            """
    
        # === Header: image + base info ===
    col_img, col_info = st.columns([0.9, 1.6], gap="large")
    
    with col_img:
        if os.path.exists(image_path):
            image = Image.open(image_path)
            max_height = 480
            if image.height > max_height:
                ratio = max_height / image.height
                image = image.resize((int(image.width * ratio), max_height), Image.LANCZOS)

            st.image(image)
        else:
            st.info(f"Image not found: {filename}")
    
    with col_info:
        actual_ctr = row["ctr"]
        impressions = row["impressions"]
        brand = row.get("brand", "—")
        year = row.get("year", "—")
        
        base_ctr = df["ctr"].mean()
        diff = actual_ctr - base_ctr

        main_object = str(row.get("main_object", "—")).replace("_", " ")

        food_type = str(row.get("food_type", "none")).replace("_", " ")
        drink_type = str(row.get("drink_type", "none")).replace("_", " ")

        if food_type != "none":
            category_type = food_type
        elif drink_type != "none":
            category_type = drink_type
        else:
            category_type = "not specified"

        if diff > 0:
            diff_color = "#1D9E75"
            diff_bg = "rgba(29,158,117,0.10)"
            diff_sign = "+"
            diff_label = "Above database average"
        elif diff < 0:
            diff_color = "#E24B4A"
            diff_bg = "rgba(226,75,74,0.10)"
            diff_sign = ""
            diff_label = "Below database average"
        else:
            diff_color = "rgba(8,8,8,0.42)"
            diff_bg = "rgba(8,8,8,0.04)"
            diff_sign = ""
            diff_label = "At database average"

        # Collect tags
        tags = {}
        for cat in CATEGORICAL_TAGS:
            tags[cat] = row[cat]
        for binary in BINARY_FEATURES:
            if binary in row.index:
                tags[binary] = bool(row[binary])

        st.markdown(f"""
<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:22px;padding:24px 24px;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:4px;background:{diff_color};"></div>

<div style="display:flex;justify-content:space-between;gap:18px;align-items:flex-start;flex-wrap:wrap;margin-bottom:20px;">
<div>
<div style="font-size:12px;color:#0009dc;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:8px;">
Selected creative
</div>
<div style="font-size:24px;font-weight:650;color:#080808;letter-spacing:-0.5px;margin-bottom:8px;">
{brand} · {year}
</div>
<div style="display:inline-flex;align-items:center;gap:8px;background:{diff_bg};color:{diff_color};border-radius:999px;padding:7px 12px;font-size:12px;font-weight:650;">
<span style="width:7px;height:7px;border-radius:50%;background:{diff_color};display:inline-block;"></span>
{diff_label}
</div>
</div>

<div style="text-align:right;">
<div style="font-size:12px;color:rgba(8,8,8,0.46);letter-spacing:1.1px;text-transform:uppercase;margin-bottom:6px;">
Actual CTR
</div>
<div style="font-size:42px;font-weight:750;color:{diff_color};line-height:1;letter-spacing:-1.2px;">
{actual_ctr:.2f}%
</div>
<div style="font-size:12px;color:rgba(8,8,8,0.52);margin-top:8px;">
{diff_sign}{diff:.2f}% vs database avg ({base_ctr:.2f}%)
</div>
</div>
</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px;">
<div style="background:#ffffff;border:1px solid #eeeeee;border-radius:16px;padding:15px 16px;">
<div style="font-size:11px;color:rgba(8,8,8,0.46);margin-bottom:5px;">Impressions</div>
<div style="font-size:22px;font-weight:650;color:#080808;">{impressions:,.0f}</div>
</div>

<div style="background:#ffffff;border:1px solid #eeeeee;border-radius:16px;padding:15px 16px;">
<div style="font-size:11px;color:rgba(8,8,8,0.46);margin-bottom:5px;">Creative type</div>
<div style="font-size:18px;font-weight:650;color:#080808;text-transform:capitalize;">{main_object}</div>
<div style="font-size:12px;color:rgba(8,8,8,0.52);margin-top:4px;text-transform:capitalize;">{category_type}</div>
</div>

<div style="font-size:12px;color:#0009dc;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:10px;">
Creative tags
</div>
</div>
""", unsafe_allow_html=True)

        st.markdown(render_tag_chips(tags), unsafe_allow_html=True)
    
    # === Active tags for analysis ===
    active_tags = get_active_tags(tags)
    
    # === Block 1: Tag correlation with CTR ===
    divider()
    render_block_tag_effects(active_tags)
    
    spacer()
    with st.expander("ℹ️ How this is calculated"):
        st.markdown("""
        For each tag of this creative, we look at all creatives in the database where this
        tag is active and calculate the average SHAP value — how much this tag tends to push
        CTR up or down compared to the baseline.
        
        Arrow indicators:
        
        - ↑↑ / ↓↓ — strong correlation (≥ 0.15 CTR percentage points)
        - ↑ / ↓ — moderate (≥ 0.05)
        - — — neutral
        
        This shows correlation in the data, not a guaranteed effect for this specific creative.
        """)
    
    # === Block 2: Creative category ===
    divider()
    render_block_segment_context(tags)
    
    spacer()
    with st.expander("ℹ️ How this is calculated"):
        st.markdown("""
        Compares the average CTR of creatives in the same food/drink category as this creative
        with the average CTR across the entire database.
        
        Positive difference (green) means creatives of this category tend to perform better
        than average. Negative (red) means they tend to underperform.
        
        Only categories with at least 5 creatives in the database are shown.
        """)
    
    # === Block 3: Tag combinations ===
    divider()
    render_block_combinations(active_tags, tags)
    
    spacer()
    with st.expander("ℹ️ How this is calculated"):
        st.markdown("""
        For each pair of active tags on this creative, we measure how much **adding the
        second tag** changes CTR beyond what each tag would contribute alone.
        
        We show the top 3 positive pairs (combinations that work well together) and the
        top 3 negative pairs (combinations that work against each other).
        
        If no significant pairs are found, the tag combination of this creative doesn't have
        strong synergies or conflicts in the data.
        """)

    # === Block 4: A/B scenarios ===
    divider()
    source_image_bytes = read_image_bytes(filename)
    render_block_ab_scenarios_library(tags, filename, source_image_bytes)
    
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
    
    # === Block 5: Similar creatives ===
    divider()
    similar, similar_scope = find_similar_creatives(
        active_tags,
        current_filename=filename,
        food_type=tags.get("food_type"),
        drink_type=tags.get("drink_type"),
        top_n=10,
    )
    render_block_similar_library(similar, filename, similar_scope, tags)
    
    spacer()
    with st.expander("ℹ️ How this is calculated"):
        st.markdown("""
        Similar creatives use a **category-priority cascade**, so the reference set stays visually relevant:
        
        1. **Same brand + same category** — the ideal case (same brand, same food/drink type)
        2. **Other brands in the same category** — added if the current brand has fewer than 5 examples in this category
        3. **Same brand only** — used if the category is too rare anywhere in the database
        4. **Whole database** — last fallback if none of the above produces enough results
        
        Similarity is computed as the overlap of tag sets, weighted by each tag's importance (average absolute SHAP across the database).
        
        The top 10 most similar creatives are shown, sorted by their actual CTR (highest first). The current creative is always excluded from the list.
        """)


# # ============================================================
# # UI: Page hero
# # ============================================================
# st.markdown(f"""
# <div style="margin:24px 0 28px 0;padding:28px 30px;border-radius:26px;background:radial-gradient(circle at 8% 18%, rgba(174,243,62,0.24), transparent 28%),radial-gradient(circle at 92% 12%, rgba(255,124,245,0.12), transparent 28%),linear-gradient(135deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;">
# <div style="display:grid;grid-template-columns:1.25fr 0.75fr;gap:28px;align-items:center;">

# <div>
# <div style="display:inline-flex;gap:8px;align-items:center;padding:7px 12px;border-radius:999px;background:#ffffff;border:1px solid #eeeeee;font-size:12px;color:rgba(8,8,8,0.58);margin-bottom:16px;">
# <span style="width:7px;height:7px;border-radius:50%;background:#aef33e;display:inline-block;"></span>
# Creative database
# </div>

# <div class="ca-kicker">
# Explore tagged creatives
# </div>

# <h1 class="ca-title">
# Creative Library
# </h1>

# <p class="ca-subtitle">
# Browse the full creative dataset, filter by brand, category, visual elements, and open any creative to inspect its CTR-related patterns.
# </p>

# <div style="margin-top:18px;">
# <span class="ca-chip" style="background:#0009dc;color:#ffffff;">{len(df)} creatives</span>
# <span class="ca-chip" style="background:#aef33e;color:#080808;">AI-tagged</span>
# <span class="ca-chip" style="background:#080808;color:#ffffff;">CTR data</span>
# <span class="ca-chip" style="background:#ff7cf5;color:#080808;">creative breakdowns</span>
# </div>
# </div>

# <div class="ca-card ca-soft-blue" style="padding:22px 22px;min-height:190px;">
# <div style="position:absolute;top:0;left:0;right:0;height:3px;background:#0009dc;"></div>
# <div style="position:absolute;top:-42px;right:-42px;width:130px;height:130px;background:#0009dc;border-radius:50%;opacity:0.14;"></div>
# <div style="position:absolute;bottom:-48px;left:-48px;width:150px;height:150px;background:#aef33e;border-radius:50%;opacity:0.16;"></div>

# <div style="position:relative;z-index:1;">
# <div style="font-size:12px;color:#0009dc;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:14px;font-weight:600;">
# Library workflow
# </div>

# <div style="font-size:14px;color:rgba(8,8,8,0.78);line-height:1.9;">
# <span style="color:#0009dc;font-weight:650;">01</span> · Filter the creative database<br>
# <span style="color:#0009dc;font-weight:650;">02</span> · Compare CTR and impressions<br>
# <span style="color:#0009dc;font-weight:650;">03</span> · Open creative breakdowns<br>
# <span style="color:#0009dc;font-weight:650;">04</span> · Explore tags and scenarios
# </div>

# <div style="height:1px;background:rgba(0,9,220,0.10);margin:16px 0 13px 0;"></div>

# <div style="font-size:12px;color:rgba(8,8,8,0.56);line-height:1.5;">
# Designed for browsing the dataset before deep-diving into a specific creative.
# </div>
# </div>
# </div>

# </div>
# </div>
# """, unsafe_allow_html=True)

# === MODE: CREATIVE DETAIL ===
if st.session_state.selected_creative is not None:
    if st.button("← Back to library"):
        st.session_state.selected_creative = None
        st.rerun()

    st.markdown("""
<div style="margin:22px 0 28px 0;">
<div style="font-size:12px;color:#0009dc;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:8px;">
Creative detail
</div>
<div style="font-size:34px;font-weight:650;color:#080808;margin-bottom:8px;letter-spacing:-0.8px;">
Creative breakdown
</div>
<div style="font-size:15px;color:rgba(8,8,8,0.62);max-width:760px;line-height:1.55;">
Inspect the selected creative, its actual CTR, visual tags, category context, and scenario testing options.
</div>
</div>
""", unsafe_allow_html=True)

    render_library_creative(st.session_state.selected_creative)

# === MODE: GALLERY ===
else:
    # ============================================================
    # UI: Page hero
    # ============================================================
    st.markdown(f"""
<div style="margin:24px 0 28px 0;padding:28px 30px;border-radius:26px;background:radial-gradient(circle at 8% 18%, rgba(174,243,62,0.24), transparent 28%),radial-gradient(circle at 92% 12%, rgba(255,124,245,0.12), transparent 28%),linear-gradient(135deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;">
<div style="display:grid;grid-template-columns:1.25fr 0.75fr;gap:28px;align-items:center;">

<div>
<div style="display:inline-flex;gap:8px;align-items:center;padding:7px 12px;border-radius:999px;background:#ffffff;border:1px solid #eeeeee;font-size:12px;color:rgba(8,8,8,0.58);margin-bottom:16px;">
<span style="width:7px;height:7px;border-radius:50%;background:#aef33e;display:inline-block;"></span>
Creative database
</div>

<div class="ca-kicker">
Explore tagged creatives
</div>

<h1 class="ca-title">
Creative Library
</h1>

<p class="ca-subtitle">
Browse the full creative dataset, filter by brand, category, visual elements, and open any creative to inspect its CTR-related patterns.
</p>

<div style="margin-top:18px;">
<span class="ca-chip" style="background:#0009dc;color:#ffffff;">{len(df)} creatives</span>
<span class="ca-chip" style="background:#aef33e;color:#080808;">AI-tagged</span>
<span class="ca-chip" style="background:#080808;color:#ffffff;">CTR data</span>
<span class="ca-chip" style="background:#ff7cf5;color:#080808;">creative breakdowns</span>
</div>
</div>

<div class="ca-card ca-soft-blue" style="padding:22px 22px;min-height:190px;">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#0009dc;"></div>
<div style="position:absolute;top:-42px;right:-42px;width:130px;height:130px;background:#0009dc;border-radius:50%;opacity:0.14;"></div>
<div style="position:absolute;bottom:-48px;left:-48px;width:150px;height:150px;background:#aef33e;border-radius:50%;opacity:0.16;"></div>

<div style="position:relative;z-index:1;">
<div style="font-size:12px;color:#0009dc;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:14px;font-weight:600;">
Library workflow
</div>

<div style="font-size:14px;color:rgba(8,8,8,0.78);line-height:1.9;">
<span style="color:#0009dc;font-weight:650;">01</span> · Filter the creative database<br>
<span style="color:#0009dc;font-weight:650;">02</span> · Compare CTR and impressions<br>
<span style="color:#0009dc;font-weight:650;">03</span> · Open creative breakdowns<br>
<span style="color:#0009dc;font-weight:650;">04</span> · Explore tags and scenarios
</div>

<div style="height:1px;background:rgba(0,9,220,0.10);margin:16px 0 13px 0;"></div>

<div style="font-size:12px;color:rgba(8,8,8,0.56);line-height:1.5;">
Designed for browsing the dataset before deep-diving into a specific creative.
</div>
</div>
</div>

</div>
</div>
""", unsafe_allow_html=True)

    # ============================================================
    # Sidebar filters
    # ============================================================
    with st.sidebar:
        with st.sidebar:
            st.markdown("""
<div style="margin:4px 0 18px 0;">
<div style="font-size:12px;color:#0009dc;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:6px;">
Library controls
</div>
<div style="font-size:22px;font-weight:650;color:#080808;letter-spacing:-0.4px;">
Filters
</div>
<div style="font-size:13px;color:rgba(8,8,8,0.56);line-height:1.45;margin-top:6px;">
Narrow the creative dataset by brand, category, status, and visual elements.
</div>
</div>
""", unsafe_allow_html=True)

        # Brand
        all_brands = ["All"] + sorted(df["brand"].dropna().unique().tolist())
        selected_brand = st.selectbox("Brand", all_brands)

        # Year
        all_years = ["All"] + sorted(df["year"].dropna().unique().tolist())
        selected_year = st.selectbox("Year", all_years)

        st.markdown('<div style="height:1px;background:#eeeeee;margin:18px 0;"></div>', unsafe_allow_html=True)


        # Filter by type
        main_objects = ["All"] + sorted(df["main_object"].dropna().unique().tolist())
        selected_main = st.selectbox("Main object", main_objects)

        food_types = ["All"] + sorted(df["food_type"].dropna().unique().tolist())
        selected_food = st.selectbox("Food type", food_types)

        drink_types = ["All"] + sorted(df["drink_type"].dropna().unique().tolist())
        selected_drink = st.selectbox("Drink type", drink_types)

        st.markdown('<div style="height:1px;background:#eeeeee;margin:18px 0;"></div>', unsafe_allow_html=True)

        # Status filter (anomaly)
        status_filter = st.selectbox(
            "Status",
            ["All creatives", "💎 Hidden gems", "📉 Underperformers"],
            help="Hidden gems — CTR above creatives with similar tags. Underperformers — CTR below."
        )

        st.markdown('<div style="height:1px;background:#eeeeee;margin:18px 0;"></div>', unsafe_allow_html=True)

        # Binary filters
        st.markdown("""
<div style="margin:2px 0 10px 0;">
<div style="font-size:12px;color:#0009dc;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:4px;">
Must have
</div>
<div style="font-size:13px;color:rgba(8,8,8,0.56);line-height:1.45;">
Require specific visual tags.
</div>
</div>
""", unsafe_allow_html=True)

        must_have = []
        for feat in BINARY_FEATURES:
            if st.checkbox(display_name(feat), key=f"mh_{feat}"):
                must_have.append(feat)

        st.markdown('<div style="height:1px;background:#eeeeee;margin:18px 0;"></div>', unsafe_allow_html=True)

        # Sort
        sort_by = st.selectbox("Sort by", ["CTR (descending)", "CTR (ascending)", "Impressions (descending)"])

        st.markdown("""
<div style="height:1px;background:#eeeeee;margin:18px 0;"></div>
<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:16px;padding:14px 14px;">
<div style="font-size:12px;color:#0009dc;letter-spacing:1.1px;text-transform:uppercase;margin-bottom:6px;">
Tip
</div>
<div style="font-size:13px;color:rgba(8,8,8,0.62);line-height:1.5;">
Use <b style="color:#080808;">Status</b> to find hidden gems or underperformers, then open a creative for full tag breakdown.
</div>
</div>
""", unsafe_allow_html=True)

    # ============================================================
    # Apply filters
    # ============================================================
    filtered = df.copy()
    if selected_brand != "All":
        filtered = filtered[filtered["brand"] == selected_brand]
    if selected_year != "All":
        filtered = filtered[filtered["year"] == selected_year]

    if selected_main != "All":
        filtered = filtered[filtered["main_object"] == selected_main]
    if selected_food != "All":
        filtered = filtered[filtered["food_type"] == selected_food]
    if selected_drink != "All":
        filtered = filtered[filtered["drink_type"] == selected_drink]

    for feat in must_have:
        filtered = filtered[filtered[feat] == True]


    if status_filter != "All creatives":
            filtered = filtered.merge(
                shap_df[["filename", "predicted_ctr"]], on="filename", how="left"
            ).dropna(subset=["predicted_ctr"])
            filtered["residual"] = filtered["ctr"] - filtered["predicted_ctr"]

            all_with_pred = df.merge(
                shap_df[["filename", "predicted_ctr"]], on="filename", how="left"
            ).dropna(subset=["predicted_ctr"])
            all_with_pred["residual"] = all_with_pred["ctr"] - all_with_pred["predicted_ctr"]
            gem_threshold = all_with_pred["residual"].quantile(0.90)
            under_threshold = all_with_pred["residual"].quantile(0.10)

            if status_filter == "💎 Hidden gems":
                filtered = filtered[(filtered["residual"] >= gem_threshold) & (filtered["residual"] > 0.20)]
            elif status_filter == "📉 Underperformers":
                filtered = filtered[(filtered["residual"] <= under_threshold) & (filtered["residual"] < -0.20)]

            filtered = filtered.drop(columns=["predicted_ctr", "residual"], errors="ignore")



    if sort_by == "CTR (descending)":
        filtered = filtered.sort_values("ctr", ascending=False)
    elif sort_by == "CTR (ascending)":
        filtered = filtered.sort_values("ctr", ascending=True)
    else:
        filtered = filtered.sort_values("impressions", ascending=False)

    st.markdown(f"**Found: {len(filtered)} creatives**")

    spacer()

    with st.expander("ℹ️ How to use the library"):
        st.markdown("""
    Browse all creatives in the database, filter by brand, year, visual elements,
    and creative status.

    **Filters:**

    - **Brand & Year** — narrow down to a specific brand or time period
    - **Main object / Food type / Drink type** — filter by the primary subject of the creative
    - **Status** — find hidden gems or underperformers
    - **Must have** — require specific visual elements such as logo, CTA, discount, etc.
    - **Sort by** — order results by CTR or impressions

    Click **Open** on any creative to see its full breakdown.
    """)

    if len(filtered) == 0:
        st.info("Nothing found — try loosening the filters")
    else:
        # ============================================================
        # Pagination
        # ============================================================
        items_per_page = 24
        total_pages = (len(filtered) - 1) // items_per_page + 1
        
        # Unique key to keep current page per filter combination
        # (so that page resets when filters change)
        filter_key = f"{selected_brand}_{selected_year}_{selected_main}_{'_'.join(must_have)}_{sort_by}_{status_filter}"

        page_state_key = f"library_page_{filter_key}"
        
        if page_state_key not in st.session_state:
            st.session_state[page_state_key] = 0
        
        # Safety: if filters changed, current page might be beyond total_pages
        page = min(st.session_state[page_state_key], total_pages - 1)
        st.session_state[page_state_key] = page
        
        start = page * items_per_page
        end = start + items_per_page
        page_data = filtered.iloc[start:end]
        
        # Top navigation
        if total_pages > 1:
            nav_cols = st.columns([1, 1, 3, 1, 1])
            
            with nav_cols[0]:
                if st.button("← Previous", disabled=(page == 0), use_container_width=True, key="page_prev"):
                    st.session_state[page_state_key] = page - 1
                    st.rerun()
            
            with nav_cols[2]:
                st.markdown(
                    f"<div style='text-align:center;padding-top:6px;font-size:14px;color:#666;'>"
                    f"Page <b style='color:#1a1a1a;'>{page + 1}</b> of <b style='color:#1a1a1a;'>{total_pages}</b> "
                    f"</div>",
                    unsafe_allow_html=True,
                )
            
            with nav_cols[4]:
                if st.button("Next →", disabled=(page >= total_pages - 1), use_container_width=True, key="page_next"):
                    st.session_state[page_state_key] = page + 1
                    st.rerun()
        # page_data = filtered.iloc[start:end]

        # ============================================================
        # Gallery — 4 columns
        # ============================================================
        cols_per_row = 4
        rows = [page_data.iloc[i:i+cols_per_row] for i in range(0, len(page_data), cols_per_row)]

        for row_data in rows:
            cols = st.columns(cols_per_row)
            for col, (_, item) in zip(cols, row_data.iterrows()):
                with col:
                    image_path = os.path.join(IMAGES_DIR, item["filename"])
                    data_url = encode_image(image_path)

                    median_ctr = df["ctr"].median()
                    ctr_color = "#1D9E75" if item["ctr"] > median_ctr else "#E24B4A"
                    ctr_label = "above median" if item["ctr"] > median_ctr else "below median"

                    brand_label = item.get("brand", "—")
                    year_label = item.get("year", "—")
                    main_object = str(item.get("main_object", "—")).replace("_", " ")

                    # Image
                    if data_url:
                        st.markdown(
                            f'<div style="aspect-ratio:1;border-radius:18px 18px 0 0;overflow:hidden;background:#ffffff;'
                            f'border:1px solid #eeeeee;border-bottom:none;display:flex;align-items:center;justify-content:center;">'
                            f'<img src="{data_url}" style="max-width:100%;max-height:100%;object-fit:contain;display:block;"/>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div style="aspect-ratio:1;border-radius:18px 18px 0 0;background:#ffffff;'
                            'border:1px solid #eeeeee;border-bottom:none;display:flex;align-items:center;justify-content:center;'
                            'color:rgba(8,8,8,0.38);font-size:12px;">no image</div>',
                            unsafe_allow_html=True
                        )

                    # Info card
                    st.markdown(
                        f'<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);'
                        f'border:1px solid #eeeeee;border-top:3px solid {ctr_color};'
                        f'border-radius:0 0 18px 18px;padding:13px 13px 14px 13px;margin-bottom:10px;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">'
                        f'<div>'
                        f'<div style="font-size:22px;font-weight:700;color:{ctr_color};line-height:1;">{item["ctr"]:.2f}%</div>'
                        f'<div style="font-size:11px;color:rgba(8,8,8,0.46);margin-top:4px;">CTR · {ctr_label}</div>'
                        f'</div>'
                        f'<div style="text-align:right;">'
                        f'<div style="font-size:11px;color:rgba(8,8,8,0.46);margin-bottom:3px;">impressions</div>'
                        f'<div style="font-size:13px;font-weight:650;color:#080808;">{item["impressions"]:,.0f}</div>'
                        f'</div>'
                        f'</div>'
                        f'<div style="height:1px;background:#eeeeee;margin:12px 0 10px 0;"></div>'
                        f'<div style="font-size:12px;color:rgba(8,8,8,0.62);line-height:1.45;min-height:34px;">'
                        f'<b style="color:#080808;">{brand_label}</b> · {year_label}<br>{main_object}'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    df_idx = item.name
                    if st.button("Open", key=f"open_{df_idx}", width="stretch"):
                        st.session_state.selected_creative = df_idx
                        st.rerun()

        # Bottom navigation (duplicates top)
        if total_pages > 1:
            st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
            nav_cols = st.columns([1, 1, 3, 1, 1])
            
            with nav_cols[0]:
                if st.button("← Previous", disabled=(page == 0), use_container_width=True, key="page_prev_bottom"):
                    st.session_state[page_state_key] = page - 1
                    st.rerun()
            
            with nav_cols[2]:
                st.markdown(
                    f"<div style='text-align:center;padding-top:6px;font-size:14px;color:#666;'>"
                    f"Page <b style='color:#1a1a1a;'>{page + 1}</b> of <b style='color:#1a1a1a;'>{total_pages}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            
            with nav_cols[4]:
                if st.button("Next →", disabled=(page >= total_pages - 1), use_container_width=True, key="page_next_bottom"):
                    st.session_state[page_state_key] = page + 1
                    st.rerun()

st.markdown("""
<div style="text-align:center;font-size:13px;color:#bbb;margin:60px 0 20px 0;padding-top:24px;border-top:1px solid #eee;">
    Creative Analyzer · Hackathon MVP · Built in Streamlit · by Viktoriia Iachmeneva · 2026
</div>
""", unsafe_allow_html=True)