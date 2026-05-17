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
    """HTML for arrows: colored double/single, or gray dash for neutral."""
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
    """Compare CTR of creative's category with the entire database."""
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
    """Top-3 positive and top-3 negative pairs among active tags."""
    if food_type and food_type != "none":
        segment_filter = (interactions_df["segment_type"] == "food") & \
                        (interactions_df["segment"] == food_type)
    elif drink_type and drink_type != "none":
        segment_filter = (interactions_df["segment_type"] == "drink") & \
                        (interactions_df["segment"] == drink_type)
    else:
        segment_filter = (interactions_df["segment_type"] == "all")
    
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
    
    positive = active_pairs[active_pairs["interaction"] > 0.01].nlargest(top_n_each, "interaction")
    negative = active_pairs[active_pairs["interaction"] < -0.01].nsmallest(top_n_each, "interaction")
    
    return positive.to_dict("records"), negative.to_dict("records")


def find_similar_creatives(active_tags, current_filename, food_type=None, drink_type=None, top_n=10):
    """Find creatives with similar tag set. Excludes the current creative itself."""
    feature_importance = {
        c: np.abs(shap_df[c]).mean() for c in feature_cols if c in shap_df.columns
    }

    # Filter database by category if available
    candidates = df.copy()
    if food_type and food_type != "none":
        candidates = candidates[candidates["food_type"] == food_type]
    elif drink_type and drink_type != "none":
        candidates = candidates[candidates["drink_type"] == drink_type]

    # If too few in segment — fallback to whole database
    if len(candidates) < 10:
        candidates = df.copy()

    # Exclude the creative we're looking at
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
    return similarities[:top_n]    

# ============================================================
# Render functions for creative breakdown blocks
# ============================================================

def divider():
    st.markdown('<div style="height:1px;background:#e8e6df;margin:36px 0 28px 0;"></div>', unsafe_allow_html=True)


def render_block_tag_effects(active_tags):
    """Tag correlation with CTR — arrows instead of numbers."""
    effects = calculate_tag_effects(active_tags)
    
    TAG_STRONG = 0.1
    TAG_WEAK = 0.05
    
    categorized = []
    for tag, val in effects:
        direction, strength = categorize_strength(val, TAG_STRONG, TAG_WEAK)
        categorized.append((tag, direction, strength))
    
    st.markdown('<div style="font-size:20px;font-weight:500;margin-bottom:6px;">Tag correlation with CTR</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:13px;color:#888;margin-bottom:20px;">'
        'Which tags appear more often in creatives with high or low CTR. '
        '<b>↑↑ / ↓↓</b> — strong correlation, <b>↑ / ↓</b> — moderate, <b>—</b> — neutral. '
        'This is a pattern in the data, not a guaranteed effect.</div>',
        unsafe_allow_html=True,
    )
    
    if not categorized:
        st.markdown(
            "<div style='color:#888;font-size:14px;'>No significant effects found</div>",
            unsafe_allow_html=True
        )
        return
    
    rows_html = ""
    for tag, direction, strength in categorized:
        rows_html += f"""
        <div style="padding:10px 0;border-bottom:1px solid #f5f3ed;display:flex;align-items:center;gap:12px;">
            <span style="width:70px;text-align:center;">{arrow_symbol(direction, strength)}</span>
            <span style="flex:1;font-size:14px;">{display_name(tag)}</span>
        </div>
        """
    st.markdown(rows_html, unsafe_allow_html=True)


def render_block_segment_context(tags):
    """Creative category — comparison with the entire database."""
    insights, base_ctr = get_segment_context(tags)
    
    if not insights:
        return
    
    st.markdown('<div style="font-size:20px;font-weight:500;margin-bottom:6px;">Creative category</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:13px;color:#888;margin-bottom:20px;">'
        'How strong creatives in the same category are within the entire database</div>',
        unsafe_allow_html=True,
    )
    
    rows_html = ""
    for ins in insights:
        diff = ins["diff"]
        is_better = diff > 0
        color = "#1D9E75" if is_better else "#E24B4A"
        sign = "+" if is_better else ""
        
        if abs(diff) < 0.1:
            verdict = "Category performs roughly on par with the rest"
            verdict_color = "#888"
        elif is_better:
            verdict = f"✓ Category outperforms others by {sign}{diff:.2f}%"
            verdict_color = "#1D9E75"
        else:
            verdict = f"✗ Category underperforms others by {diff:.2f}%"
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
                    <div style="font-size:11px;color:#888;margin-bottom:2px;">All creatives CTR</div>
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


def render_block_combinations(active_tags, tags):
    """Tag pair correlation with CTR — arrows instead of numbers."""
    positive, negative = calculate_active_combinations(
        active_tags,
        food_type=tags.get("food_type"),
        drink_type=tags.get("drink_type"),
    )
    
    COMBO_STRONG = 0.08
    COMBO_WEAK = 0.03
    
    all_combos = []
    for combo in positive + negative:
        direction, strength = categorize_strength(combo["interaction"], COMBO_STRONG, COMBO_WEAK)
        all_combos.append((combo, direction, strength))
    
    st.markdown('<div style="font-size:20px;font-weight:500;margin-bottom:6px;">Tag pair correlation with CTR</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:13px;color:#888;margin-bottom:20px;">'
        'Which tag combinations correlate with CTR more or less than the tags taken separately. '
        '<b>↑↑ / ↓↓</b> — strong, <b>↑ / ↓</b> — moderate, <b>—</b> — neutral.</div>',
        unsafe_allow_html=True,
    )
    
    if not all_combos:
        st.markdown(
            "<div style='color:#888;font-size:14px;'>No significant combinations found for this tag set</div>",
            unsafe_allow_html=True
        )
        return
    
    rows_html = ""
    for combo, direction, strength in all_combos:
        rows_html += f"""
        <div style="padding:10px 0;border-bottom:1px solid #f5f3ed;display:flex;align-items:center;gap:12px;">
            <span style="width:70px;text-align:center;">{arrow_symbol(direction, strength)}</span>
            <span style="flex:1;font-size:14px;"><b>{display_name(combo['feat1'])}</b> × <b>{display_name(combo['feat2'])}</b></span>
        </div>
        """
    st.markdown(rows_html, unsafe_allow_html=True)


def render_block_ab_scenarios_library(tags, filename, source_image_bytes):
    """A/B scenarios + AI reference for a library creative."""
    
    st.markdown('<div style="font-size:20px;font-weight:500;margin-bottom:6px;">A/B scenarios + AI reference</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:13px;color:#888;margin-bottom:20px;">'
        'Toggle tags — see how the correlation with CTR changes, '
        'and generate a visual reference based on your scenario</div>',
        unsafe_allow_html=True,
    )

    # Toggle state — uniqueness by filename
    state_key = f"lib_ab_state_{filename}"
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            feat: bool(tags.get(feat, False)) for feat in BINARY_FEATURES
        }

    # Reset button
    col_reset, _ = st.columns([1, 4])
    with col_reset:
        if st.button("↻ Reset", key=f"lib_reset_{filename}"):
            st.session_state[state_key] = {
                feat: bool(tags.get(feat, False)) for feat in BINARY_FEATURES
            }
            for key in list(st.session_state.keys()):
                if key.startswith(f"lib_ab_generated_{filename}"):
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
                key=f"lib_toggle_{filename}_{feat}",
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
        new_val = st.session_state[state_key][feat]
        if original_val != new_val:
            action = "turn on" if new_val else "turn off"
            changes.append((feat, action, new_val))

    AB_STRONG = 0.10
    AB_WEAK = 0.05
    direction, strength = categorize_strength(diff, AB_STRONG, AB_WEAK)

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

    # === AI reference generation ===
    if not changes:
        st.markdown(
            "<div style='font-size:13px;color:#888;margin-top:16px;'>"
            "Toggle some tags above — and you'll be able to generate a visual reference."
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

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

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
        
        if st.button("🔄 Regenerate", key=f"lib_regen_{filename}_{scenario_hash}"):
            del st.session_state[cache_key]
            st.rerun()

def render_block_similar_library(similar, current_filename):
    """Show similar creatives from the database with links."""
    st.markdown('<div style="font-size:20px;font-weight:500;margin-bottom:6px;">Similar creatives from the database</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px;color:#888;margin-bottom:20px;">Creatives with a similar tag set, sorted by actual CTR</div>', unsafe_allow_html=True)

    if not similar:
        st.markdown(
            "<div style='color:#888;font-size:14px;'>No similar creatives found</div>",
            unsafe_allow_html=True
        )
        return

    creatives = []
    for idx, sim_score in similar:
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
                    img_data = base64.b64encode(open(image_path, 'rb').read()).decode()
                    st.markdown(f"""
                    <div style="height:200px;background:#f5f3ed;border-radius:8px;overflow:hidden;display:flex;align-items:center;justify-content:center;">
                        <img src="data:image/png;base64,{img_data}"
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
                """, unsafe_allow_html=True)
                
                # Link to open this creative in library
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
        if badge_html:
            st.markdown(badge_html, unsafe_allow_html=True)
        actual_ctr = row["ctr"]
        impressions = row["impressions"]
        brand = row.get("brand", "—")
        year = row.get("year", "—")
        
        # Comparison with database
        base_ctr = df["ctr"].mean()
        diff = actual_ctr - base_ctr
        diff_color = "#1D9E75" if diff > 0 else "#E24B4A"
        diff_sign = "+" if diff > 0 else ""
        
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:18px;">
            <div style="background:#faf9f5;border-radius:10px;padding:14px 18px;">
                <div style="font-size:11px;color:#888;margin-bottom:4px;">Actual CTR</div>
                <div style="font-size:24px;font-weight:600;color:#1a1a1a;">{actual_ctr:.2f}%</div>
                <div style="font-size:12px;color:{diff_color};margin-top:4px;">
                    {diff_sign}{diff:.2f}% vs database ({base_ctr:.2f}%)
                </div>
            </div>
            <div style="background:#faf9f5;border-radius:10px;padding:14px 18px;">
                <div style="font-size:11px;color:#888;margin-bottom:4px;">Impressions</div>
                <div style="font-size:24px;font-weight:600;color:#1a1a1a;">{impressions:,.0f}</div>
                <div style="font-size:12px;color:#888;margin-top:4px;">
                    {brand} · {year}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Collect tags
        tags = {}
        for cat in CATEGORICAL_TAGS:
            tags[cat] = row[cat]
        for binary in BINARY_FEATURES:
            if binary in row.index:
                tags[binary] = bool(row[binary])
        
        st.markdown("<div style='font-size:14px;font-weight:500;margin-bottom:8px;'>Creative tags</div>", unsafe_allow_html=True)
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
    similar = find_similar_creatives(
        active_tags,
        current_filename=filename,
        food_type=tags.get("food_type"),
        drink_type=tags.get("drink_type"),
        top_n=10,
    )
    render_block_similar_library(similar, filename)
    
    spacer()
    with st.expander("ℹ️ How this is calculated"):
        st.markdown("""
        We compute the similarity of each creative in the database to the current one based
        on the overlap of their tag sets, weighted by tag importance (average absolute SHAP).
        
        If the current creative has a food/drink category, we first look at creatives within
        the same category. If the category has fewer than 10 creatives, we expand to the
        whole database.
        
        The top 10 most similar creatives are shown, sorted by actual CTR (highest first).
        Click "Open →" on any card to see its full breakdown.
        """)


# ============================================================
# UI: Title
# ============================================================
st.title("📚 Creative Library")
st.caption(f"{len(df)} creatives with actual CTR and AI-tagged visual elements")

# === MODE: CREATIVE DETAIL ===
if st.session_state.selected_creative is not None:
    if st.button("← Back to library"):
        st.session_state.selected_creative = None
        st.rerun()

    render_library_creative(st.session_state.selected_creative)

# === MODE: GALLERY ===
else:
    # ============================================================
    # Sidebar filters
    # ============================================================
    with st.sidebar:
        st.markdown("### Filters")

        # Brand
        all_brands = ["All"] + sorted(df["brand"].dropna().unique().tolist())
        selected_brand = st.selectbox("Brand", all_brands)

        # Year
        all_years = ["All"] + sorted(df["year"].dropna().unique().tolist())
        selected_year = st.selectbox("Year", all_years)

        st.divider()


        # Filter by type
        main_objects = ["All"] + sorted(df["main_object"].dropna().unique().tolist())
        selected_main = st.selectbox("Main object", main_objects)

        food_types = ["All"] + sorted(df["food_type"].dropna().unique().tolist())
        selected_food = st.selectbox("Food type", food_types)

        drink_types = ["All"] + sorted(df["drink_type"].dropna().unique().tolist())
        selected_drink = st.selectbox("Drink type", drink_types)

        st.divider()

        # Status filter (anomaly)
        status_filter = st.selectbox(
            "Status",
            ["All creatives", "💎 Hidden gems", "📉 Underperformers"],
            help="Hidden gems — CTR above creatives with similar tags. Underperformers — CTR below."
        )

        st.divider()

        # Binary filters
        st.markdown("**Must have**")
        must_have = []
        for feat in BINARY_FEATURES:
            if st.checkbox(feat.replace("_", " "), key=f"mh_{feat}"):
                must_have.append(feat)

        st.divider()

        # Sort
        sort_by = st.selectbox("Sort by", ["CTR (descending)", "CTR (ascending)", "Impressions"])

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
        and creative status (hidden gems / underperformers).
        
        **Filters:**
        
        - **Brand & Year** — narrow down to a specific brand or time period
        - **Main object / Food type / Drink type** — filter by what's the primary subject of the creative
        - **Status** — find anomalies: creatives that perform unusually well or poorly
          compared to others with similar tags
        - **Must have** — require specific visual elements (logo, CTA, discount, etc.)
        - **Sort by** — order results by CTR or impressions
        
        Click **"Open"** on any creative to see its full breakdown — tags, correlations,
        tag combinations, and whether it's a hidden gem or underperformer.
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
        page_data = filtered.iloc[start:end]

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

                    if data_url:
                        st.markdown(
                            '<div style="aspect-ratio:1;border-radius:8px;overflow:hidden;background:#fff;'
                            'display:flex;align-items:center;justify-content:center;">'
                            f'<img src="{data_url}" style="max-width:100%;max-height:100%;object-fit:contain;display:block;"/>'
                            '</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            '<div style="aspect-ratio:1;border-radius:8px;background:#fff;'
                            'display:flex;align-items:center;justify-content:center;color:#bbb;font-size:12px;">'
                            'no image</div>',
                            unsafe_allow_html=True
                        )

                    # CTR badge
                    ctr_color = "#1D9E75" if item["ctr"] > df["ctr"].median() else "#666"
                    st.markdown(f"""
                    <div style="font-size:18px;font-weight:600;color:{ctr_color};margin-top:4px;">
                        {item['ctr']:.2f}%
                    </div>
                    <div style="font-size:11px;color:#888;">
                        {item['main_object']} · {item['impressions']:,.0f} impressions
                    </div>
                    """, unsafe_allow_html=True)

                    # Open button
                    real_idx = filtered.index.get_loc(item.name) + start
                    df_idx = item.name
                    if st.button("Open", key=f"open_{df_idx}", width='stretch'):
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