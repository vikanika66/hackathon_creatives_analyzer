"""
Страница загрузки и анализа нового креатива — версия 3.
Чистая типографика, без белых карточек.
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
    """Кэшированный OpenAI клиент."""
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def generate_image_via_openai(prompt, source_image_bytes, quality="medium"):
    """Image-to-image. Размер подбирается под пропорции исходника."""
    client = get_openai_client()
    if client is None:
        return None, "OPENAI_API_KEY не настроен в .streamlit/secrets.toml"
    
    try:
        from io import BytesIO
        
        img = Image.open(BytesIO(source_image_bytes))
        
        # Подбираем ближайший поддерживаемый размер по аспекту
        aspect = img.width / img.height
        if aspect > 1.2:
            size = "1536x1024"   # альбомный
        elif aspect < 0.85:
            size = "1024x1536"   # портретный
        else:
            size = "1024x1024"   # квадрат
        
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


# def build_generation_prompt(tags, recs_add, recs_remove):
#     """Промпт для image-edit: что в существующем креативе изменить."""
#     add_list = [display_name(f) for f, _ in recs_add]
#     remove_list = [display_name(f) for f, _ in recs_remove]
    
#     instructions = []
#     if remove_list:
#         instructions.append(f"reduce or remove these elements: {', '.join(remove_list)}")
#     if add_list:
#         instructions.append(f"add or emphasize these elements: {', '.join(add_list)}")
    
#     if not instructions:
#         return (
#             "Refine this advertisement creative while keeping product, brand, "
#             "composition, and overall style identical."
#         )
    
#     prompt = (
#         "Modify this advertisement creative to better match high-performing patterns. "
#         f"Required changes: {'. '.join(instructions)}. "
#         "Keep the same product, brand identity, typography style, and overall composition. "
#         "Maintain commercial photography quality."
#     )
#     return prompt

st.set_page_config(
    page_title="Анализ — Creative Analyzer",
    page_icon="🚀",
    layout="wide",
)

# ============================================================
# Стили — чистая типографика
# ============================================================
PAGE_CSS = """
<style>
    .stApp { background: white; }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 1200px;
    }

    /* Разделитель между блоками */
    .section-divider {
        height: 1px;
        background: #e8e6df;
        margin: 48px 0 36px 0;
    }

    /* Заголовок секции */
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

    /* Метрики */
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

    /* Теги */
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
    .tag-absent {
        background: #f0efe9;
        color: #b5b3a8;
        text-decoration: line-through;
    }

    /* Эффекты тегов */
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

    /* Рекомендации */
    .rec-row {
        padding: 12px 0;
        border-bottom: 1px solid #f5f3ed;
        font-size: 14px;
    }
    .rec-row:last-child { border-bottom: none; }
    .rec-add { color: #1D9E75; font-weight: 600; }
    .rec-remove { color: #E24B4A; font-weight: 600; }

    /* Кнопка "Открыть" в похожих */
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

# ============================================================
# Данные
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
    """Категоризирует связь: (направление, сила).
    strong  → двойная стрелка
    weak    → одинарная стрелка
    neutral → серый дефис (связь слабая, шум)
    """
    abs_v = abs(value)
    if abs_v < weak_threshold:
        return "neutral", "none"
    direction = "up" if value > 0 else "down"
    strength = "strong" if abs_v >= strong_threshold else "weak"
    return direction, strength


def arrow_symbol(direction, strength):
    """HTML стрелок: двойная/одинарная цветная, или серый дефис для нейтральной связи."""
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
# История
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "selected_history" not in st.session_state:
    st.session_state.selected_history = None


# ============================================================
# Функции анализа
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
    """Только бинарные теги — категории сюда не идут."""
    effects = []
    for tag in active_tags:
        # Пропускаем категориальные (one-hot колонки)
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
    """Компактный waterfall график для эффектов тегов."""
    if not effects:
        return None
    
    labels = ["средний CTR"]
    values = [base_ctr]
    measure = ["absolute"]
    
    running_total = base_ctr
    for tag, val in effects:
        labels.append(display_name(tag))
        values.append(val)
        measure.append("relative")
        running_total += val
    
    labels.append("ожидаемый CTR")
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
    """Ищем похожие, опционально фильтруя по категории."""
    feature_importance = {
        c: np.abs(shap_df[c]).mean() for c in feature_cols if c in shap_df.columns
    }

    # Фильтруем базу по категории
    candidates = df.copy()
    if food_type and food_type != "none":
        candidates = candidates[candidates["food_type"] == food_type]
    elif drink_type and drink_type != "none":
        candidates = candidates[candidates["drink_type"] == drink_type]

    # Если в категории слишком мало креативов — расширяем до всей базы
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
    """Возвращает текст про категорию креатива и средний CTR."""
    base_ctr = df["ctr"].mean()
    insights = []

    # food_type
    food = tags.get("food_type", "none")
    if food != "none":
        segment_df = df[df["food_type"] == food]
        if len(segment_df) >= 5:
            seg_ctr = segment_df["ctr"].mean()
            diff = seg_ctr - base_ctr
            sign = "+" if diff > 0 else ""
            insights.append({
                "label": food.replace("_", " "),
                "category": "тип еды",
                "ctr": seg_ctr,
                "diff": diff,
                "count": len(segment_df),
            })

    # drink_type
    drink = tags.get("drink_type", "none")
    if drink != "none":
        segment_df = df[df["drink_type"] == drink]
        if len(segment_df) >= 5:
            seg_ctr = segment_df["ctr"].mean()
            diff = seg_ctr - base_ctr
            insights.append({
                "label": drink.replace("_", " "),
                "category": "тип напитка",
                "ctr": seg_ctr,
                "diff": diff,
                "count": len(segment_df),
            })

    return insights, base_ctr


def calculate_active_combinations(active_tags, food_type=None, drink_type=None, top_n_each=3):
    """Находит самые сильные комбинации с учётом сегмента."""
    
    # Определяем какой сегмент использовать
    if food_type and food_type != "none":
        segment_filter = (interactions_df["segment_type"] == "food") & \
                        (interactions_df["segment"] == food_type)
    elif drink_type and drink_type != "none":
        segment_filter = (interactions_df["segment_type"] == "drink") & \
                        (interactions_df["segment"] == drink_type)
    else:
        segment_filter = (interactions_df["segment_type"] == "all")
    
    segment_df = interactions_df[segment_filter]
    
    # Если в сегменте мало пар — фоллбэк на all
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
# Рендер блоков
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

    st.markdown('<div class="section-title">Креатив</div>', unsafe_allow_html=True)

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
                <div class="metric-mini-label">CTR похожих (мин)</div>
                <div class="metric-mini-value">{ctr_min:.2f}%</div>
            </div>
            <div class="metric-mini" style="flex:1;">
                <div class="metric-mini-label">медиана</div>
                <div class="metric-mini-value">{ctr_median:.2f}%</div>
            </div>
            <div class="metric-mini" style="flex:1;">
                <div class="metric-mini-label">CTR похожих (макс)</div>
                <div class="metric-mini-value">{ctr_max:.2f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Динамическая подпись с брендом и категорией
        food = tags.get("food_type", "none")
        drink = tags.get("drink_type", "none")
        category_label = food.replace("_", " ") if food != "none" else (drink.replace("_", " ") if drink != "none" else "")
        caption_text = f"Диапазон CTR по похожим креативам бренда <b>{selected_brand}</b>"
        if category_label:
            caption_text += f" в категории <b>{category_label}</b>"

        st.markdown(f'<div style="font-size:12px;color:#888;margin-top:-12px;margin-bottom:18px;">{caption_text}</div>', unsafe_allow_html=True)

        # Категории — зелёная строка
        cat_chips = []
        for cat in CATEGORICAL_TAGS:
            value = tags.get(cat, "none")
            if value != "none":
                cat_chips.append(f'<span class="tag-chip tag-categorical-active">{cat.replace("_"," ")}: {value}</span>')

        if cat_chips:
            st.markdown("<div style='font-size:14px;font-weight:500;margin-bottom:8px;'>Категории</div>", unsafe_allow_html=True)
            st.markdown("".join(cat_chips), unsafe_allow_html=True)
            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        # Теги — синяя строка
        tag_chips = []
        for binary in BINARY_FEATURES:
            if tags.get(binary, False):
                tag_chips.append(f'<span class="tag-chip tag-binary-active">{display_name(binary)}</span>')
        for binary in BINARY_FEATURES:
            if not tags.get(binary, False):
                tag_chips.append(f'<span class="tag-chip tag-absent">{display_name(binary)}</span>')

        st.markdown("<div style='font-size:14px;font-weight:500;margin-bottom:8px;'>Теги</div>", unsafe_allow_html=True)
        st.markdown("".join(tag_chips), unsafe_allow_html=True)
    return active_tags, similar


def render_block_tag_effects(active_tags):
    """Связь тегов с CTR — без чисел и графика, только направление и сила."""
    effects = calculate_tag_effects(active_tags)
    
    # Пороги: подбери под свои данные, если надо
    TAG_STRONG = 0.30  # двойная стрелка
    TAG_WEAK = 0.00001    # одинарая стрелка
    
    categorized = []
    for tag, val in effects:
        direction, strength = categorize_strength(val, TAG_STRONG, TAG_WEAK)
        categorized.append((tag, direction, strength))
    
    st.markdown('<div class="section-title">Связь тегов с CTR</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Какие теги чаще встречаются в креативах с высоким или низким CTR. '
        '<b>↑↑ / ↓↓</b> — сильная связь, <b>↑ / ↓</b> — умеренная. '
        'Это паттерн в данных, не гарантированный эффект.</div>',
        unsafe_allow_html=True,
    )
    
    if not categorized:
        st.markdown(
            "<div style='color:#888;font-size:14px;'>Среди найденных тегов нет значимой связи с CTR</div>",
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
    """Связь пар тегов с CTR — без чисел."""
    positive, negative = calculate_active_combinations(active_tags)
    
    # Пороги для комбинаций — обычно меньше чем для отдельных тегов
    COMBO_STRONG = 0.02
    COMBO_WEAK = 0.001
    
    all_combos = []
    for combo in positive + negative:
        direction, strength = categorize_strength(combo["interaction"], COMBO_STRONG, COMBO_WEAK)
        all_combos.append((combo, direction, strength))
    
    st.markdown('<div class="section-title">Связь пар тегов с CTR</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Какие сочетания тегов в данных бренда связаны с CTR сильнее или слабее, '
        'чем эти теги по отдельности</div>',
        unsafe_allow_html=True,
    )
    
    if not all_combos:
        st.markdown(
            "<div style='color:#888;font-size:14px;'>Среди активных пар нет значимой связи</div>",
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
    st.markdown(
    '<div class="section-subtitle">Какие теги чаще встречаются в креативах с высоким или низким CTR. '
    '<b>↑↑ / ↓↓</b> — сильная связь, <b>↑ / ↓</b> — умеренная, <b>—</b> — нейтральная. '
    'Это паттерн в данных, не гарантированный эффект.</div>',
    unsafe_allow_html=True,
)


def render_block_recommendations(tags):
    X_new = tags_to_features(tags, feature_cols)
    active_tags = get_active_tags(tags)
    recs_add, recs_remove = get_recommendations_full(active_tags, X_new)

    st.markdown('<div class="section-title">Рекомендации</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Что можно изменить, чтобы повысить ожидаемый CTR. '
        'Точный эффект для этого креатива можно посмотреть в А/B сценариях.</div>',
        unsafe_allow_html=True,
    )

    if not recs_add and not recs_remove:
        st.markdown(
            "<div style='color:#1D9E75;font-size:14px;'>Набор тегов выглядит сильным! 💪</div>",
            unsafe_allow_html=True,
        )
    else:
        rows_html = ""
        for feat, _ in recs_add:
            rows_html += f"""
            <div class="rec-row">
                <span class="rec-add">➕ Добавить</span>
                &nbsp;&nbsp;<b>{display_name(feat)}</b> — креативы с этим элементом в среднем перформят лучше
            </div>
            """
        for feat, _ in recs_remove:
            rows_html += f"""
            <div class="rec-row">
                <span class="rec-remove">➖ Убрать</span>
                &nbsp;&nbsp;<b>{display_name(feat)}</b> — креативы без него в среднем перформят лучше
            </div>
            """
        st.markdown(rows_html, unsafe_allow_html=True)

    return recs_add, recs_remove

def build_scenario_prompt(changes):
    """Промпт для генерации по А/B сценарию — берёт явный список изменений."""
    to_add = [display_name(f) for f, _, v in changes if v]      # включить
    to_remove = [display_name(f) for f, _, v in changes if not v]  # выключить
    
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
    """Объединённый блок: А/B + генерация AI-референса по выбранному сценарию."""
    
    st.markdown('<div class="section-title">А/B сценарии + AI-референс</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Поменяй теги — посмотри как изменится связь с CTR, '
        'и сгенерируй визуальный референс по своему сценарию</div>',
        unsafe_allow_html=True,
    )

    # Состояние тогглов
    state_key = f"ab_state_{entry_hash}"
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            feat: bool(tags.get(feat, False)) for feat in BINARY_FEATURES
        }

    # Кнопка сброса
    col_reset, _ = st.columns([1, 4])
    with col_reset:
        if st.button("↻ Сбросить", key=f"reset_{entry_hash}"):
            st.session_state[state_key] = {
                feat: bool(tags.get(feat, False)) for feat in BINARY_FEATURES
            }
            # Очищаем все сгенерированные референсы для этого креатива
            for key in list(st.session_state.keys()):
                if key.startswith(f"ab_generated_{entry_hash}"):
                    del st.session_state[key]
            st.rerun()

    # Тогглы
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

    # Считаем дельту
    original_X = tags_to_features(tags, feature_cols)
    original_ctr = float(model.predict(original_X)[0])

    modified_tags = dict(tags)
    for feat in BINARY_FEATURES:
        modified_tags[feat] = st.session_state[state_key][feat]

    modified_X = tags_to_features(modified_tags, feature_cols)
    modified_ctr = float(model.predict(modified_X)[0])
    diff = modified_ctr - original_ctr

    # Список изменений
    changes = []
    for feat in BINARY_FEATURES:
        original_val = bool(tags.get(feat, False))
        new_val = st.session_state[state_key][feat]
        if original_val != new_val:
            action = "включить" if new_val else "выключить"
            changes.append((feat, action, new_val))

    # Категоризация направления
    AB_STRONG = 0.30
    AB_WEAK = 0.10
    direction, strength = categorize_strength(diff, AB_STRONG, AB_WEAK)

    # Текст и стрелка
    if not changes:
        verdict_html = '<span style="color:#888;font-size:15px;">Ничего не изменено</span>'
        arrow_html = '<span style="color:#b5b3a8;font-size:36px;">—</span>'
    else:
        changes_text = ", ".join(
            f"<b>{action} {display_name(feat)}</b>" for feat, action, _ in changes
        )
        
        if direction == "neutral":
            label = "Связь почти не меняется"
            label_color = "#888"
        elif direction == "up":
            label = "Связь с CTR усиливается" if strength == "strong" else "Связь с CTR немного усиливается"
            label_color = "#1D9E75"
        else:
            label = "Связь с CTR ослабевает" if strength == "strong" else "Связь с CTR немного ослабевает"
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

    # Плашка с результатом
    st.markdown(f"""
    <div style="margin-top:24px;padding:20px;background:#faf9f5;border-radius:12px;">
        <div style="display:flex;align-items:center;gap:24px;">
            <div style="min-width:80px;text-align:center;">{arrow_html}</div>
            <div style="flex:1;">{verdict_html}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # === Генерация референса ===
    
    # Если ничего не менялось — не показываем кнопку, нечего генерить
    if not changes:
        st.markdown(
            "<div style='font-size:13px;color:#888;margin-top:16px;'>"
            "Поменяй какие-нибудь теги выше — и появится возможность сгенерировать визуальный референс."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Уникальный ключ кэша по этому набору изменений
    # — чтобы повторная генерация того же сценария отдавала из кэша
    scenario_signature = "|".join(
        f"{feat}={int(val)}" for feat, _, val in sorted(changes, key=lambda x: x[0])
    )
    scenario_hash = hashlib.md5(scenario_signature.encode()).hexdigest()[:8]
    cache_key = f"ab_generated_{entry_hash}_{scenario_hash}"

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

    if cache_key not in st.session_state:
        if st.button(
            "✨ Сгенерировать визуальный референс по этому сценарию",
            use_container_width=True,
            key=f"gen_btn_{entry_hash}_{scenario_hash}",
        ):
            with st.spinner("Генерируем (~15-30 секунд)..."):
                prompt = build_scenario_prompt(changes)
                image_bytes, error = generate_image_via_openai(prompt, source_image_bytes)
                
                if error:
                    st.error(f"Ошибка генерации: {error}")
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
            st.markdown("<div style='font-size:13px;color:#888;margin-bottom:8px;'>Исходный</div>", unsafe_allow_html=True)
            st.image(source_image_bytes, use_container_width=True)
        with col_new:
            st.markdown("<div style='font-size:13px;color:#888;margin-bottom:8px;'>AI-референс по сценарию</div>", unsafe_allow_html=True)
            st.image(result["image"], use_container_width=True)
        
        st.markdown(
            "<div style='font-size:12px;color:#888;margin-top:12px;line-height:1.6;'>"
            "Этот референс — отправная точка для дизайнера. "
            "Логотип, типографика и точная подача — работа дизайнера."
            "</div>",
            unsafe_allow_html=True,
        )
        
        with st.expander("Полный промпт"):
            st.code(result["prompt"], language=None)
        
        if st.button("🔄 Сгенерировать заново", key=f"regen_{entry_hash}_{scenario_hash}"):
            del st.session_state[cache_key]
            st.rerun()

def render_block_similar(similar):
    st.markdown('<div class="section-title">Похожие креативы из базы</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Креативы с похожим набором тегов, отсортированы по фактическому CTR</div>', unsafe_allow_html=True)

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
                    Открыть →
                </a>
                """, unsafe_allow_html=True)


def render_block_segment_context(tags):
    """Блок инсайт о сегменте — насколько эта категория удачная для бренда."""
    insights, base_ctr = get_segment_context(tags)

    if not insights:
        return

    st.markdown('<div class="section-title">Категория креатива</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Насколько креативы такой же категории сильны у этого бренда</div>',
        unsafe_allow_html=True,
    )

    rows_html = ""
    for ins in insights:
        diff = ins["diff"]
        is_better = diff > 0
        color = "#1D9E75" if is_better else "#E24B4A"
        sign = "+" if is_better else ""

        # Понятная интерпретация
        if abs(diff) < 0.1:
            verdict = "Категория работает примерно как и остальные у этого бренда"
            verdict_color = "#888"
        elif is_better:
            verdict = f"✓ Категория сильнее остальных у этого бренда на {sign}{diff:.2f}%"
            verdict_color = "#1D9E75"
        else:
            verdict = f"✗ Категория слабее остальных у этого бренда на {diff:.2f}%"
            verdict_color = "#E24B4A"

        rows_html += f"""
        <div style="padding:18px 20px;margin-bottom:12px;background:#faf9f5;border-radius:12px;">
            <div style="font-size:18px;font-weight:600;margin-bottom:14px;color:#1a1a1a;text-transform:capitalize;">
                {ins['label']}
            </div>
            <div style="display:flex;gap:24px;margin-bottom:12px;">
                <div>
                    <div style="font-size:11px;color:#888;margin-bottom:2px;">CTR этой категории</div>
                    <div style="font-size:20px;font-weight:600;color:#1a1a1a;">{ins['ctr']:.2f}%</div>
                </div>
                <div>
                    <div style="font-size:11px;color:#888;margin-bottom:2px;">CTR всех креативов бренда</div>
                    <div style="font-size:20px;font-weight:600;color:#1a1a1a;">{base_ctr:.2f}%</div>
                </div>
                <div>
                    <div style="font-size:11px;color:#888;margin-bottom:2px;">Разница</div>
                    <div style="font-size:20px;font-weight:600;color:{color};">{sign}{diff:.2f}%</div>
                </div>
            </div>
            <div style="font-size:14px;color:{verdict_color};font-weight:500;">{verdict}</div>
            <div style="font-size:12px;color:#888;margin-top:6px;">На основе {ins['count']} креативов</div>
        </div>
        """
    st.markdown(rows_html, unsafe_allow_html=True)

# def render_block_generated_alternative(tags, recs_add, recs_remove, entry_hash, source_image_bytes):
#     """Блок: AI-референс на основе исходного креатива."""
#     st.markdown('<div class="section-title">AI-референс альтернативы</div>', unsafe_allow_html=True)
#     st.markdown(
#         '<div class="section-subtitle">Модификация исходного креатива с учётом паттернов библиотеки. '
#         '<b>Референс для дизайнера</b>, не финальный креатив — возможны визуальные артефакты.</div>',
#         unsafe_allow_html=True,
#     )
    
#     cache_key = f"generated_{entry_hash}"
#     prompt = build_generation_prompt(tags, recs_add, recs_remove)
    
#     # Показываем промпт сразу
#     add_list = [display_name(f) for f, _ in recs_add]
#     remove_list = [display_name(f) for f, _ in recs_remove]
    
#     changes_html = ""
#     if add_list:
#         changes_html += (
#             "<div style='margin-bottom:8px;'>"
#             "<span style='color:#1D9E75;font-weight:600;'>➕ Добавить / усилить:</span> "
#             f"{', '.join(f'<b>{t}</b>' for t in add_list)}"
#             "</div>"
#         )
#     if remove_list:
#         changes_html += (
#             "<div style='margin-bottom:8px;'>"
#             "<span style='color:#E24B4A;font-weight:600;'>➖ Убрать / ослабить:</span> "
#             f"{', '.join(f'<b>{t}</b>' for t in remove_list)}"
#             "</div>"
#         )
#     if not changes_html:
#         changes_html = "<div style='color:#888;'>Рекомендуемых изменений нет — креатив и так в хорошем наборе тегов</div>"
    
#     st.markdown(
#         f"""
#         <div style="background:#faf9f5;border-radius:12px;padding:18px 22px;margin-bottom:16px;">
#             <div style="font-size:12px;color:#888;margin-bottom:10px;">
#                 Что попросим у AI поменять в исходном креативе
#             </div>
#             {changes_html}
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )
    
#     # Результат / кнопка
#     if cache_key not in st.session_state:
#         if st.button(
#             "✨ Сгенерировать пример более удачного креатива",
#             use_container_width=True,
#             key=f"gen_btn_{entry_hash}",
#         ):
#             with st.spinner("Генерируем референс (~15-30 секунд)..."):
#                 image_bytes, error = generate_image_via_openai(prompt, source_image_bytes)
                
#                 if error:
#                     st.error(f"Ошибка генерации: {error}")
#                 else:
#                     st.session_state[cache_key] = {
#                         "image": image_bytes,
#                         "prompt": prompt,
#                     }
#                     st.rerun()
#     else:
#         result = st.session_state[cache_key]
        
#         col_orig, col_new = st.columns(2, gap="large")
        
#         with col_orig:
#             st.markdown("<div style='font-size:13px;color:#888;margin-bottom:8px;'>Исходный</div>", unsafe_allow_html=True)
#             st.image(source_image_bytes, use_container_width=True)
        
#         with col_new:
#             st.markdown("<div style='font-size:13px;color:#888;margin-bottom:8px;'>AI-референс</div>", unsafe_allow_html=True)
#             st.image(result["image"], use_container_width=True)
        
#         st.markdown(
#             "<div style='font-size:12px;color:#888;margin-top:12px;line-height:1.6;'>"
#             "Этот референс — отправная точка для дизайнера, не готовый креатив. "
#             "Логотип, типографика и точная подача — работа дизайнера."
#             "</div>",
#             unsafe_allow_html=True,
#         )
        
#         with st.expander("Полный промпт"):
#             st.code(result["prompt"], language=None)
        
#         if st.button("🔄 Другой вариант", key=f"regen_{entry_hash}"):
#             del st.session_state[cache_key]
#             st.rerun()


def render_analysis(entry, selected_brand):
    active_tags, similar = render_block_creative_card(entry, selected_brand)
    divider()
    render_block_tag_effects(active_tags)
    divider()
    render_block_segment_context(entry["tags"])
    divider()
    render_block_combinations(active_tags)
    divider()
    recs_add, recs_remove = render_block_recommendations(entry["tags"])
    divider()
    render_block_ab_scenarios(entry["tags"], entry["hash"], entry["image_bytes"])  
    divider()
    render_block_similar(similar)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### История")
    st.caption(f"Проанализировано: {len(st.session_state.history)}")

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
                if st.button("Открыть", key=f"open_{real_idx}", width='stretch'):
                    st.session_state.selected_history = real_idx
                    st.rerun()
        st.divider()
        if st.button("🗑️ Очистить", width='stretch'):
            st.session_state.history = []
            st.session_state.selected_history = None
            st.rerun()
    else:
        st.caption("Загрузи креатив чтобы начать")


# ============================================================
# MAIN
# ============================================================
st.title("🚀 Анализ креатива")
st.caption("Загрузи рекламный баннер — узнай как визуальные элементы влияют на CTR")

# Дропдаун бренда
all_brands = sorted(df["brand"].unique().tolist())
selected_brand = st.selectbox(
    "Бренд для анализа",
    options=all_brands,
    help="Анализ будет построен на креативах выбранного бренда"
)

# Фильтруем данные
df_filtered = df[df["brand"] == selected_brand].copy()
filenames = df_filtered["filename"].tolist()
shap_df_filtered = shap_df[shap_df["filename"].isin(filenames)].copy()

# Варнинг для маленьких сегментов
if len(df_filtered) < 30:
    st.warning(f"⚠️ У бренда {selected_brand} только {len(df_filtered)} креативов. Анализ может быть неточным.")

# Подменяем глобальные переменные для функций анализа
df = df_filtered
shap_df = shap_df_filtered

st.caption(f"Анализ построен на {len(df_filtered)} креативах бренда **{selected_brand}**")

# Логика страницы
if st.session_state.selected_history is not None:
    entry = st.session_state.history[st.session_state.selected_history]
    if st.button("← Назад к загрузке"):
        st.session_state.selected_history = None
        st.rerun()
    render_analysis(entry, selected_brand)
else:
    uploaded_file = st.file_uploader(
        "Перетащи картинку или нажми чтобы выбрать",
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
            with st.spinner("Анализируем картинку..."):
                tags = tag_image_cached(image_hash, image_bytes, mime)
            entry = {
                "hash": image_hash,
                "name": uploaded_file.name,
                "image_bytes": image_bytes,
                "tags": tags,
            }
            st.session_state.history.append(entry)
            st.rerun()

        render_analysis(entry, selected_brand)