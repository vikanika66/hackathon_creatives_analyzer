"""
Страница библиотеки — галерея креативов из базы с фильтрами и SHAP-разбором.
"""

import streamlit as st
import pandas as pd
import os
from PIL import Image
import base64
from pathlib import Path

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
    page_title="Библиотека — Creative Analyzer",
    page_icon="📚",
    layout="wide",
)

st.markdown(COMMON_CSS, unsafe_allow_html=True)

# ============================================================
# Данные и модель
# ============================================================
df, shap_df, feature_cols, interactions_df = load_data()
model = load_model()
explainer = get_explainer(model)

IMAGES_DIR = "images"

# ============================================================
# Хелперы для рендера разбора креатива
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
    """Категоризирует связь: (направление, сила)."""
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
# Функции расчёта для разбора креатива
# ============================================================


def get_active_tags(tags):
    """Возвращает список активных бинарных тегов + one-hot колонок категорий."""
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
    """Средний SHAP по каждому активному бинарному тегу."""
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
    """Сравнение CTR категории креатива с базой."""
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
                "category": "тип еды",
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
                "category": "тип напитка",
                "ctr": seg_ctr,
                "diff": diff,
                "count": len(segment_df),
            })
    
    return insights, base_ctr


def calculate_active_combinations(active_tags, food_type=None, drink_type=None, top_n_each=3):
    """Топ-3 положительных и топ-3 отрицательных пары среди активных тегов."""
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

# ============================================================
# Функции рендера блоков разбора креатива
# ============================================================

def divider():
    st.markdown('<div style="height:1px;background:#e8e6df;margin:36px 0 28px 0;"></div>', unsafe_allow_html=True)


def render_block_tag_effects(active_tags):
    """Связь тегов с CTR — стрелки вместо чисел."""
    effects = calculate_tag_effects(active_tags)
    
    TAG_STRONG = 0.15
    TAG_WEAK = 0.05
    
    categorized = []
    for tag, val in effects:
        direction, strength = categorize_strength(val, TAG_STRONG, TAG_WEAK)
        categorized.append((tag, direction, strength))
    
    st.markdown('<div style="font-size:20px;font-weight:500;margin-bottom:6px;">Связь тегов с CTR</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:13px;color:#888;margin-bottom:20px;">'
        'Какие теги чаще встречаются в креативах с высоким или низким CTR. '
        '<b>↑↑ / ↓↓</b> — сильная связь, <b>↑ / ↓</b> — умеренная, <b>—</b> — нейтральная. '
        'Это паттерн в данных, не гарантированный эффект.</div>',
        unsafe_allow_html=True,
    )
    
    if not categorized:
        st.markdown(
            "<div style='color:#888;font-size:14px;'>Не нашли значимых эффектов</div>",
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
    """Категория креатива — сравнение с базой."""
    insights, base_ctr = get_segment_context(tags)
    
    if not insights:
        return
    
    st.markdown('<div style="font-size:20px;font-weight:500;margin-bottom:6px;">Категория креатива</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:13px;color:#888;margin-bottom:20px;">'
        'Насколько креативы такой же категории сильны в общей базе</div>',
        unsafe_allow_html=True,
    )
    
    rows_html = ""
    for ins in insights:
        diff = ins["diff"]
        is_better = diff > 0
        color = "#1D9E75" if is_better else "#E24B4A"
        sign = "+" if is_better else ""
        
        if abs(diff) < 0.1:
            verdict = "Категория работает примерно как и остальные"
            verdict_color = "#888"
        elif is_better:
            verdict = f"✓ Категория сильнее остальных на {sign}{diff:.2f}%"
            verdict_color = "#1D9E75"
        else:
            verdict = f"✗ Категория слабее остальных на {diff:.2f}%"
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
                    <div style="font-size:11px;color:#888;margin-bottom:2px;">CTR всех креативов</div>
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


def render_block_combinations(active_tags, tags):
    """Связь пар тегов с CTR — стрелки вместо чисел."""
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
    
    st.markdown('<div style="font-size:20px;font-weight:500;margin-bottom:6px;">Связь пар тегов с CTR</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:13px;color:#888;margin-bottom:20px;">'
        'Какие сочетания тегов связаны с CTR сильнее или слабее, чем эти теги по отдельности. '
        '<b>↑↑ / ↓↓</b> — сильная, <b>↑ / ↓</b> — умеренная, <b>—</b> — нейтральная.</div>',
        unsafe_allow_html=True,
    )
    
    if not all_combos:
        st.markdown(
            "<div style='color:#888;font-size:14px;'>Не нашли значимых комбинаций для этого набора тегов</div>",
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


# ============================================================
# Читаем query-параметр от Insights — если есть, открываем нужный креатив
# ============================================================

preselected = st.query_params.get("creative")
if preselected:
    matches = df.index[df["filename"] == preselected].tolist()
    if matches:
        st.session_state["selected_creative"] = int(matches[0])
    # очищаем query param чтобы при F5 не залипало
    st.query_params.clear()

# ============================================================
# session_state — какой креатив выбран
# ============================================================
if "selected_creative" not in st.session_state:
    st.session_state.selected_creative = None

# ============================================================
# Функция: для нормального отображения картинки
# ============================================================

@st.cache_data
def encode_image(path):
    """Конвертит картинку в data URL для inline HTML."""
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
    """Показывает разбор креатива из библиотеки (как в Upload, без прогнозов)."""
    row = df.iloc[idx]
    filename = row["filename"]
    image_path = os.path.join(IMAGES_DIR, filename)
    
    # === Шапка: картинка + базовая инфа ===
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
            st.info(f"Картинка не найдена: {filename}")
    
    with col_info:
        actual_ctr = row["ctr"]
        impressions = row["impressions"]
        brand = row.get("brand", "—")
        year = row.get("year", "—")
        
        # Сравнение с базой
        base_ctr = df["ctr"].mean()
        diff = actual_ctr - base_ctr
        diff_color = "#1D9E75" if diff > 0 else "#E24B4A"
        diff_sign = "+" if diff > 0 else ""
        
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:18px;">
            <div style="background:#faf9f5;border-radius:10px;padding:14px 18px;">
                <div style="font-size:11px;color:#888;margin-bottom:4px;">Фактический CTR</div>
                <div style="font-size:24px;font-weight:600;color:#1a1a1a;">{actual_ctr:.2f}%</div>
                <div style="font-size:12px;color:{diff_color};margin-top:4px;">
                    {diff_sign}{diff:.2f}% относительно базы ({base_ctr:.2f}%)
                </div>
            </div>
            <div style="background:#faf9f5;border-radius:10px;padding:14px 18px;">
                <div style="font-size:11px;color:#888;margin-bottom:4px;">Показов</div>
                <div style="font-size:24px;font-weight:600;color:#1a1a1a;">{impressions:,.0f}</div>
                <div style="font-size:12px;color:#888;margin-top:4px;">
                    {brand} · {year}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Собираем теги
        tags = {}
        for cat in CATEGORICAL_TAGS:
            tags[cat] = row[cat]
        for binary in BINARY_FEATURES:
            if binary in row.index:
                tags[binary] = bool(row[binary])
        
        st.markdown("<div style='font-size:14px;font-weight:500;margin-bottom:8px;'>Теги креатива</div>", unsafe_allow_html=True)
        st.markdown(render_tag_chips(tags), unsafe_allow_html=True)
    
    # === Активные теги для аналитики ===
    active_tags = get_active_tags(tags)
    
    # === Блок 1: Связь тегов с CTR ===
    divider()
    render_block_tag_effects(active_tags)
    
    # === Блок 2: Категория креатива ===
    divider()
    render_block_segment_context(tags)
    
    # === Блок 3: Сочетания тегов ===
    divider()
    render_block_combinations(active_tags, tags)


# ============================================================
# UI: Заголовок
# ============================================================
st.title("📚 Библиотека креативов")
st.caption(f"{len(df)} креативов с фактическим CTR и AI-разметкой")

# === РЕЖИМ: ПРОСМОТР КРЕАТИВА ===
if st.session_state.selected_creative is not None:
    if st.button("← Назад к библиотеке"):
        st.session_state.selected_creative = None
        st.rerun()

    render_library_creative(st.session_state.selected_creative)

# === РЕЖИМ: ГАЛЕРЕЯ ===
else:
    # ============================================================
    # Фильтры в сайдбаре
    # ============================================================
    with st.sidebar:
        st.markdown("### Фильтры")

        # Бренд
        all_brands = ["Все"] + sorted(df["brand"].dropna().unique().tolist())
        selected_brand = st.selectbox("Бренд", all_brands)

        # Год
        all_years = ["Все"] + sorted(df["year"].dropna().unique().tolist())
        selected_year = st.selectbox("Год", all_years)

        st.divider()


        # Фильтр по типу
        main_objects = ["Все"] + sorted(df["main_object"].dropna().unique().tolist())
        selected_main = st.selectbox("Главный объект", main_objects)

        food_types = ["Все"] + sorted(df["food_type"].dropna().unique().tolist())
        selected_food = st.selectbox("Тип еды", food_types)

        drink_types = ["Все"] + sorted(df["drink_type"].dropna().unique().tolist())
        selected_drink = st.selectbox("Тип напитка", drink_types)

        st.divider()

        # Бинарные фильтры
        st.markdown("**Должны быть**")
        must_have = []
        for feat in BINARY_FEATURES:
            if st.checkbox(feat.replace("_", " "), key=f"mh_{feat}"):
                must_have.append(feat)

        st.divider()

        # Сортировка
        sort_by = st.selectbox("Сортировать по", ["CTR (убыв.)", "CTR (возр.)", "Показы"])

    # ============================================================
    # Применяем фильтры
    # ============================================================
    filtered = df.copy()
    if selected_brand != "Все":
        filtered = filtered[filtered["brand"] == selected_brand]
    if selected_year != "Все":
        filtered = filtered[filtered["year"] == selected_year]

    if selected_main != "Все":
        filtered = filtered[filtered["main_object"] == selected_main]

    if selected_main != "Все":
        filtered = filtered[filtered["main_object"] == selected_main]
    if selected_food != "Все":
        filtered = filtered[filtered["food_type"] == selected_food]
    if selected_drink != "Все":
        filtered = filtered[filtered["drink_type"] == selected_drink]

    for feat in must_have:
        filtered = filtered[filtered[feat] == True]

    if sort_by == "CTR (убыв.)":
        filtered = filtered.sort_values("ctr", ascending=False)
    elif sort_by == "CTR (возр.)":
        filtered = filtered.sort_values("ctr", ascending=True)
    else:
        filtered = filtered.sort_values("impressions", ascending=False)

    st.markdown(f"**Найдено: {len(filtered)} креативов**")

    if len(filtered) == 0:
        st.info("Ничего не найдено — попробуй ослабить фильтры")
    else:
        # ============================================================
        # Пагинация
        # ============================================================
        # ============================================================
        # Пагинация
        # ============================================================
        items_per_page = 24
        total_pages = (len(filtered) - 1) // items_per_page + 1
        
        # Уникальный ключ для хранения текущей страницы по фильтрам
        # (чтобы при смене фильтров пагинация сбрасывалась)
        filter_key = f"{selected_brand}_{selected_year}_{selected_main}_{'_'.join(must_have)}_{sort_by}"
        page_state_key = f"library_page_{filter_key}"
        
        if page_state_key not in st.session_state:
            st.session_state[page_state_key] = 0
        
        # Безопасный коридор: если из-за смены фильтров текущая страница больше total_pages
        page = min(st.session_state[page_state_key], total_pages - 1)
        st.session_state[page_state_key] = page
        
        start = page * items_per_page
        end = start + items_per_page
        page_data = filtered.iloc[start:end]
        
        # Навигация сверху галереи
        if total_pages > 1:
            nav_cols = st.columns([1, 1, 3, 1, 1])
            
            with nav_cols[0]:
                if st.button("← Назад", disabled=(page == 0), use_container_width=True, key="page_prev"):
                    st.session_state[page_state_key] = page - 1
                    st.rerun()
            
            with nav_cols[2]:
                st.markdown(
                    f"<div style='text-align:center;padding-top:6px;font-size:14px;color:#666;'>"
                    f"Страница <b style='color:#1a1a1a;'>{page + 1}</b> из <b style='color:#1a1a1a;'>{total_pages}</b> "
                    f"<span style='color:#bbb;'>· показано {len(page_data)} из {len(filtered)}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            
            with nav_cols[4]:
                if st.button("Вперёд →", disabled=(page >= total_pages - 1), use_container_width=True, key="page_next"):
                    st.session_state[page_state_key] = page + 1
                    st.rerun()
        page_data = filtered.iloc[start:end]

        # ============================================================
        # Галерея — 4 колонки
        # ============================================================
        cols_per_row = 4
        rows = [page_data.iloc[i:i+cols_per_row] for i in range(0, len(page_data), cols_per_row)]

        for row_data in rows:
            cols = st.columns(cols_per_row)
            for col, (_, item) in zip(cols, row_data.iterrows()):
                with col:
                    image_path = os.path.join(IMAGES_DIR, item["filename"])
                    # st.write(f"DEBUG path: {image_path}")
                    data_url = encode_image(image_path)
                    # st.write(f"DEBUG data_url: {data_url[:50] if data_url else 'None'}")

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
                            'нет картинки</div>',
                            unsafe_allow_html=True
                        )

                    # CTR badge
                    ctr_color = "#1D9E75" if item["ctr"] > df["ctr"].median() else "#666"
                    st.markdown(f"""
                    <div style="font-size:18px;font-weight:600;color:{ctr_color};margin-top:4px;">
                        {item['ctr']:.2f}%
                    </div>
                    <div style="font-size:11px;color:#888;">
                        {item['main_object']} · {item['impressions']:,.0f} показов
                    </div>
                    """, unsafe_allow_html=True)

                    # Кнопка открыть
                    real_idx = filtered.index.get_loc(item.name) + start
                    df_idx = item.name
                    if st.button("Разобрать", key=f"open_{df_idx}", width='stretch'):
                        st.session_state.selected_creative = df_idx
                        st.rerun()

        # Навигация снизу галереи (дублирует верхнюю)
        if total_pages > 1:
            st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
            nav_cols = st.columns([1, 1, 3, 1, 1])
            
            with nav_cols[0]:
                if st.button("← Назад", disabled=(page == 0), use_container_width=True, key="page_prev_bottom"):
                    st.session_state[page_state_key] = page - 1
                    st.rerun()
            
            with nav_cols[2]:
                st.markdown(
                    f"<div style='text-align:center;padding-top:6px;font-size:14px;color:#666;'>"
                    f"Страница <b style='color:#1a1a1a;'>{page + 1}</b> из <b style='color:#1a1a1a;'>{total_pages}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            
            with nav_cols[4]:
                if st.button("Вперёд →", disabled=(page >= total_pages - 1), use_container_width=True, key="page_next_bottom"):
                    st.session_state[page_state_key] = page + 1
                    st.rerun()