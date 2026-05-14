"""
Страница Insights — готовые аналитические выводы по бренду.
Для маркетологов, клиентов и дизайнеров.
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
# Стили (как в Upload)
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


# ============================================================
# Данные
# ============================================================
df, shap_df, feature_cols, interactions_df = load_data()


# Читаемые имена тегов
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
    """Категоризирует связь: (направление, сила)."""
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
# UI: Заголовок и фильтры
# ============================================================
st.title("📊 Insights")
st.caption("Готовые аналитические выводы по выбранному бренду — для маркетинга, клиентов и дизайнеров")

# Фильтры
col_brand, col_year = st.columns([1, 1])

with col_brand:
    all_brands = sorted(df["brand"].unique().tolist())
    selected_brand = st.selectbox(
        "Бренд",
        options=all_brands,
        help="Выбери бренд для аналитики"
    )

with col_year:
    year_options = ["Все годы"] + sorted(df["year"].unique().tolist())
    selected_year = st.selectbox(
        "Год",
        options=year_options,
        help="Опционально — фильтр по году"
    )

# Фильтруем данные
df_brand = df[df["brand"] == selected_brand].copy()
if selected_year != "Все годы":
    df_brand = df_brand[df_brand["year"] == int(selected_year)]

# Соответствующие SHAP данные
filenames = df_brand["filename"].tolist()
shap_brand = shap_df[shap_df["filename"].isin(filenames)].copy()

# Базовые данные для сравнения (вся база)
df_all = df.copy()
if selected_year != "Все годы":
    df_all = df_all[df_all["year"] == int(selected_year)]

if len(df_brand) < 10:
    st.warning(f"⚠️ В выбранном сегменте только {len(df_brand)} креативов. Аналитика может быть неточной.")

st.caption(f"Аналитика на основе **{len(df_brand)}** креативов бренда **{selected_brand}**" + 
           (f" за **{selected_year}**" if selected_year != "Все годы" else ""))


# ============================================================
# БЛОК 1: Общая статистика бренда
# ============================================================
divider()

st.markdown('<div class="section-title">Общая статистика</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Как бренд выглядит относительно всей базы</div>', unsafe_allow_html=True)

# Считаем метрики
brand_avg_ctr = df_brand["ctr"].mean()
brand_median_ctr = df_brand["ctr"].median()
brand_max_ctr = df_brand["ctr"].max()
brand_min_ctr = df_brand["ctr"].min()
brand_count = len(df_brand)

industry_avg_ctr = df_all["ctr"].mean()
ctr_diff = brand_avg_ctr - industry_avg_ctr

# Сравнение со средним
diff_color = "#1D9E75" if ctr_diff > 0 else "#E24B4A"
diff_sign = "+" if ctr_diff > 0 else ""
diff_text = f"{diff_sign}{ctr_diff:.2f}% относительно всех CTR ({industry_avg_ctr:.2f}%)"

st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(4, 1fr);gap:12px;">
    <div class="stat-box">
        <div class="stat-label">Креативов</div>
        <div class="stat-value">{brand_count}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Средний CTR</div>
        <div class="stat-value">{brand_avg_ctr:.2f}%</div>
        <div class="stat-comparison" style="color:{diff_color};">{diff_text}</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Медианный CTR</div>
        <div class="stat-value">{brand_median_ctr:.2f}%</div>
    </div>
    <div class="stat-box">
        <div class="stat-label">Диапазон</div>
        <div class="stat-value" style="font-size:18px;">{brand_min_ctr:.2f}% — {brand_max_ctr:.2f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# БЛОК 2: Топ-5 сильных и слабых тегов
# ============================================================
divider()

st.markdown('<div class="section-title">Что работает у бренда</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Как часто каждый тег встречается в креативах с высоким или низким CTR. '
    'Это паттерн в данных, не гарантированный эффект.</div>',
    unsafe_allow_html=True
)

# Пороги
TAG_STRONG = 0.30
TAG_WEAK = 0.01

# Считаем средний SHAP для каждого тега когда он активен
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
    """kind: 'strong' или 'weak'"""
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
        <span style="font-size:11px;color:#888;">{row['count']} креативов</span>
    </div>
    """


TOP_N = 5

# Кандидаты в каждую колонку — строго по знаку и не ниже порога WEAK
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
        "✓ Сильные теги</div>",
        unsafe_allow_html=True,
    )
    if len(strong_candidates) == 0:
        st.markdown(
            "<div style='color:#888;font-size:14px;padding:10px 0;'>"
            "У бренда нет тегов, которые заметно связаны с высоким CTR</div>",
            unsafe_allow_html=True,
        )
    else:
        rows_html = "".join(render_tag_row(row, "strong") for _, row in strong_candidates.iterrows())
        st.markdown(rows_html, unsafe_allow_html=True)

with col_weak:
    st.markdown(
        "<div style='font-size:16px;font-weight:600;margin-bottom:12px;color:#E24B4A;'>"
        "✗ Слабые теги</div>",
        unsafe_allow_html=True,
    )
    if len(weak_candidates) == 0:
        st.markdown(
            "<div style='color:#888;font-size:14px;padding:10px 0;'>"
            "У бренда нет тегов, которые заметно связаны с низким CTR</div>",
            unsafe_allow_html=True,
        )
    else:
        rows_html = "".join(render_tag_row(row, "weak") for _, row in weak_candidates.iterrows())
        st.markdown(rows_html, unsafe_allow_html=True)


# ============================================================
# БЛОК 3: Карта взаимодействий тегов
# ============================================================
divider()

st.markdown('<div class="section-title">Карта взаимодействий тегов</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Какие пары тегов чаще встречаются вместе в успешных или слабых креативах. '
    '<b>↑↑ / ↓↓</b> — сильная связь, <b>↑ / ↓</b> — умеренная, пусто — нейтральная.</div>',
    unsafe_allow_html=True
)

INT_STRONG = 0.05
INT_WEAK = 0.01

all_binary = BINARY_FEATURES
n = len(all_binary)

# Строим матрицу значений
matrix = np.zeros((n, n))
seg_filter = interactions_df["segment_type"] == "all"
seg_data = interactions_df[seg_filter]

for _, row in seg_data.iterrows():
    if row["feat1"] in all_binary and row["feat2"] in all_binary:
        i = all_binary.index(row["feat1"])
        j = all_binary.index(row["feat2"])
        matrix[i][j] = row["interaction"]
        matrix[j][i] = row["interaction"]

# Конвертим в категории: -2, -1, 0, 1, 2
def categorize_cell(val):
    abs_v = abs(val)
    if abs_v < INT_WEAK:
        return 0
    if val > 0:
        return 2 if abs_v >= INT_STRONG else 1
    return -2 if abs_v >= INT_STRONG else -1

cat_matrix = np.vectorize(categorize_cell)(matrix)

# Символы для отображения
SYMBOLS = {-2: "↓↓", -1: "↓", 0: "", 1: "↑", 2: "↑↑"}
text_matrix = np.vectorize(lambda x: SYMBOLS[int(x)])(cat_matrix)

# Диагональ — пустые клетки без символа (тег сам с собой не сравнивается)
for i in range(n):
    text_matrix[i][i] = ""
    cat_matrix[i][i] = 0  # нейтральный белый цвет

labels = [display_name(t) for t in all_binary]

# Дискретная цветовая шкала
fig = go.Figure(data=go.Heatmap(
    z=cat_matrix,
    x=labels,
    y=labels,
    text=text_matrix,
    texttemplate="%{text}",
    textfont={"size": 16, "color": "#1a1a1a"},
    colorscale=[
        [0.0,  "#e8506e"],   # -2 сильное ослабление
        [0.25, "#f5b0bc"],   # -1 слабое
        [0.5,  "#ffffff"],   # 0 нейтрально
        [0.75, "#d5f0b0"],   # +1 слабое
        [1.0,  "#aef33e"],   # +2 сильное
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
💡 <b>Как читать:</b> Зелёные клетки со стрелками вверх — пара тегов чаще встречается в креативах с высоким CTR. 
Красные со стрелками вниз — в креативах с низким CTR. 
Пустые клетки — связь нейтральная или пары встречаются редко.
</div>
""", unsafe_allow_html=True)

# ============================================================
# БЛОК 4: Лучшие и худшие креативы бренда
# ============================================================
divider()

st.markdown('<div class="section-title">Лучшие и худшие креативы</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Топ-5 перформеров и аутсайдеров бренда — учимся на своих кейсах</div>', unsafe_allow_html=True)

# Путь к картинкам — поправь если у тебя другая папка
IMAGES_DIR = "images"

import os

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

# def render_creative_card(row, rank, tone):
#     """Карточка одного креатива. Подпись сверху, картинка снизу — так высота картинок не ломает выравнивание."""
#     color = "#1D9E75" if tone == "good" else "#E24B4A"
#     sign = "🏆" if tone == "good" else "⚠️"
    
#     # Собираем активные теги (макс 4 для компактности)
#     active_tags = []
#     for feat in BINARY_FEATURES:
#         if feat in row.index and row[feat] == True:
#             active_tags.append(display_name(feat))
#         if len(active_tags) >= 4:
#             break
    
#     tags_html = " · ".join(f"<span style='color:#666;'>{t}</span>" for t in active_tags) or "<span style='color:#bbb;'>—</span>"
    
#     # Подпись СВЕРХУ — фиксированная высота, чтобы выровнять начало картинок
#     st.markdown(f"""
#     <div style="min-height:60px;margin-bottom:8px;">
#         <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
#             <span style="font-size:14px;">{sign}</span>
#             <span style="color:{color};font-weight:600;font-size:16px;">{row['ctr']:.2f}%</span>
#             <span style="font-size:11px;color:#888;">#{rank}</span>
#         </div>
#         <div style="font-size:11px;line-height:1.5;">{tags_html}</div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Картинка снизу
#     image_path = os.path.join(IMAGES_DIR, row["filename"])
#     try:
#         st.image(image_path, use_container_width=True)
#     except Exception:
#         st.markdown(
#             "<div style='aspect-ratio:1;background:#f5f3ed;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#bbb;font-size:12px;'>нет картинки</div>",
#             unsafe_allow_html=True
#         )

def render_creative_card(row, rank, tone):
    color = "#1D9E75" if tone == "good" else "#E24B4A"
    sign = "🏆" if tone == "good" else "⚠️"
    
    # Активные теги (макс 4)
    active_tags = []
    for feat in BINARY_FEATURES:
        if feat in row.index and row[feat] == True:
            active_tags.append(display_name(feat))
        if len(active_tags) >= 4:
            break
    
    tags_html = " · ".join(f"<span style='color:#666;'>{t}</span>" for t in active_tags) or "<span style='color:#bbb;'>—</span>"
    
    # Картинка в base64 — contain, чтобы было целиком
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
            'нет картинки</div>'
        )
    
    # Карточка одной строкой, без обёртки <a>
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
    
    # Кнопка-ссылка под карточкой
    st.link_button(
        "Открыть в Library →",
        f"/Library?creative={quote(str(row['filename']))}",
        use_container_width=True,
    )


df_sorted = df_brand.sort_values("ctr", ascending=False).reset_index(drop=True)
top_5 = df_sorted.head(5)
bottom_5 = df_sorted.tail(5).iloc[::-1].reset_index(drop=True)

# Топ-5
st.markdown("<div style='font-size:15px;font-weight:600;margin:8px 0 12px 0;color:#1D9E75;'>Лучшие 5</div>", unsafe_allow_html=True)
cols = st.columns(5, gap="small")
for i, col in enumerate(cols):
    if i < len(top_5):
        with col:
            render_creative_card(top_5.iloc[i], i + 1, "good")

# Худшие 5
st.markdown("<div style='font-size:15px;font-weight:600;margin:24px 0 12px 0;color:#E24B4A;'>Худшие 5</div>", unsafe_allow_html=True)
cols = st.columns(5, gap="small")
for i, col in enumerate(cols):
    if i < len(bottom_5):
        with col:
            render_creative_card(bottom_5.iloc[i], i + 1, "bad")


# ============================================================
# БЛОК 5: Аномалии — креативы выпадающие из паттерна тегов
# ============================================================
divider()

st.markdown('<div class="section-title">Аномалии: креативы выпадающие из паттерна тегов</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Креативы, у которых CTR заметно отличается от других креативов '
    'с похожим набором тегов. Подсказывают: тут работает (или не работает) что-то <b>помимо</b> видимых элементов.</div>',
    unsafe_allow_html=True
)

# Мерджим predicted_ctr в df_brand
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
        comparison_label = "CTR выше похожих"
    else:
        color = "#E24B4A"
        icon = "📉"
        label = "Underperformer"
        comparison_label = "CTR ниже похожих"

    # Активные теги (макс 3)
    active_tags = []
    for feat in BINARY_FEATURES:
        if feat in row.index and row[feat] == True:
            active_tags.append(display_name(feat))
        if len(active_tags) >= 3:
            break

    tags_html = " · ".join(f"<span style='color:#666;'>{t}</span>" for t in active_tags) or "<span style='color:#bbb;'>—</span>"

    # Картинка
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
            'нет картинки</div>'
        )

    # Стрелка по силе аномалии
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
        "Открыть в Library →",
        f"/Library?creative={quote(str(row['filename']))}",
        use_container_width=True,
    )


# Hidden gems
st.markdown(
    "<div style='font-size:15px;font-weight:600;margin:8px 0 12px 0;color:#1D9E75;'>"
    "💎 Hidden gems — CTR выше похожих по тегам</div>",
    unsafe_allow_html=True,
)
if len(hidden_gems) == 0:
    st.info("Недостаточно данных для определения hidden gems в этом сегменте")
else:
    cols = st.columns(3, gap="small")
    for i, col in enumerate(cols):
        if i < len(hidden_gems):
            with col:
                render_anomaly_card(hidden_gems.iloc[i], "gem")
    st.markdown(
        "<div style='font-size:13px;color:#666;margin:12px 0 24px 0;'>"
        "💡 У этих креативов CTR заметно выше, чем у других с похожим набором тегов. "
        "Это значит, что работает что-то <b>помимо</b> видимых тегов — копирайт, эмоция, исполнение. "
        "Стоит разобрать руками: что в них особенного, что нельзя описать тегом?"
        "</div>",
        unsafe_allow_html=True
    )

# Underperformers
st.markdown(
    "<div style='font-size:15px;font-weight:600;margin:8px 0 12px 0;color:#E24B4A;'>"
    "📉 Underperformers — CTR ниже похожих по тегам</div>",
    unsafe_allow_html=True,
)
if len(underperformers) == 0:
    st.info("Недостаточно данных для определения underperformers в этом сегменте")
else:
    cols = st.columns(3, gap="small")
    for i, col in enumerate(cols):
        if i < len(underperformers):
            with col:
                render_anomaly_card(underperformers.iloc[i], "underperformer")
    st.markdown(
        "<div style='font-size:13px;color:#666;margin-top:12px;'>"
        "💡 У этих креативов CTR заметно ниже, чем у других с похожим набором тегов. "
        "Концепция (набор тегов) похожа на сильные креативы, но <b>исполнение</b> подкачало — "
        "качество фотографии, нечитаемый текст, слабый CTA, неудачная композиция."
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# БЛОК 6: Тренд по годам (не зависит от фильтра по году)
# ============================================================
divider()

st.markdown('<div class="section-title">Тренд по годам</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Как менялась позиция бренда в индустрии и какие теги становились важнее. '
    '<b>Фильтр по году не применяется</b> — тренд всегда по всем годам бренда.</div>',
    unsafe_allow_html=True,
)

# Игнорируем фильтр по году — тренд считаем по всем данным бренда
df_brand_all_years = df[df["brand"] == selected_brand].copy()
filenames_all = df_brand_all_years["filename"].tolist()
shap_brand_all = shap_df[shap_df["filename"].isin(filenames_all)].copy()

years_available = sorted(df_brand_all_years["year"].unique().tolist())

if len(years_available) < 2:
    st.info(
        f"У бренда {selected_brand} креативы только за один год "
        f"({years_available[0] if years_available else '—'}). Тренд построить нельзя."
    )
else:
    # === Часть 1: Позиция бренда в индустрии ===
    st.markdown(
        "<div style='font-size:15px;font-weight:600;margin:8px 0 12px 0;'>"
        "Позиция бренда в индустрии</div>",
        unsafe_allow_html=True,
    )
    
    # Считаем позицию бренда (перцентиль) в каждом году
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
        if p >= 75: return ("Топ индустрии", "🏆", "#1D9E75")
        if p >= 50: return ("Выше среднего", "📈", "#7ABE5C")
        if p >= 25: return ("Ниже среднего", "📊", "#E8A24A")
        return ("Нижний сегмент", "⚠️", "#E24B4A")
    
    # Берём предпоследний и последний год — сравниваем актуальную динамику
    first_row = position_df.iloc[-2]
    last_row = position_df.iloc[-1]
    
    first_label, first_icon, first_color = position_label(first_row["percentile"])
    last_label, last_icon, last_color = position_label(last_row["percentile"])
    
    # CTR бренда и индустрии для каждого года
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
            return f'<span style="color:#1D9E75;font-weight:600;">+{diff:.2f}%</span> над индустрией'
        elif diff < 0:
            return f'<span style="color:#E24B4A;font-weight:600;">{diff:.2f}%</span> к индустрии'
        else:
            return '<span style="color:#888;font-weight:600;">на уровне индустрии</span>'
    
    # Направление изменения
    diff_p = last_row["percentile"] - first_row["percentile"]
    if diff_p > 10:
        trend_arrow = '<span style="color:#1D9E75;font-size:42px;font-weight:600;">↑</span>'
        trend_text = "укрепил позиции"
        trend_color = "#1D9E75"
    elif diff_p < -10:
        trend_arrow = '<span style="color:#E24B4A;font-size:42px;font-weight:600;">↓</span>'
        trend_text = "сдал позиции"
        trend_color = "#E24B4A"
    else:
        trend_arrow = '<span style="color:#888;font-size:42px;">→</span>'
        trend_text = "позиция стабильна"
        trend_color = "#888"
    
    # Плашки: было → стрелка → стало
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 80px 1fr;gap:12px;align-items:center;margin-bottom:16px;">
        <div style="background:#faf9f5;border-radius:12px;padding:24px;text-align:center;">
            <div style="font-size:12px;color:#888;margin-bottom:8px;">{first_year_int}</div>
            <div style="font-size:42px;margin-bottom:8px;">{first_icon}</div>
            <div style="font-size:18px;font-weight:600;color:{first_color};margin-bottom:14px;">{first_label}</div>
            <div style="font-size:13px;color:#666;line-height:1.6;padding-top:12px;border-top:1px solid #eae8df;">
                <div style="margin-bottom:4px;">
                    CTR бренда: <b style="color:#1a1a1a;">{first_brand_ctr:.2f}%</b>
                </div>
                <div style="margin-bottom:4px;color:#888;">
                    Индустрия: {first_ind_ctr:.2f}%
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
                    CTR бренда: <b style="color:#1a1a1a;">{last_brand_ctr:.2f}%</b>
                </div>
                <div style="margin-bottom:4px;color:#888;">
                    Индустрия: {last_ind_ctr:.2f}%
                </div>
                <div style="margin-top:6px;font-size:13px;">
                    {diff_html(last_diff)}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Итоговая фраза
    st.markdown(
        f"<div style='font-size:14px;color:#1a1a1a;margin:8px 0 24px 0;'>"
        f"С {first_year_int} по {last_year_int} бренд "
        f"<b style='color:{trend_color};'>{trend_text}</b> в индустрии."
        f"</div>",
        unsafe_allow_html=True,
    )
    
    # === Часть 2: Что изменилось в работе тегов ===
    first_year = years_available[-2]
    last_year = years_available[-1]
    
    st.markdown(
        f"<div style='font-size:15px;font-weight:600;margin:24px 0 4px 0;'>"
        f"Что изменилось: {first_year} → {last_year}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:13px;color:#888;margin-bottom:12px;'>"
        "Какие теги стали сильнее или слабее связаны с CTR. Стрелки показывают направление изменения.</div>",
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
            f"Недостаточно данных для сравнения {first_year} и {last_year}. "
            "Нужно минимум 2 креатива с каждым тегом в каждом из годов."
        )
    else:
        DELTA_STRONG = 0.20
        DELTA_WEAK = 0.05
        
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
        
        # Краткий вывод
        growth = deltas_df[deltas_df["delta"] >= DELTA_WEAK]
        decline = deltas_df[deltas_df["delta"] <= -DELTA_WEAK]
        
        summary_html = "<div style='font-size:13px;color:#666;margin-top:16px;line-height:1.7;'>"
        if len(growth) > 0:
            top_growth_tags = ", ".join(f"<b>{display_name(t)}</b>" for t in growth.head(3)["tag"])
            summary_html += f"📈 <b style='color:#1D9E75;'>Усилили влияние:</b> {top_growth_tags}<br>"
        if len(decline) > 0:
            top_decline_tags = ", ".join(f"<b>{display_name(t)}</b>" for t in decline.tail(3)["tag"])
            summary_html += f"📉 <b style='color:#E24B4A;'>Потеряли вес:</b> {top_decline_tags}"
        summary_html += "</div>"
        st.markdown(summary_html, unsafe_allow_html=True)