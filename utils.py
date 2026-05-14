"""
Общие функции и константы для всех страниц приложения.
Импортируется в каждой странице.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import base64
import io
from openai import OpenAI
import shap
import plotly.graph_objects as go
from PIL import Image


# ============================================================
# КОНСТАНТЫ
# ============================================================

# Промпт для тегирования
TAGGING_PROMPT = """You are analyzing a food/beverage advertising creative (banner ad).
Look at this image carefully and classify it using the following tags.
Return ONLY a valid JSON object, no markdown, no explanation.

Rules:
- For categorical tags: pick exactly ONE value from the allowed list, or "none" if not applicable.
- For binary tags: answer true or false.
- If there is no food visible, set food_type to "none".
- If there is no drink visible, set drink_type to "none".
- If there are no people visible, set person_who, person_emotion, person_action to "none".
- Classify by visual appearance, not literal text. Non-alcoholic beer, 0% beer, alcohol-free wine should be tagged as "alcohol".
- main_object: choose what dominates the image. A bottle of soda = "drink", a burger = "food", a person holding food = "person".
- has_person: ONLY real human people count. Cartoon characters, mascots, illustrated figures, brand characters, and animals = false.
- person_who: "woman" = one woman, "man" = one man, "multiple" = more than one person.
- clean_background: ANY simple solid color or gradient counts as clean. Only false if there are real-world objects, rooms, scenes, or people visible behind the main subject.
- cta: true if there is any call to action — buttons, "Order now", delivery service logos with action text, or any prompt encouraging action.

{
  "main_object": "food | drink | person",
  "food_type": "fast_food | dessert | healthy | main_dish | snack | none",
  "drink_type": "coffee_tea | soda_juice | alcohol | water_energy | none",
  "has_person": true/false,
  "person_who": "woman | man | multiple | none",
  "person_emotion": "positive | neutral | none",
  "person_action": "eating | cooking | posing | none",
  "close_up": true/false,
  "top_down": true/false,
  "bright_colors": true/false,
  "warm_tones": true/false,
  "dark_moody": true/false,
  "clean_background": true/false,
  "packaged": true/false,
  "text_overlay": true/false,
  "price_discount": true/false,
  "brand_logo": true/false,
  "cta": true/false,
  "multiple_items": true/false
}"""

# Бинарные фичи (для рекомендаций и преобразований)
BINARY_FEATURES = [
    "has_person", "close_up", "top_down", "warm_tones",
    "clean_background", "packaged", "text_overlay",
    "price_discount", "brand_logo", "cta", "multiple_items"
]

# Категориальные теги (для one-hot)
CATEGORICAL_TAGS = [
    "main_object", "food_type", "drink_type",
    "person_who", "person_emotion", "person_action"
]


# ============================================================
# ОБЩИЕ СТИЛИ
# ============================================================
COMMON_CSS = """
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1300px;
    }
    .metric-box {
        background: #f5f5f0;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 12px;
        color: #666;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 500;
        line-height: 1.1;
    }
    .metric-value.good { color: #1D9E75; }
    .metric-value.bad { color: #E24B4A; }
    .tag-chip {
        display: inline-block;
        padding: 5px 12px;
        margin: 3px;
        border-radius: 14px;
        font-size: 12px;
        font-weight: 500;
    }
    .tag-binary { background: #E1F5EE; color: #085041; }
    .tag-categorical { background: #E6F1FB; color: #0C447C; }
    .rec-card {
        padding: 12px 16px;
        margin: 8px 0;
        background: #E1F5EE;
        border-radius: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .rec-text { font-size: 14px; color: #1a1a1a; }
    .rec-delta { font-weight: 600; color: #1D9E75; font-size: 14px; }
    h2, h3 { font-weight: 500; }
</style>
"""


# ============================================================
# ЗАГРУЗКА ДАННЫХ И МОДЕЛИ (с кэшем)
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("creatives_with_ctr.csv")
    shap_df = pd.read_csv("shap_values.csv")
    interactions_df = pd.read_csv("interactions.csv")
    with open("feature_cols.json", "r") as f:
        feature_cols = json.load(f)
    return df, shap_df, feature_cols, interactions_df


@st.cache_resource
def load_model():
    """Загружает обученную LightGBM модель."""
    return joblib.load("lgbm_model.pkl")


@st.cache_resource
def get_openai_client():
    """Создаёт OpenAI клиент. cache_resource — потому что объект."""
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


@st.cache_resource
def get_explainer(_model):
    """Создаёт SHAP explainer."""
    return shap.TreeExplainer(_model)


# ============================================================
# ТЕГИРОВАНИЕ КАРТИНКИ
# ============================================================

@st.cache_data(show_spinner=False)
def tag_image_cached(image_hash, image_bytes, mime_type):
    """Кэшируется по hash картинки — повторный вызов мгновенный."""
    client = get_openai_client()
    image_data = base64.b64encode(image_bytes).decode()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=1024,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": TAGGING_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime_type};base64,{image_data}"
                }}
            ]
        }]
    )
    return json.loads(response.choices[0].message.content.strip())


# ============================================================
# ПРЕОБРАЗОВАНИЯ
# ============================================================

def tags_to_features(tags, feature_cols):
    """Превращает словарь тегов в DataFrame для модели."""
    features = pd.DataFrame(0, index=[0], columns=feature_cols)

    # Бинарные → 1/0
    for col in BINARY_FEATURES:
        if col in feature_cols:
            features[col] = int(tags.get(col, False))

    # Категориальные → one-hot
    for cat in CATEGORICAL_TAGS:
        value = tags.get(cat, "none")
        col_name = f"{cat}_{value}"
        if col_name in feature_cols:
            features[col_name] = 1

    return features


# ============================================================
# WATERFALL ГРАФИК
# ============================================================

def make_waterfall(shap_values_obj, feature_names, max_features=10):
    """Создаёт интерактивный waterfall в стиле прототипа."""
    base = float(shap_values_obj.base_values)
    contributions = list(zip(feature_names, shap_values_obj.values))

    significant = [(n, v) for n, v in contributions if abs(v) > 0.005]
    significant.sort(key=lambda x: abs(x[1]), reverse=True)
    top = significant[:max_features]
    predicted = base + sum(v for _, v in contributions)

    items = [{"label": "средний CTR", "value": base, "type": "base"}]
    for name, val in top:
        clean = name.replace("_", " ").replace("food type ", "").replace("drink type ", "")
        items.append({"label": clean, "value": val, "type": "rel"})
    items.append({"label": "итого (теги)", "value": predicted, "type": "total"})

    fig = go.Figure()
    max_val = max(base + sum(max(0, v) for _, v in top), predicted) * 1.1
    running = base

    for i, item in enumerate(items):
        y_pos = len(items) - i - 1
        if item["type"] == "base":
            fig.add_trace(go.Bar(
                y=[y_pos], x=[item["value"]], orientation='h',
                marker_color="#E8E5DC",
                text=f"{item['value']:.1f}%",
                textposition="inside",
                textfont=dict(color="#666", size=12),
                showlegend=False, hoverinfo='skip',
            ))
        elif item["type"] == "total":
            fig.add_trace(go.Bar(
                y=[y_pos], x=[item["value"]], orientation='h',
                marker_color="#E6F1FB",
                text=f"{item['value']:.2f}%",
                textposition="inside",
                textfont=dict(color="#0C447C", size=13),
                showlegend=False, hoverinfo='skip',
            ))
        else:
            v = item["value"]
            if v > 0:
                fig.add_trace(go.Bar(
                    y=[y_pos], x=[v], base=running, orientation='h',
                    marker_color="#D4F0E5",
                    text=f"+{v:.2f}",
                    textposition="outside",
                    textfont=dict(color="#085041", size=12),
                    showlegend=False, hoverinfo='skip',
                ))
            else:
                fig.add_trace(go.Bar(
                    y=[y_pos], x=[abs(v)], base=running + v, orientation='h',
                    marker_color="#FCEBEB",
                    text=f"{v:.2f}",
                    textposition="outside",
                    textfont=dict(color="#791F1F", size=12),
                    showlegend=False, hoverinfo='skip',
                ))
            running += v

    fig.update_layout(
        height=max(280, len(items) * 38),
        margin=dict(l=130, r=60, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(range=[0, max_val], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(items))),
            ticktext=[items[len(items)-1-i]["label"] for i in range(len(items))],
            showgrid=False,
        ),
        bargap=0.25,
        font=dict(size=12),
    )
    return fig


# ============================================================
# РЕКОМЕНДАЦИИ
# ============================================================

def get_recommendations(X_new, df, shap_df, feature_cols, top_n=5):
    """Возвращает список (тег, потенциал) для рекомендаций."""
    recs = []
    for feat in BINARY_FEATURES:
        if feat in feature_cols and X_new[feat].iloc[0] == 0:
            shap_when_true = shap_df[shap_df.index.isin(
                df[df[feat] == True].index
            )][feat].mean()
            if shap_when_true > 0.05:
                recs.append((feat, shap_when_true))

    recs.sort(key=lambda x: x[1], reverse=True)
    return recs[:top_n]


# ============================================================
# РЕНДЕР ТЕГОВ
# ============================================================

def render_tag_chips(tags):
    """Возвращает HTML с пилюлями тегов."""
    chips = []
    for k, v in tags.items():
        if v == True:
            chips.append(f'<span class="tag-chip tag-binary">{k.replace("_"," ")}</span>')
        elif isinstance(v, str) and v != "none":
            chips.append(f'<span class="tag-chip tag-categorical">{k.replace("_"," ")}: {v}</span>')
    return "".join(chips)