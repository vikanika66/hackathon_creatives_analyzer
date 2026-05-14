"""
Главная страница — лендинг с описанием проекта.
"""

import streamlit as st
from utils import COMMON_CSS, load_data

st.set_page_config(
    page_title="Creative Analyzer",
    page_icon="🍔",
    layout="wide",
)

st.markdown(COMMON_CSS, unsafe_allow_html=True)

# Загружаем данные чтобы показать статистику
df, shap_df, feature_cols, interactions_df = load_data()

# ============================================================
# HERO
# ============================================================
st.markdown("""
<div style="text-align:center;padding:40px 0 20px 0;">
    <div style="font-size:64px;">🍔</div>
    <h1 style="font-size:42px;margin:8px 0;font-weight:600;">Creative Analyzer</h1>
    <p style="font-size:18px;color:#666;max-width:700px;margin:0 auto;">
        AI-анализ рекламных креативов для Food & Beverage.
        Прогнозируй CTR и получай рекомендации до запуска кампании.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ============================================================
# СТАТИСТИКА
# ============================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-box" style="text-align:center;">
        <div class="metric-label">Креативов в базе</div>
        <div class="metric-value">{len(df)}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box" style="text-align:center;">
        <div class="metric-label">Тегов на креатив</div>
        <div class="metric-value">{len(feature_cols)}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box" style="text-align:center;">
        <div class="metric-label">Средний CTR</div>
        <div class="metric-value">{df['ctr'].mean():.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-box" style="text-align:center;">
        <div class="metric-label">Точность модели</div>
        <div class="metric-value">±0.18%</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ============================================================
# ПРОБЛЕМА
# ============================================================
st.markdown("## Проблема")
st.markdown("""
Маркетологи запускают рекламные кампании и узнают какой креатив сработал
**после** того как потратили бюджет. CTR можно посмотреть только постфактум,
а понять *почему* один креатив работает лучше другого — почти невозможно
без углублённого анализа.

В результате:
- Бюджеты тратятся на креативы со слабым потенциалом
- Успешные элементы (улыбка, скидка, крупный план) не масштабируются на новые креативы
- Решения принимаются интуитивно, а не на основе данных
""")

st.divider()

# ============================================================
# РЕШЕНИЕ
# ============================================================
st.markdown("## Решение")

col_l, col_r = st.columns([1, 1])
with col_l:
    st.markdown("""
    Creative Analyzer использует **AI-тегирование** и **SHAP-анализ** чтобы:

    - **Автоматически размечать** каждый креатив по визуальным элементам (что на картинке, какие цвета, есть ли текст и CTA)
    - **Прогнозировать CTR** нового креатива до запуска
    - **Объяснять**, какие именно элементы тянут CTR вверх или вниз
    - **Рекомендовать** изменения на основе паттернов из истории
    """)

with col_r:
    st.markdown("""
    **Под капотом:**

    - GPT-4o-mini для тегирования картинок
    - LightGBM для предсказания CTR
    - SHAP values для объяснения прогноза
    - 1000 реальных рекламных креативов из Meta Ad Library
    """)

st.divider()

# ============================================================
# КАК ЭТО РАБОТАЕТ
# ============================================================
st.markdown("## Как это работает")

step_col1, step_col2, step_col3 = st.columns(3)

with step_col1:
    st.markdown("""
    <div style="padding:20px;background:#f5f5f0;border-radius:12px;height:180px;">
        <div style="font-size:32px;margin-bottom:8px;">📤</div>
        <div style="font-weight:600;font-size:16px;margin-bottom:6px;">1. Загрузи креатив</div>
        <div style="font-size:13px;color:#666;">Перетащи рекламный баннер в формате PNG, JPG или WebP</div>
    </div>
    """, unsafe_allow_html=True)

with step_col2:
    st.markdown("""
    <div style="padding:20px;background:#f5f5f0;border-radius:12px;height:180px;">
        <div style="font-size:32px;margin-bottom:8px;">🤖</div>
        <div style="font-weight:600;font-size:16px;margin-bottom:6px;">2. AI размечает</div>
        <div style="font-size:13px;color:#666;">GPT определяет визуальные элементы: персонажи, эмоции, тип еды, композиция, CTA</div>
    </div>
    """, unsafe_allow_html=True)

with step_col3:
    st.markdown("""
    <div style="padding:20px;background:#f5f5f0;border-radius:12px;height:180px;">
        <div style="font-size:32px;margin-bottom:8px;">📊</div>
        <div style="font-weight:600;font-size:16px;margin-bottom:6px;">3. Получи разбор</div>
        <div style="font-size:13px;color:#666;">Прогноз CTR, разложение на компоненты, рекомендации что улучшить</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ============================================================
# CTA
# ============================================================
st.markdown("""
<div style="text-align:center;padding:30px 0;">
    <h3 style="font-weight:500;margin-bottom:8px;">Готова попробовать?</h3>
    <p style="color:#666;margin-bottom:20px;">Выбери раздел в меню слева</p>
</div>
""", unsafe_allow_html=True)

cta_col1, cta_col2, cta_col3 = st.columns([1, 1, 1])
with cta_col2:
    if st.button("🚀 Анализировать креатив", width='stretch', type="primary"):
        st.switch_page("pages/1_Upload.py")

with cta_col1:
    if st.button("📚 Открыть библиотеку", width='stretch'):
        st.switch_page("pages/2_Library.py")