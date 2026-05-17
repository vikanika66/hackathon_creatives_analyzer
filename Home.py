"""
Главная страница — лендинг с описанием проекта.
"""

import streamlit as st
from utils import COMMON_CSS, load_data, BINARY_FEATURES

st.set_page_config(
    page_title="Creative Analyzer",
    page_icon="🍔",
    layout="wide",
)

st.markdown(COMMON_CSS, unsafe_allow_html=True)

st.markdown("""
<style>
    /* Кастомный цвет для primary-кнопок на главной */
    .stButton > button[kind="primary"] {
        background-color: #0009dc !important;
        border-color: #0009dc !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #0007b8 !important;
        border-color: #0007b8 !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:active,
    .stButton > button[kind="primary"]:focus {
        background-color: #0006a3 !important;
        border-color: #0006a3 !important;
        color: white !important;
        box-shadow: 0 0 0 0.2rem rgba(0, 9, 220, 0.25) !important;
    }
</style>
""", unsafe_allow_html=True)

# Загружаем данные чтобы показать статистику
df, shap_df, feature_cols, interactions_df = load_data()

# Считаем что показать
n_creatives = len(df)
n_brands = df["brand"].nunique()
n_tags = len(BINARY_FEATURES) + 6
n_food = df["food_type"].nunique() + df["drink_type"].nunique() - 2  # минус "none" в обоих


# ============================================================
# HERO
# ============================================================
st.markdown("""
<div style="text-align:center;padding:60px 0 8px 0;">
    <div style="font-size:64px;line-height:1;">🍔</div>
    <h1 style="font-size:46px;margin:16px 0 12px 0;font-weight:600;letter-spacing:-1px;">
        Creative Analyzer
    </h1>
    <p style="font-size:20px;color:#666;max-width:720px;margin:0 auto;line-height:1.5;">
        Understand which visual elements correlate with high CTR in your ad creatives —
        before you spend the budget.
    </p>
    <div style="font-size:17px;color:#0009dc;margin-top:14px;">
        Focused on <b style="color:#aef33e;">Food & Drinks</b> advertising category
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# БОЛЬШИЕ ЦИФРЫ
# ============================================================
st.markdown(f"""
<div style="display:grid;grid-template-columns:repeat(5, 1fr);gap:12px;margin:40px 0 8px 0;">
    <div style="background:#faf9f5;border-radius:14px;padding:24px 16px;text-align:center;">
        <div style="font-size:36px;font-weight:600;color:#1a1a1a;line-height:1;">{n_brands}</div>
        <div style="font-size:12px;color:#888;margin-top:8px;">brands</div>
    </div>
    <div style="background:#faf9f5;border-radius:14px;padding:24px 16px;text-align:center;">
        <div style="font-size:36px;font-weight:600;color:#1a1a1a;line-height:1;">{n_creatives:,}</div>
        <div style="font-size:12px;color:#888;margin-top:8px;">creatives</div>
    </div>
    <div style="background:#faf9f5;border-radius:14px;padding:24px 16px;text-align:center;">
        <div style="font-size:36px;font-weight:600;color:#1a1a1a;line-height:1;">100%</div>
        <div style="font-size:12px;color:#888;margin-top:8px;">AI-tagged</div>
    </div>
    <div style="background:#faf9f5;border-radius:14px;padding:24px 16px;text-align:center;">
        <div style="font-size:36px;font-weight:600;color:#1a1a1a;line-height:1;">{n_food}</div>
        <div style="font-size:12px;color:#888;margin-top:8px;">food &amp; drink categories</div>
    </div>
    <div style="background:#faf9f5;border-radius:14px;padding:24px 16px;text-align:center;">
        <div style="font-size:36px;font-weight:600;color:#1a1a1a;line-height:1;">{n_tags}</div>
        <div style="font-size:12px;color:#888;margin-top:8px;">visual tags per creative</div>
    </div>
</div>

<div style="text-align:center;font-size:12px;color:#aaa;margin:0 0 60px 0;">
    Demo uses synthetic CTR data generated based on real ad creative structure from Meta Ad Library
</div>
""", unsafe_allow_html=True)

# ============================================================
# ДЛЯ КОГО ЭТО
# ============================================================
st.divider()
st.markdown("""
<div style="margin:20px 0 32px 0;">
    <div style="font-size:28px;font-weight:600;margin-bottom:8px;">Who it's for</div>
    <div style="font-size:15px;color:#666;max-width:700px;">
        Three perspectives on the same data — each role gets the view they need.
    </div>
</div>
""", unsafe_allow_html=True)

audience_col1, audience_col2, audience_col3 = st.columns(3, gap="medium")

with audience_col1:
    st.markdown("""
    <div style="background:#faf9f5;border-radius:14px;padding:28px 24px;height:280px;">
        <div style="font-size:32px;margin-bottom:12px;">📈</div>
        <div style="font-size:18px;font-weight:600;margin-bottom:10px;color:#1a1a1a;">
            Marketing manager
        </div>
        <div style="font-size:14px;color:#555;line-height:1.6;">
            See which visual elements correlate with higher CTR across your brand portfolio.
            Spot underperforming creatives and understand patterns in your top performers.
        </div>
    </div>
    """, unsafe_allow_html=True)

with audience_col2:
    st.markdown("""
    <div style="background:#faf9f5;border-radius:14px;padding:28px 24px;height:280px;">
        <div style="font-size:32px;margin-bottom:12px;">🎯</div>
        <div style="font-size:18px;font-weight:600;margin-bottom:10px;color:#1a1a1a;">
            Brand client
        </div>
        <div style="font-size:14px;color:#555;line-height:1.6;">
            Get a transparent view of how your creatives perform versus the industry.
            Track your brand's position year over year and discover what's working.
        </div>
    </div>
    """, unsafe_allow_html=True)

with audience_col3:
    st.markdown("""
    <div style="background:#faf9f5;border-radius:14px;padding:28px 24px;height:280px;">
        <div style="font-size:32px;margin-bottom:12px;">🎨</div>
        <div style="font-size:18px;font-weight:600;margin-bottom:10px;color:#1a1a1a;">
            Creative designer
        </div>
        <div style="font-size:14px;color:#555;line-height:1.6;">
            Test A/B scenarios on visual elements before production.
            Generate AI reference mockups based on data-driven recommendations.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# ЧТО ВНУТРИ ПРОДУКТА
# ============================================================
st.divider()
st.markdown("""
<div style="margin:60px 0 32px 0;">
    <div style="font-size:28px;font-weight:600;margin-bottom:8px;">What's inside</div>
    <div style="font-size:15px;color:#666;max-width:700px;">
        Three connected views that work together — analyze a new creative,
        explore the library, and zoom out to brand-level insights.
    </div>
</div>
""", unsafe_allow_html=True)


# === Upload ===
upload_col1, upload_col2 = st.columns([1.2, 1], gap="large")

with upload_col1:
    st.markdown("""
    <div style="padding-top:10px;">
        <div style="font-size:13px;color:#888;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:8px;">
            For designers · For testing ideas · For clients
        </div>
        <div style="font-size:24px;font-weight:600;margin-bottom:12px;color:#1a1a1a;">
            📤 Upload &amp; Analyze
        </div>
        <div style="font-size:15px;color:#555;line-height:1.7;margin-bottom:20px;">
            Drop a new creative and let AI tag its visual elements automatically.
            See how each tag correlates with CTR, test A/B scenarios by toggling tags on and off,
            and generate AI reference mockups based on the strongest patterns from real data.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Open Upload →", key="cta_upload", type="primary"):
        st.switch_page("pages/1_Upload.py")

with upload_col2:
    st.markdown("""
    <div style="background:#faf9f5;border-radius:14px;padding:24px;font-size:13px;color:#666;line-height:1.8;">
        <div style="font-weight:600;color:#1a1a1a;margin-bottom:10px;">Key features</div>
        • Auto-tagging via GPT-4o<br>
        • Tag correlation with CTR<br>
        • A/B scenario testing<br>
        • AI reference mockup generation
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div style="height:48px;"></div>', unsafe_allow_html=True)


# === Library ===
lib_col1, lib_col2 = st.columns([1.2, 1], gap="large")

with lib_col1:
    st.markdown(f"""
    <div style="padding-top:10px;">
        <div style="font-size:13px;color:#888;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:8px;">
            For all roles · Explore the data
        </div>
        <div style="font-size:24px;font-weight:600;margin-bottom:12px;color:#1a1a1a;">
            📚 Library
        </div>
        <div style="font-size:15px;color:#555;line-height:1.7;margin-bottom:20px;">
            Browse {n_creatives:,} real ad creatives from {n_brands} F&amp;B brands with full tag data and CTR.
            Filter by brand, year, visual elements, and open any creative to see its full breakdown —
            how each tag relates to CTR, what category it belongs to, and how its tag combinations behave.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Open Library →", key="cta_library", type="primary"):
        st.switch_page("pages/2_Library.py")

with lib_col2:
    st.markdown("""
    <div style="background:#faf9f5;border-radius:14px;padding:24px;font-size:13px;color:#666;line-height:1.8;">
        <div style="font-weight:600;color:#1a1a1a;margin-bottom:10px;">Key features</div>
        • Filter by brand, year, tags<br>
        • Full creative breakdown<br>
        • Tag-level correlation<br>
        • Tag combination patterns
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div style="height:48px;"></div>', unsafe_allow_html=True)


# === Insights ===
ins_col1, ins_col2 = st.columns([1.2, 1], gap="large")

with ins_col1:
    st.markdown("""
    <div style="padding-top:10px;">
        <div style="font-size:13px;color:#888;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:8px;">
            For MPO · For clients
        </div>
        <div style="font-size:24px;font-weight:600;margin-bottom:12px;color:#1a1a1a;">
            📊 Insights
        </div>
        <div style="font-size:15px;color:#555;line-height:1.7;margin-bottom:20px;">
            Zoom out to the brand level. See what works and what doesn't across the brand's portfolio,
            spot hidden gems and underperformers, and track how the brand's position in the industry
            evolved over time. Each block answers a specific question a marketing manager would ask.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Open Insights →", key="cta_insights", type="primary"):
        st.switch_page("pages/3_Insights.py")

with ins_col2:
    st.markdown("""
    <div style="background:#faf9f5;border-radius:14px;padding:24px;font-size:13px;color:#666;line-height:1.8;">
        <div style="font-weight:600;color:#1a1a1a;margin-bottom:10px;">Key features</div>
        • Brand-level statistics<br>
        • Strong &amp; weak tags<br>
        • Tag interactions heatmap<br>
        • Hidden gems &amp; underperformers<br>
        • Year-over-year position trend
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Футер с техничсеким описанием 
# ============================================================

st.markdown('<div style="height:48px;"></div>', unsafe_allow_html=True)

with st.expander("🔧 Under the hood — for technical reviewers"):
    st.markdown("""
    **AI tagging.** Each creative is automatically tagged using OpenAI GPT-4o vision model.
    The taxonomy includes 11 binary tags (`has_person`, `close_up`, `warm_tones`, `text_overlay`,
    `price_discount`, `brand_logo`, `cta`, etc.) and 6 categorical tags (`main_object`, `food_type`,
    `drink_type`, `person_who`, `person_emotion`, `person_action`).
    
    **CTR modeling.** A LightGBM regressor is trained on the tagged dataset.
    The model serves two purposes: (1) predicting CTR for unseen creatives, (2) providing SHAP
    values that show how each tag contributes to the prediction.
    
    **Honest framing.** We deliberately don't show raw CTR predictions or SHAP numbers in the UI.
    Reason: CTR depends heavily on targeting, budget, and audience — not just creative.
    Instead, we show *patterns in the data* using arrow indicators (↑↑ / ↑ / — / ↓ / ↓↓)
    that communicate strength of correlation without claiming causation.
    
    **AI mockup generation.** Reference mockups are generated via OpenAI gpt-image-1 (image-to-image)
    using the original creative as base and a prompt built from the active tags of the chosen A/B scenario.
    
    **Stack:** Streamlit · pandas · LightGBM · SHAP · OpenAI GPT-4o + gpt-image-1 · Plotly · PIL
    
    **Data source.** Synthetic CTR data generated for 641 real ad creatives collected from the
    Meta Ad Library, focused on the Food & Drinks vertical across 7 brands.
    """)


st.markdown('<div style="height:60px;"></div>', unsafe_allow_html=True)
st.markdown('<div style="height:1px;background:#eee;margin-bottom:32px;"></div>', unsafe_allow_html=True)

footer_col1, footer_col2 = st.columns([1, 4], gap="medium")

with footer_col1:
    try:
        st.image("assets/me.jpg", width=600)
    except Exception:
        # Если фото не найдено — показываем placeholder
        st.markdown(
            '<div style="width:140px;height:140px;border-radius:50%;background:#faf9f5;'
            'display:flex;align-items:center;justify-content:center;color:#bbb;font-size:13px;">'
            'photo</div>',
            unsafe_allow_html=True
        )

with footer_col2:
    st.markdown("""
    <div style="padding-top:8px;">
        <div style="font-size:12px;color:#888;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:6px;">
            Built by
        </div>
        <div style="font-size:22px;font-weight:600;color:#1a1a1a;margin-bottom:4px;">
            Viktoriia Iachmeneva
        </div>
        <div style="font-size:14px;color:#666;margin-bottom:12px;">
            Product Analyst · Product &amp; Innovations
        </div>
        <div style="font-size:13px;color:#888;">
            ✉️ <a href="mailto:viktoriia.iachmeneva@aidigital.com" style="color:#888;text-decoration:none;border-bottom:1px solid #ddd;">viktoriia.iachmeneva@aidigital.com</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# st.markdown("""
# <div style="text-align:center;font-size:12px;color:#bbb;margin:40px 0 20px 0;">
#     Hackathon MVP · Built in Streamlit
# </div>
# """, unsafe_allow_html=True)


st.markdown("""
<div style="text-align:center;font-size:13px;color:#bbb;margin:60px 0 20px 0;padding-top:24px;border-top:1px solid #eee;">
    Creative Analyzer · Hackathon MVP · Built in Streamlit · by Viktoriia Iachmeneva · 2026
</div>
""", unsafe_allow_html=True)