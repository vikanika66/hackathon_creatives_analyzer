"""
Главная страница — лендинг с описанием проекта.
"""

import streamlit as st
from utils import COMMON_CSS, load_data, BINARY_FEATURES

st.set_page_config(
    page_title="Creative Analyzer",
    page_icon="🎨",
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
<div style="margin:32px 0 18px 0;padding:42px 36px 38px 36px;border-radius:26px;background:radial-gradient(circle at 18% 18%, rgba(174,243,62,0.32), transparent 26%),radial-gradient(circle at 82% 12%, rgba(0,9,220,0.16), transparent 30%),linear-gradient(135deg, #f9f9f9 0%, #ffffff 100%);border:1px solid #eeeeea;text-align:center;">
<div style="display:inline-flex;gap:8px;align-items:center;padding:7px 13px;border-radius:999px;background:rgba(255,255,255,0.82);border:1px solid #e8e8e2;font-size:12px;color:rgba(8,8,8,0.58);margin-bottom:18px;">
<span style="width:7px;height:7px;border-radius:50%;background:#aef33e;display:inline-block;"></span>
AI-powered creative intelligence
</div>
<div style="font-size:13px;color:#0009dc;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:10px;">
Welcome to
</div>
<h1 style="font-size:44px;margin:0 0 14px 0;font-weight:650;letter-spacing:-1.3px;color:#080808;">
Creative Analyzer
</h1>
<p style="font-size:18px;color:rgba(8,8,8,0.68);max-width:720px;margin:0 auto;line-height:1.55;">
A workspace for exploring what makes ad creatives perform — from visual tags to CTR patterns and AI-generated reference ideas.
</p>
<div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin-top:24px;">
<span style="background:#0009dc;color:white;padding:7px 12px;border-radius:999px;font-size:12px;">visual tags</span>
<span style="background:#aef33e;color:#111;padding:7px 12px;border-radius:999px;font-size:12px;">CTR patterns</span>
<span style="background:#111;color:white;padding:7px 12px;border-radius:999px;font-size:12px;">creative insights</span>
<span style="background:white;color:rgba(8,8,8,0.68);padding:7px 12px;border-radius:999px;font-size:12px;border:1px solid #ddd;">AI mockups</span>
</div>
<div style="font-size:15px;color:#0009dc;margin-top:22px;">
Focused on <b style="color:#080808;background:#aef33e;padding:2px 6px;border-radius:6px;">Food &amp; Drinks</b> advertising category
</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# БОЛЬШИЕ ЦИФРЫ
# ============================================================
st.markdown(f"""
<div style="margin:34px 0 10px 0;">
<div style="text-align:center;margin-bottom:18px;">
<div style="font-size:14px;color:#0009dc;letter-spacing:1.3px;text-transform:uppercase;margin-bottom:9px;">Dataset snapshot</div>
<div style="font-size:15px;color:rgba(8,8,8,0.68);">A quick look at the creative universe behind the demo</div>
</div>

<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(150px, 1fr));gap:12px;">

<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:22px 14px;text-align:center;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#0009dc;"></div>
<div style="font-size:34px;font-weight:650;color:#080808;line-height:1;">{n_brands}</div>
<div style="font-size:12px;color:rgba(8,8,8,0.58);margin-top:8px;">brands</div>
</div>

<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:22px 14px;text-align:center;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#aef33e;"></div>
<div style="font-size:34px;font-weight:650;color:#080808;line-height:1;">{n_creatives:,}</div>
<div style="font-size:12px;color:rgba(8,8,8,0.58);margin-top:8px;">creatives</div>
</div>

<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:22px 14px;text-align:center;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#ff7cf5;"></div>
<div style="font-size:34px;font-weight:650;color:#080808;line-height:1;">100%</div>
<div style="font-size:12px;color:rgba(8,8,8,0.58);margin-top:8px;">AI-tagged</div>
</div>

<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:22px 14px;text-align:center;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#080808;"></div>
<div style="font-size:34px;font-weight:650;color:#080808;line-height:1;">{n_food}</div>
<div style="font-size:12px;color:rgba(8,8,8,0.58);margin-top:8px;">food &amp; drink categories</div>
</div>

<div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:22px 14px;text-align:center;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:3px;background:#0009dc;"></div>
<div style="font-size:34px;font-weight:650;color:#080808;line-height:1;">{n_tags}</div>
<div style="font-size:12px;color:rgba(8,8,8,0.58);margin-top:8px;">visual tags per creative</div>
</div>

</div>

<div style="text-align:center;font-size:12px;color:#999;margin:14px 0 52px 0;">
Demo uses synthetic CTR data generated based on real ad creative structure from Meta Ad Library
</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# ДЛЯ КОГО ЭТО
# ============================================================
st.divider()

st.markdown("""
<div style="margin:24px 0 28px 0;text-align:center;">
<div style="font-size:14px;color:#0009dc;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:9px;">Who it's for</div>
<div style="font-size:28px;font-weight:650;color:#080808;margin-bottom:8px;">Built for different creative decisions</div>
<div style="font-size:15px;color:rgba(8,8,8,0.62);max-width:720px;margin:0 auto;line-height:1.55;">
Three perspectives on the same creative data — strategy, client reporting, and design exploration.
</div>
</div>
""", unsafe_allow_html=True)

audience_col1, audience_col2, audience_col3 = st.columns(3, gap="medium")

with audience_col1:
    st.markdown("""
    <div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:20px;padding:26px 24px;height:270px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;left:0;right:0;height:3px;background:#0009dc;"></div>
    <div style="font-size:28px;margin-bottom:14px;">📈</div>
    <div style="font-size:11px;color:#0009dc;letter-spacing:1.1px;text-transform:uppercase;margin-bottom:8px;">Strategy view</div>
    <div style="font-size:19px;font-weight:650;margin-bottom:10px;color:#080808;">
    Marketing manager
    </div>
    <div style="font-size:14px;color:rgba(8,8,8,0.66);line-height:1.65;">
    See which visual elements correlate with higher CTR across the brand portfolio.
    Spot underperforming creatives and understand patterns in top performers.
    </div>
    </div>
    """, unsafe_allow_html=True)

with audience_col2:
    st.markdown("""
    <div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:20px;padding:26px 24px;height:270px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;left:0;right:0;height:3px;background:#aef33e;"></div>
    <div style="font-size:28px;margin-bottom:14px;">🎯</div>
    <div style="font-size:11px;color:#080808;letter-spacing:1.1px;text-transform:uppercase;margin-bottom:8px;">Client view</div>
    <div style="font-size:19px;font-weight:650;margin-bottom:10px;color:#080808;">
    Brand client
    </div>
    <div style="font-size:14px;color:rgba(8,8,8,0.66);line-height:1.65;">
    Get a transparent view of creative performance versus the industry.
    Track the brand's position over time and see which directions are working.
    </div>
    </div>
    """, unsafe_allow_html=True)

with audience_col3:
    st.markdown("""
    <div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:20px;padding:26px 24px;height:270px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;left:0;right:0;height:3px;background:#ff7cf5;"></div>
    <div style="font-size:28px;margin-bottom:14px;">🎨</div>
    <div style="font-size:11px;color:#080808;letter-spacing:1.1px;text-transform:uppercase;margin-bottom:8px;">Design view</div>
    <div style="font-size:19px;font-weight:650;margin-bottom:10px;color:#080808;">
    Creative designer
    </div>
    <div style="font-size:14px;color:rgba(8,8,8,0.66);line-height:1.65;">
    Test A/B scenarios before production.
    Explore visual tags, compare creative directions, and generate AI reference mockups.
    </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# ЧТО ВНУТРИ ПРОДУКТА
# ============================================================
st.divider()

st.markdown("""
<div style="margin:56px 0 32px 0;text-align:center;">
<div style="font-size:14px;color:#0009dc;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:7px;">What's inside</div>
<div style="font-size:28px;font-weight:650;color:#080808;margin-bottom:8px;">Three connected product views</div>
<div style="font-size:15px;color:rgba(8,8,8,0.62);max-width:760px;margin:0 auto;line-height:1.55;">
Analyze a new creative, explore the tagged library, and zoom out to brand-level insights.
</div>
</div>
""", unsafe_allow_html=True)


# === Upload ===
upload_col1, upload_col2 = st.columns([1.25, 1], gap="large")

with upload_col1:
    st.markdown("""
    <div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:22px;padding:28px 26px;min-height:245px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;left:0;right:0;height:3px;background:#0009dc;"></div>
    <div style="font-size:11px;color:#0009dc;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:10px;">
    For designers · For testing ideas · For clients
    </div>
    <div style="font-size:24px;font-weight:650;margin-bottom:12px;color:#080808;">
    📤 Upload &amp; Analyze
    </div>
    <div style="font-size:15px;color:rgba(8,8,8,0.66);line-height:1.7;margin-bottom:20px;">
    Drop a new creative and let AI tag its visual elements automatically.
    See how each tag relates to CTR, test A/B scenarios by toggling tags on and off,
    and generate AI reference mockups based on the strongest patterns from the dataset.
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

    if st.button("Open Upload →", key="cta_upload", type="primary"):
        st.switch_page("pages/1_Upload.py")

with upload_col2:
    st.markdown("""
    <div style="background:#080808;border-radius:22px;padding:26px 24px;min-height:245px;color:#ffffff;position:relative;overflow:hidden;">
    <div style="position:absolute;top:-40px;right:-40px;width:140px;height:140px;background:#0009dc;border-radius:50%;opacity:0.32;"></div>
    <div style="position:absolute;bottom:-50px;left:-50px;width:150px;height:150px;background:#aef33e;border-radius:50%;opacity:0.22;"></div>
    <div style="position:relative;z-index:1;">
    <div style="font-size:12px;color:#aef33e;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:12px;">Key features</div>
    <div style="font-size:14px;line-height:1.9;color:rgba(255,255,255,0.82);">
    <span style="color:#ffffff;">01</span> · Auto-tagging via GPT-4o<br>
    <span style="color:#ffffff;">02</span> · Tag correlation with CTR<br>
    <span style="color:#ffffff;">03</span> · A/B scenario testing<br>
    <span style="color:#ffffff;">04</span> · AI reference mockup generation
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div style="height:48px;"></div>', unsafe_allow_html=True)


# === Library ===
lib_col1, lib_col2 = st.columns([1.25, 1], gap="large")

with lib_col1:
    st.markdown(f"""
    <div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:22px;padding:28px 26px;min-height:245px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;left:0;right:0;height:3px;background:#aef33e;"></div>
    <div style="font-size:11px;color:#080808;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:10px;">
    For all roles · Explore the data
    </div>
    <div style="font-size:24px;font-weight:650;margin-bottom:12px;color:#080808;">
    📚 Library
    </div>
    <div style="font-size:15px;color:rgba(8,8,8,0.66);line-height:1.7;margin-bottom:20px;">
    Browse {n_creatives:,} real ad creatives from {n_brands} F&amp;B brands with full tag data and CTR.
    Filter by brand, year, and visual elements, then open any creative to see its full breakdown.
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

    if st.button("Open Library →", key="cta_library", type="primary"):
        st.switch_page("pages/2_Library.py")

with lib_col2:
    st.markdown("""
    <div style="background:#ffffff;border:1px solid #eeeeee;border-radius:22px;padding:26px 24px;min-height:245px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:-45px;right:-45px;width:145px;height:145px;background:#aef33e;border-radius:50%;opacity:0.34;"></div>
    <div style="position:absolute;bottom:-55px;left:-55px;width:160px;height:160px;background:#ff7cf5;border-radius:50%;opacity:0.18;"></div>
    <div style="position:relative;z-index:1;">
    <div style="font-size:12px;color:#0009dc;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:12px;">Key features</div>
    <div style="font-size:14px;line-height:1.9;color:rgba(8,8,8,0.68);">
    <span style="color:#080808;">01</span> · Filter by brand, year, tags<br>
    <span style="color:#080808;">02</span> · Full creative breakdown<br>
    <span style="color:#080808;">03</span> · Tag-level correlation<br>
    <span style="color:#080808;">04</span> · Tag combination patterns
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div style="height:48px;"></div>', unsafe_allow_html=True)


# === Insights ===
ins_col1, ins_col2 = st.columns([1.25, 1], gap="large")

with ins_col1:
    st.markdown("""
    <div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:22px;padding:28px 26px;min-height:245px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;left:0;right:0;height:3px;background:#ff7cf5;"></div>
    <div style="font-size:11px;color:#080808;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:10px;">
    For MPO · For clients
    </div>
    <div style="font-size:24px;font-weight:650;margin-bottom:12px;color:#080808;">
    📊 Insights
    </div>
    <div style="font-size:15px;color:rgba(8,8,8,0.66);line-height:1.7;margin-bottom:20px;">
    Zoom out to the brand level. See what works and what doesn't across the portfolio,
    spot hidden gems and underperformers, and track how the brand's position evolves over time.
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

    
    if st.button("Open Insights →", key="cta_insights", type="primary"):
        st.switch_page("pages/3_Insights.py")

with ins_col2:
    st.markdown("""
    <div style="background:#080808;border-radius:22px;padding:26px 24px;min-height:245px;color:#ffffff;position:relative;overflow:hidden;">
    <div style="position:absolute;top:-45px;right:-45px;width:145px;height:145px;background:#ff7cf5;border-radius:50%;opacity:0.28;"></div>
    <div style="position:absolute;bottom:-55px;left:-55px;width:160px;height:160px;background:#0009dc;border-radius:50%;opacity:0.28;"></div>
    <div style="position:relative;z-index:1;">
    <div style="font-size:12px;color:#ff7cf5;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:12px;">Key features</div>
    <div style="font-size:14px;line-height:1.9;color:rgba(255,255,255,0.82);">
    <span style="color:#ffffff;">01</span> · Brand-level statistics<br>
    <span style="color:#ffffff;">02</span> · Strong &amp; weak tags<br>
    <span style="color:#ffffff;">03</span> · Tag interactions heatmap<br>
    <span style="color:#ffffff;">04</span> · Hidden gems &amp; underperformers<br>
    <span style="color:#ffffff;">05</span> · Year-over-year position trend
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# Футер с техничсеким описанием 
# ============================================================

st.markdown('<div style="height:48px;"></div>', unsafe_allow_html=True)

st.markdown("""
<style>
div[data-testid="stExpander"] {
    border: 1px solid #eeeeee !important;
    border-radius: 18px !important;
    overflow: hidden !important;
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
</style>
""", unsafe_allow_html=True)

with st.expander("🔧 Under the hood — for technical reviewers"):
    st.markdown("""
        <div style="margin:6px 0 18px 0;">
        <div style="font-size:12px;color:#0009dc;letter-spacing:1.3px;text-transform:uppercase;margin-bottom:6px;">Technical layer</div>
        <div style="font-size:20px;font-weight:650;color:#080808;margin-bottom:8px;">How the prototype works</div>
        <div style="font-size:14px;color:rgba(8,8,8,0.62);line-height:1.55;max-width:780px;">
        A compact view of the AI tagging, CTR modeling, explainability layer, and demo data behind Creative Analyzer.
        </div>
        </div>

        <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(260px, 1fr));gap:12px;margin-top:18px;">

        <div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:20px 18px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:#0009dc;"></div>
        <div style="font-size:12px;color:#0009dc;letter-spacing:1.1px;text-transform:uppercase;margin-bottom:8px;">01 · AI tagging</div>
        <div style="font-size:14px;color:rgba(8,8,8,0.68);line-height:1.65;">
        Each creative is automatically tagged using GPT-4o vision. The taxonomy includes binary visual tags and categorical fields such as object, food type, drink type, person, emotion, and action.
        </div>
        </div>

        <div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:20px 18px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:#aef33e;"></div>
        <div style="font-size:12px;color:#080808;letter-spacing:1.1px;text-transform:uppercase;margin-bottom:8px;">02 · CTR modeling</div>
        <div style="font-size:14px;color:rgba(8,8,8,0.68);line-height:1.65;">
        A LightGBM regressor is trained on the tagged dataset to estimate creative-level CTR patterns and support scenario comparison for unseen creatives.
        </div>
        </div>

        <div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:20px 18px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:#ff7cf5;"></div>
        <div style="font-size:12px;color:#080808;letter-spacing:1.1px;text-transform:uppercase;margin-bottom:8px;">03 · Explainability</div>
        <div style="font-size:14px;color:rgba(8,8,8,0.68);line-height:1.65;">
        SHAP values are used to show how each visual tag contributes to the model output, but the interface avoids raw precision claims and focuses on pattern strength.
        </div>
        </div>

        <div style="background:linear-gradient(180deg,#ffffff 0%,#f9f9f9 100%);border:1px solid #eeeeee;border-radius:18px;padding:20px 18px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:#ff7cf5;"></div>
        <div style="font-size:12px;color:#080808;letter-spacing:1.1px;text-transform:uppercase;margin-bottom:8px;">04 · Honest framing</div>
        <div style="font-size:14px;color:rgba(8,8,8,0.68);line-height:1.65;">
        CTR depends on targeting, budget, and audience. The product therefore communicates correlations and directional signals instead of claiming creative-only causation.
        </div>
        </div>
        </div>

        </div>

        <div style="margin:18px 0 8px 0;background:#ffffff;border:1px solid #eeeeee;border-radius:18px;padding:18px 18px;">
        <div style="font-size:12px;color:#0009dc;letter-spacing:1.1px;text-transform:uppercase;margin-bottom:8px;">Stack &amp; data</div>
        <div style="font-size:14px;color:rgba(8,8,8,0.68);line-height:1.7;">
        <b style="color:#080808;">Stack:</b> Streamlit · pandas · LightGBM · SHAP · OpenAI GPT-4o + gpt-image-1 · Plotly · PIL<br>
        <b style="color:#080808;">Data source:</b> Synthetic CTR data generated for 641 real ad creatives collected from the Meta Ad Library, focused on the Food &amp; Drinks vertical across 7 brands.
        </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# ОБО МНЕ
# ============================================================

# ============================================================
# FOOTER
# ============================================================

st.markdown('<div style="height:44px;"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="margin:0 0 20px 0;padding:26px 8px 0 8px;border-top:1px solid #eeeeee;">
""", unsafe_allow_html=True)

footer_col1, footer_col2 = st.columns([0.9, 4], gap="large")

with footer_col1:
    try:
        st.image("assets/me.jpg", width=600)
    except Exception:
        st.markdown(
            '<div style="width:150px;height:150px;border-radius:24px;background:#ffffff;'
            'border:1px solid #eeeeee;display:flex;align-items:center;justify-content:center;'
            'color:rgba(8,8,8,0.42);font-size:13px;">photo</div>',
            unsafe_allow_html=True
        )

with footer_col2:
    st.markdown("""
    <div style="padding-top:8px;">
    <div style="font-size:12px;color:#0009dc;letter-spacing:1.3px;text-transform:uppercase;margin-bottom:8px;">
    Built by
    </div>

    <div style="font-size:26px;font-weight:650;color:#080808;margin-bottom:12px;letter-spacing:-0.4px;">
    Viktoriia Iachmeneva
    </div>

    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;">
    <span style="background:#0009dc;color:#ffffff;padding:7px 12px;border-radius:999px;font-size:12px;">Product Analyst</span>
    <span style="background:#aef33e;color:#080808;padding:7px 12px;border-radius:999px;font-size:12px;">Product &amp; Innovations</span>
    </div>

    <div style="font-size:13px;color:rgba(8,8,8,0.58);">
    ✉️ <a href="mailto:viktoriia.iachmeneva@aidigital.com" style="color:#0009dc;text-decoration:none;border-bottom:1px solid rgba(0,9,220,0.28);">viktoriia.iachmeneva@aidigital.com</a>
    </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:22px;padding-top:16px;border-top:1px solid #eeeeee;text-align:center;font-size:15px;color:rgba(8,8,8,0.42);">
Creative Analyzer · Hackathon MVP · Built in Streamlit · 2026
</div>
</div>
""", unsafe_allow_html=True)