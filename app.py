import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import Counter
import re

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake vs Real News Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

  .block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }

  /* Header */
  .dash-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    display: flex;
    align-items: center;
    gap: 1.5rem;
  }
  .dash-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #fff 60%, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .dash-header p { margin: 0.3rem 0 0; opacity: 0.75; font-size: 0.95rem; }

  /* KPI cards */
  .kpi-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
  .kpi-card {
    flex: 1; min-width: 150px;
    background: #1e1b4b;
    border: 1px solid #3730a3;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    color: white;
    text-align: center;
  }
  .kpi-card .kpi-val {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #a78bfa;
  }
  .kpi-card .kpi-label { font-size: 0.78rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

  /* Section headers */
  .section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #312e81;
    border-left: 4px solid #7c3aed;
    padding-left: 10px;
    margin: 1.5rem 0 0.8rem;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0f0c29 !important;
    color: white !important;
  }
  section[data-testid="stSidebar"] * { color: white !important; }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stMultiSelect label { color: #c4b5fd !important; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 1px; }

  /* Chart containers */
  .chart-box {
    background: #fafafa;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 1rem;
  }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    fb   = pd.read_csv("facebook-fact-check.csv")

    fake["label"] = "Fake"
    true["label"] = "Real"

    # ── FIX 1: strip trailing whitespace from dates (True.csv has trailing spaces)
    for df in [fake, true]:
        df["date"] = df["date"].astype(str).str.strip()

    # ── FIX 2: normalise subject so Real subjects are visible in sidebar
    # True.csv uses 'politicsNews'/'worldnews' — keep as-is but strip whitespace
    for df in [fake, true]:
        df["subject"] = df["subject"].astype(str).str.strip()

    combined = pd.concat([fake, true], ignore_index=True)

    # Parse dates
    for df in [fake, true, combined]:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"]  = df["date"].dt.year
        df["month"] = df["date"].dt.to_period("M").astype(str)
        df["word_count"] = df["text"].fillna("").apply(lambda x: len(x.split()))
        df["title_len"]  = df["title"].fillna("").apply(len)

    fb["Date Published"] = pd.to_datetime(fb["Date Published"], errors="coerce")
    fb["month"] = fb["Date Published"].dt.to_period("M").astype(str)
    fb["total_engagement"] = fb[["share_count","reaction_count","comment_count"]].fillna(0).sum(axis=1)

    return fake, true, combined, fb

fake_df, true_df, combined_df, fb_df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filters")
    selected_label = st.multiselect("Article Type", ["Fake", "Real"], default=["Fake", "Real"])
    all_subjects = sorted(combined_df["subject"].dropna().unique())
    selected_subjects = st.multiselect("Subjects / Topics", all_subjects, default=all_subjects)
    st.caption("ℹ️ Real articles use subjects: **politicsNews**, **worldnews**")
    year_min, year_max = int(combined_df["year"].min()), int(combined_df["year"].max())
    year_range = st.slider("Year Range", year_min, year_max, (year_min, year_max))

    st.markdown("---")
    st.markdown("### 📡 Facebook Filter")
    fb_categories = st.multiselect("FB Category", sorted(fb_df["Category"].dropna().unique()), default=sorted(fb_df["Category"].dropna().unique()))
    fb_ratings = st.multiselect("FB Rating", sorted(fb_df["Rating"].dropna().unique()), default=sorted(fb_df["Rating"].dropna().unique()))

# ── Apply Filters ─────────────────────────────────────────────────────────────
filtered = combined_df[
    (combined_df["label"].isin(selected_label)) &
    (combined_df["subject"].isin(selected_subjects)) &
    (combined_df["year"].between(year_range[0], year_range[1]))
].copy()

fb_filtered = fb_df[
    (fb_df["Category"].isin(fb_categories)) &
    (fb_df["Rating"].isin(fb_ratings))
].copy()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <div>
    <h1>🔍 Fake vs Real News Analyzer</h1>
    <p>Multi-source analysis · Facebook Fact-Check · Article-Level Intelligence</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
total   = len(filtered)
n_fake  = len(filtered[filtered["label"] == "Fake"])
n_real  = len(filtered[filtered["label"] == "Real"])
avg_wc  = int(filtered["word_count"].mean()) if total else 0
fb_eng  = int(fb_filtered["total_engagement"].mean()) if len(fb_filtered) else 0

kpi_html = f"""
<div class="kpi-row">
  <div class="kpi-card"><div class="kpi-val">{total:,}</div><div class="kpi-label">Total Articles</div></div>
  <div class="kpi-card"><div class="kpi-val" style="color:#f87171">{n_fake:,}</div><div class="kpi-label">Fake Articles</div></div>
  <div class="kpi-card"><div class="kpi-val" style="color:#34d399">{n_real:,}</div><div class="kpi-label">Real Articles</div></div>
  <div class="kpi-card"><div class="kpi-val">{avg_wc:,}</div><div class="kpi-label">Avg Word Count</div></div>
  <div class="kpi-card"><div class="kpi-val">{fb_eng:,}</div><div class="kpi-label">Avg FB Engagement</div></div>
  <div class="kpi-card"><div class="kpi-val">{len(fb_filtered):,}</div><div class="kpi-label">FB Posts Analyzed</div></div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Distribution", "📈 Trends", "🏷️ Category",
    "📱 Platform", "💬 Engagement", "📝 Text Analysis", "🔥 Advanced"
])

COLORS = {"Fake": "#f87171", "Real": "#34d399"}
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#1e1b4b"),
    margin=dict(t=40, b=30, l=30, r=20),
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 – DISTRIBUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Fake vs Real — Donut</div>', unsafe_allow_html=True)
        dist = filtered["label"].value_counts().reset_index()
        dist.columns = ["label","count"]
        fig = px.pie(dist, names="label", values="count", hole=0.55,
                     color="label", color_discrete_map=COLORS)
        fig.update_traces(textinfo="percent+label", pull=[0.04, 0.04])
        fig.update_layout(**PLOTLY_THEME, showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Subject Distribution by Label</div>', unsafe_allow_html=True)
        subj = filtered.groupby(["subject","label"]).size().reset_index(name="count")
        fig2 = px.bar(subj, x="subject", y="count", color="label",
                      color_discrete_map=COLORS, barmode="group")
        fig2.update_layout(**PLOTLY_THEME, height=320,
                           xaxis=dict(tickangle=-35), legend_title="")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-title">Word Count Distribution</div>', unsafe_allow_html=True)
        fig3 = px.histogram(filtered, x="word_count", color="label",
                            color_discrete_map=COLORS, nbins=60, barmode="overlay", opacity=0.75)
        fig3.update_layout(**PLOTLY_THEME, height=300,
                           xaxis_title="Word Count", yaxis_title="Articles")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<div class="section-title">Title Length Distribution</div>', unsafe_allow_html=True)
        fig4 = px.box(filtered, x="label", y="title_len", color="label",
                      color_discrete_map=COLORS, points="outliers")
        fig4.update_layout(**PLOTLY_THEME, height=300,
                           yaxis_title="Title Length (chars)", showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 – TRENDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown('<div class="section-title">Monthly Article Volume — Fake vs Real</div>', unsafe_allow_html=True)
    trend = filtered.groupby(["month","label"]).size().reset_index(name="count")
    trend = trend.sort_values("month")
    fig5 = px.line(trend, x="month", y="count", color="label",
                   color_discrete_map=COLORS, markers=True)
    fig5.update_layout(**PLOTLY_THEME, height=340,
                       xaxis_title="Month", yaxis_title="Article Count", legend_title="")
    st.plotly_chart(fig5, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Yearly Article Count</div>', unsafe_allow_html=True)
        yearly = filtered.groupby(["year","label"]).size().reset_index(name="count")
        fig6 = px.bar(yearly, x="year", y="count", color="label",
                      color_discrete_map=COLORS, barmode="stack")
        fig6.update_layout(**PLOTLY_THEME, height=300, legend_title="")
        st.plotly_chart(fig6, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Cumulative Article Growth</div>', unsafe_allow_html=True)
        cum = trend.copy()
        cum = cum.sort_values(["label","month"])
        cum["cumulative"] = cum.groupby("label")["count"].cumsum()
        fig7 = px.area(cum, x="month", y="cumulative", color="label",
                       color_discrete_map=COLORS)
        fig7.update_layout(**PLOTLY_THEME, height=300, legend_title="")
        st.plotly_chart(fig7, use_container_width=True)

    st.markdown('<div class="section-title">Facebook Post Trend Over Time</div>', unsafe_allow_html=True)
    fb_trend = fb_filtered.groupby(["month","Category"]).size().reset_index(name="count").sort_values("month")
    fig8 = px.line(fb_trend, x="month", y="count", color="Category",
                   color_discrete_sequence=px.colors.qualitative.Vivid, markers=True)
    fig8.update_layout(**PLOTLY_THEME, height=300, legend_title="")
    st.plotly_chart(fig8, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 – CATEGORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Top Fake News Subjects</div>', unsafe_allow_html=True)
        fake_subj = fake_df["subject"].value_counts().head(8).reset_index()
        fake_subj.columns = ["subject","count"]
        fig9 = px.bar(fake_subj, x="count", y="subject", orientation="h",
                      color="count", color_continuous_scale="Reds")
        fig9.update_layout(**PLOTLY_THEME, height=320, coloraxis_showscale=False,
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig9, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Top Real News Subjects</div>', unsafe_allow_html=True)
        true_subj = true_df["subject"].value_counts().head(8).reset_index()
        true_subj.columns = ["subject","count"]
        fig10 = px.bar(true_subj, x="count", y="subject", orientation="h",
                       color="count", color_continuous_scale="Greens")
        fig10.update_layout(**PLOTLY_THEME, height=320, coloraxis_showscale=False,
                            yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig10, use_container_width=True)

    st.markdown('<div class="section-title">Facebook Rating Breakdown by Category</div>', unsafe_allow_html=True)
    rating_cat = fb_filtered.groupby(["Category","Rating"]).size().reset_index(name="count")
    fig11 = px.bar(rating_cat, x="Category", y="count", color="Rating",
                   barmode="stack", color_discrete_sequence=px.colors.qualitative.Set2)
    fig11.update_layout(**PLOTLY_THEME, height=320, legend_title="Rating")
    st.plotly_chart(fig11, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 – PLATFORM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Facebook Post Types</div>', unsafe_allow_html=True)
        pt = fb_filtered["Post Type"].value_counts().reset_index()
        pt.columns = ["type","count"]
        fig12 = px.pie(pt, names="type", values="count", hole=0.45,
                       color_discrete_sequence=px.colors.qualitative.Pastel)
        fig12.update_layout(**PLOTLY_THEME, height=300)
        st.plotly_chart(fig12, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Political Lean (FB Category)</div>', unsafe_allow_html=True)
        cat_ct = fb_filtered["Category"].value_counts().reset_index()
        cat_ct.columns = ["category","count"]
        fig13 = px.bar(cat_ct, x="category", y="count",
                       color="category",
                       color_discrete_map={"mainstream":"#6366f1","right":"#ef4444","left":"#3b82f6"})
        fig13.update_layout(**PLOTLY_THEME, height=300, showlegend=False)
        st.plotly_chart(fig13, use_container_width=True)

    st.markdown('<div class="section-title">Top 15 Facebook Pages by Post Count</div>', unsafe_allow_html=True)
    top_pages = fb_filtered["Page"].value_counts().head(15).reset_index()
    top_pages.columns = ["page","count"]
    fig14 = px.bar(top_pages, x="count", y="page", orientation="h",
                   color="count", color_continuous_scale="Purples")
    fig14.update_layout(**PLOTLY_THEME, height=420, coloraxis_showscale=False,
                        yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig14, use_container_width=True)

    st.markdown('<div class="section-title">Post Type vs Rating Heatmap</div>', unsafe_allow_html=True)
    heat = fb_filtered.groupby(["Post Type","Rating"]).size().unstack(fill_value=0)
    fig15 = px.imshow(heat, color_continuous_scale="Purples", text_auto=True, aspect="auto")
    fig15.update_layout(**PLOTLY_THEME, height=300)
    st.plotly_chart(fig15, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 – ENGAGEMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Avg Engagement by Rating</div>', unsafe_allow_html=True)
        eng_rating = fb_filtered.groupby("Rating")[["share_count","reaction_count","comment_count"]].mean().reset_index()
        eng_m = eng_rating.melt(id_vars="Rating", var_name="metric", value_name="avg")
        fig16 = px.bar(eng_m, x="Rating", y="avg", color="metric", barmode="group",
                       color_discrete_sequence=["#7c3aed","#a78bfa","#c4b5fd"])
        fig16.update_layout(**PLOTLY_THEME, height=320, xaxis_tickangle=-25, legend_title="")
        st.plotly_chart(fig16, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Avg Engagement by Category</div>', unsafe_allow_html=True)
        eng_cat = fb_filtered.groupby("Category")[["share_count","reaction_count","comment_count"]].mean().reset_index()
        eng_cm = eng_cat.melt(id_vars="Category", var_name="metric", value_name="avg")
        fig17 = px.bar(eng_cm, x="Category", y="avg", color="metric", barmode="group",
                       color_discrete_sequence=["#0ea5e9","#38bdf8","#7dd3fc"])
        fig17.update_layout(**PLOTLY_THEME, height=320, legend_title="")
        st.plotly_chart(fig17, use_container_width=True)

    st.markdown('<div class="section-title">Engagement Scatter: Shares vs Reactions (sized by Comments)</div>', unsafe_allow_html=True)
    fb_scat = fb_filtered.dropna(subset=["share_count","reaction_count","comment_count"])
    fig18 = px.scatter(fb_scat, x="share_count", y="reaction_count",
                       size="comment_count", color="Category",
                       hover_data=["Page","Rating"],
                       color_discrete_map={"mainstream":"#6366f1","right":"#ef4444","left":"#3b82f6"},
                       size_max=30)
    fig18.update_layout(**PLOTLY_THEME, height=380,
                        xaxis_title="Share Count", yaxis_title="Reaction Count")
    st.plotly_chart(fig18, use_container_width=True)

    st.markdown('<div class="section-title">Top 10 Pages by Total Engagement</div>', unsafe_allow_html=True)
    top_eng = fb_filtered.groupby("Page")["total_engagement"].sum().nlargest(10).reset_index()
    fig19 = px.bar(top_eng, x="total_engagement", y="Page", orientation="h",
                   color="total_engagement", color_continuous_scale="Teal")
    fig19.update_layout(**PLOTLY_THEME, height=340, coloraxis_showscale=False,
                        yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig19, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6 – TEXT ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab6:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Average Word Count by Subject & Label</div>', unsafe_allow_html=True)
        wc_subj = filtered.groupby(["subject","label"])["word_count"].mean().reset_index()
        fig20 = px.bar(wc_subj, x="subject", y="word_count", color="label",
                       color_discrete_map=COLORS, barmode="group")
        fig20.update_layout(**PLOTLY_THEME, height=320, xaxis_tickangle=-35, legend_title="")
        st.plotly_chart(fig20, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Word Count Violin Plot</div>', unsafe_allow_html=True)
        fig21 = px.violin(filtered, x="label", y="word_count", color="label",
                          box=True, points=False, color_discrete_map=COLORS)
        fig21.update_layout(**PLOTLY_THEME, height=320, showlegend=False)
        st.plotly_chart(fig21, use_container_width=True)

    # Top words
    st.markdown('<div class="section-title">Top 20 Words — Fake vs Real (excluding stopwords)</div>', unsafe_allow_html=True)
    STOPWORDS = set(["the","a","an","to","of","and","in","is","it","that","this",
                     "was","for","on","are","with","as","at","be","by","or","from",
                     "have","had","he","she","they","we","you","i","said","his",
                     "her","their","has","but","not","will","its","been","one",
                     "all","which","who","would","can","more","also","were","do",
                     "about","up","when","there","so","than","then","if","no","s",
                     "our","after","out","into","my","what","your","how","over",
                     "new","amp","could","did","some","other","just","like","trump",
                     "people","us","president","donald"])

    def top_words(df, n=20):
        words = " ".join(df["text"].fillna("").str.lower().values)
        tokens = re.findall(r"[a-z]{4,}", words)
        return Counter(t for t in tokens if t not in STOPWORDS).most_common(n)

    col1, col2 = st.columns(2)
    with col1:
        fw = pd.DataFrame(top_words(fake_df), columns=["word","freq"])
        figW1 = px.bar(fw, x="freq", y="word", orientation="h",
                       color="freq", color_continuous_scale="Reds",
                       title="Fake News — Top Words")
        figW1.update_layout(**PLOTLY_THEME, height=450, coloraxis_showscale=False,
                            yaxis=dict(autorange="reversed"))
        st.plotly_chart(figW1, use_container_width=True)

    with col2:
        tw = pd.DataFrame(top_words(true_df), columns=["word","freq"])
        figW2 = px.bar(tw, x="freq", y="word", orientation="h",
                       color="freq", color_continuous_scale="Greens",
                       title="Real News — Top Words")
        figW2.update_layout(**PLOTLY_THEME, height=450, coloraxis_showscale=False,
                            yaxis=dict(autorange="reversed"))
        st.plotly_chart(figW2, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 7 – ADVANCED
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab7:
    st.markdown('<div class="section-title">Fake vs Real Ratio Heatmap (Subject × Year)</div>', unsafe_allow_html=True)
    pivot = filtered.groupby(["year","subject","label"]).size().unstack(fill_value=0)
    if "Fake" in pivot.columns and "Real" in pivot.columns:
        pivot["fake_ratio"] = pivot["Fake"] / (pivot["Fake"] + pivot["Real"]).replace(0, np.nan)
        ratio_pivot = pivot["fake_ratio"].unstack(level="subject").fillna(0)
        figH = px.imshow(ratio_pivot, color_continuous_scale="RdYlGn_r",
                         zmin=0, zmax=1, text_auto=".2f", aspect="auto",
                         labels=dict(color="Fake Ratio"))
        figH.update_layout(**PLOTLY_THEME, height=350)
        st.plotly_chart(figH, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Word Count vs Title Length Scatter</div>', unsafe_allow_html=True)
        samp = filtered.sample(min(2000, len(filtered)), random_state=42)
        figS = px.scatter(samp, x="title_len", y="word_count", color="label",
                          color_discrete_map=COLORS, opacity=0.5)
        figS.update_layout(**PLOTLY_THEME, height=300, legend_title="")
        st.plotly_chart(figS, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Engagement Correlation Matrix (FB)</div>', unsafe_allow_html=True)
        corr = fb_filtered[["share_count","reaction_count","comment_count"]].corr()
        figC = px.imshow(corr, text_auto=".2f", color_continuous_scale="Purples",
                         zmin=0, zmax=1)
        figC.update_layout(**PLOTLY_THEME, height=300)
        st.plotly_chart(figC, use_container_width=True)

    st.markdown('<div class="section-title">Monthly Fake-to-Real Ratio Over Time</div>', unsafe_allow_html=True)
    ratio_trend = filtered.groupby(["month","label"]).size().unstack(fill_value=0)
    if "Fake" in ratio_trend.columns and "Real" in ratio_trend.columns:
        ratio_trend["ratio"] = ratio_trend["Fake"] / (ratio_trend["Real"].replace(0, np.nan))
        ratio_trend = ratio_trend.reset_index().sort_values("month")
        figR = px.line(ratio_trend, x="month", y="ratio", markers=True,
                       color_discrete_sequence=["#f59e0b"])
        figR.add_hline(y=1, line_dash="dash", line_color="#6b7280",
                       annotation_text="Equal ratio", annotation_position="right")
        figR.update_layout(**PLOTLY_THEME, height=300,
                           yaxis_title="Fake / Real Ratio")
        st.plotly_chart(figR, use_container_width=True)

    st.markdown('<div class="section-title">Facebook Debate Flag Analysis</div>', unsafe_allow_html=True)
    if "Debate" in fb_filtered.columns:
        deb = fb_filtered.copy()
        deb["Debate"] = deb["Debate"].fillna("No")
        deb_cnt = deb.groupby(["Rating","Debate"]).size().reset_index(name="count")
        figD = px.bar(deb_cnt, x="Rating", y="count", color="Debate",
                      barmode="stack",
                      color_discrete_map={"yes":"#f87171","no":"#a5b4fc","No":"#a5b4fc"})
        figD.update_layout(**PLOTLY_THEME, height=300, xaxis_tickangle=-20, legend_title="Debate")
        st.plotly_chart(figD, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#9ca3af; font-size:0.8rem; margin-top:2rem; padding-top:1rem; border-top:1px solid #e5e7eb;">
  Fake News Dashboard · Built with Streamlit & Plotly · Data: Fake.csv, True.csv, facebook-fact-check.csv
</div>
""", unsafe_allow_html=True)