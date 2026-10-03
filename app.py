import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.forecast import load_data, build_features, predict, actuals, grap_stage
from src.advisor import answer

st.set_page_config(page_title="Delhi AQI Early Warning", page_icon="🌫️", layout="wide",
                   initial_sidebar_state="expanded")

# custom css, cards aur header ke liye
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1200px;}
.hero {padding: 1.6rem 2rem; border-radius: 18px; margin-bottom: 1rem;
       background: linear-gradient(120deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
       border: 1px solid rgba(255,255,255,0.08);}
.hero h1 {margin: 0; font-size: 2.1rem; color: #fff;}
.hero p {margin: 0.4rem 0 0.8rem 0; color: #b8c7d9; font-size: 1.02rem;}
.pill {display: inline-block; padding: 0.15rem 0.75rem; border-radius: 999px; font-size: 0.78rem;
       margin-right: 0.4rem; background: rgba(255,255,255,0.1); color: #dbe7f5;}
.banner {padding: 0.9rem 1.2rem; border-radius: 12px; margin-bottom: 1rem; font-size: 1rem; color: #fff;}
.card {background: #121a2b; border: 1px solid rgba(255,255,255,0.07); border-radius: 16px; padding: 1.1rem 1.3rem;}
.card .lbl {color: #9fb0c7; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em;}
.card .date {color: #e8eef9; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.3rem;}
.card .big {font-size: 3.3rem; font-weight: 800; line-height: 1.05;}
.card .cat {color: #9fb0c7; font-size: 0.9rem; margin-bottom: 0.7rem;}
.card .row {display: flex; justify-content: space-between; color: #c9d6e8; font-size: 0.92rem;
            padding: 0.35rem 0; border-top: 1px solid rgba(255,255,255,0.06);}
.badge {display: inline-block; padding: 0.2rem 0.75rem; border-radius: 999px; font-size: 0.8rem;
        font-weight: 700; color: #fff;}
</style>
""", unsafe_allow_html=True)


def aqi_color(a):
    # cpcb aqi categories ke rang
    if a is None: return "#6b7a90"
    if a <= 50: return "#00b050"
    if a <= 100: return "#7cc04b"
    if a <= 200: return "#e0b800"
    if a <= 300: return "#ff9900"
    if a <= 400: return "#ff4d4d"
    return "#a50021"


def aqi_cat(a):
    if a is None: return "n/a"
    if a <= 50: return "Good"
    if a <= 100: return "Satisfactory"
    if a <= 200: return "Moderate"
    if a <= 300: return "Poor"
    if a <= 400: return "Very Poor"
    return "Severe"


def stage_badge(stage, aqi):
    if stage is None:
        return '<span class="badge" style="background:#2e7d4f">No GRAP stage</span>'
    return f'<span class="badge" style="background:{aqi_color(aqi)}">GRAP {stage}</span>'


@st.cache_data
def get_data():
    df = load_data()
    return df, build_features(df)


df, feat = get_data()
STAGES = ["Stage I", "Stage II", "Stage III", "Stage IV"]
DAYS = ["day1", "day2", "day3"]

# sidebar
with st.sidebar:
    st.header("Replay date")
    d = st.date_input("Forecast made at 11 PM on",
                      value=dt.date(2025, 11, 10),
                      min_value=dt.date(2025, 1, 4), max_value=dt.date(2025, 12, 28))
    st.caption("2025 is the test year, so the model never saw these days during training. "
               "Try 10 Nov 2025 for a winter pollution episode.")
    st.divider()
    st.markdown("**How to read this**")
    st.caption("Each card is the 24-hour average AQI, which is what GRAP uses to decide the stage. "
               "Expected = median model. Worst case = 80th percentile model.")

ts = pd.Timestamp(d) + pd.Timedelta(hours=23)
pred = predict(feat, ts)

# hero
st.markdown("""
<div class="hero">
  <h1>🌫️ Delhi Air Quality Early Warning</h1>
  <p>3-day AQI forecast with GRAP stage alerts and an advisor that answers from the official CAQM document.</p>
  <span class="pill">XGBoost + quantile models</span>
  <span class="pill">RAG on CAQM GRAP (rev. 29.09.2026)</span>
  <span class="pill">Replay demo</span>
</div>
""", unsafe_allow_html=True)

st.info(f"Replay mode: showing the forecast made at 11 PM on {d.strftime('%d %b %Y')}. "
        "Change the date from the sidebar on the left.")

if pred is None:
    st.warning("Data for this date is incomplete, please pick another date.")
    st.stop()

act = actuals(df, ts)
day_label = {k: (ts + pd.Timedelta(days=i + 1)).strftime("%a, %d %b") for i, k in enumerate(DAYS)}

# alert banner, worst case ke hisaab se
top_idx, top_day = -1, None
for k in DAYS:
    s = pred[k]["stage_worst"]
    if s and STAGES.index(s) > top_idx:
        top_idx, top_day = STAGES.index(s), k
if top_idx < 0:
    st.markdown('<div class="banner" style="background:#1f7a4d">✅ No GRAP stage expected in the next 3 days, '
                'even in the worst-case estimate.</div>', unsafe_allow_html=True)
else:
    col = aqi_color([250, 350, 420, 480][top_idx])
    st.markdown(f'<div class="banner" style="background:{col}">⚠️ Worst case reaches GRAP {STAGES[top_idx]} '
                f'on {day_label[top_day]}. Open the GRAP Advisor tab to see what applies.</div>',
                unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🤖 GRAP Advisor", "ℹ️ About"])

# forecast tab
with tab1:
    cols = st.columns(3)
    for col, k in zip(cols, DAYS):
        p = pred[k]
        a = act[k]
        actual_txt = (f"{a} &nbsp;<span style='color:#9fb0c7'>(error {p['expected'] - a:+d})</span>"
                      if a is not None else "n/a")
        col.markdown(f"""
<div class="card">
  <div class="lbl">{['Next 24 hours', '24 to 48 hours', '48 to 72 hours'][DAYS.index(k)]}</div>
  <div class="date">{day_label[k]}</div>
  <div class="big" style="color:{aqi_color(p['expected'])}">{p['expected']}</div>
  <div class="cat">expected AQI · {aqi_cat(p['expected'])}</div>
  <div class="row"><span>Worst case</span><b style="color:{aqi_color(p['worst_case'])}">{p['worst_case']}</b></div>
  <div class="row"><span>Actual (replay)</span><b>{actual_txt}</b></div>
  <div class="row"><span>Expected stage</span>{stage_badge(p['stage_expected'], p['expected'])}</div>
  <div class="row"><span>Worst-case stage</span>{stage_badge(p['stage_worst'], p['worst_case'])}</div>
</div>
""", unsafe_allow_html=True)

    st.write("")
    hist = df["AQI"].loc[ts - pd.Timedelta(days=7): ts]
    fut = df["AQI"].loc[ts: ts + pd.Timedelta(hours=72)]
    mid = [ts + pd.Timedelta(hours=12 + 24 * i) for i in range(3)]
    exp = [pred[k]["expected"] for k in DAYS]
    up = [pred[k]["worst_case"] - pred[k]["expected"] for k in DAYS]

    fig = go.Figure()
    # aqi category ke bands peeche
    for y0, y1, c in [(0, 50, "#00b050"), (50, 100, "#7cc04b"), (100, 200, "#e0b800"),
                      (200, 300, "#ff9900"), (300, 400, "#ff4d4d"), (400, 520, "#a50021")]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=c, opacity=0.07, line_width=0)
    # grap thresholds
    for y, name in [(200, "Stage I"), (300, "Stage II"), (400, "Stage III"), (450, "Stage IV")]:
        fig.add_hline(y=y, line_dash="dot", line_color="rgba(255,255,255,0.25)",
                      annotation_text=name, annotation_position="top left",
                      annotation_font_color="#9fb0c7")

    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="Observed AQI (hourly)",
                             line=dict(color="#4cc9f0", width=1.6)))
    fig.add_trace(go.Scatter(x=fut.index, y=fut.values, mode="lines", name="Actual after forecast (replay)",
                             line=dict(color="#9fb0c7", width=1.4, dash="dash")))
    fig.add_trace(go.Scatter(x=mid, y=exp, mode="lines+markers", name="Forecast (bar goes up to worst case)",
                             line=dict(color="#ffb703", width=2), marker=dict(size=11, color="#ffb703"),
                             error_y=dict(type="data", symmetric=False, array=up, arrayminus=[0, 0, 0],
                                          color="#ffb703", thickness=2, width=7)))
    fig.add_shape(type="line", x0=ts, x1=ts, y0=0, y1=1, yref="paper",
                  line=dict(color="#ffffff", width=1, dash="dot"))
    fig.add_annotation(x=ts, y=1, yref="paper", text="forecast starts", showarrow=False,
                       yanchor="bottom", font=dict(size=11, color="#b8c7d9"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=440, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      yaxis=dict(range=[0, 520], title="AQI"),
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig)

# advisor tab
with tab2:
    c1, c2 = st.columns([1, 1])
    day = c1.selectbox("Which day", DAYS, format_func=lambda x: f"{day_label[x]} ({['next 24h', '24 to 48h', '48 to 72h'][DAYS.index(x)]})")
    which = c2.radio("Which estimate", ["expected", "worst_case"],
                     format_func=lambda x: "Expected" if x == "expected" else "Worst case", horizontal=True)
    aqi_pick = pred[day][which]
    st.markdown(f"Advisor is answering for **AQI {aqi_pick}** ({aqi_cat(aqi_pick)}) &nbsp; "
                f"{stage_badge(grap_stage(aqi_pick), aqi_pick)}", unsafe_allow_html=True)
    st.caption("The stage is decided by code from the document thresholds. "
               "The language model only explains what the document says for that stage.")

    chips = ["What restrictions apply?", "Can I use diesel generators?",
             "Which construction activities are banned?", "Are trucks allowed to enter Delhi?"]
    chip_cols = st.columns(len(chips))
    chip_q = None
    for cc, text in zip(chip_cols, chips):
        if cc.button(text, key="chip_" + text):
            chip_q = text

    if "messages" not in st.session_state:
        st.session_state.messages = []

    typed = st.chat_input("Ask about GRAP restrictions...")
    question = typed or chip_q

    # demo ke liye session limit
    if "asked" not in st.session_state:
        st.session_state.asked = 0
    if question and st.session_state.asked >= 10:
        st.warning("Demo limit reached for this session (10 questions). Refresh the page to start again.")
        question = None

    if question:
        question = question[:300]
        st.session_state.asked += 1
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Reading the GRAP document..."):
            reply = answer(aqi_pick, question)
        st.session_state.messages.append({
            "role": "assistant", "content": reply,
            "meta": f"Answered for {day_label[day]}, {which.replace('_', ' ')} AQI {aqi_pick}"})
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("meta"):
                st.caption(m["meta"])

    if st.session_state.messages and st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# about tab
with tab3:
    a1, a2, a3 = st.columns(3)
    a1.markdown("**Data**\n\nHourly AQI for Lodhi Road (IMD) from 2017 to 2025 plus weather from Open-Meteo. "
                "2024-25 AQI was calculated from PM2.5 and PM10 using CPCB breakpoints.")
    a2.markdown("**Forecast**\n\nXGBoost models predict the 24-hour average AQI for the next 3 days. "
                "A second set of 80th percentile models gives the worst case.")
    a3.markdown("**Advisor**\n\nRAG over the CAQM GRAP document. Retrieval is filtered by stage, "
                "and lower stages are included because GRAP actions are cumulative.")

    st.markdown("#### Results on 2025 (test year)")
    st.table(pd.DataFrame({
        "Horizon": ["Next 24h", "24 to 48h", "48 to 72h"],
        "MAE model": [27.8, 37.5, 40.8],
        "MAE baseline": [32.3, 44.5, 50.9],
        "Warning recall: median": [0.80, 0.56, 0.49],
        "Warning recall: worst case": [0.91, 0.87, 0.85],
        "Warning recall: baseline": [0.71, 0.59, 0.53],
    }))
    st.caption("Baseline = average AQI of the last 24 hours. Warning = Stage II or above (AQI above 300).")

    st.markdown("#### Limitations")
    st.markdown("- Severe days (Stage III and IV) are rare in the test year and the models do not catch them reliably.\n"
                "- Test period is a single year and does not include the monsoon gap in the data.\n"
                "- Replay demo only. It does not use live data yet.\n"
                "- Not official advice, check caqm.nic.in for the actual orders.")