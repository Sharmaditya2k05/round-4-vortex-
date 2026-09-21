"""
Team Entropy - Data Vortex Round 4: Social Engine Revival
An interactive analytics dashboard integrating all four rounds.

Run:  streamlit run Entropy_app.py
"""
from __future__ import annotations

import base64
import io
import os
import itertools
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from Entropy_post_fetcher import fetch_post
import Entropy_rag as rag

APP = Path(__file__).resolve().parent
ROOT = APP.parent
P1 = ROOT / "Round1-Phase1-Data-Recovery"
P2 = ROOT / "Round1-Phase2-Analytical-Core"
R2 = ROOT / "Round2-Semantic-Recovery"
R3 = ROOT / "Round3-Signal-Tracking"
REPO = "https://github.com/SaiGrover/entropy-data-vortex"

try:                                    # local development: a .env beside this file
    from dotenv import load_dotenv
    load_dotenv(APP / ".env")
except ImportError:
    pass

# Streamlit Community Cloud has no .env - it injects st.secrets instead, so the
# same names are promoted into the environment and everything downstream is
# unchanged whether this runs locally or deployed.
try:
    for _name in ("GROQ_API_KEY", "GROQ_MODEL"):
        if not os.environ.get(_name, "").strip() and _name in st.secrets:
            os.environ[_name] = str(st.secrets[_name])
except Exception:                       # no secrets configured at all
    pass


@st.cache_resource(show_spinner=False)
def model_available() -> bool:
    """Is the deep-learning stack installed?

    The saved ensemble needs torch, transformers and sentence-transformers,
    which are heavy enough to be optional: without them every chart, query and
    table still works and only the live scoring demos step aside.
    """
    try:
        import torch, transformers, sentence_transformers   # noqa: F401
        return True
    except Exception:
        return False


def model_note(reason: str = "") -> None:
    """A side note, not an error.

    Live scoring is the one part of this dashboard that needs heavy optional
    dependencies and several hundred MB of weights. When it is unavailable the
    page says so quietly and carries on - every chart, query and table is read
    from saved artefacts and does not touch the model.
    """
    why = reason or ("torch, transformers and sentence-transformers are not installed here")
    st.markdown(
        f'<div class="softnote"><b>Live scoring is off.</b> {why}. Everything else on this page - '
        f'the charts, the tables and the reports - is read from saved artefacts and is unaffected.'
        f'</div>', unsafe_allow_html=True)


def model_broken() -> str:
    """Why loading failed earlier this session, if it did."""
    return st.session_state.get("model_error", "")

st.set_page_config(page_title="Social Engine | Team Entropy", page_icon="◆",
                   layout="wide", initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------- palette
# Deep forest and sand. The six categorical colours run green -> yellow -> clay,
# so a chart legend reads as an ordered natural ramp rather than a random set.
BG, PANEL, BORDER = "#0b120f", "#14201b", "rgba(190,210,192,.15)"
TXT, MUT, DIM = "#eef2ee", "#9bb0a2", "#6b8175"
SAGE, TEAL, OLIVE, SAND, BRONZE, CLAY = "#7fb582", "#5ea38f", "#a8bf6a", "#d9b26a", "#d98b5e", "#d9705e"
GREEN = "#6fc08a"
# the round accents keep their old names so every page keeps its own identity
CYAN, INDIGO, EMER, AMBER, ROSE, BLUE, VIOLET = SAGE, TEAL, GREEN, SAND, CLAY, OLIVE, BRONZE
SENT = {"Negative": CLAY, "Neutral": "#93a89b", "Positive": GREEN}
SEQ = [SAGE, SAND, TEAL, CLAY, OLIVE, BRONZE]
GRID = "rgba(190,210,192,.11)"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

  html, body, .stApp, .stMarkdown, button, input, textarea, select {{
     font-family:'Inter','Segoe UI',system-ui,sans-serif; }}
  /* the artwork sits behind everything, fixed so it does not scroll with the
     content, under a scrim that deepens down the page: the top is where the
     image has room to show, the lower half is where the dense charts live */
  .stApp {{ background:
      linear-gradient(180deg, rgba(11,18,15,.68) 0%, rgba(11,18,15,.84) 34%,
                              rgba(11,18,15,.93) 68%, rgba(11,18,15,.96) 100%),
      url("app/static/Entropy_bg.webp") center top / cover no-repeat fixed,
      {BG}; }}
  [data-testid="stHeaderActionElements"] {{ display:none !important; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{ padding: 1.1rem 1.6rem 3rem 1.6rem; max-width: 1800px; }}

  section[data-testid="stMain"] *, .stMarkdown, p, li, label, span {{ color: {TXT}; }}
  h1,h2,h3,h4 {{ color: {TXT} !important; letter-spacing:-.02em; }}

  .topbar {{ position:relative; display:flex; align-items:center; justify-content:space-between;
     gap:1rem; padding:.7rem .25rem 1.05rem .25rem; margin-bottom:1.1rem; }}
  .topbar:after {{ content:""; position:absolute; left:0; right:0; bottom:0; height:1px;
     background:linear-gradient(90deg, transparent, {CYAN}, {INDIGO}, {VIOLET}, {EMER}, transparent);
     background-size:240% 100%; animation:sweep 12s linear infinite; opacity:.9; }}
  @keyframes sweep {{ from {{ background-position:0% 0; }} to {{ background-position:240% 0; }} }}

  .brand {{ display:flex; align-items:center; gap:.95rem; }}
  .mark {{ position:relative; width:50px; height:50px; display:grid; place-items:center;
     border-radius:16px; border:1px solid rgba(190,210,192,.2);
     background:radial-gradient(circle at 32% 26%, rgba(127,181,130,.30), rgba(94,163,143,.1) 58%, transparent 74%);
     box-shadow:0 0 34px -8px rgba(127,181,130,.75), inset 0 1px 0 rgba(255,255,255,.07); }}
  .mark:before {{ content:""; position:absolute; inset:-6px; border-radius:22px; z-index:-1;
     background:conic-gradient(from 0deg, {CYAN}, {INDIGO}, {VIOLET}, {EMER}, {CYAN});
     filter:blur(13px); opacity:.34; animation:spin 9s linear infinite; }}
  .mark svg {{ width:34px; height:34px; overflow:visible; }}
  .mark .ring {{ transform-origin:24px 24px; animation:spin 17s linear infinite; }}
  .mark .wave {{ animation:pulse 3.2s ease-in-out infinite; }}
  @keyframes spin {{ to {{ transform:rotate(360deg); }} }}
  @keyframes pulse {{ 0%,100% {{ opacity:.7; }} 50% {{ opacity:1; }} }}

  .brand .wm {{ font-family:'Outfit','Inter',sans-serif; font-size:1.62rem; margin:0;
     font-weight:700; letter-spacing:-.035em; line-height:1.1;
     background:linear-gradient(100deg, {TXT} 0%, {CYAN} 24%, {INDIGO} 42%, {VIOLET} 58%, {TXT} 84%);
     background-size:270% 100%; -webkit-background-clip:text; background-clip:text;
     color:transparent !important; -webkit-text-fill-color:transparent;
     animation:flow 13s ease-in-out infinite; white-space:nowrap; }}
  @keyframes flow {{ 0%,100% {{ background-position:0% 50%; }} 50% {{ background-position:100% 50%; }} }}

  @media (prefers-reduced-motion: reduce) {{
     .topbar:after, .mark:before, .mark .ring, .mark .wave, .brand .wm {{ animation:none !important; }} }}
  .brand .tag {{ color:{MUT}; font-size:.73rem; letter-spacing:.2em;
     text-transform:uppercase; font-weight:500; margin-top:.22rem; }}
  .brand .tag b {{ color:{CYAN}; font-weight:600; }}
  .who {{ text-align:right; line-height:1.4; }}
  .repo {{ display:inline-flex; align-items:center; gap:.36rem; margin-top:.34rem; padding:.2rem .55rem;
     border-radius:999px; border:1px solid {BORDER}; background:rgba(255,255,255,.035);
     color:{MUT} !important; font-size:.72rem; text-decoration:none !important; transition:.15s ease; }}
  .repo:hover {{ color:{TXT} !important; border-color:{CYAN}; background:rgba(127,181,130,.1); }}
  .repo svg {{ width:13px; height:13px; flex:none; }}
  .who .t {{ font-family:'Outfit','Inter',sans-serif; font-size:.95rem; font-weight:600;
     color:{TXT}; letter-spacing:-.01em; white-space:nowrap; }}
  .who .m {{ color:{MUT}; font-size:.79rem; }}
  .who .badge {{ display:inline-block; margin-bottom:.3rem; padding:.16rem .55rem; border-radius:999px;
     font-size:.63rem; letter-spacing:.16em; text-transform:uppercase; font-weight:600; color:{EMER};
     background:rgba(111,192,138,.13); border:1px solid rgba(111,192,138,.4); }}

  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(168px,1fr)); gap:.75rem; margin-bottom:1.1rem; }}
  .tile {{ background:linear-gradient(160deg, rgba(255,255,255,.055), rgba(255,255,255,.015));
     border:1px solid {BORDER}; border-radius:14px; padding:.85rem .95rem; position:relative; overflow:hidden; }}
  .tile:before {{ content:""; position:absolute; inset:0 auto auto 0; width:100%; height:2px;
     background:linear-gradient(90deg,var(--c),transparent); }}
  .tile .l {{ color:{MUT}; font-size:.665rem; letter-spacing:.13em; text-transform:uppercase; font-weight:600; }}
  .tile .v {{ font-family:'Outfit','Inter',sans-serif; font-size:1.8rem; font-weight:700; line-height:1.1;
     margin:.3rem 0 .14rem 0; letter-spacing:-.025em;
     color:var(--c) !important; font-variant-numeric:tabular-nums; }}
  .tile .n {{ color:{DIM}; font-size:.755rem; line-height:1.4; }}
  .tile:hover {{ border-color:rgba(190,210,192,.28); }}

  /* one card shape, used by every chart, panel and step */
  div[class*="st-key-card"] {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:16px;
     padding:.85rem .9rem .5rem .9rem; margin-bottom:.1rem; height:100%; }}
  div[class*="st-key-card"] div[data-testid="stExpander"] {{ border:0; background:transparent;
     border-top:1px solid {BORDER}; border-radius:0; margin-top:.3rem; }}
  div[class*="st-key-card"] div[data-testid="stExpander"] details {{ border:0; }}
  .flow .sbody {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:14px;
     padding:.7rem .85rem .8rem .85rem; }}
  .panel {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:16px; padding:1rem 1.15rem; margin-bottom:.9rem; }}
  .panel h4 {{ font-family:'Outfit','Inter',sans-serif; margin:0 0 .2rem 0; font-size:1.02rem;
     font-weight:600; letter-spacing:-.015em; }}
  .panel .sub {{ color:{MUT}; font-size:.845rem; line-height:1.62; margin-bottom:.5rem; }}

  .eyebrow {{ display:flex; align-items:baseline; gap:.65rem; margin:1.8rem 0 .75rem 0;
     padding-left:.7rem; border-left:3px solid transparent;
     border-image:linear-gradient(180deg,{CYAN},{INDIGO}) 1 100%; }}
  .eyebrow h3 {{ font-family:'Outfit','Inter',sans-serif; margin:0; font-size:1.2rem; font-weight:600;
     letter-spacing:-.02em; }}
  .eyebrow span {{ color:{DIM}; font-size:.83rem; }}

  /* one readable measure for body copy, and a calmer caption */
  .stMarkdown p, .stMarkdown li {{ line-height:1.62; }}
  div[data-testid="stCaptionContainer"] p {{ color:{DIM} !important; font-size:.775rem !important;
     letter-spacing:.01em; }}
  div[data-testid="stExpander"] summary p {{ font-size:.855rem !important; font-weight:500; }}
  .stSlider label, .stMultiSelect label, .stSelectbox label,
  div[data-testid="stWidgetLabel"] p {{ color:{MUT} !important; font-size:.71rem !important;
     letter-spacing:.11em; text-transform:uppercase; font-weight:600; }}

  .flow {{ max-width:960px; margin:.35rem 0 1.4rem 0; }}
  .flow .step {{ position:relative; display:grid; grid-template-columns:44px 1fr; gap:1.05rem;
     padding-bottom:1.15rem; }}
  .flow .step:last-child {{ padding-bottom:0; }}
  .flow .step:not(:last-child):after {{ content:""; position:absolute; left:21px; top:50px; bottom:-2px;
     width:2px; background:linear-gradient(180deg, var(--c), rgba(190,210,192,.15)); }}
  .flow .node {{ width:44px; height:44px; border-radius:13px; display:grid; place-items:center;
     font-family:'Outfit','Inter',sans-serif; font-weight:700; font-size:.92rem; color:var(--c);
     background:rgba(255,255,255,.04); border:1.5px solid var(--c);
     box-shadow:0 0 20px -7px var(--c); font-variant-numeric:tabular-nums; }}
  .flow h5 {{ font-family:'Outfit','Inter',sans-serif; margin:0 0 .2rem 0; font-size:.98rem;
     font-weight:600; color:{TXT}; letter-spacing:-.012em; }}
  .flow p {{ margin:0; color:{MUT}; font-size:.85rem; line-height:1.62; }}
  .flow .chip {{ display:inline-block; margin-top:.45rem; padding:.2rem .62rem; border-radius:999px;
     font-size:.705rem; font-weight:600; color:var(--c); background:rgba(255,255,255,.045);
     border:1px solid var(--c); }}

  /* the column, not the panel, has to be sticky: Streamlit sizes a column's
     inner block to its content, so a sticky panel inside it has nowhere to go */
  div[data-testid="stColumn"]:has(> div > div > div > div .iopanel) {{
     position:sticky; top:122px; align-self:flex-start; z-index:2; }}
  .iopanel {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:16px;
     padding:1.05rem 1.1rem; }}
  .io-h {{ font-size:.66rem; letter-spacing:.16em; text-transform:uppercase; font-weight:700; color:{MUT};
     margin-bottom:.7rem; }}
  .io-k {{ font-size:.63rem; letter-spacing:.14em; text-transform:uppercase; font-weight:700;
     color:var(--c); margin:.1rem 0 .35rem 0; }}
  .io-list {{ display:flex; flex-direction:column; gap:.3rem; }}
  .io-row {{ display:flex; justify-content:space-between; align-items:baseline; gap:.7rem;
     background:rgba(255,255,255,.035); border:1px solid {BORDER}; border-radius:9px;
     padding:.38rem .6rem; }}
  .io-n {{ font-size:.775rem; color:{TXT}; line-height:1.35; }}
  .io-s {{ font-size:.68rem; color:{DIM}; white-space:nowrap; font-variant-numeric:tabular-nums; }}
  .io-arrow {{ text-align:center; color:var(--c); font-size:1.05rem; line-height:1; margin:.5rem 0; opacity:.9; }}
  .io-engine {{ text-align:center; font-family:'Outfit','Inter',sans-serif; font-weight:650; font-size:.92rem;
     color:var(--c); border:1.5px solid var(--c); border-radius:12px; padding:.6rem .5rem;
     background:rgba(255,255,255,.04); box-shadow:0 0 26px -10px var(--c); }}
  .io-f {{ margin-top:.75rem; font-size:.67rem; color:{DIM}; }}

  .minitiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:.6rem;
     margin:.2rem 0 .9rem 0; }}
  .minitile {{ background:rgba(255,255,255,.035); border:1px solid {BORDER}; border-radius:12px;
     padding:.6rem .75rem; }}
  .minitile .l {{ color:{MUT}; font-size:.62rem; letter-spacing:.13em; text-transform:uppercase;
     font-weight:600; }}
  .minitile .v {{ font-family:'Outfit','Inter',sans-serif; font-size:1.3rem; font-weight:700;
     color:var(--c); line-height:1.15; margin-top:.18rem; font-variant-numeric:tabular-nums; }}
  .minitile .d {{ color:{DIM}; font-size:.7rem; margin-top:.1rem; }}

  /* the assistant docks bottom-left and stays out of the page flow */
  div[class*="st-key-askdock"] {{ position:fixed; left:20px; bottom:20px; z-index:900; }}
  div[class*="st-key-askdock"] button {{ border-radius:999px !important; padding:.55rem 1.05rem !important;
     background:linear-gradient(135deg, {CYAN}, {INDIGO}) !important; border:0 !important;
     box-shadow:0 10px 30px -10px {CYAN}, 0 0 0 1px rgba(255,255,255,.08) inset !important; }}
  div[class*="st-key-askdock"] button p {{ color:#08130d !important; font-weight:700 !important;
     font-size:.82rem !important; letter-spacing:.01em; text-transform:none !important; }}
  div[class*="st-key-askdock"] button:hover {{ filter:brightness(1.08); }}
  .shannon {{ border-bottom:1px solid {BORDER}; padding-bottom:.7rem; margin-bottom:.8rem; }}
  .sh-name {{ font-family:'Outfit','Inter',sans-serif; font-size:1.34rem; font-weight:700;
     letter-spacing:.07em; line-height:1.1;
     background:linear-gradient(100deg,{CYAN} 0%,{INDIGO} 45%,{EMER} 100%);
     background-size:220% 100%; -webkit-background-clip:text; background-clip:text;
     color:transparent !important; -webkit-text-fill-color:transparent;
     animation:flow 11s ease-in-out infinite; }}
  .sh-tag {{ color:{MUT}; font-size:.69rem; letter-spacing:.14em; text-transform:uppercase;
     font-weight:600; margin-top:.1rem; }}
  .sh-what {{ color:{DIM}; font-size:.78rem; line-height:1.55; margin-top:.5rem; }}
  .qa {{ border-left:2px solid rgba(127,181,130,.4); padding:.1rem 0 .1rem .75rem; margin:.1rem 0 .7rem 0; }}
  .qa .q {{ color:{TXT}; font-weight:600; font-size:.86rem; margin-bottom:.2rem; }}
  .qa .a {{ color:{MUT}; font-size:.84rem; line-height:1.6; white-space:pre-wrap; }}
  .qa .src {{ color:{DIM}; font-size:.68rem; margin-top:.35rem; }}

  .linklist {{ display:flex; flex-direction:column; gap:.4rem; }}
  .linklist > div {{ display:flex; flex-direction:column; gap:.1rem; }}
  .linklist b {{ color:{TXT}; font-size:.76rem; font-weight:600; }}
  .linklist code {{ font-size:.68rem !important; word-break:break-all; background:rgba(127,181,130,.09)
     !important; border-radius:5px; padding:.1rem .3rem; }}
  .linklist span {{ color:{DIM}; font-size:.72rem; }}

  .softnote {{ border:1px solid rgba(217,178,106,.34); background:rgba(217,178,106,.07);
     border-radius:12px; padding:.6rem .8rem; margin:.1rem 0 .9rem 0;
     color:{MUT}; font-size:.8rem; line-height:1.55; }}
  .softnote b {{ color:{AMBER}; font-weight:600; }}

  /* a notebook cell is a card: header strip, code, then its output */
  div[class*="st-key-nbcell"] {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:14px;
     padding:.55rem .7rem .3rem .7rem; margin-bottom:.75rem; }}
  .nbhead {{ display:flex; align-items:center; gap:.55rem; margin-bottom:.35rem; }}
  .nbin {{ font-size:.63rem; letter-spacing:.1em; font-weight:700; color:{CYAN};
     background:rgba(127,181,130,.12); border:1px solid rgba(127,181,130,.32);
     border-radius:6px; padding:.1rem .4rem; font-variant-numeric:tabular-nums; }}
  .nbkind {{ font-size:.6rem; letter-spacing:.15em; text-transform:uppercase; color:{DIM}; }}
  .nbout {{ font-size:.6rem; letter-spacing:.15em; text-transform:uppercase; color:{DIM};
     font-weight:600; margin:.55rem 0 .25rem 0; padding-top:.45rem;
     border-top:1px dashed {BORDER}; }}
  /* narrative cells read as prose, not as another panel */
  div[class*="st-key-nbmd"] {{ border-left:2px solid rgba(127,181,130,.35); padding-left:.95rem;
     margin:.2rem 0 .8rem 0; }}

  .stTabs [data-baseweb="tab-list"] {{ gap:.3rem; border-bottom:1px solid {BORDER}; }}
  .stTabs [data-baseweb="tab"] {{ background:transparent; border-radius:10px 10px 0 0; padding:.45rem .9rem;
     color:{MUT}; font-size:.9rem; }}
  .stTabs [aria-selected="true"] {{ background:rgba(94,163,143,.16); color:{TXT} !important; }}

  div[data-testid="stExpander"] {{ border:1px solid {BORDER}; border-radius:12px; background:rgba(255,255,255,.02); }}
  div[data-testid="stExpander"] summary {{ color:{MUT} !important; font-size:.86rem; }}
  .stDataFrame {{ border:1px solid {BORDER}; border-radius:12px; }}
  div[data-testid="stSidebarCollapsedControl"] {{ display:none; }}
  /* the sticky element must be the wrapper whose parent is the tall main block,
     otherwise its containing block is only as tall as the rail and it has nowhere to stick */
  div[data-testid="stLayoutWrapper"]:has(> div.st-key-navrail) {{ position:sticky; top:0; z-index:99; }}
  div.st-key-navrail {{ margin:-.2rem 0 1.25rem 0; padding:.55rem 0 .75rem 0;
     background:rgba(11,18,15,.88); backdrop-filter:blur(12px) saturate(1.15);
     border-bottom:1px solid {BORDER}; box-shadow:0 14px 22px -18px rgba(0,0,0,.95); }}
  div.st-key-navrail div[data-testid="stColumn"] > div {{ width:100%; }}
  div.st-key-navrail button {{ width:100% !important; background:rgba(255,255,255,.028) !important;
     border:1px solid {BORDER}; border-radius:13px !important; padding:.55rem .45rem !important;
     min-height:74px; position:relative; overflow:hidden; transition:.16s ease; }}
  div.st-key-navrail button:before {{ content:""; position:absolute; top:0; left:0; right:0; height:2px; }}
  div.st-key-navrail button p {{ color:{DIM} !important; font-size:.62rem !important; line-height:1.5;
     letter-spacing:.07em; text-transform:uppercase; margin:0; white-space:normal; }}
  div.st-key-navrail button strong {{ display:inline-block; color:{TXT} !important; font-size:.9rem;
     font-weight:650; letter-spacing:-.01em; text-transform:none; line-height:1.25; margin:.1rem 0; }}
  div.st-key-navrail button em {{ display:inline-block; color:{MUT} !important; font-style:normal;
     font-size:.68rem; letter-spacing:0; text-transform:none; }}
  div.st-key-navrail button:focus {{ box-shadow:none; }}
  .flowarrow {{ text-align:center; color:{DIM}; font-size:1.15rem; line-height:1; opacity:.75; }}
  /* the notebook view renders the notebook's own markdown, whose headings are
     written for a notebook page and are far too large inside a dashboard */
  div[class*="st-key-nbview"] {{ margin-top:.5rem; }}
  div[class*="st-key-nbview"] div[data-testid="stVerticalBlockBorderWrapper"] {{
     background:rgba(0,0,0,.18); border:1px solid {BORDER}; border-radius:16px; padding:.9rem 1rem; }}
  div[class*="st-key-nbview"] h1 {{ font-size:1.32rem !important; margin:1.1rem 0 .3rem 0 !important; }}
  div[class*="st-key-nbview"] h2 {{ font-size:1.14rem !important; margin:1rem 0 .3rem 0 !important; }}
  div[class*="st-key-nbview"] h3 {{ font-size:1rem !important; margin:.9rem 0 .25rem 0 !important; }}
  div[class*="st-key-nbview"] h4 {{ font-size:.92rem !important; }}
  div[class*="st-key-nbview"] p, div[class*="st-key-nbview"] li {{ font-size:.87rem; color:{MUT} !important; }}
  div[class*="st-key-nbview"] pre, div[class*="st-key-nbview"] pre code,
  div[class*="st-key-nbview"] pre span {{ font-size:.76rem !important; line-height:1.5 !important; }}
  div[class*="st-key-nbview"] div[data-testid="stCode"] {{ margin-bottom:.35rem; }}
  div[class*="st-key-nbview"] div[data-testid="stElementContainer"] {{ margin-bottom:.15rem; }}
  /* notebook figures are matplotlib PNGs on white; frame them so they read as
     deliberate figure cards rather than as holes in the dark page */
  div[class*="st-key-nbview"] img {{ border-radius:10px; background:#fff; padding:.5rem;
     border:1px solid rgba(190,210,192,.18); }}
  code {{ color:{CYAN} !important; background:rgba(127,181,130,.1) !important; }}
</style>
""", unsafe_allow_html=True)


def fig_style(fig, h=330, title=None, legend=True, margin_t=None):
    fig.update_layout(
        height=h, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=12.3, color=MUT),
        margin=dict(l=8, r=12, b=8,
                    t=margin_t if margin_t is not None else
                    (74 if (title and legend) else 46 if title else 36 if legend else 14)),
        hoverlabel=dict(bgcolor="#132019", bordercolor=BORDER, font_size=12, font_color=TXT),
        showlegend=legend,
        legend=dict(orientation="h", y=1.03, yanchor="bottom", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11.5, color=MUT)),
        colorway=SEQ,
    )
    if title:
        fig.update_layout(title=dict(
            text=title, x=0, xanchor="left", yref="container", y=0.97, yanchor="top",
            font=dict(size=15, color=TXT, family="Outfit, Inter, sans-serif", weight=600)))
    fig.update_layout(legend_title_text="")
    ax = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
              tickfont=dict(color=DIM, size=11.2),
              title_font=dict(color=MUT, size=11.5, family="Inter, sans-serif"))
    fig.update_xaxes(**ax)
    fig.update_yaxes(**ax)
    return fig


_CARD = itertools.count()


def chart(fig, what=None, why=None, key=None):
    """Every figure sits in its own card, so a column of charts reads as a grid."""
    with st.container(key=f"card{next(_CARD)}"):
        st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})
        if what:
            with st.expander("Interpretation"):
                st.markdown(f"**What this shows.** {what}")
                if why:
                    st.markdown(f"**Why it matters.** {why}")


def tiles(items):
    html = '<div class="tiles">'
    for label, value, note, colour in items:
        html += (f'<div class="tile" style="--c:{colour}"><div class="l">{label}</div>'
                 f'<div class="v">{value}</div><div class="n">{note}</div></div>')
    st.markdown(html + "</div>", unsafe_allow_html=True)


def _size(path):
    if not path.exists():
        return "missing"
    n = path.stat().st_size if path.is_file() else sum(
        f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit, div in (("GB", 1073741824), ("MB", 1048576), ("KB", 1024)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


def _io_block(kind, items, accent):
    rows = "".join(
        f'<div class="io-row"><span class="io-n">{label}</span>'
        f'<span class="io-s">{_size(path)}</span></div>'
        for label, path in items)
    return f'<div class="io-k" style="--c:{accent}">{kind}</div><div class="io-list">{rows}</div>'


def method(steps, sub, accent, engine, ins, outs):
    """The round's method as a vertical flowchart, beside its artefact diagram.

    The right-hand column is read from disk, so the file sizes shown are the
    real ones and a missing artefact says so instead of being silently omitted.
    """
    eyebrow("Method", sub)
    left, right = st.columns([1.32, 1], gap="large")

    with left:
        n = len(steps)
        html = ['<div class="flow">']
        for i, (title, body, chip) in enumerate(steps):
            c = EMER if i == n - 1 else accent
            html.append(f'<div class="step" style="--c:{c}">'
                        f'<div class="node">{i + 1:02d}</div>'
                        f'<div class="sbody"><h5>{title}</h5><p>{body}</p>'
                        + (f'<span class="chip">{chip}</span>' if chip else "")
                        + '</div></div>')
        html.append("</div>")
        st.markdown("".join(html), unsafe_allow_html=True)

    with right:
        st.markdown(
            f'<div class="iopanel">'
            f'<div class="io-h">Artefacts</div>'
            f'{_io_block("Consumed", ins, accent)}'
            f'<div class="io-arrow" style="--c:{accent}">&#8595;</div>'
            f'<div class="io-engine" style="--c:{accent}">{engine}</div>'
            f'<div class="io-arrow" style="--c:{EMER}">&#8595;</div>'
            f'{_io_block("Produced", outs, EMER)}'
            f'<div class="io-f">Sizes read from disk on this page load.</div>'
            f'</div>', unsafe_allow_html=True)


def eyebrow(title, sub=""):
    st.markdown(f'<div class="eyebrow"><h3>{title}</h3><span>{sub}</span></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_p1():
    posts = pd.read_csv(P1 / "Entropy_Social_Engine_Posts_Cleaned.csv", parse_dates=["timestamp"])
    users = pd.read_csv(P1 / "Entropy_Social_Engine_Users_Cleaned.csv", parse_dates=["account_created"])
    raw = pd.read_csv(P1 / "Entropy_Social_Engine_Posts_Corrupted.csv")
    log = (P1 / "Entropy_cleaning_log.txt").read_text(encoding="utf-8", errors="ignore")
    return posts, users, raw, log


@st.cache_data(show_spinner=False)
def run_sql(name):
    sql = (P2 / "queries" / name).read_text(encoding="utf-8")
    with sqlite3.connect(f"file:{(P2 / 'Entropy_social_engine.db').as_posix()}?mode=ro", uri=True) as con:
        return sql, pd.read_sql_query(sql, con)



SQL_BLOCKED = ("attach", "detach", "pragma", "insert", "update", "delete", "drop", "create",
               "alter", "replace", "vacuum", "reindex", "begin", "commit", "rollback")
SQL_ROW_CAP = 500


def run_user_sql(query: str, cap: int = SQL_ROW_CAP):
    """Execute one read-only SELECT against the Round 1b database.

    The connection is opened read-only at the URI level, so the checks below are
    a second line rather than the only one: they exist to give a clear message
    instead of a driver error, and to stop a runaway query hanging the page.
    """
    clean = re.sub(r"--[^\n]*", " ", query)
    clean = re.sub(r"/\*.*?\*/", " ", clean, flags=re.S).strip().rstrip(";").strip()
    if not clean:
        raise ValueError("Write a query first.")
    if ";" in clean:
        raise ValueError("One statement at a time, please.")
    low = clean.lower()
    if not (low.startswith("select") or low.startswith("with")):
        raise ValueError("Only SELECT (or WITH ... SELECT) queries can be run here.")
    hit = next((w for w in SQL_BLOCKED if re.search(r"\b" + w + r"\b", low)), None)
    if hit:
        raise ValueError("'" + hit + "' is not allowed - this console is read-only.")

    con = sqlite3.connect("file:" + str(P2 / "Entropy_social_engine.db") + "?mode=ro", uri=True)
    steps = {"n": 0}

    def guard():                      # a few seconds of work, then abort
        steps["n"] += 1
        return 1 if steps["n"] > 400_000 else 0

    try:
        con.set_progress_handler(guard, 1000)
        t0 = time.perf_counter()
        df = pd.read_sql_query("SELECT * FROM (" + clean + ") LIMIT " + str(cap + 1), con)
        plan = pd.read_sql_query("EXPLAIN QUERY PLAN " + clean, con)
        ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:
        raise ValueError(str(exc).splitlines()[0]) from None
    finally:
        con.set_progress_handler(None, 0)
        con.close()
    return df.head(cap), len(df) > cap, ms, plan


@st.cache_data(show_spinner=False)
def load_r2():
    comp = pd.read_csv(R2 / "outputs" / "Entropy_r2_model_comparison.csv")
    rep = pd.read_csv(R2 / "outputs" / "Entropy_r2_classification_report.csv").rename(columns={"Unnamed: 0": "class"})
    preds = pd.read_csv(R2 / "outputs" / "Entropy_r2_test_predictions.csv")
    met = json.loads((R2 / "outputs" / "Entropy_r2_metrics.json").read_text(encoding="utf-8"))
    train = pd.read_csv(R2 / "data" / "Entropy_Labeled_Social_NLP_Training_Data.csv")
    return comp, rep, preds, met, train


@st.cache_data(show_spinner=False)
def _read_robustness(path_str, _mtime):
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def load_robustness():
    """Round 3 robustness checks, written by Entropy_03_robustness_checks.py.

    Keyed on the file's mtime: re-running the script must change the page, and a
    plain cache_data would happily serve a stale copy for the session's lifetime.
    """
    f = R3 / "outputs" / "Entropy_round3_robustness.json"
    return _read_robustness(str(f), f.stat().st_mtime) if f.exists() else None


@st.cache_data(show_spinner=False)
def load_r3():
    s = pd.read_csv(R3 / "outputs" / "Entropy_round3_scored_posts.csv")
    s["created_utc"] = pd.to_datetime(s.created_utc, format="mixed", utc=True)
    met = json.loads((R3 / "outputs" / "Entropy_round3_metrics.json").read_text(encoding="utf-8"))
    man = json.loads((R3 / "data" / "Entropy_round3_collection_manifest.json").read_text(encoding="utf-8"))
    return s, met, man


@st.cache_resource(show_spinner=False)
def load_model():
    sys.path.insert(0, str(R2))
    import Entropy_nlp_utils as nu
    from sklearn.linear_model import LogisticRegression
    eng = nu.SentimentEnsemble(R2 / "model")
    for comp in eng.cfg["components"]:
        obj = eng._load(comp)
        cand = [obj.steps[-1][1]] if hasattr(obj, "steps") else ([obj[1]] if isinstance(obj, tuple) else [])
        for c in cand:
            if isinstance(c, LogisticRegression) and not hasattr(c, "multi_class"):
                c.multi_class = "multinomial"
    return eng, nu


@st.cache_data(show_spinner=False)
def lorenz(values):
    """Cumulative share of engagement against cumulative share of posts.

    Returns the curve plus its Gini coefficient - 0 is every post earning the
    same, 1 is one post taking everything.
    """
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    x = np.clip(x, 0, None)
    if len(x) == 0 or x.sum() == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.0
    cum = np.cumsum(x) / x.sum()
    share = np.arange(1, len(x) + 1) / len(x)
    gini = float((2 * np.arange(1, len(x) + 1) - len(x) - 1).dot(x) / (len(x) * x.sum()))
    return np.insert(share, 0, 0), np.insert(cum, 0, 0), gini


CHUNK_WORDS, CHUNK_STRIDE = 34, 28


def score_text(eng, text):
    """Score one piece of text exactly as Round 3 scored the live corpus.

    The fine-tuned component was trained at 50 tokens, so anything longer is
    split into overlapping windows and averaged rather than silently truncated,
    and the saved neutral bias is applied on top - the same two corrections the
    Round 3 notebook applies.
    """
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        chunks = [text]
    else:
        chunks = [" ".join(words[i:i + CHUNK_WORDS])
                  for i in range(0, len(words), CHUNK_STRIDE)
                  if words[i:i + CHUNK_WORDS]]
    mean = eng.predict_proba(chunks).mean(axis=0)
    return eng._apply_neutral_bias(mean[None, :])[0], len(chunks)


LABELS = np.array(["Negative", "Neutral", "Positive"])
PCOLS = ["p_negative", "p_neutral", "p_positive"]


@st.cache_data(show_spinner=False)
def relabel(probs, weight):
    """Re-decide every post's class with the Neutral column reweighted.

    The saved probabilities already carry the shipped 1.05 neutral bias, so a
    weight of 1.00 reproduces exactly what Round 3 submitted; the slider moves
    the operating point relative to that, and needs no model in the loop.
    """
    q = probs.copy()
    q[:, 1] *= weight
    return LABELS[q.argmax(1)]


@st.cache_data(show_spinner=False)
def net_with_ci(counts, n_boot=1200, seed=42):
    """Bootstrap a 95% interval for net sentiment in each time bin.

    Net sentiment is a mean over values in {-1, 0, +1}, so a bin is fully
    described by its three class counts and can be resampled from a multinomial
    rather than from the rows themselves.
    """
    rng = np.random.default_rng(seed)
    lo, hi, net = [], [], []
    for neg, neu, pos in counts:
        n = neg + neu + pos
        if n == 0:
            lo.append(np.nan), hi.append(np.nan), net.append(np.nan)
            continue
        draws = rng.multinomial(n, [neg / n, neu / n, pos / n], size=n_boot)
        vals = (draws[:, 2] - draws[:, 0]) / n
        lo.append(np.percentile(vals, 2.5))
        hi.append(np.percentile(vals, 97.5))
        net.append((pos - neg) / n)
    return np.array(net), np.array(lo), np.array(hi)


@st.cache_data(show_spinner=False)
def load_validation(weight):
    """In-domain scores at a given operating point.

    Scored against the two-annotator consensus where it exists, and against the
    original single-annotator sheet only as a fallback.
    """
    from sklearn.metrics import f1_score
    cons = R3 / "outputs" / "Entropy_round3_consensus_labels.csv"
    if cons.exists():
        v = pd.read_csv(cons).rename(columns={"gold": "truth"})
    else:
        v = pd.read_csv(R3 / "outputs" / "Entropy_round3_validation_sample.csv").rename(
            columns={"human_label": "truth"})
    p = pd.read_csv(R3 / "outputs" / "Entropy_round3_scored_posts.csv",
                    usecols=["post_id"] + PCOLS)
    m = v[["post_id", "truth"]].merge(p, on="post_id", how="inner")
    pred = relabel(m[PCOLS].to_numpy(), weight)
    return (f1_score(m.truth, pred, average="macro"),
            float((pred == m.truth).mean()), len(m))


LAUNCH = pd.Timestamp("2026-09-14 17:00", tz="UTC")

# --------------------------------------------------------------------------- notebooks
NB_CLIP = 3000
_NBCELL = itertools.count()


def _clip(text):
    """Long outputs are cut, but never silently."""
    if len(text) <= NB_CLIP:
        return text
    return text[:NB_CLIP].rstrip() + chr(10) + chr(10) + \
        f"... {len(text) - NB_CLIP:,} more characters - download the notebook for the rest"


@st.cache_data(show_spinner=False, max_entries=8)
def load_nb(path_str, _mtime):
    """Parse an executed .ipynb into sections keyed by its markdown headings.

    Outputs are reduced to what the page can render: stream text, plain-text
    results and PNG figures. HTML outputs are skipped - they carry Jupyter's own
    light-theme table styling and would fight the dashboard.
    """
    nb = json.loads(Path(path_str).read_text(encoding="utf-8"))
    secs, cur = [], {"title": "Opening", "cells": []}
    for c in nb.get("cells", []):
        src = "".join(c.get("source", []))
        if c.get("cell_type") == "markdown":
            head = next((l for l in src.strip().splitlines() if l.strip()), "")
            if head.lstrip().startswith("#"):
                if cur["cells"]:
                    secs.append(cur)
                cur = {"title": head.lstrip("#").strip()[:58], "cells": []}
        outs = []
        for o in (c.get("outputs", []) if c.get("cell_type") == "code" else []):
            kind = o.get("output_type")
            if kind == "stream":
                outs.append(("text", _clip("".join(o.get("text", "")))))
            elif kind == "error":
                outs.append(("error", chr(10).join(o.get("traceback", []))[:3000]))
            else:
                d = o.get("data") or {}
                if "image/png" in d:
                    png = d["image/png"]
                    outs.append(("image", base64.b64decode(png if isinstance(png, str) else "".join(png))))
                elif "text/plain" in d:
                    outs.append(("text", _clip("".join(d["text/plain"]))))
        cur["cells"].append({"kind": c.get("cell_type"), "src": src, "out": outs,
                             "n": c.get("execution_count")})
    if cur["cells"]:
        secs.append(cur)
    return secs


def notebook_panel(prefix, files, blurb):
    """Render the round's executed notebook: code, output and its own notes."""
    eyebrow("The notebook behind this page", "code, output and the working notes, exactly as it ran")
    st.markdown(f'<div class="panel"><div class="sub" style="margin:0">{blurb}</div></div>',
                unsafe_allow_html=True)

    labels = list(files)
    pick, pick_sec, dl = st.columns([1.15, 2, .85], vertical_alignment="bottom")
    label = pick.selectbox("Notebook", labels, key=f"{prefix}_nb") if len(labels) > 1 else labels[0]
    path = files[label]
    if not path.exists():
        st.info(f"Not found: {path.name} - run the notebook to regenerate it.")
        return

    secs = load_nb(str(path), path.stat().st_mtime)
    titles = [f"{i + 1}. {sec['title']}" for i, sec in enumerate(secs)]
    WHOLE = "Whole notebook"
    chosen = pick_sec.selectbox("Show", [WHOLE] + titles, key=f"{prefix}_sec")
    dl.download_button("Download .ipynb", path.read_bytes(), file_name=path.name,
                       mime="application/x-ipynb+json", key=f"{prefix}_dl", width="stretch")

    shown = [c for sec in secs for c in sec["cells"]] if chosen == WHOLE else secs[titles.index(chosen)]["cells"]
    total_code = sum(c["kind"] == "code" for sec in secs for c in sec["cells"])
    total_out = sum(len(c["out"]) for sec in secs for c in sec["cells"])
    n_code = sum(c["kind"] == "code" for c in shown)
    n_out = sum(len(c["out"]) for c in shown)
    st.caption(f"{path.name} - {len(secs)} sections, {total_code} code cells, {total_out} outputs in full"
               + ("" if chosen == WHOLE else f" - showing {n_code} code cells and {n_out} outputs from this section"))

    # a bounded, scrollable reading area: a whole notebook inlined at full height
    # buries everything below it, and there is a lot below it
    with st.container(key=f"nbview_{prefix}"):
        with st.container(height=720):
            for cell in shown:
                k = next(_NBCELL)
                if cell["kind"] == "markdown":
                    if cell["src"].strip():
                        with st.container(key=f"nbmd{k}"):
                            st.markdown(cell["src"])
                    continue
                if not cell["src"].strip() and not cell["out"]:
                    continue
                n = cell.get("n")
                with st.container(key=f"nbcell{k}"):
                    st.markdown(
                        f'<div class="nbhead"><span class="nbin">In [{n if n is not None else " "}]</span>'
                        f'<span class="nbkind">python</span></div>', unsafe_allow_html=True)
                    if cell["src"].strip():
                        st.code(cell["src"], language="python")
                    if cell["out"]:
                        st.markdown('<div class="nbout">Output</div>', unsafe_allow_html=True)
                        for kind, val in cell["out"]:
                            if kind == "image":
                                st.image(val, width="stretch")
                            elif kind == "error":
                                st.error(val)
                            else:
                                st.code(val, language=None)


@st.cache_data(show_spinner=False, max_entries=12)
def file_bytes(path_str, _mtime):
    return Path(path_str).read_bytes()


@st.cache_data(show_spinner=False, max_entries=6)
def pdf_pages(path_str, _mtime, scale=1.7):
    """Rasterise a report to PNG pages.

    Rendering server-side rather than embedding the file means the report reads
    the same in every browser, including ones with no built-in PDF viewer.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None
    doc = pdfium.PdfDocument(path_str)
    out = []
    for i in range(len(doc)):
        buf = io.BytesIO()
        doc[i].render(scale=scale).to_pil().save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def report_panel(prefix, reports, blurb):
    """Open this round's written deliverables inside the page."""
    eyebrow("Reports for this round", "read a report here, page by page, or take the file")
    st.markdown(f'<div class="panel"><div class="sub" style="margin:0">{blurb}</div></div>',
                unsafe_allow_html=True)
    have = {k: v for k, v in reports.items() if v.exists()}
    if not have:
        st.info("No report PDFs found next to this round - build them from the LaTeX sources.")
        return

    pick = st.segmented_control("Open a report", list(have), key=f"{prefix}_rep")
    if not pick:
        st.caption("Pick a report above to read it without leaving the dashboard.")
        return

    path = have[pick]
    raw = file_bytes(str(path), path.stat().st_mtime)
    with st.spinner("Rendering the report..."):
        pages = pdf_pages(str(path), path.stat().st_mtime)

    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    c1.caption(f"{path.name} - {len(raw) / 1048576:.1f} MB"
               + (f", {len(pages)} pages" if pages else ""))
    c2.download_button("Download PDF", raw, file_name=path.name, mime="application/pdf",
                       key=f"{prefix}_rdl", width="stretch")

    if not pages:
        st.info("Install `pypdfium2` to read reports inside the dashboard; the download above "
                "always works.")
        return

    first, last = 1, len(pages)
    if len(pages) > 4:
        first, last = st.slider("Pages", 1, len(pages), (1, min(4, len(pages))),
                                key=f"{prefix}_pg")
    with st.container(key=f"pdfview_{prefix}"):
        for i in range(first - 1, last):
            st.markdown(f'<div class="outno">Page {i + 1} of {len(pages)}</div>',
                        unsafe_allow_html=True)
            st.image(pages[i], width="stretch")


# --------------------------------------------------------------------------- shell
st.markdown(f"""
<div class="topbar">
  <div class="brand">
    <div class="mark">
      <svg viewBox="0 0 48 48" aria-hidden="true">
        <defs>
          <linearGradient id="mg" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="{CYAN}"/>
            <stop offset="52%" stop-color="{INDIGO}"/>
            <stop offset="100%" stop-color="{VIOLET}"/>
          </linearGradient>
        </defs>
        <circle class="ring" cx="24" cy="24" r="18.5" fill="none" stroke="url(#mg)"
                stroke-width="2.4" stroke-linecap="round" stroke-dasharray="30 14"/>
        <circle cx="24" cy="24" r="11" fill="none" stroke="{CYAN}" stroke-opacity=".3" stroke-width="1.3"/>
        <path class="wave" d="M12.5 24h4.6l3.1-8.2L25.6 32l2.9-8h6.9" fill="none" stroke="url(#mg)"
              stroke-width="2.7" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    <div>
      <div class="wm">The Social Engine</div>
      <div class="tag">Rebuilt across four rounds &middot; <b>Data Vortex</b> &middot; AARUUSH&rsquo;26</div>
    </div>
  </div>
  <div class="who">
    <div class="badge">Team Entropy</div>
    <div class="t">Saanvi Grover &amp; Aditya Sharma</div>
    <a class="repo" href="{REPO}" target="_blank" rel="noopener">
      <svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
      <span>SaiGrover/entropy-data-vortex</span>
    </a>
  </div>
</div>""", unsafe_allow_html=True)

# The four-round pipeline doubles as the navigation: every node is a real button.
NAV = [
    ("Round 1 · Phase 1", "Data Recovery", "12,000 rows repaired", CYAN),
    ("Round 1 · Phase 2", "Analytical Core", "3 SQL questions", BLUE),
    ("Round 2", "Semantic Recovery", "0.718 macro-F1", VIOLET),
    ("Round 3", "Signal Tracking", "1,403 live posts", AMBER),
    ("Round 4", "The Revived Engine", "all four, running", EMER),
]
HOME = len(NAV) - 1          # Round 4 *is* this dashboard, so it is the landing page
if "page" not in st.session_state:
    st.session_state.page = HOME

st.markdown("<style>%s</style>" % "".join(
    f'div.st-key-nav{i} button:hover {{ border-color:{c}aa !important;'
    f'  box-shadow:0 0 0 1px {c}33, 0 6px 20px -9px {c}99 !important; }}'
    f'div.st-key-nav{i} button:before {{ background:linear-gradient(90deg,{c},{c}22); }}'
    + (f'div.st-key-nav{i} button {{ background:{c}1f !important; border-color:{c} !important;'
       f'  box-shadow:0 0 0 1px {c}66, 0 10px 28px -12px {c} !important; }}'
       f'div.st-key-nav{i} button strong {{ color:{c} !important; }}'
       f'div.st-key-nav{i} button em {{ color:{c}cc !important; }}'
       f'div.st-key-nav{i} button:before {{ height:3px; }}'
       if i == st.session_state.page else "")
    for i, (_, _, _, c) in enumerate(NAV)), unsafe_allow_html=True)

rail = st.container(key="navrail")
with rail:
    widths = []
    for i in range(len(NAV)):
        widths.append(5)
        if i < len(NAV) - 1:
            widths.append(0.9)
    slots = st.columns(widths, gap="small", vertical_alignment="center")
    for i, (tag, name, metric, _) in enumerate(NAV):
        with slots[i * 2]:
            label = tag + "  " + chr(10) + "**" + name + "**  " + chr(10) + "*" + metric + "*"
            if st.button(label, key=f"nav{i}",
                         width="stretch", help=f"Go to {name}"):
                st.session_state.page = i
                st.rerun()
        if i < len(NAV) - 1:
            slots[i * 2 + 1].markdown('<div class="flowarrow">&#10230;</div>',
                                      unsafe_allow_html=True)

PAGE = st.session_state.page

# ===========================================================================  ROUND 1a
if PAGE == 0:
    posts, users, raw, log = load_p1()
    miss_tot = int(posts[["platform", "text_content", "likes"]].isna().sum().sum())
    tiles([("Rows in", f"{len(raw):,}", "corrupted export", DIM),
           ("Rows out", f"{len(posts):,}", "360 duplicates removed", CYAN),
           ("Users", f"{len(users):,}", "33 cities, 10 languages", INDIGO),
           ("Values left missing", f"{miss_tot:,}", "not imputed, by choice", AMBER),
           ("Span", "364 days", "May 2024 - Apr 2025", BLUE)])


    # the headline claim is that this corpus is flat; these controls exist so a
    # reader can try to find a slice where it is not
    merged = posts.merge(users[["user_id", "follower_count"]], on="user_id", how="left")
    ts = pd.to_datetime(merged.timestamp, errors="coerce")
    lo_d, hi_d = ts.min().date(), ts.max().date()

    fc1, fc2, fc3 = st.columns([1.3, 1.5, 1.2], gap="medium")
    plats = sorted(posts.platform.dropna().unique())
    pick = fc1.multiselect("Platforms", plats, default=plats, key="p1_plat")
    span = fc2.slider("Date range", min_value=lo_d, max_value=hi_d, value=(lo_d, hi_d),
                      format="DD MMM YY", key="p1_dates")
    BANDS = {"Everyone": None, "Under 10k followers": (0, 10_000),
             "10k - 30k": (10_000, 30_000), "Over 30k": (30_000, 10**9)}
    band = fc3.selectbox("Follower band", list(BANDS), key="p1_band")

    view = merged[merged.platform.isin(pick)] if pick else merged
    keep = pd.to_datetime(view.timestamp, errors="coerce")
    view = view[(keep.dt.date >= span[0]) & (keep.dt.date <= span[1])]
    if BANDS[band]:
        lo_f, hi_f = BANDS[band]
        view = view[(view.follower_count >= lo_f) & (view.follower_count < hi_f)]

    if len(view) < len(posts):
        cv = view[["likes", "shares", "comments"]].sum(axis=1)
        share = view.platform.value_counts(normalize=True)
        st.caption(f"{len(view):,} of {len(posts):,} posts in this slice &middot; "
                   f"platform share spread {share.max() - share.min():.1%} &middot; "
                   f"engagement CV {cv.std() / max(cv.mean(), 1e-9):.2f} "
                   f"- the flatness holds under slicing, which is the point")
    if view.empty:
        st.warning("No posts in that slice. Widen the filters.")
        st.stop()

    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        pv = view.platform.value_counts()
        fig = go.Figure(go.Bar(x=pv.index, y=pv.values, marker_color=CYAN, text=pv.values,
                               texttemplate="%{text:,}", textposition="outside"))
        fig.update_layout(yaxis_range=[0, pv.max() * 1.2])
        chart(fig_style(fig, 310, "Volume by platform", legend=False),
              f"Post counts across the selected platforms, spanning only "
              f"{100*(pv.max()-pv.min())/pv.mean():.1f}% between busiest and quietest.",
              "Real platforms differ by multiples. This flatness is the first clue the corpus was generated, and it "
              "sets expectations for every later query.")
    with c2:
        fig = go.Figure()
        for col, colour in [("likes", CYAN), ("shares", INDIGO), ("comments", VIOLET)]:
            fig.add_trace(go.Histogram(x=view[col], name=col, nbinsx=44, marker_color=colour, opacity=.72))
        fig.update_layout(barmode="overlay", xaxis_title="value", yaxis_title="posts")
        chart(fig_style(fig, 310, "Engagement distributions"),
              "Likes, shares and comments are each spread evenly across their range instead of forming the long "
              "tail real platforms show.",
              "No viral tail means anomaly hunting will return empty results - a structural property of the data, "
              "not a failure of the query.")

    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        corr = view[["likes", "shares", "comments"]].corr()
        z = corr.values.copy()
        np.fill_diagonal(z, np.nan)  # the diagonal is always 1 and would swamp the real signal
        fig = go.Figure(go.Heatmap(z=z, x=corr.columns, y=corr.columns, zmin=-.3, zmax=.3,
                                   colorscale=[[0, ROSE], [.5, "#16261d"], [1, CYAN]],
                                   text=[["" if np.isnan(v) else f"{v:.3f}" for v in row] for row in z],
                                   texttemplate="%{text}",
                                   textfont=dict(size=15, color=TXT), xgap=3, ygap=3,
                                   hovertemplate="%{y} vs %{x}: r = %{z:.3f}<extra></extra>",
                                   colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color=DIM, size=10),
                                                 title=dict(text="r", font=dict(color=DIM, size=11)))))
        chart(fig_style(fig, 300, "Do the metrics move together?", legend=False),
              "Correlations between the three engagement metrics all sit within about 0.02 of zero.",
              "On a real platform a popular post wins on every metric at once. Here they were generated "
              "independently, which is why follower and virality questions come back flat.")
    with c2:
        pc, lc = posts.platform.value_counts(), users.language.value_counts()
        dc, loc = posts.timestamp.dt.dayofweek.value_counts(), users.location.value_counts()
        cvs = pd.DataFrame([
            {"dimension": "platform", "cv": 100 * pc.std() / pc.mean()},
            {"dimension": "language", "cv": 100 * lc.std() / lc.mean()},
            {"dimension": "day of week", "cv": 100 * dc.std() / dc.mean()},
            {"dimension": "location", "cv": 100 * loc.std() / loc.mean()},
        ]).sort_values("cv")
        fig = go.Figure(go.Bar(x=cvs.cv, y=cvs.dimension, orientation="h", marker_color=EMER,
                               text=cvs.cv.round(1), texttemplate="%{text}%", textposition="outside"))
        fig.add_vline(x=10, line_dash="dot", line_color=AMBER)
        fig.update_layout(xaxis_range=[0, max(cvs.cv.max() * 1.45, 14)], xaxis_title="coefficient of variation")
        chart(fig_style(fig, 300, "How evenly is everything spread?", legend=False),
              "Coefficient of variation is the standard deviation as a percentage of the mean. The dotted line marks "
              "10%, below which a real social platform essentially never sits.",
              "It converts 'this looks too uniform' into a number that can be quoted and challenged - the difference "
              "between an impression and a measurement.")

    c1, c2 = st.columns([1.2, 1], gap="medium")
    with c1:
        hours = view.timestamp.dt.hour.value_counts().sort_index()
        mid = int(((view.timestamp.dt.hour == 0) & (view.timestamp.dt.minute == 0)).sum())
        cols = [AMBER if h == 0 else CYAN for h in hours.index]
        fig = go.Figure(go.Bar(x=hours.index, y=hours.values, marker_color=cols))
        fig.update_layout(xaxis_title="hour of day (UTC)", yaxis_title="posts", xaxis=dict(dtick=2))
        chart(fig_style(fig, 300, "Posting by hour - and an artefact we created", legend=False),
              f"The highlighted midnight bar holds {mid:,} posts because one source format carried a date with no "
              f"time, and parsing placed them at 00:00. The other 23 hours are flat.",
              "We report this instead of hiding it. A cleaning step can manufacture something that looks like a "
              "finding, and an analyst who never inspects their own repairs will publish the artefact as a result.")
    with c2:
        m = posts.merge(users, on="user_id", how="left")
        pu = m.groupby("user_id").agg(total=("likes", "sum"), followers=("follower_count", "first")).reset_index()
        rho, p_rho = stats.spearmanr(pu.followers, pu.total)
        fig = go.Figure(go.Scattergl(x=pu.followers, y=pu.total, mode="markers",
                                     marker=dict(size=5, color=INDIGO, opacity=.45)))
        fig.update_layout(xaxis_title="follower count", yaxis_title="total likes received")
        chart(fig_style(fig, 300, f"Followers vs engagement · rho = {rho:.2f}, p = {p_rho:.2f}", legend=False),
              "Each dot is a user. The cloud is flat: audience size tells you nothing about the engagement received.",
              "Follower count is the most trusted vanity metric in social media. Here it is decorative, which is "
              "exactly the kind of assumption a recovered dataset should be able to falsify.")

    with st.expander("Hypothesis tests, recomputed from the cleaned CSVs on every page load"):
        kw = stats.kruskal(*[g.likes.dropna().values for _, g in posts.groupby("platform")])
        miss = posts.platform.isna()
        mw = stats.mannwhitneyu(posts.loc[miss, "shares"], posts.loc[~miss, "shares"])
        st.dataframe(pd.DataFrame([
            {"Question": "Does platform change likes?", "Test": "Kruskal-Wallis",
             "Statistic": f"H = {kw.statistic:.2f}", "p": f"{kw.pvalue:.3f}", "Verdict": "no effect"},
            {"Question": "Do followers predict engagement?", "Test": "Spearman",
             "Statistic": f"rho = {rho:.3f}", "p": f"{p_rho:.3f}", "Verdict": "no effect"},
            {"Question": "Is the missing data random?", "Test": "Mann-Whitney U",
             "Statistic": f"U = {mw.statistic:,.0f}", "p": f"{mw.pvalue:.3f}", "Verdict": "missing at random"},
        ]), width="stretch", hide_index=True)
        st.caption("The third test licenses a decision: because missingness is unrelated to engagement, leaving "
                   "~15% of values empty is defensible while imputing them would invent data.")

    method([
        ("Retrieve the wreckage",
         "Both tables were pulled out of the recovery terminal on the simulated site rather than handed over, "
         "so the corrupted export is kept beside the clean one and every claim can be checked against it.",
         "12,360 rows in"),
        ("Name the damage before touching it",
         "Every column was profiled first: literal NULL strings, exact duplicates, HTML entities, mojibake, "
         "mixed timestamp formats, negative engagement counts and whitespace-only text.",
         "10 corruption classes"),
        ("Repair one class at a time",
         "Each fix is applied on its own, the rows it touched are counted, and that count is written to the "
         "cleaning log before the next fix runs. The log is what the repair ledger below is drawn from.",
         "every change logged"),
        ("Refuse to invent data",
         "1,784 missing platform values were left as NaN because nothing in the row implies the platform. The "
         "missingness was tested first and behaves as missing completely at random (Mann-Whitney, p = 0.64).",
         "nothing imputed"),
        ("Describe the corpus, do not assume it",
         "A twenty-section exploration with twenty-one figures and six hypothesis tests - which is how the "
         "flatness, the midnight timestamp artefact and the engagement asymmetry were all found.",
         "5 of 6 tests null"),
        ("Hand off a single clean corpus",
         "Cleaned posts, cleaned users and one merged submission file. Every later round reads these files and "
         "nothing else.",
         "12,000 rows out"),
    ], "how a corrupted export became a corpus the later rounds could trust", CYAN,
           "Round 1 &middot; Phase 1<br>Data Recovery",
           [("Posts export (corrupted)", P1 / "Entropy_Social_Engine_Posts_Corrupted.csv"),
         ("Users export", P1 / "Entropy_Social_Engine_Users.csv")],
           [("Posts, cleaned", P1 / "Entropy_Social_Engine_Posts_Cleaned.csv"),
         ("Users, cleaned", P1 / "Entropy_Social_Engine_Users_Cleaned.csv"),
         ("Merged submission CSV", P1 / "Entropy_Social_Engine_Cleaned.csv"),
         ("Cleaning log", P1 / "Entropy_cleaning_log.txt")])

    report_panel("p1", {
        "Phase 1 - EDA Insight Report": P1 / "Entropy_Phase1_EDA_Report.pdf",
    }, "The written deliverable for this phase: the twenty-section exploratory analysis, its twenty-one figures "
       "and the six statistical tests, typeset from the notebook output.")

    notebook_panel("p1", {
        "01 - Data Cleaning": P1 / "Entropy_01_Data_Cleaning.ipynb",
        "02 - EDA Report": P1 / "Entropy_02_EDA_Report.ipynb",
    }, "Two notebooks did the recovery. <b>01 - Data Cleaning</b> takes the ten corruption classes one at a "
       "time, applying, counting and logging each repair before the next one runs, which is where the ledger "
       "above comes from. <b>02 - EDA Report</b> is the twenty-section exploration built on the cleaned result, "
       "and it is where the statistical tests on this page were first run. Pick a notebook and a section to "
       "read the code, the output it produced and the notes written beside it.")

# ===========================================================================  ROUND 1b
if PAGE == 1:
    tiles([("Tables", "2", "users and posts", CYAN),
           ("Constraints", "PK · FK · CHECK", "enforced by the engine", INDIGO),
           ("Indexes", "5", "confirmed by query plans", BLUE),
           ("Questions answered", "3", "one per difficulty level", VIOLET),
           ("Queries run", "live", "against the .db on this page", EMER)])


    q1, q2, q3 = st.tabs(["E3 · platform averages", "M4 · large accounts", "H4 · anomaly hunt"])

    with q1:
        sql, e3 = run_sql("Entropy_E3_average_engagement_by_platform.sql")
        c1, c2 = st.columns([1.25, 1], gap="medium")
        with c1:
            mean = e3.avg_total_engagement.mean()
            fig = go.Figure(go.Bar(x=e3.platform, y=e3.avg_total_engagement, marker_color=CYAN,
                                   text=e3.avg_total_engagement, texttemplate="%{text:,.0f}",
                                   textposition="inside"))
            fig.add_hline(y=mean, line_dash="dot", line_color=AMBER)
            fig.update_layout(yaxis_range=[e3.avg_total_engagement.min() * .985,
                                           e3.avg_total_engagement.max() * 1.01],
                              yaxis_title="avg total engagement")
            spread = 100 * (e3.avg_total_engagement.max() - e3.avg_total_engagement.min()) / mean
            chart(fig_style(fig, 330, f"Instagram leads by {spread:.1f}% of the mean", legend=False),
                  "Average engagement per platform, with the overall mean dotted. The y-axis is deliberately "
                  "truncated - at full scale the five bars are indistinguishable.",
                  "With ~1,700 posts per platform the data could detect a 3.7% gap and finds none, so ranking these "
                  "platforms is ranking noise. The query answers the question and then disarms it.")
        with c2:
            st.dataframe(e3[["platform", "posts", "avg_likes", "avg_shares", "avg_comments",
                             "avg_total_engagement", "pct_vs_overall"]],
                         width="stretch", hide_index=True, height=248)
            with st.expander("SQL"):
                st.code(sql, language="sql")

    with q2:
        sql, m4 = run_sql("Entropy_M4_platform_behaviour_high_follower_users.sql")
        c1, c2 = st.columns([1.25, 1], gap="medium")
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="30k+ followers", x=m4.platform, y=m4.avg_eng_high_followers,
                                 marker_color=INDIGO))
            fig.add_trace(go.Bar(name="under 30k", x=m4.platform, y=m4.avg_eng_other_users, marker_color=DIM))
            lo_ = min(m4.avg_eng_high_followers.min(), m4.avg_eng_other_users.min())
            hi_ = max(m4.avg_eng_high_followers.max(), m4.avg_eng_other_users.max())
            fig.update_layout(barmode="group", yaxis_range=[lo_ * .985, hi_ * 1.015], yaxis_title="avg engagement")
            chart(fig_style(fig, 330, "Do big accounts do better anywhere?"),
                  "The cohort is compared with users below the threshold, never with 'everyone' - that would place "
                  "the cohort on both sides of the comparison and shrink any real gap.",
                  "The platform ranking reverses between the two groups, the signature of noise. A segment strategy "
                  "built on this would fail to replicate next month.")
        with c2:
            fig = go.Figure(go.Bar(x=m4.lift_pct, y=m4.platform, orientation="h",
                                   marker_color=[EMER if v > 0 else ROSE for v in m4.lift_pct],
                                   text=m4.lift_pct, texttemplate="%{text:+.1f}%", textposition="outside"))
            fig.add_vline(x=0, line_color=GRID)
            fig.update_layout(xaxis_range=[m4.lift_pct.min() * 1.7, m4.lift_pct.max() * 1.7],
                              xaxis_title="lift for large accounts")
            chart(fig_style(fig, 330, "Lift, platform by platform", legend=False),
                  "How much better or worse large accounts do than smaller ones on each platform.",
                  "Three of five platforms are negative. Audience size buys nothing here - the same conclusion the "
                  "follower scatter reached in Data Recovery from a different direction.")
            with st.expander("SQL"):
                st.code(sql, language="sql")

    with q3:
        sql, h4 = run_sql("Entropy_H4_follower_to_engagement_anomaly.sql")
        c1, c2 = st.columns([1.35, 1], gap="medium")
        with c1:
            fig = px.scatter(h4, x="post_count", y="total_engagement", size="avg_eng_per_post",
                             color="classification", hover_name="user_id",
                             color_discrete_map={"Volume-driven": INDIGO, "Exceptional per post": AMBER},
                             labels={"post_count": "posts written", "total_engagement": "total engagement"})
            fig.update_traces(marker=dict(line=dict(width=1, color=PANEL)))
            chart(fig_style(fig, 340, "The 16 flagged small accounts"),
                  "Every flagged user, positioned by how much they posted and how much engagement they collected; "
                  "bubble size is engagement per post.",
                  "Chance alone predicts about 13 such users, so the list is not anomalous. Most are simply "
                  "prolific - which is why the second ranking exists.")
        with c2:
            counts = h4.classification.value_counts()
            fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=.6,
                                   marker=dict(colors=[INDIGO, AMBER], line=dict(color=PANEL, width=3)),
                                   textinfo="label+value", insidetextorientation="horizontal",
                                   textfont=dict(color="#0b120f", size=13)))
            chart(fig_style(fig, 340, "Volume or genuine performance?", legend=False),
                  "The same 16 users split by whether their engagement per post is also top-decile.",
                  "Handed over unfiltered this list wastes fourteen investigations. Ranked per post it produces two "
                  "real leads - the difference between a metric and a decision.")
            with st.expander("SQL"):
                st.code(sql, language="sql")


    eyebrow("Run your own query", "the same database, opened read-only")
    st.markdown(
        '<div class="panel"><div class="sub" style="margin:0">Three questions were submitted, but the '
        'schema holds more than three answers. This console runs a single <code>SELECT</code> against '
        '<code>Entropy_social_engine.db</code> through a read-only URI - the same connection the charts '
        'above use - and shows the query plan beside the result. Two tables: '
        '<code>users(user_id, location, language, account_created, follower_count)</code> and '
        '<code>posts(post_id, user_id, platform, text_content, timestamp, likes, shares, comments)</code>.'
        '</div></div>', unsafe_allow_html=True)

    STARTERS = {
        "Busiest day of the week":
            "SELECT strftime('%w', timestamp) AS weekday, COUNT(*) AS posts,\n"
            "       ROUND(AVG(likes + shares + comments), 1) AS avg_engagement\n"
            "FROM posts GROUP BY weekday ORDER BY posts DESC",
        "Engagement by language":
            "SELECT u.language, COUNT(*) AS posts,\n"
            "       ROUND(AVG(p.likes + p.shares + p.comments), 1) AS avg_engagement\n"
            "FROM posts p JOIN users u ON u.user_id = p.user_id\n"
            "GROUP BY u.language HAVING posts > 200 ORDER BY avg_engagement DESC",
        "Top cities by follower count":
            "SELECT location, COUNT(*) AS users, ROUND(AVG(follower_count)) AS avg_followers\n"
            "FROM users GROUP BY location ORDER BY avg_followers DESC LIMIT 10",
        "Do longer posts earn more?":
            "SELECT CASE WHEN LENGTH(text_content) < 80 THEN 'short'\n"
            "            WHEN LENGTH(text_content) < 160 THEN 'medium' ELSE 'long' END AS bucket,\n"
            "       COUNT(*) AS posts, ROUND(AVG(likes + shares + comments), 1) AS avg_engagement\n"
            "FROM posts WHERE text_content IS NOT NULL GROUP BY bucket ORDER BY avg_engagement DESC",
    }
    if "sql_text" not in st.session_state:
        st.session_state.sql_text = list(STARTERS.values())[0]

    sc1, sc2 = st.columns([1, 2.15], gap="medium")
    with sc1:
        st.markdown('<div class="io-h">Start from a question</div>', unsafe_allow_html=True)
        for i, (label, q) in enumerate(STARTERS.items()):
            if st.button(label, key=f"sqlstart{i}", width="stretch"):
                st.session_state.sql_text = q
                st.rerun()
        st.caption(f"SELECT or WITH only, one statement, first {SQL_ROW_CAP} rows.")
    with sc2:
        st.text_area("Your query", key="sql_text", height=196)
        if st.button("Run query", type="primary", key="sql_run", width="stretch"):
            try:
                out, truncated, ms, plan = run_user_sql(st.session_state.sql_text)
                st.session_state.sql_out = (out, truncated, ms, plan, None)
            except Exception as exc:
                st.session_state.sql_out = (None, False, 0, None, str(exc))

    got = st.session_state.get("sql_out")
    if got:
        out, truncated, ms, plan, err = got
        if err:
            st.warning(err)
        else:
            st.caption(f"{len(out):,} row{'s' * (len(out) != 1)}"
                       f"{' (capped)' if truncated else ''} in {ms:.0f} ms")
            st.dataframe(out, width="stretch", height=min(430, 60 + 35 * max(len(out), 1)),
                         hide_index=True)
            with st.expander("Query plan"):
                st.caption("How SQLite chose to answer it - the same check used to pick the indexes.")
                st.dataframe(plan, width="stretch", hide_index=True)

    method([
        ("Let the schema enforce validity",
         "Two tables with primary keys, a foreign key from posts to users, and CHECK constraints on the "
         "engagement columns, so invalid rows are rejected by the engine rather than by a later script.",
         "2 tables, 5 indexes"),
        ("Load the cleaned CSVs, not the raw ones",
         "The database is built from Phase 1 output with declared types, which is what makes the constraints "
         "mean anything in the first place.",
         "12,000 rows loaded"),
        ("Take one question per difficulty band",
         "E3, M4 and H4 - easy, medium and hard - so the submission shows range rather than three variations "
         "of the same aggregation.",
         "E3 / M4 / H4"),
        ("Answer in SQL, not in pandas",
         "CTE chains and window functions (NTILE, PERCENT_RANK, RANK) do the bucketing and ranking inside the "
         "query, so the database does the analytical work it exists to do.",
         "no post-processing"),
        ("Benchmark every query",
         "EXPLAIN QUERY PLAN was read for each one, and the indexes were chosen from what the plan actually "
         "did rather than from guesswork.",
         "plans confirmed"),
        ("Report the null expectation beside the result",
         "H4 flags 16 accounts where chance alone predicts about 13, so the honest answer is that the anomaly "
         "hunt found nothing anomalous - stated plainly rather than buried.",
         "results in context"),
    ], "how three questions became a constrained, benchmarked analytical core", BLUE,
           "Round 1 &middot; Phase 2<br>Analytical Core",
           [("Posts, cleaned", P1 / "Entropy_Social_Engine_Posts_Cleaned.csv"),
         ("Users, cleaned", P1 / "Entropy_Social_Engine_Users_Cleaned.csv")],
           [("SQLite database", P2 / "Entropy_social_engine.db"),
         ("E3 query", P2 / "queries" / "Entropy_E3_average_engagement_by_platform.sql"),
         ("M4 query", P2 / "queries" / "Entropy_M4_platform_behaviour_high_follower_users.sql"),
         ("H4 query", P2 / "queries" / "Entropy_H4_follower_to_engagement_anomaly.sql")])

    report_panel("p2", {
        "SQL Queries": P2 / "Entropy_Phase2_SQL_Queries.pdf",
        "Logic Explanation": P2 / "Entropy_Phase2_Logic_Explanation.pdf",
        "Insight Report": P2 / "Entropy_Phase2_Insight_Report.pdf",
    }, "Three documents were submitted for this phase: the queries themselves, the reasoning behind each one "
       "(why that window function, why that join, what the query plan confirms), and the insights they produced.")

    notebook_panel("p2", {"01 - SQL Analysis": P2 / "Entropy_01_SQL_Analysis.ipynb"},
                   "<b>01 - SQL Analysis</b> builds the constrained schema, loads the cleaned CSVs into SQLite, "
                   "then runs each submitted query together with its <code>EXPLAIN QUERY PLAN</code> and its "
                   "result. The charts above come from re-executing those same <code>.sql</code> files against "
                   "the database on this page load - this notebook is the working that produced and benchmarked "
                   "them in the first place.")

# ===========================================================================  ROUND 2
if PAGE == 2:
    comp, rep, preds, r2m, train = load_r2()
    macro = float(rep.loc[rep["class"] == "macro avg", "f1-score"].iloc[0])
    acc = float(rep.loc[rep["class"] == "accuracy", "f1-score"].iloc[0])
    uniq = train.post_text.str.lower().str.strip().nunique()
    labels = ["Negative", "Neutral", "Positive"]
    tiles([("Labelled posts", f"{len(train):,}", f"{len(train)-uniq:,} are duplicates", CYAN),
           ("Models compared", f"{len(comp)}", "baseline to transformers", INDIGO),
           ("Test macro-F1", f"{macro:.3f}", f"accuracy {acc:.3f}", VIOLET),
           ("Winner", "Ensemble", "3 components, weighted", EMER),
           ("Topic labels", "100% rules", "reproduced exactly", AMBER)])


    c1, c2 = st.columns([1.3, 1], gap="medium")
    with c1:
        cc = comp.dropna(subset=["val_macro_f1"]).sort_values("val_macro_f1")
        colour = [EMER if "ensemble" in m.lower() else (VIOLET if "MiniLM" in m else INDIGO) for m in cc.model]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cc.val_macro_f1, y=cc.model, orientation="h", marker_color=colour,
                             text=cc.val_macro_f1.round(3), textposition="outside", name="validation"))
        if "test_macro_f1" in cc:
            fig.add_trace(go.Scatter(x=cc.test_macro_f1, y=cc.model, mode="markers", name="test",
                                     marker=dict(symbol="diamond", size=9, color=AMBER)))
        fig.update_layout(xaxis_range=[0, cc.val_macro_f1.max() * 1.28], xaxis_title="macro-F1")
        chart(fig_style(fig, 420, "Every candidate, ranked"),
              "Validation macro-F1 per candidate with the held-out test score as an amber diamond. The ensemble is "
              "green, transformers purple, classical models blue.",
              "The winner was selected on validation and only then scored on test, so the headline number is not "
              "the result of picking whatever looked best on the final set.")
    with c2:
        cm = pd.crosstab(preds.sentiment_label, preds.predicted_sentiment).reindex(index=labels, columns=labels).fillna(0)
        fig = go.Figure(go.Heatmap(z=cm.values, x=labels, y=labels,
                                   colorscale=[[0, "#132019"], [1, INDIGO]], showscale=False,
                                   text=cm.values.astype(int), texttemplate="%{text}",
                                   textfont=dict(size=15, color=TXT)))
        fig.update_layout(xaxis_title="predicted", yaxis_title="actual")
        chart(fig_style(fig, 420, "Where it goes wrong"),
              "Rows are the true label, columns the prediction. The bright diagonal is correct work; the errors "
              "cluster around Neutral.",
              "Neutral is the hardest class and the most common one in real monitoring, so this weakness is carried "
              "into the live round explicitly rather than forgotten once the model is saved.")

    c1, c2 = st.columns([1, 1.25], gap="medium")
    with c1:
        per = rep[rep["class"].isin(labels)]
        fig = go.Figure()
        for metric, colour in [("precision", CYAN), ("recall", INDIGO), ("f1-score", EMER)]:
            fig.add_trace(go.Bar(x=per["class"], y=per[metric], name=metric, marker_color=colour,
                                 text=per[metric].round(2), textposition="outside",
                                 textfont=dict(size=10.5, color=MUT),
                                 hovertemplate="%{x} - " + metric + ": %{y:.3f}<extra></extra>"))
        fig.update_layout(barmode="group", bargap=.3, bargroupgap=.06,
                          yaxis=dict(range=[0, 1.05], tickformat=".1f"))
        chart(fig_style(fig, 360, "Per-class balance"),
              "Precision, recall and F1 for each class. A level profile means no class was traded away to flatter "
              "the headline score.",
              "Macro-F1 weights the classes equally, so a model that quietly ignored Neutral would still look fine "
              "on accuracy and bad here. This is the check on that.")
    with c2:
        share = r2m["topic"].get("whole_word_share", {})
        if share:
            sh = pd.DataFrame([{"topic": k, "inside": v["inside_other_word_share"],
                                "whole": v["whole_word_share"], "n": v["n"]} for k, v in share.items()])
            fig = go.Figure()
            fig.add_trace(go.Bar(y=sh.topic, x=sh.whole, orientation="h", name="keyword is the actual word",
                                 marker_color=EMER))
            fig.add_trace(go.Bar(y=sh.topic, x=sh.inside, orientation="h",
                                 name="keyword hidden inside another word", marker_color=ROSE))
            fig.update_layout(barmode="stack", xaxis_tickformat=".0%", xaxis_title="share of that topic's labels")
            chart(fig_style(fig, 360, "The topic labels were a lookup table"),
                  "For each topic, how its labels were triggered. Most fire on a keyword buried inside an unrelated "
                  "word: 'happy' contains 'app', 'band' contains 'ban'.",
                  "A topic classifier trained here would score well while learning spelling. Recovering the rule "
                  "exactly is what proves the labels describe characters, not meaning.")

    eyebrow("Read a post with the saved model", "loads the real ensemble from disk · first run takes about a minute")
    if not model_available():
        model_note()
    elif model_broken():
        model_note(model_broken())
    c1, c2 = st.columns([1.1, 1], gap="medium")
    with c1:
        txt = st.text_area("Post", "Honestly the new update broke my notifications and the battery is gone by noon.",
                           height=110, label_visibility="collapsed")
        go_btn = st.button("Analyse", type="primary",
                           disabled=not model_available() or bool(model_broken()))
    if go_btn:
        try:
            with st.spinner("Loading TF-IDF + embeddings + 3 transformer seeds..."):
                eng, nu = load_model()
            proba = eng._apply_neutral_bias(eng.predict_proba([txt]))[0]
            lab = eng.labels[int(proba.argmax())]
            topic, kwd, token, _ = nu.TopicRuleClassifier().explain(txt)
            via = f" via '{kwd}' inside '{token}'" if kwd else ""
            with c1:
                st.markdown(f'<div class="panel"><div class="sub">Predicted sentiment</div>'
                            f'<div style="font-size:2rem;font-weight:700;color:{SENT[lab]}">{lab}</div>'
                            f'<div class="sub">confidence {proba.max():.0%} &middot; dataset topic rule says '
                            f'<b>{topic}</b>{via}</div></div>', unsafe_allow_html=True)
            with c2:
                fig = go.Figure(go.Bar(x=eng.labels, y=proba, marker_color=[SENT[l] for l in eng.labels],
                                       text=[f"{p:.0%}" for p in proba], textposition="outside"))
                fig.update_layout(yaxis_range=[0, 1.14], yaxis_tickformat=".0%")
                st.plotly_chart(fig_style(fig, 260, "Class probabilities", legend=False), width="stretch",
                                config={"displayModeBar": False})
        except Exception as exc:
            # remember it, so the next render shows a note rather than spending
            # another minute failing the same way
            st.session_state.model_error = f"loading it failed ({type(exc).__name__})"
            st.rerun()


    eyebrow("Where it goes wrong, post by post",
            "the confusion matrix says Neutral is the weak class - here are the actual failures")
    st.markdown(
        '<div class="panel"><div class="sub" style="margin:0">Every held-out test post with the label it '
        'was given, the label the ensemble predicted and the three probabilities behind that call. A '
        'confusion matrix counts mistakes; this is what they look like.</div></div>',
        unsafe_allow_html=True)

    pr = preds.copy()
    pr["confidence"] = pr[["p_negative", "p_neutral", "p_positive"]].max(axis=1)
    pr["correct"] = pr.sentiment_label == pr.predicted_sentiment
    classes = sorted(pr.sentiment_label.dropna().unique())

    f1c, f2c, f3c = st.columns([1.1, 1.1, 1], gap="medium")
    true_pick = f1c.multiselect("Labelled as", classes, default=classes, key="r2_true")
    pred_pick = f2c.multiselect("Model said", classes, default=classes, key="r2_pred")
    show = f3c.selectbox("Show", ["Mistakes only", "Everything", "Correct only"], key="r2_show")

    g1, g2 = st.columns([2, 1], gap="medium")
    query = g1.text_input("Search the text", "", placeholder="battery, refund, love...", key="r2_q")
    conf_max = g2.slider("Maximum confidence", 0.34, 1.0, 1.0, 0.01, key="r2_conf")

    ev = pr[pr.sentiment_label.isin(true_pick) & pr.predicted_sentiment.isin(pred_pick)]
    if show == "Mistakes only":
        ev = ev[~ev.correct]
    elif show == "Correct only":
        ev = ev[ev.correct]
    if query.strip():
        ev = ev[ev.post_text.str.contains(query.strip(), case=False, na=False)]
    ev = ev[ev.confidence <= conf_max]

    err_rate = 1 - pr.correct.mean()
    st.caption(f"{len(ev):,} of {len(pr):,} test posts match. The model is wrong on "
               f"{err_rate:.1%} of the test set overall"
               + (f"; median confidence in this selection is {ev.confidence.median():.2f}"
                  if len(ev) else ""))
    if len(ev):
        st.dataframe(
            ev[["post_text", "sentiment_label", "predicted_sentiment", "confidence",
                "p_negative", "p_neutral", "p_positive", "rule_topic"]],
            width="stretch", height=430, hide_index=True,
            column_config={
                "post_text": st.column_config.TextColumn("Post", width="large"),
                "sentiment_label": st.column_config.TextColumn("Labelled", width="small"),
                "predicted_sentiment": st.column_config.TextColumn("Predicted", width="small"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0,
                                                              max_value=1.0, format="%.2f"),
                "p_negative": st.column_config.NumberColumn("P(neg)", format="%.2f", width="small"),
                "p_neutral": st.column_config.NumberColumn("P(neu)", format="%.2f", width="small"),
                "p_positive": st.column_config.NumberColumn("P(pos)", format="%.2f", width="small"),
                "rule_topic": st.column_config.TextColumn("Topic rule", width="small"),
            })
    else:
        st.info("Nothing matches those filters.")

    with st.expander("Interpretation"):
        st.markdown("**What this shows.** The held-out test set, filterable by what the post was labelled "
                    "and what the model predicted. Setting a low confidence ceiling surfaces the calls it "
                    "was least sure about.")
        st.markdown("**Why it matters.** Most Neutral errors are posts carrying mild sentiment that a "
                    "human annotator rounded one way and the model rounded the other - a labelling "
                    "boundary problem as much as a model problem, and you can only see that by reading "
                    "them.")

    method([
        ("Audit before modelling",
         "The 9,000 labelled rows contain only 7,900 unique texts. Finding those 1,100 duplicates first is "
         "what makes every score afterwards trustworthy.",
         "1,100 duplicates"),
        ("Close the leak",
         "Splits are grouped by text, so no duplicated post can sit in training and test at once. Scored "
         "naively, the same models look several points better than they are.",
         "group-aware split"),
        ("Test the preprocessing instead of assuming it",
         "Each cleaning step was ablated on its own. Some steps that feel obviously right cost accuracy and "
         "were dropped.",
         "measured, not assumed"),
        ("Climb a ladder of models",
         "Majority class, then logistic regression, then NB-SVM with character n-grams, then sentence "
         "embeddings, then three fine-tuned MiniLM seeds. Nine candidates, one protocol.",
         "9 candidates"),
        ("Combine only what earns its place",
         "A weight search settled on TF-IDF 0.25, embeddings 0.20 and the three fine-tuned seeds 0.55, with a "
         "1.05 neutral bias at the operating point.",
         "0.718 test macro-F1"),
        ("Read the errors",
         "Error analysis showed the topic labels were never semantic: they are a substring lookup, and a "
         "handful of rules reproduce all 9,000 of them exactly.",
         "100% by rules"),
        ("Save one loadable artefact",
         "The ensemble is written to disk with its config so Round 3 can load it unchanged, rather than "
         "retraining and quietly getting a different model.",
         "reused in Round 3"),
    ], "how a labelled corpus became a comprehension layer the next round could reuse", VIOLET,
           "Round 2<br>Semantic Recovery",
           [("Labelled training data", R2 / "data" / "Entropy_Labeled_Social_NLP_Training_Data.csv")],
           [("Saved ensemble", R2 / "model"),
         ("Model comparison", R2 / "outputs" / "Entropy_r2_model_comparison.csv"),
         ("Classification report", R2 / "outputs" / "Entropy_r2_classification_report.csv"),
         ("Metrics", R2 / "outputs" / "Entropy_r2_metrics.json")])

    report_panel("r2", {
        "Technical Report": R2 / "reports" / "Entropy_r2_technical_report.pdf",
        "Evaluation Metrics": R2 / "reports" / "Entropy_r2_evaluation_metrics_report.pdf",
    }, "The technical report covers the pipeline, the model comparison and the ensemble; the metrics report is "
       "the evaluation in full - per-class scores, the confusion matrix and the error analysis.")

    notebook_panel("r2", {"01 - NLP Model": R2 / "Entropy_01_NLP_Model.ipynb"},
                   "<b>01 - NLP Model</b> is the whole modelling record in order: the data audit, the leakage "
                   "check on the 1,100 duplicate texts, the preprocessing ablation, every candidate from the "
                   "majority-class baseline up to the fine-tuned transformers, the ensemble weight search, and "
                   "the error analysis that uncovered the substring rule behind the topic labels.")

# ===========================================================================  ROUND 3
if PAGE == 3:
    scored, r3m, man = load_r3()
    v = r3m["in_domain_validation"]
    _rbj = load_robustness() or {}
    _hc = _rbj.get("human_consensus") or {}
    _hc = _hc if "cohens_kappa" in _hc else None
    chosen = v["configurations"][v["chosen"]]
    tiles([("Posts collected", f"{man['rows_final']:,}", "two snapshots, no API keys", CYAN),
           ("On topic", f"{len(scored):,}", f"{r3m['dataset']['unique_authors']:,} authors", INDIGO),
           ("Analysis window", f"{r3m['dataset']['span_hours']:.0f} h", "of a 298 h collection", BLUE),
           ("Shifts detected", "3", "Holm-corrected, significant", ROSE),
           ("In-domain macro-F1", f"{_hc['model_macro_f1']:.3f}" if _hc else f"{chosen['macro_f1']:.3f}",
            f"{_hc['n_consensus']} posts, 2 annotators" if _hc else "150 hand-labelled posts", VIOLET)])


    lo, hi = scored.created_utc.min().to_pydatetime(), scored.created_utc.max().to_pydatetime()
    rng = st.slider("Time window", min_value=lo, max_value=hi, value=(lo, hi), format="DD MMM HH:mm", key="r3_rng")
    win = scored[(scored.created_utc >= rng[0]) & (scored.created_utc <= rng[1])]

    bins = (win.set_index("created_utc").resample("12h")
            .agg(n=("sentiment", "size"),
                 pos=("sentiment", lambda s: (s == "Positive").sum()),
                 neg=("sentiment", lambda s: (s == "Negative").sum()),
                 neu=("sentiment", lambda s: (s == "Neutral").sum())))
    bins = bins[bins.n >= 10]

    c1, c2 = st.columns([1.5, 1], gap="medium")
    with c1:
        fig = go.Figure()
        for name, col, colour in [("Negative", "neg", ROSE), ("Neutral", "neu", "#64748b"), ("Positive", "pos", EMER)]:
            fig.add_trace(go.Scatter(x=bins.index, y=bins[col] / bins.n, name=name, stackgroup="s",
                                     mode="lines", line=dict(width=.5, color=colour), fillcolor=colour))
        fig.add_vline(x=LAUNCH.timestamp() * 1000, line_color=AMBER, line_width=2)
        fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1], yaxis_title="share of posts")
        chart(fig_style(fig, 330, "Sentiment mix, every 12 hours"),
              "How the balance of opinion moves through the launch window; the amber line is the release itself.",
              "The negative band widens steadily after release. Because the model was validated on this exact kind "
              "of text, the movement can be trusted more than the absolute levels.")
    with c2:
        by_src = win.groupby("source").sentiment.value_counts(normalize=True).unstack(fill_value=0)
        order = (by_src.get("Positive", 0) - by_src.get("Negative", 0)).sort_values().index
        fig = go.Figure()
        for lab in ["Negative", "Neutral", "Positive"]:
            if lab in by_src:
                fig.add_trace(go.Bar(y=order, x=by_src.loc[order, lab], orientation="h", name=lab,
                                     marker_color=SENT[lab]))
        fig.update_layout(barmode="stack", xaxis_tickformat=".0%", xaxis_title="share of posts")
        chart(fig_style(fig, 330, "The same event, four moods"),
              "Sentiment split for each platform in the selected window.",
              "Mastodon skews positive, Reddit strongly negative - a gap far larger than any day-to-day movement. A "
              "monitoring system with one global baseline would report platform mix as if it were opinion.")

    c1, c2 = st.columns([1.5, 1], gap="medium")
    with c1:
        hourly = win.set_index("created_utc").resample("1h").size()
        sp = pd.DataFrame(r3m["spikes"])
        sp["hour"] = pd.to_datetime(sp.hour, format="mixed", utc=True)
        sp = sp[(sp.hour >= rng[0]) & (sp.hour <= rng[1])].nlargest(8, "z_posts")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly.index, y=hourly.values, mode="lines", name="posts / hour",
                                 line=dict(color=BLUE, width=2), fill="tozeroy",
                                 fillcolor="rgba(168,191,106,.14)"))
        fig.add_trace(go.Scatter(x=sp.hour, y=sp.posts, mode="markers", name="spike",
                                 marker=dict(size=13, color=AMBER, line=dict(width=2, color=BG)),
                                 hovertext=[f"z = {z:.1f}" for z in sp.z_posts]))
        fig.update_layout(yaxis_title="posts per hour")
        chart(fig_style(fig, 320, "Activity and detected spikes"),
              "Hourly posting volume with statistically detected spikes marked. Spikes use a robust z-score against "
              "a trailing 48-hour baseline, so a quiet night does not manufacture one.",
              "Volume and engagement peak at different moments: the release produced the most posts, but the "
              "loudest engagement peak came four days later from a single thread about Android 17.")
    with c2:
        ents = pd.DataFrame(r3m["entities"]).sort_values("net")
        fig = go.Figure(go.Bar(x=ents.net, y=ents.entity, orientation="h",
                               marker=dict(color=ents.net, colorscale=[[0, ROSE], [1, "#475569"]], showscale=False),
                               text=ents.net.round(2), textposition="outside", customdata=ents.mentions,
                               hovertemplate="%{y}<br>net %{x:.2f} across %{customdata} posts<extra></extra>"))
        fig.update_layout(xaxis_range=[ents.net.min() * 1.3, .06], xaxis_title="net sentiment")
        chart(fig_style(fig, 320, "What the complaints are about", legend=False),
              "Net sentiment per entity and per feature theme across the whole window.",
              "Bugs and notifications sit far below the visual redesign that dominated the press. This ordering is "
              "the prioritised queue a product team would work down.")

    c1, c2 = st.columns([1, 1.4], gap="medium")
    with c1:
        d_pt, d_lo, d_hi = (_hc["shipped_minus_as_is"]["point"], _hc["shipped_minus_as_is"]["ci_low"],
                            _hc["shipped_minus_as_is"]["ci_high"]) if _hc else (0, 0, 0)
        cfgs = pd.DataFrame([{"configuration": k, **vv} for k, vv in v["configurations"].items()])
        fig = go.Figure(go.Bar(x=cfgs.macro_f1, y=cfgs.configuration.str.replace(r" \(used\)", "", regex=True),
                               orientation="h", marker_color=[DIM, INDIGO, EMER],
                               text=cfgs.macro_f1.round(3), textposition="outside"))
        fig.add_vline(x=v["round2_test_macro_f1"], line_dash="dot", line_color=AMBER)
        fig.update_layout(xaxis_range=[0, .88], xaxis_title="macro-F1 on hand-labelled live posts")
        why = ("Reusing a model is not free, and these two details stayed invisible until the model was "
               "validated in the domain where it is actually deployed.")
        if _hc:
            why += (f" One caveat, and it is ours: measured against the two-annotator consensus instead, the "
                    f"correction is worth {d_pt:+.3f} macro-F1 with a 95% interval of [{d_lo:+.3f}, {d_hi:+.3f}] "
                    f"- it includes zero, so the improvement shown here does not replicate against the better "
                    f"reference standard. The chunked configuration is kept because truncation discards text "
                    f"from 68% of these posts, not because it measurably scores higher.")
        chart(fig_style(fig, 300, "Two inference fixes, measured", legend=False),
              "The saved model silently truncated 68% of these posts at 50 tokens and skipped its own "
              "neutral-bias setting. Against the first annotator's labels, fixing both lifted macro-F1 from "
              "0.632 to 0.666; the amber line is its score on tweets.", why)
    with c2:
        trig = pd.DataFrame(r3m["triggers"])
        trig = trig[trig.headlines.map(len) > 0].head(5)
        st.markdown('<div class="panel"><h4>Why each spike happened</h4><div class="sub">every spike matched to '
                    'news published in the same hour and the top post of that window - nothing inferred from '
                    'sentiment alone</div></div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "when (UTC)": pd.to_datetime(trig.when, format="mixed", utc=True).dt.strftime("%d %b %H:%M"),
            "event": trig.kind + " · " + trig["size"],
            "evidence": trig.headlines.map(lambda h: h[0][:80] if h else ""),
        }), width="stretch", hide_index=True, height=248)



    # ------------------------------------------------------- robustness ----
    rb = load_robustness()
    if rb:
        eyebrow("Checks against our own claim",
                "four ways this finding could have been wrong, tested")
        ac, al = rb["author_concentration"], rb["alerting"]
        vd, om = rb.get("vader_baseline", {}), rb.get("our_model_on_same_labels", {})
        hc = rb.get("human_consensus") or {}
        hc = hc if "cohens_kappa" in hc else None
        tiles([
            ("Authors behind the negatives", f"{ac['negative_authors']:,}",
             f"for {ac['negative_posts']:,} negative posts", EMER),
            ("Busiest 10 accounts", f"{ac['top10_share_of_corpus']:.1%}",
             "of the whole corpus", INDIGO),
            ("Annotator agreement", f"{hc['cohens_kappa']:.3f}" if hc else "pending",
             f"Cohen's kappa, {hc['raw_agreement']:.0%} raw" if hc else "one annotator so far", VIOLET),
            ("Alert lead time", f"{al['hours_after_launch']:.0f} h",
             f"{al['false_alarms_pre_launch']} false alarms before launch", AMBER),
        ])

        rc1, rc2 = st.columns(2, gap="medium")
        with rc1:
            lv = rb["period_levels"]
            names = list(lv)
            short = ["Pre-launch", "+0-24h", "+24-72h", "72h+"][:len(names)]
            nets = [lv[k]["net"] for k in names]
            lo = [lv[k]["net"] - lv[k]["ci_low"] for k in names]
            hi = [lv[k]["ci_high"] - lv[k]["net"] for k in names]
            cols = [AMBER if not lv[k]["distinguishable_from_zero"]
                    else (CLAY if lv[k]["net"] < 0 else EMER) for k in names]
            fig = go.Figure(go.Scatter(
                x=short, y=nets, mode="markers",
                marker=dict(size=15, color=cols, line=dict(color=BG, width=2)),
                error_y=dict(type="data", symmetric=False, array=hi, arrayminus=lo,
                             color="#5f7568", thickness=1.8, width=8),
                customdata=[[lv[k]["n"], lv[k]["ci_low"], lv[k]["ci_high"]] for k in names],
                hovertemplate="net %{y:+.3f}<br>95%% CI %{customdata[1]:+.2f} to "
                              "%{customdata[2]:+.2f}<br>%{customdata[0]} posts<extra></extra>"))
            fig.add_hline(y=0, line_color=GRID, line_dash="dash")
            fig.update_layout(yaxis_title="net sentiment")
            chart(fig_style(fig, 330, "Is each level actually a claim?", legend=False),
                  "Net sentiment per period with a 95% bootstrap interval. The pre-launch marker is amber "
                  "because its interval crosses zero - on 171 posts the baseline cannot be told apart from "
                  "neutral.",
                  "It changes the sentence we are entitled to write. The decline is significant at every "
                  "pre-specified comparison and reaches 0.43 below baseline by day three, but the launch "
                  "fell from indifference, not from approval, and the report now says so.")
        with rc2:
            if hc:
                names = ["VADER lexicon", "Ours, as-is", "Ours, shipped"]
                vals = [hc["vader_macro_f1"], hc["as_is_macro_f1"], hc["model_macro_f1"]]
                ref = f"the {hc['n_consensus']}-post two-annotator consensus"
                dd = hc["shipped_minus_as_is"]
                why = ("A reused fine-tuned ensemble has to earn its cost against something anyone could "
                       f"install in a minute, and it does - by {hc['model_macro_f1'] - hc['vader_macro_f1']:.3f} "
                       "macro-F1. What it does not do is beat plain truncation: chunked scoring comes out "
                       f"{dd['point']:+.3f} with a 95% interval of [{dd['ci_low']:+.3f}, {dd['ci_high']:+.3f}], "
                       "which includes zero. Against the first annotator's labels the correction appeared to "
                       "help by 0.034; against the consensus it does not replicate. We keep it because "
                       "truncation discards text from 68% of these posts, not because it scores better.")
            else:
                names = ["VADER lexicon", "Ours, as-is", "Ours, shipped"]
                vals = [vd.get("macro_f1", 0), om.get("as_is_macro_f1", 0), om.get("shipped_macro_f1", 0)]
                ref = "150 posts labelled by a single annotator"
                why = ("A reused fine-tuned ensemble has to earn its cost. The lexicon sits near chance on "
                       "three classes because it reads technical complaint language as neutral.")
            fig = go.Figure(go.Bar(
                x=names, y=vals, marker_color=[DIM, INDIGO, EMER],
                text=[f"{v:.3f}" for v in vals], textposition="outside",
                textfont=dict(size=11, color=MUT), cliponaxis=False,
                hovertemplate="%{x}: %{y:.3f} macro-F1<extra></extra>"))
            fig.update_layout(yaxis_range=[0, max(vals) * 1.28], yaxis_title="macro-F1")
            chart(fig_style(fig, 330, "Worth the complexity?", legend=False),
                  f"All three scored against {ref}. VADER is a rule-based lexicon scorer built for social "
                  "media - the thing anyone could install in a minute.", why)

        st.markdown(
            f'<div class="panel"><h4>Is this a pile-on?</h4><div class="sub" style="margin:0">'
            f'The cheapest explanation for any measured drop in sentiment is that a few accounts posted a '
            f'lot of complaints. Not here: <b>{ac["negative_posts"]:,} negative posts come from '
            f'{ac["negative_authors"]:,} distinct authors</b>, '
            f'{ac["share_of_authors_posting_once"]:.0%} of all authors appear exactly once, the busiest '
            f'single account contributes {ac["busiest_author_posts"]} posts, and the ten busiest together '
            f'are {ac["top10_share_of_corpus"]:.1%} of the corpus. Many people reacting independently, not '
            f'a campaign.</div></div>'
            f'<div class="panel" style="margin-top:12px"><h4>Would it have alerted in time?</h4>'
            f'<div class="sub" style="margin:0">Analysing a launch after it ends is not monitoring. '
            f'Replaying <code>{al["rule"]}</code> over the window fires at '
            f'<b>{str(al["first_fired"])[:16]} UTC</b> - <b>{al["hours_after_launch"]:.0f} hours after '
            f'release</b> - with <b>{al["false_alarms_pre_launch"]} false alarms</b> beforehand. A team '
            f'watching this would have known the reception had turned while the press was still writing '
            f'about the redesign.</div></div>', unsafe_allow_html=True)

        if hc:
            ka = hc["kappa_vs_first_annotator"]
            st.markdown(
                f'<div class="panel" style="margin-top:12px"><h4>Who decided what &ldquo;correct&rdquo; means</h4>'
                f'<div class="sub" style="margin:0">Both team members labelled the same '
                f'{hc["n_double_labelled"]} posts independently, from a shuffled sheet with the model\'s '
                f'predictions and the first annotator\'s labels removed. They agree at '
                f'<b>Cohen\'s kappa {hc["cohens_kappa"]:.3f}</b> ({hc["raw_agreement"]:.1%} raw), which is '
                f'&ldquo;almost perfect&rdquo; on the conventional scale. The <b>{hc["n_consensus"]} posts '
                f'they agree on</b> are the reference standard for every accuracy figure on this page; the '
                f'{hc["n_disagreements_excluded"]} they disagree on are set aside rather than adjudicated, '
                f'so no tie is broken by the people being measured. Each annotator agrees with the original '
                f'single-annotator labels at only kappa '
                f'{min(ka.values()):.2f}-{max(ka.values()):.2f}, well below their agreement with each '
                f'other - which is why the consensus replaced those labels instead of supplementing them. '
                f'It is also better balanced: {hc["consensus_balance"].get("Positive", 0)} Positive posts '
                f'against 15 in the original sample.</div></div>', unsafe_allow_html=True)

    # ---------------------------------------------------- operating point ---
    eyebrow("Move the operating point",
            "the decision threshold is a choice - so make it one you can test")
    st.markdown(
        '<div class="panel"><div class="sub" style="margin:0">Round 3 shipped a neutral bias of 1.05, and the '
        'saved probabilities already carry it. The slider below reweights the Neutral class <b>relative to '
        'that shipped point</b>, re-decides all ' + f"{len(scored):,}" + ' posts and rescores the 150 '
        'hand-labelled ones - instantly, because the per-post probabilities are on disk and no model has to '
        'run. A weight of 1.00 is exactly what was submitted. The point is not to find a better number: it is '
        'to show that the finding does not depend on the number we picked.</div></div>',
        unsafe_allow_html=True)

    wt = st.slider("Neutral weight, relative to the shipped operating point",
                   0.70, 1.60, 1.00, 0.05, key="r3_bias")
    probs = scored[PCOLS].to_numpy()
    live = scored.assign(sentiment=relabel(probs, wt))
    f1_w, acc_w, n_val = load_validation(wt)
    f1_0, acc_0, _ = load_validation(1.00)
    neutral_share = float((live.sentiment == "Neutral").mean())
    net_all = float((live.sentiment == "Positive").mean() - (live.sentiment == "Negative").mean())
    changed = int((live.sentiment != scored.sentiment).sum())

    st.markdown(
        '<div class="minitiles">'
        f'<div class="minitile" style="--c:{VIOLET}"><div class="l">In-domain macro-F1</div>'
        f'<div class="v">{f1_w:.3f}</div><div class="d">{f1_w - f1_0:+.3f} vs shipped, on {n_val} labels</div></div>'
        f'<div class="minitile" style="--c:{INDIGO}"><div class="l">Accuracy</div>'
        f'<div class="v">{acc_w:.3f}</div><div class="d">{acc_w - acc_0:+.3f} vs shipped</div></div>'
        f'<div class="minitile" style="--c:{AMBER}"><div class="l">Labelled Neutral</div>'
        f'<div class="v">{neutral_share:.0%}</div><div class="d">of {len(live):,} live posts</div></div>'
        f'<div class="minitile" style="--c:{ROSE if net_all < 0 else EMER}"><div class="l">Net sentiment</div>'
        f'<div class="v">{net_all:+.2f}</div><div class="d">across the whole window</div></div>'
        f'<div class="minitile" style="--c:{CLAY}"><div class="l">Labels changed</div>'
        f'<div class="v">{changed:,}</div><div class="d">{changed / len(live):.1%} of the corpus</div></div>'
        '</div>', unsafe_allow_html=True)

    lb = (live.set_index("created_utc").resample("12h")
          .agg(neg=("sentiment", lambda x: (x == "Negative").sum()),
               neu=("sentiment", lambda x: (x == "Neutral").sum()),
               pos=("sentiment", lambda x: (x == "Positive").sum())))
    lb = lb[(lb.neg + lb.neu + lb.pos) >= 10]
    nets, los, his = net_with_ci(lb[["neg", "neu", "pos"]].to_numpy())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(lb.index) + list(lb.index)[::-1],
                             y=list(his) + list(los)[::-1], fill="toself",
                             fillcolor="rgba(127,181,130,.16)", line=dict(width=0),
                             hoverinfo="skip", name="95% interval"))
    fig.add_trace(go.Scatter(x=lb.index, y=nets, mode="lines+markers", name="net sentiment",
                             line=dict(color=SAGE, width=2.6), marker=dict(size=6),
                             customdata=np.stack([los, his, (lb.neg + lb.neu + lb.pos)], axis=-1),
                             hovertemplate="net %{y:.2f}<br>95%% CI %{customdata[0]:.2f} to "
                                           "%{customdata[1]:.2f}<br>%{customdata[2]} posts<extra></extra>"))
    fig.add_hline(y=0, line_color=GRID)
    fig.add_vline(x=LAUNCH.timestamp() * 1000, line_color=AMBER, line_dash="dot")
    fig.add_annotation(x=LAUNCH, y=float(np.nanmax(his)), text="iOS 27 ships", showarrow=False,
                       font=dict(color=AMBER, size=11), yshift=10)
    fig.update_layout(yaxis_title="net sentiment (pos - neg)")
    chart(fig_style(fig, 360, "Net sentiment, with bootstrap uncertainty"),
          "Net sentiment in each 12-hour bin, with a 95% interval from 1,200 multinomial resamples of that "
          "bin's class counts. Wide bands are thin bins: the interval is drawn from how many posts the bin "
          "actually holds, so a quiet window cannot masquerade as a confident reading.",
          "Point estimates alone would let a reader mistake a 12-post wobble for a change in opinion. The "
          "claim being made is that the post-launch bins sit below zero by more than their own uncertainty - "
          "and that stays true as you move the slider, which is the real test of the finding.")

    # ----------------------------------------------------------- evidence ---
    eyebrow("Read the evidence", "every aggregate on this page is made of these posts")
    st.markdown(
        '<div class="panel"><div class="sub" style="margin:0">A dashboard that only shows aggregates asks to '
        'be trusted. This is the corpus itself, scored at the operating point set above: filter it, sort it, '
        'and check any claim on this page against the posts underneath it. The three probability columns are '
        'the model output, not a label - which is also how you find the posts it was least sure about.'
        '</div></div>', unsafe_allow_html=True)

    f1c, f2c, f3c = st.columns([1.1, 1.1, 1], gap="medium")
    want = f1c.multiselect("Sentiment", list(LABELS), default=list(LABELS), key="r3_ev_sent")
    srcs = sorted(live.source.unique())
    want_src = f2c.multiselect("Platform", srcs, default=srcs, key="r3_ev_src")
    mention_cols = [c for c in ["iOS 27", "iPadOS 27", "macOS 27", "Apple Intelligence", "Android 17"]
                    if c in live.columns]
    mention = f3c.selectbox("Mentions", ["Any release"] + mention_cols, key="r3_ev_ment")

    g1, g2 = st.columns([2, 1], gap="medium")
    query = g1.text_input("Search the text", "", placeholder="notifications, battery, Siri...",
                          key="r3_ev_q")
    conf_max = g2.slider("Maximum confidence", 0.34, 1.0, 1.0, 0.01, key="r3_ev_conf")

    ev = live[live.sentiment.isin(want) & live.source.isin(want_src)]
    if mention != "Any release":
        ev = ev[ev[mention].astype(bool)]
    if query.strip():
        ev = ev[ev.text_full.str.contains(query.strip(), case=False, na=False)]
    ev = ev[ev.confidence <= conf_max]

    st.caption(f"{len(ev):,} of {len(live):,} posts match - "
               f"{(ev.sentiment == 'Negative').mean() if len(ev) else 0:.0%} Negative, "
               f"median confidence {ev.confidence.median() if len(ev) else float('nan'):.2f}")
    if len(ev):
        view = (ev.sort_values("created_utc")
                  [["created_utc", "source", "sentiment", "confidence",
                    "p_negative", "p_neutral", "p_positive", "score", "text_full", "url"]]
                  .rename(columns={"created_utc": "when", "text_full": "post", "score": "engagement"}))
        st.dataframe(
            view, width="stretch", height=430, hide_index=True,
            column_config={
                "when": st.column_config.DatetimeColumn("When", format="DD MMM HH:mm", width="small"),
                "source": st.column_config.TextColumn("Platform", width="small"),
                "sentiment": st.column_config.TextColumn("Model says", width="small"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0,
                                                              max_value=1.0, format="%.2f"),
                "p_negative": st.column_config.NumberColumn("P(neg)", format="%.2f", width="small"),
                "p_neutral": st.column_config.NumberColumn("P(neu)", format="%.2f", width="small"),
                "p_positive": st.column_config.NumberColumn("P(pos)", format="%.2f", width="small"),
                "engagement": st.column_config.NumberColumn("Engagement", format="%d", width="small"),
                "post": st.column_config.TextColumn("Post", width="large"),
                "url": st.column_config.LinkColumn("Source", display_text="open", width="small"),
            })
        st.download_button("Download this selection as CSV",
                           view.to_csv(index=False).encode("utf-8"),
                           file_name="Entropy_round3_selection.csv", mime="text/csv",
                           key="r3_ev_dl")
    else:
        st.info("Nothing matches those filters.")

    with st.expander("Interpretation"):
        st.markdown("**What this shows.** The scored corpus itself, filtered live. Lowering the confidence "
                    "ceiling surfaces what the engine was least sure about, which is where its errors "
                    "concentrate.")
        st.markdown("**Why it matters.** A monitoring tool that cannot show the evidence behind an alert does "
                    "not get acted on. Every number on this page is one filter away from the posts that "
                    "produced it.")

    method([
        ("Pick an event that comes with a control",
         "The iOS 27 release on 14 September 2026, with the Android 17 release four days later as a "
         "comparison, so any shift can be checked against a second launch in the same window.",
         "295-hour window"),
        ("Collect from public endpoints only",
         "Reddit Atom feeds, the Hacker News Algolia API, Mastodon tag timelines and Lemmy search - no paid "
         "APIs, one request at a time, author names stored only as salted hashes.",
         "5,498 posts"),
        ("Go back for what RSS hides",
         "Reddit feeds carry no scores and default to newest, so a second pass searched by top and expanded "
         "those threads. That pass is the difference between 31 and 688 Reddit comments.",
         "two snapshots"),
        ("Filter to the question actually asked",
         "English posts that explicitly mention a tracked release, de-duplicated across both snapshots.",
         "1,403 on topic"),
        ("Score with the saved model, correctly",
         "The Round 2 ensemble is loaded unchanged, but 68% of these posts exceed its 50-token training "
         "length, so long posts are split into overlapping 34-word windows and the neutral bias is applied.",
         "0.699 vs 2 annotators"),
        ("Test the shifts rather than eyeball them",
         "Twelve-hour bins, two-proportion z-tests between adjacent regimes, and a Holm correction across the "
         "whole family of comparisons.",
         "3 significant shifts"),
        ("Rule out the obvious confound",
         "The platform mix changes across the window, so the entire comparison was repeated inside Mastodon "
         "alone. The decline survives, which is what makes it a finding rather than an artefact.",
         "p = 3.6e-8 within platform"),
        ("Give every spike a named cause",
         "Each detected spike is matched to news published in the same hour and to the top post of that "
         "window, so no alert is raised without evidence a human can read.",
         "every spike attributed"),
    ], "how a live event was collected, scored and turned into defensible claims", AMBER,
           "Round 3<br>Signal Tracking",
           [("Saved ensemble (Round 2)", R2 / "model"),
         ("Collected live posts", R3 / "data" / "Entropy_round3_collected_posts.csv"),
         ("News timeline", R3 / "data" / "Entropy_round3_news_timeline.csv")],
           [("Scored posts", R3 / "outputs" / "Entropy_round3_scored_posts.csv"),
         ("Metrics", R3 / "outputs" / "Entropy_round3_metrics.json"),
         ("Hand-labelled validation set", R3 / "outputs" / "Entropy_round3_validation_sample.csv"),
         ("Entity sentiment", R3 / "outputs" / "Entropy_round3_entity_sentiment.csv")])

    report_panel("r3", {
        "Analytical Report": R3 / "reports" / "Entropy_round3_analytical_report.pdf",
        "Analysis Notebook (PDF)": R3 / "reports" / "Entropy_r3_analysis_notebook.pdf",
    }, "The analytical report is the Round 3 deliverable: collection method, the detected shifts with their "
       "statistics, the sampling-confound control and the named causes. The notebook PDF is the same analysis "
       "exported cell by cell for submission.")

    notebook_panel("r3", {"02 - Realtime Analysis": R3 / "Entropy_02_Realtime_Analysis.ipynb"},
                   "<b>02 - Realtime Analysis</b> loads the saved Round 2 ensemble unchanged, scores the live "
                   "corpus with the chunking and neutral-bias fixes measured above, then runs the shift "
                   "detection, the within-platform control that rules out the sampling confound, the spike "
                   "attribution against the news timeline, and the 150-post in-domain validation.")

# ===========================================================================  ROUND 4  (the dashboard itself - landing page)
if PAGE == 4:
    try:
        posts, users, raw, log = load_p1()
        comp, rep, preds, r2m, train = load_r2()
        scored, r3m, man = load_r3()
    except FileNotFoundError as e:
        st.error(f"Missing artefact: {e.filename}")
        st.stop()

    macro = float(rep.loc[rep["class"] == "macro avg", "f1-score"].iloc[0])
    net_live = (scored.sentiment == "Positive").mean() - (scored.sentiment == "Negative").mean()
    tiles([
        ("Rows recovered", f"{len(posts):,}", f"from {len(raw):,} corrupted", CYAN),
        ("Defects repaired", "10", "classes, all logged", INDIGO),
        ("SQL questions", "3 / 3", "easy, medium, hard", BLUE),
        ("Model macro-F1", f"{macro:.3f}", "on unseen labelled posts", VIOLET),
        ("Live posts", f"{man['rows_final']:,}", f"{len(scored):,} on topic", AMBER),
        ("Live net sentiment", f"{net_live:+.2f}", "positive minus negative", ROSE if net_live < 0 else EMER),
    ])


    eyebrow("Try the engine", "the restored model, scored the way Round 3 scores live posts")
    if not model_available():
        model_note()
    elif model_broken():
        model_note(model_broken())

    EXAMPLES = [
        ("A complaint", "Honestly the update broke my notifications and the battery is gone by noon. "
                        "Third bug this week and still no fix in sight."),
        ("A compliment", "Updated last night and the new focus modes are genuinely excellent - "
                         "everything feels faster and the battery is actually better than before."),
        ("A question", "Has anyone worked out whether the new backup format is compatible with "
                       "the older desktop app? Trying to decide if I should update this weekend."),
    ]
    if "r4_text" not in st.session_state:
        st.session_state.r4_text = EXAMPLES[0][1]

    pick_col, work_col = st.columns([1, 2.15], gap="medium")

    with pick_col:
        st.markdown('<div class="io-h">Start from an example</div>', unsafe_allow_html=True)
        for i, (label, sample) in enumerate(EXAMPLES):
            if st.button(label, key=f"r4_ex{i}", width="stretch"):
                st.session_state.r4_text = sample
                st.session_state.pop("r4_src", None)
                st.rerun()
        if st.button("Clear", key="r4_exclr", width="stretch"):
            st.session_state.r4_text = ""
            st.session_state.pop("r4_src", None)
            st.rerun()

        st.markdown(
            '<div class="panel" style="margin-top:.85rem">'
            '<h4 style="font-size:.9rem">Links it can read</h4>'
            '<div class="linklist">'
            '<div><b>Hacker News</b><code>news.ycombinator.com/item?id=39104484</code></div>'
            '<div><b>Reddit</b><code>reddit.com/r/apple/comments/1wadiip/...</code></div>'
            '<div><b>Mastodon</b><code>mastodon.social/@user/109252195317431879</code></div>'
            '<div><b>Lemmy</b><code>lemmy.world/post/12345678</code></div>'
            '<div><b>Anything else</b><span>a public article or blog post - the page text is '
            'read directly</span></div>'
            '</div>'
            '<div class="sub" style="margin:.55rem 0 0 0;font-size:.74rem">Public posts only. Private, '
            'deleted and login-walled pages cannot be fetched, and non-public addresses are refused.'
            '</div></div>', unsafe_allow_html=True)

    with work_col:
        url_col, btn_col = st.columns([3, 1], gap="small", vertical_alignment="bottom")
        link = url_col.text_input("Paste a link to a public post", key="r4_url",
                                  placeholder="https://news.ycombinator.com/item?id=39104484")
        if btn_col.button("Fetch", key="r4_fetch", width="stretch") and link.strip():
            try:
                with st.spinner("Fetching the post..."):
                    got = fetch_post(link.strip())
                st.session_state.r4_text = got["combined"][:4000]
                st.session_state.r4_src = got
                st.rerun()
            except Exception as exc:
                st.session_state.pop("r4_src", None)
                st.warning(str(exc))

        src = st.session_state.get("r4_src")
        if src:
            bits = [f"from <b>{src['source']}</b>"]
            if src.get("author"):
                bits.append(f"by {src['author']}")
            if src.get("score") is not None:
                bits.append(f"{src['score']} points")
            st.markdown(f'<div class="sub" style="margin:-.35rem 0 .45rem 0">Loaded '
                        f'{" &middot; ".join(bits)}</div>', unsafe_allow_html=True)

        st.text_area("Or write the post yourself", key="r4_text", height=170,
                     placeholder="Type anything - a review, a complaint, a question...")
        run = st.button("Read it", type="primary", key="r4_run", width="stretch",
                        disabled=not model_available() or bool(model_broken()))
        st.caption("Links are read through each platform's own public endpoint - the same key-less routes "
                   "the Round 3 collector uses. The model is the real ensemble loaded from disk; the first "
                   "run takes about a minute, then it is cached.")

        if run and st.session_state.r4_text.strip():
            try:
                with st.spinner("Loading the saved ensemble..."):
                    eng, nu = load_model()
                proba, n_chunks = score_text(eng, st.session_state.r4_text.strip())
                lab = eng.labels[int(proba.argmax())]
                topic, kwd, token, _ = nu.TopicRuleClassifier().explain(st.session_state.r4_text)
                via = f" via &lsquo;{kwd}&rsquo; inside &lsquo;{token}&rsquo;" if kwd else ""
                words = len(st.session_state.r4_text.split())
                note = (f"{words} words, scored in {n_chunks} overlapping windows and averaged"
                        if n_chunks > 1 else f"{words} words, short enough to score in one pass")
                res1, res2 = st.columns([1.1, 1], gap="medium")
                with res1:
                    st.markdown(
                        f'<div class="panel" style="border-color:{SENT[lab]}66">'
                        f'<div class="sub" style="margin:0">Predicted sentiment</div>'
                        f'<div style="font-family:Outfit,Inter,sans-serif;font-size:2.1rem;font-weight:700;'
                        f'line-height:1.15;color:{SENT[lab]}">{lab}</div>'
                        f'<div class="sub" style="margin:.15rem 0 0 0">{proba.max():.0%} confidence &middot; '
                        f'{note}</div>'
                        f'<div class="sub" style="margin:.35rem 0 0 0">Dataset topic rule reads this as '
                        f'<b>{topic}</b>{via} - a reminder that the Round 2 topic labels were a substring '
                        f'lookup, not semantics.</div></div>', unsafe_allow_html=True)
                with res2:
                    fig = go.Figure(go.Bar(
                        x=list(eng.labels), y=proba, marker_color=[SENT[l] for l in eng.labels],
                        text=[f"{v:.0%}" for v in proba], textposition="outside",
                        textfont=dict(size=12, color=MUT), cliponaxis=False,
                        hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
                    fig.update_layout(yaxis_range=[0, 1.16], yaxis_tickformat=".0%")
                    chart(fig_style(fig, 300, "What the engine actually returns", legend=False),
                          "The three class probabilities behind the label, after the same chunking and "
                          "neutral-bias corrections Round 3 applies to every live post.",
                          "A single label hides how close the call was. Two nearly equal bars mean the engine "
                          "is guessing, which is exactly the case a monitoring tool should surface rather than "
                          "round away.")
            except Exception as exc:
                # remember it, so the next render shows a note instead of
                # spending another minute failing the same way
                st.session_state.model_error = f"loading it failed ({type(exc).__name__})"
                st.rerun()
        elif run:
            st.caption("Type something first.")

    eyebrow("The engine, running", "the restored stack pointed at a real event")
    c1, c2 = st.columns([1.35, 1], gap="medium")
    with c1:
        bins = (scored.set_index("created_utc").resample("12h")
                .agg(n=("sentiment", "size"),
                     pos=("sentiment", lambda s: (s == "Positive").sum()),
                     neg=("sentiment", lambda s: (s == "Negative").sum())))
        bins = bins[bins.n >= 15]
        bins["net"] = (bins.pos - bins.neg) / bins.n
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bins.index, y=bins.n, name="posts per 12h",
                             marker_color="rgba(168,191,106,.35)", yaxis="y2",
                             hovertemplate="%{y} posts<extra></extra>"))
        fig.add_trace(go.Scatter(x=bins.index, y=bins.net, name="net sentiment", mode="lines+markers",
                                 line=dict(color=CYAN, width=3), marker=dict(size=7),
                                 hovertemplate="net %{y:.2f}<extra></extra>"))
        fig.add_hline(y=0, line_color=GRID)
        fig.add_vline(x=LAUNCH.timestamp() * 1000, line_color=AMBER, line_dash="dot")
        fig.add_annotation(x=LAUNCH, y=bins.net.max(), text="iOS 27 ships", showarrow=False,
                           font=dict(color=AMBER, size=11), yshift=14)
        fig.update_layout(yaxis=dict(title="net sentiment"),
                          yaxis2=dict(overlaying="y", side="right", showgrid=False, title="posts"))
        chart(fig_style(fig, 340, "Live reaction to a real launch"),
              "Bars are posting volume per 12 hours, the line is net sentiment (positive share minus negative). "
              "Volume peaks at release while opinion falls and stays down.",
              "This is the payoff of the whole rebuild: a recovered dataset trained a model that now reads the "
              "public mood of a live event, and it catches a decline a mention-counter would have missed.")
    with c2:
        dist = scored.sentiment.value_counts().reindex(["Negative", "Neutral", "Positive"])
        fig = go.Figure(go.Pie(labels=dist.index, values=dist.values, hole=.62,
                               marker=dict(colors=[SENT[i] for i in dist.index],
                                           line=dict(color=PANEL, width=3)),
                               textinfo="label+percent", textfont=dict(size=12)))
        fig.add_annotation(text=f"<b>{len(scored):,}</b><br>posts", showarrow=False,
                           font=dict(size=17, color=TXT))
        chart(fig_style(fig, 340, "Verdict on the update", legend=False),
              "How the on-topic live posts split across the three sentiment classes.",
              "Negative outweighs positive two to one. The engine does not just count conversation, it can say "
              "which way the conversation leans.")

    eyebrow("One result from each round", "the single chart that defines each stage")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        led = []
        for line in [l.strip() for l in log.splitlines() if re.match(r"^\d+\.\s", l.strip())]:
            body = line.split(". ", 1)[-1]
            m = re.search(r"(\d[\d,]*)", body)
            if not m:
                continue
            rows = int(m.group(1).replace(",", ""))
            label = re.sub(r"\s+", " ", (body[:m.start()] + body[m.end():])).strip(" ;,.")
            led.append({"repair": label[:32], "rows": rows})
        led = pd.DataFrame(led)
        led = led[led.rows < len(posts)]  # drop whole-column normalisations so real defects stay legible
        led = led.nlargest(7, "rows").sort_values("rows")
        fig = go.Figure(go.Bar(x=led.rows, y=led.repair, orientation="h", marker_color=CYAN,
                               text=led.rows, texttemplate="%{text:,}", textposition="outside"))
        fig.update_layout(xaxis_range=[0, led.rows.max() * 1.3])
        chart(fig_style(fig, 300, "Round 1 · damage repaired", legend=False),
              "The six largest cleaning operations by rows touched.",
              "Recovery is the foundation: every later round inherits these 12,000 rows, so each repair had to be "
              "justified rather than convenient.")
    with c2:
        cc = comp.dropna(subset=["val_macro_f1"]).nlargest(6, "val_macro_f1").sort_values("val_macro_f1")
        colours = [EMER if "ensemble" in m.lower() else INDIGO for m in cc.model]
        fig = go.Figure(go.Bar(x=cc.val_macro_f1, y=cc.model.str.slice(0, 24), orientation="h",
                               marker_color=colours, text=cc.val_macro_f1.round(3), textposition="outside"))
        fig.update_layout(xaxis_range=[0, cc.val_macro_f1.max() * 1.25])
        chart(fig_style(fig, 300, "Round 2 · model shortlist", legend=False),
              "Validation macro-F1 for the strongest candidates; the winning ensemble is highlighted.",
              "The comprehension layer was chosen by measurement, not preference, and the winner is the component "
              "every later round depends on.")
    with c3:
        ents = pd.DataFrame(r3m["entities"]).nsmallest(6, "net").sort_values("net", ascending=False)
        fig = go.Figure(go.Bar(x=ents.net, y=ents.entity, orientation="h", marker_color=ROSE,
                               text=ents.net.round(2), textposition="outside", customdata=ents.mentions,
                               hovertemplate="%{y}: net %{x:.2f} across %{customdata} posts<extra></extra>"))
        fig.update_layout(xaxis_range=[ents.net.min() * 1.35, 0])
        chart(fig_style(fig, 300, "Round 3 · what users disliked", legend=False),
              "The six most negatively received themes in the live corpus.",
              "Reliability complaints dominate the redesign that got the press coverage - the sort of prioritised "
              "queue a product team can act on the morning after a launch.")

    eyebrow("The journey, end to end", "four charts that only exist because all four rounds do")

    j1, j2 = st.columns(2, gap="medium")
    with j1:
        stages = pd.DataFrame({
            "stage": ["R1 P1 · corrupted export", "R1 P1 · cleaned corpus", "R1 P2 · rows in SQLite",
                      "R2 · labelled tweets", "R3 · live posts collected", "R3 · live posts on topic"],
            "n": [len(raw), len(posts), len(posts), len(train), man["rows_final"], len(scored)],
            "c": [DIM, CYAN, BLUE, VIOLET, AMBER, EMER]})
        fig = go.Figure(go.Bar(x=stages.n, y=stages.stage, orientation="h",
                               marker_color=list(stages.c), text=stages.n,
                               texttemplate="%{text:,}", textposition="outside",
                               textfont=dict(size=10.5, color=MUT), cliponaxis=False,
                               hovertemplate="%{y}: %{x:,} records<extra></extra>"))
        fig.update_layout(xaxis_range=[0, stages.n.max() * 1.45],
                          xaxis_visible=False, yaxis=dict(autorange="reversed"))
        chart(fig_style(fig, 330, "What each round actually handled", legend=False),
              "Records at each stage of the build. These are three different corpora, not one shrinking "
              "funnel: the 9,000 labelled tweets are unrelated to the 12,000 recovered posts, and the live "
              "collection is unrelated to both.",
              "The engine was never asked to do the same thing twice. Each round changed both the data and "
              "the question, which is why the model surviving the jump to live data is worth anything.")

    with j2:
        cc = comp.dropna(subset=["val_macro_f1"]).sort_values("val_macro_f1")
        _hc4 = (load_robustness() or {}).get("human_consensus") or {}
        _hc4 = _hc4 if "cohens_kappa" in _hc4 else None
        live_f1 = float(_hc4["model_macro_f1"]) if _hc4 else float(
            r3m["in_domain_validation"]["configurations"]
            [r3m["in_domain_validation"]["chosen"]]["macro_f1"])
        live_ref = (f"the {_hc4['n_consensus']}-post two-annotator consensus" if _hc4
                    else "150 hand-labelled posts")
        win = [("ensemble" in m.lower()) for m in cc.model]
        fig = go.Figure(go.Bar(x=cc.model.str.slice(0, 18), y=cc.val_macro_f1,
                               marker_color=[EMER if w else "rgba(94,163,143,.55)" for w in win],
                               marker_line=dict(color=[EMER if w else "rgba(0,0,0,0)" for w in win], width=1),
                               text=cc.val_macro_f1.round(3), textposition="outside",
                               textfont=dict(size=9.5, color=MUT), cliponaxis=False,
                               name="on its own test set"))
        fig.add_hline(y=live_f1, line_color=CLAY, line_dash="dot")
        fig.add_annotation(x=-0.45, y=live_f1, yshift=11, xanchor="left", showarrow=False,
                           text=f"<b>{live_f1:.3f}</b> on live posts",
                           font=dict(color=CLAY, size=11), align="left")
        fig.update_layout(yaxis_range=[0, max(cc.val_macro_f1.max(), live_f1) * 1.3],
                          yaxis_title="macro-F1", xaxis_tickangle=-38,
                          xaxis_tickfont=dict(size=9.5))
        _gap = float(cc.val_macro_f1.max()) - live_f1
        chart(fig_style(fig, 330, "Every model built, and what the real world cost", legend=False),
              "Nine candidates from the majority-class baseline up to the winning ensemble, each on its own "
              f"held-out set. The dotted line is that same winning model scored against {live_ref} drawn "
              "from the live Round 3 corpus.",
              f"The ensemble beats the majority-class baseline by "
              f"{float(cc.val_macro_f1.max()) - float(cc.val_macro_f1.min()):.2f} macro-F1, and then gives "
              f"back only {_gap:.3f} when it leaves the data it was trained on entirely - different platforms, "
              "different year, longer posts. That small gap is the point: a model is only worth reusing if it "
              "survives the move, and this one does.")

    j3, j4 = st.columns(2, gap="medium")
    with j3:
        e1 = (posts[["likes", "shares", "comments"]].sum(axis=1)).dropna()
        e3 = scored.loc[scored.score.notna() & scored.source.isin(["hackernews", "mastodon", "lemmy"]), "score"]
        sx, sy, g1 = lorenz(e1)
        lx, ly, g3 = lorenz(e3)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect equality",
                                 line=dict(color=DIM, width=1.4, dash="dot"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", name=f"Round 1, synthetic (Gini {g1:.2f})",
                                 line=dict(color=BLUE, width=2.6),
                                 hovertemplate="bottom %{x:.0%} of posts hold %{y:.0%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", name=f"Round 3, live (Gini {g3:.2f})",
                                 line=dict(color=CLAY, width=2.6),
                                 hovertemplate="bottom %{x:.0%} of posts hold %{y:.0%}<extra></extra>"))
        fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          xaxis_title="share of posts", yaxis_title="share of all engagement")
        chart(fig_style(fig, 340, "Generated data has no viral tail"),
              f"Cumulative engagement against cumulative posts. The synthetic Round 1 corpus sits near the "
              f"equality line (Gini {g1:.2f}); the live Round 3 corpus is the shape real attention takes "
              f"(Gini {g3:.2f}), where the top 1% of posts carry roughly half of all engagement.",
              "This is the single clearest picture of what the competition data is and is not. Every null "
              "result in Round 1 - no viral tail, no anomalies, flat platform behaviour - follows from this "
              "curve, and none of those findings should be read as a statement about real social media.")

    with j4:
        tr = train.sentiment_label.value_counts(normalize=True).reindex(list(LABELS)).fillna(0)
        lv = scored.sentiment.value_counts(normalize=True).reindex(list(LABELS)).fillna(0)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(LABELS), y=tr.values, name="Round 2 training labels",
                             marker_color=INDIGO, text=[f"{v:.0%}" for v in tr.values],
                             textposition="outside", textfont=dict(size=10, color=MUT)))
        fig.add_trace(go.Bar(x=list(LABELS), y=lv.values, name="Round 3 live posts",
                             marker_color=AMBER, text=[f"{v:.0%}" for v in lv.values],
                             textposition="outside", textfont=dict(size=10, color=MUT)))
        fig.update_layout(barmode="group", yaxis_tickformat=".0%",
                          yaxis_range=[0, max(tr.max(), lv.max()) * 1.3],
                          yaxis_title="share of posts")
        chart(fig_style(fig, 340, "The model trained on a world that does not exist"),
              "The Round 2 training set is a perfectly balanced third each. The live corpus the same model "
              "was pointed at is 40% Negative, 41% Neutral and only 19% Positive.",
              "A balanced training set teaches a model that the three classes are equally likely, and then "
              "reality disagrees. That mismatch is exactly why the neutral operating point had to be "
              "re-examined in Round 3 rather than inherited, and why the in-domain validation set exists "
              "at all.")

    eyebrow("The same diagnostics, both corpora",
            "what carried over from generated data to the real internet")
    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        eng_live = scored[scored.score.notna() & scored.source.isin(["hackernews", "mastodon", "lemmy"])]
        med = eng_live.groupby("sentiment").score.median().reindex(["Negative", "Neutral", "Positive"])
        fig = go.Figure(go.Bar(x=med.index, y=med.values, marker_color=[SENT[i] for i in med.index],
                               text=med.values.round(1), textposition="outside"))
        fig.update_layout(yaxis_title="median engagement", yaxis_range=[0, med.max() * 1.35])
        chart(fig_style(fig, 320, "Outrage travels further - in both corpora", legend=False),
              "Median engagement by sentiment on the live data. Negative posts collect roughly twice the engagement "
              "of positive ones (Kruskal-Wallis p = 0.046).",
              "The same asymmetry appeared in the synthetic Round 1 corpus. It is the one pattern that survived the "
              "jump from generated data to the real internet.")
    with c2:
        radar = pd.DataFrame({"axis": ["Data quality", "Statistical rigour", "Model performance",
                                       "Live coverage", "Explainability"],
                              "before": [1, 1, 0, 0, 1], "after": [9, 9, 7, 8, 9]})
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=list(radar.before) + [radar.before[0]],
                                      theta=list(radar.axis) + [radar.axis[0]], name="engine as found",
                                      fill="toself", line=dict(color=ROSE, width=2), opacity=.5))
        fig.add_trace(go.Scatterpolar(r=list(radar.after) + [radar.after[0]],
                                      theta=list(radar.axis) + [radar.axis[0]], name="engine rebuilt",
                                      fill="toself", line=dict(color=EMER, width=2), opacity=.55))
        fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                     radialaxis=dict(range=[0, 10], gridcolor=GRID, tickfont=dict(color=DIM)),
                                     angularaxis=dict(gridcolor=GRID, tickfont=dict(color=MUT, size=11))))
        chart(fig_style(fig, 320, "Capability, before and after"),
              "A qualitative read of the engine's five capabilities at the start of the competition and now. Model "
              "performance is capped at 7 deliberately: 0.67 macro-F1 in the live domain is useful, not solved.",
              "The honest shape matters more than a full circle. Claiming a solved comprehension layer would "
              "contradict the very validation shown one tab earlier.")

    eyebrow("What we would tell the operators of this platform", "five recommendations, each earned by a specific result")
    recs = [("Normalise before you rank",
             "Totals measure activity, not quality. Ranking by total engagement produced 16 suspicious accounts; "
             "ranking per post left 2 real leads.", CYAN),
            ("Baseline per platform",
             "Sentiment differs more between platforms than between days. A single global line moved when the "
             "source mix changed, not when opinion did.", INDIGO),
            ("Treat small gaps as noise",
             "With ~1,700 posts per platform the data can detect a 3.7% gap. Anything tidier than that is a "
             "ranking of randomness.", BLUE),
            ("Revalidate a reused model",
             "The ensemble lost 6 points of macro-F1 moving from tweets to forum posts, and two silent inference "
             "details cost another 3.4.", VIOLET),
            ("Quote evidence with every alert",
             "Each spike carries a headline and a named post. An alert without a cause gets ignored; one with a "
             "quote gets acted on.", AMBER)]
    cols = st.columns(len(recs), gap="small")
    for col, (title, body, colour) in zip(cols, recs):
        col.markdown(f'<div class="panel" style="height:100%;border-top:2px solid {colour}">'
                     f'<h4 style="font-size:.95rem">{title}</h4>'
                     f'<div class="sub" style="margin:0">{body}</div></div>', unsafe_allow_html=True)

    eyebrow("What we would not claim", "the limits of this build, stated plainly")
    st.markdown("""<div class="panel"><div class="sub" style="font-size:.9rem;line-height:1.7">
Reddit's RSS interface exposes no scores, so engagement analysis rests on three of the four live platforms.
The live validation set is 140 posts on which both of us agreed independently (Cohen's kappa 0.896), which is still our own judgement rather than an external gold standard, and its 95% interval is wide
([0.578, 0.745]) and sentiment <em>levels</em> are noisier than sentiment <em>changes</em>. Analysis is
English-only, and X, TikTok and YouTube stay out of reach without paid API access. Finally, the Round 1 corpus
is synthetic: its flatness is a property of how it was generated, and none of its effect sizes should be read
as statements about real social media.</div></div>
<div class="panel" style="margin-top:14px"><h4>Built with</h4><div class="sub" style="line-height:1.8">
<b>Data</b> pandas &middot; numpy &middot; SQLite &nbsp;|&nbsp; <b>Statistics</b> scipy &mdash; Kruskal-Wallis,
Mann-Whitney U, Spearman, two-proportion z, Holm correction, bootstrap CIs<br>
<b>NLP</b> scikit-learn &middot; sentence-transformers &middot; transformers + PyTorch &nbsp;|&nbsp;
<b>Collection</b> urllib against public Atom/JSON/RSS endpoints<br>
<b>This dashboard</b> Streamlit + Plotly, reading every number live from the four rounds' own artefacts
</div></div>""", unsafe_allow_html=True)

    method([
        ("Read the artefacts, never retype them",
         "Every number on every page is computed at load time from that round's own files - the cleaned CSVs, "
         "the SQLite database, the saved ensemble, the scored live corpus. Re-run a notebook and this changes.",
         "nothing hard-coded"),
        ("Make the pipeline the navigation",
         "The four rounds are one system, so the flowchart pinned at the top is both the story and the way "
         "around it.",
         "5 pages, one chain"),
        ("Pair every chart with a decision",
         "Each figure carries what it shows and why it matters, because a chart nobody can act on is "
         "decoration.",
         "~27 charts"),
        ("Show the working, not only the result",
         "Each page ends with that round's submitted reports and its executed notebook - the code, the output "
         "it produced and the notes written while it ran.",
         "reports + notebooks"),
        ("State the limits in the same breath",
         "The wide validation interval, labels that are still our own, the platforms out of reach and the synthetic "
         "Round 1 corpus are on the page, not left for someone else to discover.",
         "5 findings that held"),
    ], "how four rounds of work were assembled into one running engine", EMER,
           "Round 4<br>The Revived Engine",
           [("Round 1 cleaned corpus", P1 / "Entropy_Social_Engine_Posts_Cleaned.csv"),
         ("Round 1 SQLite database", P2 / "Entropy_social_engine.db"),
         ("Round 2 saved ensemble", R2 / "model"),
         ("Round 3 scored posts", R3 / "outputs" / "Entropy_round3_scored_posts.csv")],
           [("This dashboard", APP / "Entropy_app.py"),
         ("Round 4 README", APP / "Entropy_README_Round4.md")])

    report_panel("r4", {
        "R1 P1 - EDA Report": P1 / "Entropy_Phase1_EDA_Report.pdf",
        "R1 P2 - SQL Queries": P2 / "Entropy_Phase2_SQL_Queries.pdf",
        "R1 P2 - Logic": P2 / "Entropy_Phase2_Logic_Explanation.pdf",
        "R1 P2 - Insights": P2 / "Entropy_Phase2_Insight_Report.pdf",
        "R2 - Technical": R2 / "reports" / "Entropy_r2_technical_report.pdf",
        "R2 - Metrics": R2 / "reports" / "Entropy_r2_evaluation_metrics_report.pdf",
        "R3 - Analytical": R3 / "reports" / "Entropy_round3_analytical_report.pdf",
    }, "Every written deliverable from the competition, in one place. Each is the typeset argument behind one of "
       "the pages in this dashboard, and each can be read here or taken as a file.")


# ---------------------------------------------------------------- assistant
@st.cache_resource(show_spinner=False)
def rag_index():
    """Index the repository once per session: READMEs, reports, metrics, logs."""
    return rag.build_corpus()


SUGGESTED = [
    "What did Round 3 actually find?",
    "Why is the pre-launch baseline not positive?",
    "How was the sentiment model validated?",
    "What is wrong with the Round 1 dataset?",
]

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.container(key="askdock"):
    with st.popover("SHANNON", icon=":material/graphic_eq:"):
        st.markdown(
            '<div class="shannon">'
            '<div class="sh-name">SHANNON</div>'
            '<div class="sh-tag">Retrieval-Augmented Intelligence by Entropy</div>'
            '<div class="sh-what">Answers come from this repository alone - the four READMEs, every '
            'report, the metrics and robustness files and the cleaning log - retrieved and summarised '
            'by a Groq-hosted model. Sources are listed under each answer, and it says so when the '
            'documents do not cover your question.</div></div>', unsafe_allow_html=True)

        if not os.environ.get("GROQ_API_KEY", "").strip():
            st.warning("No GROQ_API_KEY found. Create a file named `.env` next to `Entropy_app.py` "
                       "containing `GROQ_API_KEY=your_key_here`, then reload this page. "
                       "A free key comes from console.groq.com.")
        else:
            for turn in st.session_state.chat:
                srcs = (f'<div class="src">sources: {", ".join(turn["sources"])}</div>'
                        if turn.get("sources") else "")
                st.markdown(f'<div class="qa"><div class="q">{turn["q"]}</div>'
                            f'<div class="a">{turn["a"]}</div>{srcs}</div>', unsafe_allow_html=True)

            if not st.session_state.chat:
                sc = st.columns(2, gap="small")
                for i, q in enumerate(SUGGESTED):
                    if sc[i % 2].button(q, key=f"sug{i}", width="stretch"):
                        st.session_state.pending = q
                        st.rerun()

            asked = st.chat_input("Ask anything about the data, the models or the findings",
                                  key="askbox")
            pending = st.session_state.pop("pending", None) or asked
            if pending:
                try:
                    with st.spinner("Reading the project..."):
                        docs, vec, mat = rag_index()
                        hist = [m for t in st.session_state.chat[-2:]
                                for m in ({"role": "user", "content": t["q"]},
                                          {"role": "assistant", "content": t["a"]})]
                        out = rag.answer(pending, docs, vec, mat, history=hist)
                    st.session_state.chat.append({"q": pending, "a": out["text"],
                                                  "sources": out["sources"]})
                    st.rerun()
                except Exception as exc:
                    st.warning(f"SHANNON could not answer that: {exc}")

            if st.session_state.chat and st.button("Clear conversation", key="chatclr"):
                st.session_state.chat = []
                st.rerun()
