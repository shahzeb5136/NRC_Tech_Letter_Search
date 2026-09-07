"""Visual theme: deep-teal corporate palette with high-contrast, status-coded badges."""

from __future__ import annotations

import html

import streamlit as st

TEAL = "#00A3AD"
GREEN = "#8CC63F"
GOLD = "#F2B134"
RED = "#E5484D"
BLUE = "#5AB0FF"
MUTED = "#9DB7B9"

STATUS_COLORS = {
    "verified": GREEN,
    "approximate": GOLD,
    "figure": BLUE,
    "connective": MUTED,
    "withheld": RED,
    "unsupported": RED,
    "exact": GREEN,
    "fuzzy": GOLD,
    "failed": RED,
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
.stApp { background: linear-gradient(160deg, #012a2c 0%, #023a3d 55%, #013236 100%); }
[data-testid="stSidebar"] { background: #01272a; border-right: 1px solid rgba(0,163,173,0.25); }
[data-testid="stSidebar"] * { color: #E6F2F2; }
h1, h2, h3, h4 { color: #FFFFFF !important; letter-spacing: -0.01em; }
p, li, label, .stMarkdown { color: #E6F2F2; }
a { color: #7FDCE2; }
code, pre { font-family: 'JetBrains Mono', ui-monospace, monospace; }

.stTextArea textarea, .stTextInput input {
  background: rgba(0, 24, 26, 0.75) !important; color: #FFFFFF !important;
  border: 1px solid rgba(0,163,173,0.45) !important; border-radius: 10px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus { border-color: #8CC63F !important; box-shadow: 0 0 0 3px rgba(140,198,63,0.18) !important; }
.stButton > button {
  border-radius: 10px; font-weight: 600; border: 1px solid rgba(0,163,173,0.5);
  background: rgba(0,163,173,0.12); color: #FFFFFF;
}
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #00787f 0%, #00A3AD 100%); border: none; }
.stButton > button:hover { border-color: #8CC63F; color: #FFFFFF; }
div[data-testid="stExpander"] { border: 1px solid rgba(0,163,173,0.28); border-radius: 12px; background: rgba(255,255,255,0.025); }
div[data-testid="stExpander"] summary { color: #E6F2F2; }
div[data-testid="stMetric"] { background: rgba(255,255,255,0.04); border: 1px solid rgba(0,163,173,0.25); border-radius: 12px; padding: 10px 14px; }
div[data-testid="stMetric"] label { color: #9DB7B9 !important; }
div[data-testid="stMetricValue"] { color: #FFFFFF; }
hr { border-color: rgba(0,163,173,0.3); }
[data-testid="stDataFrame"] { border-radius: 10px; }

.nrc-hero { display:flex; align-items:center; gap:18px; margin: 4px 0 6px 0; }
.nrc-title { font-size: 1.9rem; font-weight: 700; color:#fff; line-height:1.15; margin:0; }
.nrc-sub { color:#9DB7B9; font-size: 0.98rem; margin: 4px 0 0 0; }
.nrc-badge { display:inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.72rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; color:#04191b; }
.nrc-pill { display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem; border:1px solid rgba(0,163,173,0.45); color:#E6F2F2; background: rgba(0,163,173,0.10); margin-right:6px; }
.nrc-card { background: rgba(255,255,255,0.035); border:1px solid rgba(0,163,173,0.25); border-radius: 14px; padding: 18px 20px; margin: 10px 0; }
.nrc-banner { border-radius: 12px; padding: 12px 16px; margin: 8px 0 14px 0; font-weight: 600; border:1px solid; }
.nrc-claim { border-left: 4px solid; padding: 8px 14px; margin: 8px 0; border-radius: 0 10px 10px 0; background: rgba(255,255,255,0.03); font-size: 1.02rem; line-height: 1.55; color:#F3FAFA; }
.nrc-claim sup { color:#7FDCE2; font-weight:700; margin-left:2px; }
.nrc-claim.connective { color:#9DB7B9; font-style: italic; border-left-style: dashed; }
.nrc-quote { border-left: 3px solid #8CC63F; padding: 8px 12px; margin: 6px 0; background: rgba(140,198,63,0.07); color:#EEF7EE; border-radius: 0 8px 8px 0; font-size: 0.95rem; }
.nrc-quote.fuzzy { border-color: #F2B134; background: rgba(242,177,52,0.08); }
.nrc-quote.failed { border-color: #E5484D; background: rgba(229,72,77,0.08); text-decoration: line-through; color:#d9b8b9; }
.nrc-meta { color:#9DB7B9; font-size: 0.85rem; }
.nrc-withheld { border-left: 4px solid #E5484D; padding: 8px 14px; margin: 8px 0; background: rgba(229,72,77,0.06); border-radius: 0 10px 10px 0; color:#F1D5D6; }
.nrc-kbd { font-family:'JetBrains Mono', monospace; font-size:0.8rem; color:#7FDCE2; }
.nrc-footer { color:#6f8a8c; font-size:0.8rem; margin-top: 30px; text-align:center; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def badge(status: str, label: str | None = None) -> str:
    color = STATUS_COLORS.get(status, MUTED)
    return f'<span class="nrc-badge" style="background:{color}">{esc(label or status)}</span>'


def pill(text: str) -> str:
    return f'<span class="nrc-pill">{esc(text)}</span>'


def banner(text: str, color: str) -> str:
    return f'<div class="nrc-banner" style="border-color:{color}; background:{color}22; color:#FFFFFF">{text}</div>'


def hero(logo_b64: str | None, title: str, subtitle: str) -> str:
    img = f'<img src="data:image/png;base64,{logo_b64}" style="height:56px;border-radius:8px" />' if logo_b64 else ""
    return f'<div class="nrc-hero">{img}<div><p class="nrc-title">{esc(title)}</p><p class="nrc-sub">{esc(subtitle)}</p></div></div>'
