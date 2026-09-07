"""NRC Technical Letter Query System - Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="NRC Technical Letter Query System",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Grounded, citation-verified question answering over NRC Technical Letter Reports."},
)

from ui import page_ask, page_audit, page_library, page_method  # noqa: E402

pages = [
    st.Page(page_ask.render, title="Ask", icon="🔎", url_path="ask", default=True),
    st.Page(page_library.render, title="Document Library", icon="📚", url_path="library"),
    st.Page(page_audit.render, title="Audit Trail", icon="🧾", url_path="audit"),
    st.Page(page_method.render, title="Methodology & Guarantees", icon="🛡️", url_path="methodology"),
]

# Sidebar logo: wide wordmark if available, otherwise the square icon.
for _logo in (ROOT / "assets" / "logo_wide.png", ROOT / "logo.png"):
    if _logo.exists():
        try:
            st.logo(str(_logo), icon_image=str(ROOT / "logo.png"))
        except TypeError:  # older Streamlit without icon_image
            st.logo(str(_logo))
        except Exception:
            pass
        break

st.navigation(pages).run()
