"""Figure images, resolved on demand.

Ingestion writes a PNG per figure so that later steps have something stable to
hash and describe. Those PNGs are large in aggregate (~70 MB for this corpus)
and entirely derivable: the source PDF is already present and every figure
carries the bounding box it was cut from. So they are treated as a cache, not as
part of the index - when the file is missing, the crop is re-rendered from the
PDF at request time. That is what lets a deployment ship the index without the
image directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from nrc_rag.render.page_render import render_region

log = logging.getLogger(__name__)

DEFAULT_DPI = 150


def figure_png(store, figure_id: str, dpi: int = DEFAULT_DPI) -> Optional[bytes]:
    """PNG bytes for a figure: the cached file if present, else cropped from the PDF."""
    fig = store.get_figure(figure_id)
    if fig is None:
        return None

    if fig.image_path:
        p = Path(fig.image_path)
        if p.exists():
            try:
                return p.read_bytes()
            except Exception as exc:  # pragma: no cover
                log.warning("could not read cached figure %s: %s", figure_id, exc)

    doc = store.get_document(fig.doc_id)
    if doc is None or not doc.path or not Path(doc.path).exists():
        log.warning("no source PDF available to render figure %s", figure_id)
        return None
    if not fig.bbox or len(fig.bbox) != 4:
        return None
    try:
        return render_region(doc.path, fig.page_number, fig.bbox, dpi=dpi, margin=10.0)
    except Exception as exc:  # pragma: no cover
        log.warning("could not render figure %s from %s: %s", figure_id, doc.path, exc)
        return None
