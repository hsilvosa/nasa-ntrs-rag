"""Query routing and metadata filter extraction (fast deterministic regex + optional LLM)."""

from __future__ import annotations

import re

from nasa_ntrs_rag.schemas import QueryFilters

_YEAR_BETWEEN_RE = re.compile(
    r"\b(?:between|from)\s+(19\d{2}|20\d{2})\s+(?:and|to|-)\s+(19\d{2}|20\d{2})\b",
    re.I,
)
_YEAR_SINCE_RE = re.compile(r"\b(?:since|after|from)\s+(19\d{2}|20\d{2})\b", re.I)
_YEAR_BEFORE_RE = re.compile(r"\b(?:before|until|prior to)\s+(19\d{2}|20\d{2})\b", re.I)
_NTRS_ID_RE = re.compile(r"\b((?:19|20)\d{9})\b")

_NASA_CENTERS: dict[str, str] = {
    "langley": "Langley Research Center",
    "ames": "Ames Research Center",
    "glenn": "Glenn Research Center",
    "armstrong": "Armstrong Flight Research Center",
    "dryden": "Dryden Flight Research Center",
    "johnson": "Johnson Space Center",
    "marshall": "Marshall Space Flight Center",
    "kennedy": "Kennedy Space Center",
    "goddard": "Goddard Space Flight Center",
    "jpl": "Jet Propulsion Laboratory",
}


def extract_query_filters(query: str) -> QueryFilters:
    """Extract explicit temporal, NASA center, and document ID filters deterministically."""
    filters = QueryFilters()

    m_between = _YEAR_BETWEEN_RE.search(query)
    if m_between:
        y1, y2 = int(m_between.group(1)), int(m_between.group(2))
        filters.year_min, filters.year_max = min(y1, y2), max(y1, y2)
    else:
        m_since = _YEAR_SINCE_RE.search(query)
        if m_since:
            filters.year_min = int(m_since.group(1))
        m_before = _YEAR_BEFORE_RE.search(query)
        if m_before:
            filters.year_max = int(m_before.group(1))

    doc_ids = _NTRS_ID_RE.findall(query)
    if doc_ids:
        filters.doc_ids = list(dict.fromkeys(doc_ids))

    q_lower = query.lower()
    for alias, canonical in _NASA_CENTERS.items():
        if re.search(rf"\b{re.escape(alias)}\b", q_lower):
            filters.center = canonical
            break

    return filters
