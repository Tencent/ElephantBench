"""Deterministic normalization and lightweight factual-slot signals."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

DATE_RE = re.compile(
    r"\b(?:"
    r"(?:18|19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+"
    r"\d{1,2},?\s+(?:18|19|20)\d{2}"
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"[a-z]*\.?\s+(?:18|19|20)\d{2}"
    r")\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?[$£€¥]?\d[\d,]*(?:\.\d+)?%?"
    r"(?:\s*(?:million|billion|thousand|km|kg|miles|years?|days?|months?))?",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_'-]{1,}")

SLOT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("age", re.compile(r"\b(?:age|aged|years old|year-old)\b", re.I)),
    ("birth", re.compile(r"\b(?:born|birth)\b", re.I)),
    ("capacity", re.compile(r"\b(?:capacity|seats|attendance)\b", re.I)),
    ("cause", re.compile(r"\b(?:cause|caused by|due to|resulted from)\b", re.I)),
    ("date", re.compile(r"\b(?:date|dated|occurred|happened|took place)\b", re.I)),
    ("death", re.compile(r"\b(?:died|dead|death|killed|assassinated)\b", re.I)),
    ("duration", re.compile(r"\b(?:duration|lasted|lasting|length of time)\b", re.I)),
    ("founded", re.compile(r"\b(?:founded|established|created|launched)\b", re.I)),
    ("height", re.compile(r"\b(?:height|tall)\b", re.I)),
    ("location", re.compile(r"\b(?:located|location|site|venue|headquartered)\b", re.I)),
    ("membership", re.compile(r"\b(?:members|membership|employees|staff)\b", re.I)),
    (
        "price",
        re.compile(r"\b(?:price|cost|costs|costing|priced|rent|rents|rental|worth|value)\b", re.I),
    ),
    ("rank", re.compile(r"\b(?:rank|ranked|ranking|no\.)\b", re.I)),
    ("record", re.compile(r"\b(?:record|wins|losses|victory|defeat)\b", re.I)),
    ("release", re.compile(r"\b(?:released|release|opens|opened|premiere|debut)\b", re.I)),
    ("score", re.compile(r"\b(?:score|scored|points|goals|yards)\b", re.I)),
    ("selection", re.compile(r"\b(?:selected|drafted|draft|picked|appointed|elected)\b", re.I)),
    ("sentence", re.compile(r"\b(?:sentenced|convicted|prison|jail)\b", re.I)),
    ("size", re.compile(r"\b(?:size|length|width|weight|weighs|area)\b", re.I)),
    ("votes", re.compile(r"\b(?:votes|voted|ballot|election|majority|tally)\b", re.I)),
)

STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "among",
    "because",
    "before",
    "being",
    "between",
    "could",
    "from",
    "have",
    "into",
    "more",
    "most",
    "other",
    "over",
    "said",
    "same",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "through",
    "under",
    "were",
    "which",
    "with",
    "would",
}

GENERIC_SUBJECTS = {
    "associated press",
    "breaking news",
    "read more",
    "reuters",
    "the associated press",
    "text size",
    "view full size",
}

SUBJECT_LABEL_PRIORITY = {
    "person": 11,
    "organization": 10,
    "event": 9,
    "product": 8,
    "work_of_art": 8,
    "law": 7,
    "facility": 6,
    "location": 5,
    "misc": 4,
    "norp": 3,
    "language": 1,
}


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_subject(value: str) -> str:
    value = unicodedata.normalize("NFKC", normalize_space(value))
    value = value.strip(" \t\r\n,;:.()[]{}<>\"'`“”‘’")
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.I)
    return value.casefold()


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def source_domain(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").casefold().strip(".")
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def extract_slots(text: str) -> list[str]:
    return [name for name, pattern in SLOT_PATTERNS if pattern.search(text or "")]


def extract_dates(text: str, limit: int = 12) -> list[str]:
    values: list[str] = []
    for match in DATE_RE.finditer(text or ""):
        value = normalize_space(match.group(0)).casefold()
        if value not in values:
            values.append(value)
        if len(values) >= limit:
            break
    return values


def extract_numbers(text: str, limit: int = 16) -> list[str]:
    values: list[str] = []
    for match in NUMBER_RE.finditer(text or ""):
        value = normalize_space(match.group(0)).casefold().replace(",", "")
        if value and value not in values:
            values.append(value)
        if len(values) >= limit:
            break
    return values


def content_tokens(text: str, limit: int = 64) -> list[str]:
    counts = Counter(
        token
        for match in TOKEN_RE.finditer(text or "")
        if len(token := match.group(0).casefold()) > 2 and token not in STOPWORDS
    )
    ordered = sorted(counts, key=lambda token: (-counts[token], token))
    return ordered[:limit]


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def select_primary_tag(tags: Any) -> dict[str, Any] | None:
    if not isinstance(tags, list):
        return None
    valid = [tag for tag in tags if isinstance(tag, dict) and tag.get("subfield")]
    if not valid:
        return None

    def key(tag: dict[str, Any]) -> tuple[float, str, str, str]:
        return (
            -float(tag.get("confidence") or 0.0),
            str(tag.get("discipline") or "").casefold(),
            str(tag.get("field") or "").casefold(),
            str(tag.get("subfield") or "").casefold(),
        )

    return min(valid, key=key)


def knowledge_point_key(tag: dict[str, Any]) -> str:
    return "\t".join(
        normalize_space(str(tag.get(level) or "")).casefold()
        for level in ("discipline", "field", "subfield")
    )


def _entity_forms(entity: dict[str, Any]) -> list[str]:
    values = [str(entity.get("text") or "")]
    forms = entity.get("surface_forms")
    if isinstance(forms, list):
        values.extend(str(value) for value in forms)
    return [normalize_space(value) for value in values if normalize_space(value)]


def select_subjects(
    evidence: str,
    entities: Any,
    max_subjects: int = 2,
) -> list[dict[str, Any]]:
    if not isinstance(entities, list):
        return []
    evidence_folded = unicodedata.normalize("NFKC", evidence).casefold()
    ranked: dict[str, tuple[tuple[int, int, int, int, str], dict[str, Any]]] = {}
    for raw in entities:
        if not isinstance(raw, dict):
            continue
        forms = _entity_forms(raw)
        matched = [form for form in forms if form.casefold() in evidence_folded]
        if not matched:
            continue
        text = str(raw.get("text") or matched[0])
        normalized = normalize_subject(text)
        if len(normalized) < 3 or normalized in GENERIC_SUBJECTS:
            continue
        label = str(raw.get("label") or "misc").casefold()
        count = int(raw.get("count") or 1)
        first_start = int(raw.get("first_start") or 10**9)
        display = max(matched, key=lambda value: (len(value), value.casefold()))
        score = (
            SUBJECT_LABEL_PRIORITY.get(label, 0),
            count,
            len(display.split()),
            -first_start,
            normalized,
        )
        record = {
            "text": display,
            "norm": normalized,
            "label": label,
            "mention_count": count,
            "first_start": None if first_start == 10**9 else first_start,
        }
        current = ranked.get(normalized)
        if current is None or score > current[0]:
            ranked[normalized] = (score, record)
    ordered = sorted(ranked.values(), key=lambda item: item[0], reverse=True)
    return [record for _, record in ordered[:max_subjects]]


def pair_scores(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    overlap = jaccard(left.get("tokens") or [], right.get("tokens") or [])
    left_dates, right_dates = set(left.get("dates") or []), set(right.get("dates") or [])
    left_numbers, right_numbers = set(left.get("numbers") or []), set(right.get("numbers") or [])
    date_mismatch = bool(left_dates and right_dates and left_dates != right_dates)
    number_mismatch = bool(left_numbers and right_numbers and left_numbers != right_numbers)
    same_domain = bool(
        left.get("source_domain") and left.get("source_domain") == right.get("source_domain")
    )
    confidence = math.sqrt(
        max(0.0, float(left.get("confidence") or 0.0))
        * max(0.0, float(right.get("confidence") or 0.0))
    )
    domain_bonus = 0.5 if not same_domain else -0.75
    same_values = bool(
        (left_dates or left_numbers) and left_dates == right_dates and left_numbers == right_numbers
    )
    support_score = overlap * 6.0 + confidence + domain_bonus + (1.0 if same_values else 0.0)
    conflict_score = (
        overlap * 6.0
        + confidence
        + domain_bonus
        + (2.0 if date_mismatch else 0.0)
        + (1.5 if number_mismatch else 0.0)
    )
    return {
        "token_jaccard": round(overlap, 6),
        "date_mismatch": date_mismatch,
        "number_mismatch": number_mismatch,
        "same_source_domain": same_domain,
        "support_score": round(support_score, 6),
        "conflict_score": round(conflict_score, 6),
    }
