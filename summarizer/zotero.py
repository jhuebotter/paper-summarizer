"""Bibliographic metadata from the local Zotero library.

PDFs collected from Zotero storage are named ``<KEY>__<file>.pdf`` (see
``collect_pdfs.sh``), where ``<KEY>`` is the attachment's item key.  Zotero 7+
serves a read-only API on ``localhost:23119`` while it runs; this module maps
each PDF to its parent item there, so title, authors, year, venue and citation
key can come from the curated library instead of the LLM.

Lookups are best-effort: when Zotero isn't running, the run continues with the
LLM's metadata and a single warning.
"""

import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

LOCAL_API = "http://localhost:23119/api"
_TIMEOUT_S = 3
_KEY_PREFIX = re.compile(r"^([A-Z0-9]{8})__")
_VENUE_FIELDS = (
    "publicationTitle",
    "proceedingsTitle",
    "conferenceName",
    "bookTitle",
    "repository",
    "university",
)


@dataclass(frozen=True)
class ZoteroRecord:
    """Metadata of the Zotero item a PDF is attached to (empty values are unknown)."""

    item: str  # e.g. "groups/5824653/items/YXWFBPTP"
    citation_key: str
    title: str
    authors: list[str]
    year: int | None
    venue: str


def lookup_all(pdfs: list[Path]) -> dict[Path, ZoteroRecord]:
    """Map each PDF whose name starts with a Zotero key to its parent item's metadata."""
    keyed = {pdf: m.group(1) for pdf in pdfs if (m := _KEY_PREFIX.match(pdf.name))}
    if not keyed:
        return {}
    try:
        groups = _get("users/0/groups") or []
    except (OSError, ValueError) as exc:
        logger.warning("Zotero is not reachable (%s); using the LLM's metadata", exc)
        return {}
    libraries = ["users/0"] + [f"groups/{g['id']}" for g in groups]
    records = {}
    for pdf, key in keyed.items():
        try:
            record = _find(key, libraries)
        except (OSError, ValueError) as exc:
            logger.warning("Zotero lookup failed for %s: %s", pdf.name, exc)
            continue
        if record is not None:
            records[pdf] = record
    logger.info("Zotero: matched %d of %d PDFs with a Zotero key", len(records), len(keyed))
    return records


def _find(key: str, libraries: list[str]) -> ZoteroRecord | None:
    for library in libraries:
        entry = _get(f"{library}/items/{key}")
        if entry is None:
            continue
        data = entry.get("data", {})
        if data.get("itemType") != "attachment" or not data.get("parentItem"):
            return None  # a standalone attachment or not an attachment: no bibliographic item
        parent = _get(f"{library}/items/{data['parentItem']}")
        return _record(f"{library}/items/{data['parentItem']}", parent) if parent else None
    return None


def _record(item: str, entry: dict) -> ZoteroRecord:
    data, meta = entry.get("data", {}), entry.get("meta", {})
    authors = [
        c.get("name") or f"{c.get('firstName', '')} {c.get('lastName', '')}".strip()
        for c in data.get("creators", [])
        if c.get("creatorType") == "author"
    ]
    year = str(meta.get("parsedDate", ""))[:4]
    venue = next((data[f] for f in _VENUE_FIELDS if data.get(f)), "")
    return ZoteroRecord(
        item=item,
        citation_key=str(data.get("citationKey") or ""),
        title=str(data.get("title") or ""),
        authors=[a for a in authors if a],
        year=int(year) if year.isdigit() else None,
        venue=str(venue),
    )


def _get(path: str) -> dict | list | None:
    """GET ``LOCAL_API/path`` as JSON; ``None`` when the item doesn't exist (404)."""
    try:
        with urllib.request.urlopen(f"{LOCAL_API}/{path}", timeout=_TIMEOUT_S) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
