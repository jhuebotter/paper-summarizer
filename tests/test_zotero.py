"""Tests for summarizer/zotero.py (the local Zotero API is mocked)."""

import io
import json
import logging
import urllib.error
from pathlib import Path
from unittest.mock import patch

from summarizer.zotero import LOCAL_API, ZoteroRecord, lookup_all

GROUP = "groups/5824653"
PARENT = {
    "data": {
        "itemType": "journalArticle",
        "citationKey": "oikonomou2023Hybrid",
        "title": "A Hybrid Reinforcement Learning Approach",
        "creators": [
            {"creatorType": "author", "firstName": "Katerina Maria", "lastName": "Oikonomou"},
            {"creatorType": "author", "name": "Sanaullah"},
            {"creatorType": "editor", "firstName": "Ed", "lastName": "Itor"},
        ],
        "publicationTitle": "IEEE Robotics and Automation Letters",
    },
    "meta": {"parsedDate": "2023-05"},
}


def _fake_api(items: dict, groups=None):
    """urlopen stand-in serving ``items`` ({path: json}); other paths are 404."""
    routes = {"users/0/groups": [{"id": 5824653}] if groups is None else groups, **items}

    def urlopen(url, timeout):
        path = url.removeprefix(f"{LOCAL_API}/")
        if path not in routes:
            raise urllib.error.HTTPError(url, 404, "Not found", {}, None)
        if isinstance(routes[path], Exception):
            raise routes[path]
        return io.BytesIO(json.dumps(routes[path]).encode())

    return patch("summarizer.zotero.urllib.request.urlopen", side_effect=urlopen)


def _attachment(parent="YXWFBPTP"):
    return {"data": {"itemType": "attachment", "parentItem": parent}}


def test_attachment_key_resolves_to_its_parent_in_a_group():
    """The user library 404s; the group has the attachment, whose parent gives the metadata."""
    pdf = Path("LUN2TVN5__Oikonomou et al. - 2023 - A Hybrid.pdf")
    items = {f"{GROUP}/items/LUN2TVN5": _attachment(), f"{GROUP}/items/YXWFBPTP": PARENT}
    with _fake_api(items):
        records = lookup_all([pdf])
    assert records == {
        pdf: ZoteroRecord(
            item=f"{GROUP}/items/YXWFBPTP",
            citation_key="oikonomou2023Hybrid",
            title="A Hybrid Reinforcement Learning Approach",
            authors=["Katerina Maria Oikonomou", "Sanaullah"],  # editors are not authors
            year=2023,
            venue="IEEE Robotics and Automation Letters",
        )
    }


def test_pdfs_without_a_zotero_key_prefix_make_no_requests():
    with _fake_api({}) as urlopen:
        assert lookup_all([Path("paper.pdf"), Path("2411.17006v1.pdf")]) == {}
    urlopen.assert_not_called()


def test_unreachable_zotero_warns_once_and_matches_nothing(caplog):
    pdfs = [Path("AAAAAAAA__a.pdf"), Path("BBBBBBBB__b.pdf")]
    with _fake_api({"users/0/groups": ConnectionRefusedError("refused")}) as urlopen:
        with caplog.at_level(logging.WARNING, logger="summarizer.zotero"):
            assert lookup_all(pdfs) == {}
    assert urlopen.call_count == 1
    assert [r.message for r in caplog.records if "not reachable" in r.message]


def test_standalone_attachment_and_non_attachment_keys_are_not_matched():
    items = {
        f"{GROUP}/items/AAAAAAAA": {"data": {"itemType": "attachment"}},  # no parent
        f"{GROUP}/items/BBBBBBBB": {"data": {"itemType": "note"}},
    }
    with _fake_api(items):
        assert lookup_all([Path("AAAAAAAA__a.pdf"), Path("BBBBBBBB__b.pdf")]) == {}


def test_a_failing_item_lookup_skips_only_that_pdf(caplog):
    ok = Path("LUN2TVN5__ok.pdf")
    items = {
        f"{GROUP}/items/LUN2TVN5": _attachment(),
        f"{GROUP}/items/YXWFBPTP": PARENT,
        "users/0/items/CCCCCCCC": urllib.error.HTTPError("u", 500, "boom", {}, None),
    }
    with _fake_api(items), caplog.at_level(logging.WARNING, logger="summarizer.zotero"):
        records = lookup_all([Path("CCCCCCCC__bad.pdf"), ok])
    assert list(records) == [ok]
    assert any("lookup failed" in r.message for r in caplog.records)


def test_missing_fields_stay_empty_and_venue_falls_back():
    parent = {
        "data": {"itemType": "preprint", "title": "T", "creators": [], "repository": "arXiv"},
        "meta": {"parsedDate": ""},
    }
    items = {f"{GROUP}/items/AAAAAAAA": _attachment("PPPPPPPP"), f"{GROUP}/items/PPPPPPPP": parent}
    with _fake_api(items):
        record = lookup_all([Path("AAAAAAAA__a.pdf")])[Path("AAAAAAAA__a.pdf")]
    assert (record.authors, record.year, record.venue, record.citation_key) == (
        [],
        None,
        "arXiv",
        "",
    )


def test_child_notes_are_not_attachments():
    note = {"data": {"itemType": "note", "parentItem": "YXWFBPTP"}}
    with _fake_api({f"{GROUP}/items/AAAAAAAA": note, f"{GROUP}/items/YXWFBPTP": PARENT}):
        assert lookup_all([Path("AAAAAAAA__a.pdf")]) == {}


def test_conference_venue_prefers_the_proceedings_title():
    parent = {
        "data": {
            "itemType": "conferencePaper",
            "title": "T",
            "creators": [],
            "proceedingsTitle": "Proceedings of ICML",
            "conferenceName": "ICML 2018",
        },
        "meta": {},
    }
    items = {f"{GROUP}/items/AAAAAAAA": _attachment("PPPPPPPP"), f"{GROUP}/items/PPPPPPPP": parent}
    with _fake_api(items):
        assert (
            lookup_all([Path("AAAAAAAA__a.pdf")])[Path("AAAAAAAA__a.pdf")].venue
            == "Proceedings of ICML"
        )


def test_only_zotero_key_prefixes_are_looked_up():
    names = [
        "aaaaaaaa__lower.pdf",
        "AAAAAAAA_single.pdf",
        "AAAAAAA__seven.pdf",
        "AAAAAAAAB__nine.pdf",
    ]
    with _fake_api({}) as urlopen:
        assert lookup_all([Path(n) for n in names]) == {}
    urlopen.assert_not_called()


def test_a_timeout_stops_further_lookups_but_keeps_earlier_matches(caplog):
    """Review finding: a hung Zotero stalled the run for the timeout on every PDF."""
    items = {
        f"{GROUP}/items/LUN2TVN5": _attachment(),
        f"{GROUP}/items/YXWFBPTP": PARENT,
        "users/0/items/BBBBBBBB": TimeoutError("timed out"),
    }
    pdfs = [Path("LUN2TVN5__a.pdf"), Path("BBBBBBBB__b.pdf"), Path("CCCCCCCC__c.pdf")]
    with _fake_api(items) as urlopen, caplog.at_level(logging.WARNING, logger="summarizer.zotero"):
        records = lookup_all(pdfs)
    assert list(records) == [pdfs[0]]
    assert not any("CCCCCCCC" in c.args[0] for c in urlopen.call_args_list)
    assert all(c.kwargs["timeout"] == 3 for c in urlopen.call_args_list)


def test_items_in_the_trash_are_used_with_a_warning(caplog):
    trashed = {**PARENT, "data": {**PARENT["data"], "deleted": True}}
    items = {f"{GROUP}/items/LUN2TVN5": _attachment(), f"{GROUP}/items/YXWFBPTP": trashed}
    with _fake_api(items), caplog.at_level(logging.WARNING, logger="summarizer.zotero"):
        assert lookup_all([Path("LUN2TVN5__a.pdf")])
    assert any("trash" in r.message for r in caplog.records)


def test_local_api_disabled_hints_at_the_setting(caplog):
    forbidden = urllib.error.HTTPError("u", 403, "Forbidden", {}, None)
    with (
        _fake_api({"users/0/groups": forbidden}),
        caplog.at_level(logging.WARNING, "summarizer.zotero"),
    ):
        assert lookup_all([Path("AAAAAAAA__a.pdf")]) == {}
    assert any("Allow other applications" in r.message for r in caplog.records)
