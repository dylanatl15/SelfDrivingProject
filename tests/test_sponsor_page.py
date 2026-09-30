"""Guards on the public sponsor page.

This page is the one artifact in the repository that strangers with money read, so the
things that would embarrass the team are pinned here rather than left to review. Three of
them are not style questions:

- A money total turns a parts list into a fundraiser. The page counts parts.
- A vendor who declined or never answered must be indistinguishable from one nobody has
  emailed yet. Publishing "declined" is both rude and a bad negotiating position.
- Notes the team writes for itself must not render, even though the repository is public.
"""

from __future__ import annotations

import re

import pytest
import yaml

from selfdrive.web.sponsor import DATA, KNOWN_STATES, build, public_state, render

REAL = yaml.safe_load(DATA.read_text())


def _minimal(state: str = "needed", **extra) -> dict:
    item = {
        "part": "Widget",
        "qty": 1,
        "why": "Because.",
        "state": state,
        "options": [{"name": "Acme Widget", "price": "$10", "url": "https://example.invalid/w"}],
    }
    item.update(extra)
    return {
        "meta": REAL["meta"],
        "status": ["fine"],
        "groups": [{"name": "Only", "items": [item]}],
        "covered": [],
        "sponsor_gets": [],
    }


def test_real_data_builds_and_every_state_is_known():
    page, counts = render(REAL)
    assert "<html" in page
    assert counts.needed + counts.pledged + counts.received == sum(
        len(g["items"]) for g in REAL["groups"]
    )
    for group in REAL["groups"]:
        for item in group["items"]:
            assert item.get("state", "needed") in KNOWN_STATES, item["part"]


def test_page_counts_parts_and_never_totals_money():
    """No dollar figure on the page may exceed the priciest single option, which a sum
    of the lines necessarily would."""
    page, counts = render(REAL)
    assert f"{counts.needed}</strong> parts still needed" in page

    def amounts(text: str) -> list[float]:
        return [float(m.replace(",", "")) for m in re.findall(r"\$([\d,]+(?:\.\d+)?)", text)]

    on_page = amounts(page)
    priciest = max(
        amount
        for group in REAL["groups"]
        for item in group["items"]
        for opt in item.get("options", [])
        for amount in amounts(str(opt.get("price", "")))
    )
    assert on_page, "the page lost its prices"
    assert max(on_page) <= priciest, f"a figure larger than any single part: {max(on_page)}"


@pytest.mark.parametrize("hidden", ["asked", "declined", "no_reply"])
def test_cold_asks_are_indistinguishable_from_untouched_lines(hidden):
    needed, _ = render(_minimal("needed"))
    quiet, _ = render(_minimal(hidden, by="Acme Corporation"))
    assert quiet == needed, f"state {hidden!r} renders differently from 'needed'"
    assert "Acme Corporation" not in quiet
    assert "Still needed" in quiet


def test_pledged_and_received_name_the_sponsor():
    pledged, counts = render(_minimal("pledged", by="Acme Corporation"))
    assert "Pledged by Acme Corporation" in pledged
    assert counts.pledged == 1 and counts.needed == 0

    got, counts = render(_minimal("received", by="Acme Corporation"))
    assert "thank you, Acme Corporation" in got
    assert counts.received == 1 and counts.needed == 0


def test_team_notes_do_not_render():
    page, _ = render(_minimal("asked", team_note="emailed 2026-09-29, chased twice, nothing"))
    assert "chased twice" not in page


def test_unverified_prices_are_marked_and_the_check_date_is_shown():
    data = _minimal()
    data["groups"][0]["items"][0]["options"] = [
        {"name": "Guess", "price": "~$20", "verified": False},
    ]
    page, _ = render(data)
    assert "price not verified" in page
    assert str(REAL["meta"]["prices_checked"]) in page


def test_unknown_state_is_a_hard_error():
    with pytest.raises(ValueError, match="unknown state"):
        public_state("maybe")
    with pytest.raises(ValueError):
        render(_minimal("probably"))


def test_data_is_escaped_not_injected():
    data = _minimal(why='Sneaky <script>alert("x")</script> & co')
    page, _ = render(data)
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page


def test_build_writes_a_self_contained_directory(tmp_path):
    out = tmp_path / "dist"
    counts = build(DATA, out)
    assert (out / "index.html").read_text().startswith("<!DOCTYPE html>")
    assert counts.needed > 0
    # One page and its clips; no stylesheet or script to go missing behind the tunnel.
    assert {p.name for p in out.iterdir()} <= {
        "index.html",
        "demo-loop.mp4",
        "demo-poster.jpg",
        "demo.mp4",
    }
    for name in ("demo-loop.mp4", "demo-poster.jpg"):
        assert f'"{name}"' in (out / "index.html").read_text(), f"{name} copied but unreferenced"


def test_covered_lines_are_never_asks():
    page, counts = render(REAL)
    for entry in REAL["covered"]:
        assert entry["part"] in page
    covered_section = page.split('id="covered"')[1].split("</section>")[0]
    assert "Still needed" not in covered_section
    assert counts.needed == sum(
        1
        for g in REAL["groups"]
        for i in g["items"]
        if public_state(i.get("state", "needed")) == "needed"
    )


def test_every_contact_address_is_a_recipient_of_every_button():
    """Both addresses go in To:, on every button.

    One address is where outreach is sent from and the other is where replies are read. A
    button that drops either one either loses the reply or makes a vendor wonder why a
    stranger is answering the email they got - so this is pinned, not reviewed.
    """
    page, _ = render(REAL)
    addresses = REAL["meta"]["contact"]
    assert len(addresses) >= 2
    links = re.findall(r'href="mailto:([^"?]+)', page)
    assert links, "the page has no mail button at all"
    for link in links:
        assert sorted(link.split(",")) == sorted(addresses)


def test_the_subject_survives_a_space_intolerant_mail_client():
    page, _ = render(REAL)
    subjects = re.findall(r"\?subject=([^\"]+)", page)
    assert subjects
    for subject in subjects:
        assert " " not in subject
