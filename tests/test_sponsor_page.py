"""Guards on the public sponsor page.

This page is the one artifact in the repository that strangers with money read, so the
things that would embarrass the team are pinned here rather than left to review. Three of
them are not style questions:

- A money total turns a parts list into a fundraiser. The page counts parts.
- A vendor who declined or never answered must be indistinguishable from one nobody has
  emailed yet. Publishing "declined" is both rude and a bad negotiating position.
- Notes the team writes for itself must not render, even though the repository is public.
- An offer is published as a fact, never as a figure. No name, no amount, no kind; the
  badge is derived from a live offer, so it cannot be set by hand or outlive the offer.
- A discount must not buy what a donation buys. The logo tier is for a part given or lent;
  a discount earns a named thank-you and nothing more. Tier belongs to the vendor, best
  contribution winning, so a donor who also discounts something stays a donor.
"""

from __future__ import annotations

import re
import subprocess
from datetime import date

import pytest
import yaml

from selfdrive.web.offers import load_overlay, load_parts
from selfdrive.web.sponsor import (
    DATA,
    KNOWN_CONTRIBUTIONS,
    KNOWN_STATES,
    LOGO_TIER,
    REPO_ROOT,
    build,
    public_state,
    render,
)

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
        "sponsor_gets": {"donated": [], "discounted": []},
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
    pledged, counts = render(
        _minimal("pledged", by="Acme Corporation", contribution="donated")
    )
    assert "Donation pledged by Acme Corporation" in pledged
    assert counts.pledged == 1 and counts.needed == 0

    got, counts = render(
        _minimal("received", by="Acme Corporation", contribution="donated")
    )
    assert "Donated by Acme Corporation - thank you" in got
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


def _two_vendors(*items: dict) -> dict:
    base = _minimal()
    base["groups"] = [{"name": "Only", "items": list(items)}]
    return base


def _part(name: str, **extra) -> dict:
    item = {
        "part": name,
        "why": "Because.",
        "options": [{"name": "Acme Widget", "price": "$10"}],
    }
    item.update(extra)
    return item


@pytest.mark.parametrize("contribution", sorted(KNOWN_CONTRIBUTIONS))
def test_every_contribution_renders_a_badge_that_says_what_happened(contribution):
    for state in ("pledged", "received"):
        page, _ = render(_minimal(state, by="Acme Corporation", contribution=contribution))
        assert '<span class="badge' in page
        if contribution == "purchased":
            assert "Acme Corporation" not in page, "a part we paid full price for names nobody"
        else:
            assert "Acme Corporation" in page


def test_a_discount_never_earns_what_a_donation_earns():
    """The whole point of the tier split: a discounted part must not put the vendor in the
    donated-or-loaned line, which is the line that carries the logo."""
    page, counts = render(
        _two_vendors(
            _part("Camera", state="received", by="Generous Co", contribution="donated"),
            _part("Motor", state="received", by="Cheaper Co", contribution="discounted"),
        )
    )
    assert counts.tier(LOGO_TIER) == ["Generous Co"]
    assert counts.tier(("discounted",)) == ["Cheaper Co"]
    donated_line = page.split("Parts donated or loaned by")[1].split("</p>")[0]
    assert "Generous Co" in donated_line
    assert "Cheaper Co" not in donated_line
    assert "Cheaper Co" in page.split("With thanks also to")[1].split("</p>")[0]


def test_a_loan_sits_in_the_logo_tier_but_is_described_as_a_loan():
    page, counts = render(_minimal("received", by="Lender Co", contribution="loaned"))
    assert counts.tier(LOGO_TIER) == ["Lender Co"]
    assert "On loan from Lender Co" in page
    assert "Donated by Lender Co" not in page, "a loan must not read as a gift"


def test_a_vendor_who_donates_and_discounts_is_a_donor_outright():
    _, counts = render(
        _two_vendors(
            _part("Camera", state="received", by="Mixed Co", contribution="discounted"),
            _part("Motor", state="pledged", by="Mixed Co", contribution="donated"),
        )
    )
    assert counts.tier(LOGO_TIER) == ["Mixed Co"]
    assert counts.tier(("discounted",)) == []


def test_the_credits_block_is_absent_while_nothing_has_been_given():
    page, counts = render(REAL)
    assert counts.vendors == {}
    assert "Who has helped so far" not in page, "no empty trophy cabinet"


def test_a_named_part_must_say_how_it_came_to_us():
    """Without this, a forgotten field defaults to the most generous reading and credits a
    discount as a donation."""
    with pytest.raises(ValueError, match="no contribution"):
        render(_minimal("pledged", by="Acme Corporation"))
    with pytest.raises(ValueError, match="unknown contribution"):
        render(_minimal("pledged", by="Acme Corporation", contribution="sponsored"))
    with pytest.raises(ValueError, match="still needed but has contribution"):
        render(_minimal("needed", contribution="donated"))


def test_both_tiers_are_published_on_the_page():
    page, _ = render(REAL)
    assert "If you donate or lend a part" in page
    assert "If you offer a discount" in page
    for line in REAL["sponsor_gets"]["discounted"]:
        assert line.split(".")[0][:40] in page.replace("&#x27;", "'")


OFFER = {
    "by": "Quietly Generous Ltd",
    "kind": "discounted",
    "unit_price": 199.0,
    "shipping": 12.5,
    "expires": date(2026, 12, 1),
    "checked": date(2026, 9, 30),
    "note": "40 % off list for a student team",
}
TODAY = date(2026, 9, 30)


def _with_offer(offer: dict | None = OFFER) -> dict:
    data = _minimal()
    data["groups"][0]["items"][0]["options"][0]["offers"] = [offer or OFFER]
    return data


def test_an_offer_is_published_as_a_fact_and_never_as_a_figure():
    """The one thing the page may say is that an offer exists. Who made it, how much and
    what kind are all theirs to share rather than ours."""
    page, counts = render(_with_offer(), today=TODAY)
    assert "Offer received - still open" in page
    assert counts.offered == 1
    for secret in ("Quietly Generous", "199", "12.5", "discounted", "40 %", "2026-12-01"):
        assert secret not in page, f"the page leaked {secret!r}"


def test_the_badge_says_the_line_is_still_open():
    """Without those two words the badge reads as 'slot taken' and costs us the second
    offer, which is the only reason to show anything."""
    page, counts = render(_with_offer(), today=TODAY)
    row = page.split('<li class="opt">')[1]
    badge = row.split('class="badge offer">')[1].split("</span>")[0]
    assert "still open" in badge.lower()
    # Still an ask, still counted as one.
    assert counts.needed == 1
    assert "<strong>1</strong> parts still needed" in page


def test_an_expired_offer_leaves_no_trace_on_the_page():
    """A badge that outlives its offer is the page lying on our behalf."""
    lapsed = dict(OFFER, expires=date(2026, 9, 29))
    page, counts = render(_with_offer(lapsed), today=TODAY)
    assert "Offer received" not in page
    assert "Still needed" in page
    assert counts.offered == 0
    # It was true the day before.
    assert render(_with_offer(lapsed), today=date(2026, 9, 29))[1].offered == 1


def test_the_badge_is_explained_only_on_a_page_that_shows_one():
    """An explanation of a badge the reader cannot see is a hint about what we are not
    telling them."""
    with_offer, _ = render(_with_offer(), today=TODAY)
    without, _ = render(_minimal(), today=TODAY)
    assert "not accepted anything yet" in with_offer
    assert "nothing has been accepted" not in without


def test_the_offer_badge_is_derived_so_it_cannot_be_set_by_hand():
    """`offered` is not a state. If it were, somebody would set it and forget it."""
    assert "offered" not in KNOWN_STATES
    with pytest.raises(ValueError, match="unknown state"):
        public_state("offered")


def test_internal_offer_fields_never_reach_the_page_from_the_real_data():
    """Run against the merged data, overlay included, so this guards the real quotes.

    A vendor's own name is the one thing not asserted here: several of these products are
    named after the company that makes them, so "Luxonis" is on the page either way. That
    is the documented cost of the badge. Every field that is actually a secret - what we
    would pay, what they charge to ship, when it lapses, and the note that carries the
    discount code and the contact who sent it - must be absent.
    """
    merged = load_parts(DATA)
    page, _ = render(merged, today=date.today())
    for group in merged["groups"]:
        for item in group["items"]:
            for opt in item.get("options", []):
                for offer in opt.get("offers", []):
                    where = f"{item['part']} / {opt['name']}"
                    assert str(offer["unit_price"]) not in page, where
                    assert str(offer["expires"]) not in page, where
                    assert str(offer["checked"]) not in page, where
                    assert offer["kind"] not in page, where
                    for word in str(offer.get("note", "")).split():
                        if len(word) > 12:  # a code, an email, a long surname
                            assert word not in page, f"{where}: {word}"


def test_the_overlay_of_real_quotes_is_not_committed():
    """The reason the overlay exists at all: this repository is public."""
    ignored = (REPO_ROOT / ".gitignore").read_text()
    assert "web/sponsor/offers.local.yaml" in ignored
    tracked = subprocess.run(
        ["git", "ls-files", "web/sponsor/offers.local.yaml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert tracked.stdout.strip() == "", "the real quotes are committed to a public repo"


def _vendors(overlay: dict) -> set[str]:
    named = {o["by"] for opts in (overlay.get("offers") or {}).values()
             for offers in opts.values() for o in offers}
    return named | {lead["by"] for lead in overlay.get("leads") or []}


def test_the_example_overlay_is_committed_and_carries_no_real_quote():
    """It documents the shape. If somebody puts a real figure in it, it gets published.

    Compared vendor by vendor rather than by searching the text, because the example
    points at a real option name and several of those are a manufacturer's own.
    """
    example = REPO_ROOT / "web" / "sponsor" / "offers.local.example.yaml"
    text = example.read_text()
    assert "gitignored" in text
    assert not _vendors(yaml.safe_load(text)) & _vendors(load_overlay())


def _two_options(offer_on: int) -> dict:
    data = _minimal()
    item = data["groups"][0]["items"][0]
    item["options"] = [
        {"name": "Luxonis OAK-D S2", "price": "$329"},
        {"name": "Orbbec Gemini 335", "price": "$264"},
    ]
    item["options"][offer_on]["offers"] = [OFFER]
    return data


def test_the_badge_sits_on_the_option_offered_and_not_on_its_siblings():
    """The point of putting it on the option: an offer on one camera says nothing about
    the camera listed beside it, which is still wide open."""
    page, _ = render(_two_options(offer_on=1), today=TODAY)
    rows = page.split('<li class="opt">')[1:]
    assert len(rows) == 2
    oak, orbbec = rows
    assert "OAK-D S2" in oak and "Gemini 335" in orbbec
    assert "Offer received" not in oak, "an untouched option was badged"
    assert "Offer received" in orbbec
    assert page.count("badge offer") == 2, "one on the option, one in the explainer"


def test_a_part_is_still_published_as_needed_while_an_option_has_an_offer():
    """Because it is. Nothing has been accepted, and the headline must not shrink."""
    page, counts = render(_two_options(offer_on=0), today=TODAY)
    header = page.split('<article class="item">')[1].split("</header>")[0]
    assert "Still needed" in header
    assert "Offer received" not in header
    assert counts.needed == 1 and counts.offered == 1
    assert "<strong>1</strong> parts still needed" in page


def test_the_explainer_says_the_other_options_are_untouched():
    page, _ = render(_two_options(offer_on=1), today=TODAY)
    note = page.split('class="gblurb offnote"')[1].split("</p>")[0]
    assert "every other option on the same part is wide open" in note
    assert "a gift always outranks a discount" in note
