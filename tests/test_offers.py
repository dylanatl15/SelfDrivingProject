"""Guards on the internal offer ranking.

This is the table the team decides on, so the ways it can be wrong are the ways we end up
buying the wrong thing:

- Ranking by percent off picks the worse deal, and percent is the one figure a vendor
  controls both halves of.
- A cheaper part that forces a more expensive part elsewhere is not cheaper.
- A line that lists a kit rather than alternatives must be summed, not minimised, or the
  plan reports an $8 answer to a $61 problem.
- An offer with no end date never goes stale, so the badge on the public page keeps
  claiming a lead that went cold months ago.
- An offer whose part was renamed would vanish without a word, and the plan would go on
  recommending a list price we are not paying.
- A standing public code is a price, not a sponsorship. Badging one would claim this
  project has a sponsor it does not have.
"""

from __future__ import annotations

from datetime import date

import pytest
import yaml

from selfdrive.web.offers import (
    GIFT_KINDS,
    KNOWN_OFFER_KINDS,
    PUBLIC_KINDS,
    Offer,
    attach_offers,
    has_live_offer,
    item_paths,
    item_plan,
    load_overlay,
    load_parts,
    option_has_live_offer,
    option_qty,
    options_mode,
    parse_money,
    parse_qty,
    plan_text,
)
from selfdrive.web.sponsor import DATA, KNOWN_CONTRIBUTIONS

TODAY = date(2026, 9, 30)
SOON = date(2026, 12, 1)
GONE = date(2026, 9, 1)
REAL = yaml.safe_load(DATA.read_text())


def _offer(by: str, kind: str = "discounted", price: float = 100.0, **kw) -> dict:
    offer = {"by": by, "kind": kind, "unit_price": price, "expires": SOON, "checked": TODAY}
    offer.update(kw)
    return offer


def _item(*options: dict, **kw) -> dict:
    item = {"part": "Depth camera", "qty": 1, "options": list(options)}
    item.update(kw)
    return item


def test_the_bigger_discount_can_be_the_worse_deal():
    """The whole reason this module exists. 20 % off $334 is $267; 10 % off $264 is $238."""
    item = _item(
        {"name": "RealSense D435i", "price": "$334", "offers": [_offer("Lens Co", price=267.20)]},
        {"name": "Orbbec Gemini 335", "price": "$264", "offers": [_offer("Cam Co", price=237.60)]},
    )
    best = item_paths(item, TODAY)[0]
    assert best.offer is not None
    assert best.offer.by == "Cam Co", "ranked by percent off instead of by money"
    assert best.cost == pytest.approx(237.60)


def test_shipping_is_part_of_the_price():
    item = _item(
        {"name": "A", "price": "$300", "offers": [_offer("Cheap Co", price=250, shipping=60)]},
        {"name": "B", "price": "$300", "offers": [_offer("Fair Co", price=260, shipping=0)]},
    )
    assert item_paths(item, TODAY)[0].offer.by == "Fair Co"


def test_a_part_that_makes_another_part_dearer_is_not_cheaper():
    """The OAK runs the network on-camera, so the computer below can be the small one."""
    item = _item(
        {"name": "OAK-D S2", "price": "$329"},
        {
            "name": "Plain stereo cam",
            "price": "$264",
            "knock_on": 65,
            "knock_on_why": "forces the 8 GB Pi",
        },
    )
    best = item_paths(item, TODAY)[0]
    assert best.option == "OAK-D S2"
    assert item_plan(item, TODAY)[1] == pytest.approx(329)


def test_a_gift_wins_with_no_special_case():
    item = _item({"name": "A", "price": "$300", "offers": [_offer("Giver", "donated", 0)]})
    best = item_paths(item, TODAY)[0]
    assert best.cost == 0 and best.offer.kind == "donated"


def test_an_expired_offer_is_gone_everywhere():
    item = _item({"name": "A", "price": "$300", "offers": [_offer("Lapsed", expires=GONE)]})
    assert not has_live_offer(item, TODAY)
    assert [p.offer for p in item_paths(item, TODAY)] == [None]
    assert item_plan(item, TODAY)[1] == pytest.approx(300)
    # ... and it was a real offer the day before it lapsed.
    assert has_live_offer(item, date(2026, 8, 31))


def test_the_list_price_stays_in_the_running_behind_an_offer():
    """An offer can lapse before we order, and then list is what we actually pay."""
    item = _item({"name": "A", "price": "$300", "offers": [_offer("Co", price=200)]})
    assert [p.cost for p in item_paths(item, TODAY)] == [200, 300]


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"kind": "sponsored"}, "kind"),
        ({"by": ""}, "no 'by'"),
        ({"kind": "donated", "unit_price": 50}, "a gift lands at 0"),
        ({"unit_price": -5}, "negative"),
        ({"expires": None}, "expires"),
        ({"checked": None}, "checked"),
        ({"expires": "soon"}, "expires"),
    ],
)
def test_an_offer_that_cannot_be_chased_or_retired_is_an_error(bad, match):
    raw = _offer("Acme")
    raw.update(bad)
    with pytest.raises(ValueError, match=match):
        Offer.parse(raw, "a part")


def test_a_kit_is_summed_and_alternatives_are_not():
    radio = _item(
        {"name": "Transmitter", "price": "$53"},
        {"name": "Kill switch", "price": "$8"},
        options_are="all_needed",
    )
    assert item_plan(radio, TODAY)[1] == pytest.approx(61)
    choice = dict(radio, options_are="alternatives")
    assert item_plan(choice, TODAY)[1] == pytest.approx(8)


def test_an_option_may_carry_its_own_quantity():
    """Three packs and one charger is one line. Pricing the charger three times is how a
    plan table starts lying."""
    batteries = _item(
        {"name": "Pack", "price": "$40"},
        {"name": "Charger", "price": "$35", "qty": 1},
        qty="3 packs, 1 charger",
        options_are="all_needed",
    )
    assert option_qty(batteries["options"][0], batteries) == 3
    assert option_qty(batteries["options"][1], batteries) == 1
    assert item_plan(batteries, TODAY)[1] == pytest.approx(3 * 40 + 35)


@pytest.mark.parametrize(
    "text, value, exact",
    [
        ("$329", 329.0, True),
        ("$1,234.50", 1234.50, True),
        ("$14.95 each", 14.95, True),
        ("~$75 for the set", 75.0, False),
        ("$69-81", 69.0, False),
        ("free", None, False),
        (None, None, False),
    ],
)
def test_a_hedged_price_is_flagged_rather_than_believed(text, value, exact):
    assert parse_money(text) == (value, exact)


def test_an_approximate_figure_loses_a_tie():
    item = _item(
        {"name": "Guess", "price": "~$100"},
        {"name": "Quote", "price": "$100"},
    )
    assert item_paths(item, TODAY)[0].option == "Quote"


def test_quantity_survives_prose():
    assert parse_qty("3 packs, 1 charger") == 3
    assert parse_qty(4) == 4
    assert parse_qty(None) == 1
    assert parse_qty(0) == 1


def test_an_unknown_options_mode_is_a_hard_error():
    with pytest.raises(ValueError, match="options_are"):
        options_mode(_item(options_are="whichever"))


def test_the_real_parts_file_costs_out_without_guessing_a_kit():
    """Every line prices, and the two composite lines are marked as kits - otherwise the
    plan quietly reports the cheapest component as the cost of the whole line."""
    table = plan_text(REAL, TODAY)
    assert "No priced option:" not in table
    kits = [
        i["part"]
        for g in REAL["groups"]
        for i in g["items"]
        if options_mode(i) == "all_needed"
    ]
    assert sorted(kits) == [
        "Battery packs and a charger",
        "Radio transmitter, receiver and a hard kill switch",
    ]
    for group in REAL["groups"]:
        for item in group["items"]:
            assert item_plan(item, TODAY)[0], f"{item['part']} has no priced path"


def test_the_plan_says_in_its_own_header_that_it_is_internal():
    """Somebody will paste this into an email one day. Make them read it first."""
    assert "Nothing here reaches the page" in plan_text(REAL, TODAY)


# --- the overlay: real figures live outside the public repository -------------------


def _overlay_data():
    """Two parts, two options each, straight out of the same shape as `parts.yaml`."""
    return {
        "groups": [
            {
                "items": [
                    {
                        "part": "Depth camera",
                        "options": [
                            {"name": "Cheap Cam", "price": "$264"},
                            {"name": "Dear Cam", "price": "$329"},
                        ],
                    },
                    {"part": "GPS receiver", "options": [{"name": "A Receiver", "price": "$70"}]},
                ]
            }
        ]
    }


OVERLAY = {
    "offers": {
        "Depth camera": {
            "Dear Cam": [
                {
                    "by": "Quietly Generous Ltd",
                    "kind": "discounted",
                    "unit_price": 249.0,
                    "shipping": 0,
                    "expires": SOON,
                    "checked": TODAY,
                }
            ]
        }
    }
}


def test_the_real_figures_are_not_in_the_public_parts_file():
    """The whole reason the overlay exists. `parts.yaml` is committed and world-readable."""
    assert "offers" not in DATA.read_text().split("groups:")[1]
    for group in REAL["groups"]:
        for item in group["items"]:
            for opt in item.get("options", []):
                assert "offers" not in opt, f"{item['part']} / {opt['name']} carries a quote"


def test_an_overlay_offer_lands_on_the_option_it_names():
    data = attach_offers(_overlay_data(), OVERLAY)
    camera = data["groups"][0]["items"][0]
    cheap, dear = camera["options"]
    assert not cheap.get("offers")
    assert len(dear["offers"]) == 1
    # And it wins: $249 beats the $264 list price of the option beside it.
    chosen, cost, _ = item_plan(camera, TODAY)
    assert chosen[0].option == "Dear Cam"
    assert cost == pytest.approx(249.0)


def test_an_offer_on_a_part_that_no_longer_exists_is_a_hard_error():
    """Silence here is the dangerous failure: we would keep paying list and not know."""
    bad = {"offers": {"Depth camrea": {"Dear Cam": []}}}
    with pytest.raises(ValueError, match="no part named 'Depth camrea'"):
        attach_offers(_overlay_data(), bad)


def test_an_offer_on_an_option_that_no_longer_exists_is_a_hard_error():
    bad = {"offers": {"Depth camera": {"Dearest Cam": []}}}
    with pytest.raises(ValueError, match="no option named 'Dearest Cam'"):
        attach_offers(_overlay_data(), bad)


def test_a_malformed_offer_fails_the_build_rather_than_un_badging_itself():
    """An unparseable date would otherwise read as 'not live' and quietly drop the badge."""
    bad = {
        "offers": {
            "Depth camera": {
                "Dear Cam": [
                    {"by": "Acme", "kind": "discounted", "unit_price": 1, "expires": "soon"}
                ]
            }
        }
    }
    with pytest.raises(ValueError):
        attach_offers(_overlay_data(), bad)


def test_an_unknown_top_level_key_is_a_hard_error(tmp_path):
    path = tmp_path / "offers.local.yaml"
    path.write_text("offer:\n  Depth camera: {}\n")
    with pytest.raises(ValueError, match="unknown top-level key"):
        load_overlay(path)


def test_a_missing_overlay_is_normal_and_changes_nothing(tmp_path):
    """A fresh clone has no overlay. It must still build the page, minus a badge."""
    assert load_overlay(tmp_path / "nothing.yaml") == {}
    plain = load_parts(DATA, tmp_path / "nothing.yaml")
    for group in plain["groups"]:
        for item in group["items"]:
            assert not has_live_offer(item, TODAY)


def test_a_lead_is_listed_and_costed_nowhere():
    """A vendor who named no product cannot be ranked against one."""
    lead = {
        "by": "Vague Components Inc",
        "heard": "2026-09-30",
        "said": "50% off MSRP on what you need",
        "next": "ask which SKUs",
    }
    with_lead = plan_text(_overlay_data(), TODAY, leads=[lead])
    without = plan_text(_overlay_data(), TODAY)
    assert "Vague Components Inc" in with_lead
    assert "50% off MSRP" in with_lead
    assert "ask which SKUs" in with_lead
    # It moves no money: the totals are identical with and without it.
    assert with_lead.split("Leads -")[0].strip() == without.strip()


# --- a sale is not a sponsorship ---------------------------------------------------


def test_a_public_code_lowers_the_price_and_badges_nothing():
    """The whole distinction in one test: it is real money, and nobody chose us.

    Badging a public price would say this project has a sponsor when it does not, which
    is unfair to the companies that do give.
    """
    item = _item(
        {"name": "Dear Cam", "price": "$329", "offers": [_offer("Code Co", kind="public_discount",
                                                                price=249.0)]},
        {"name": "Cheap Cam", "price": "$264"},
    )
    chosen, cost, _ = item_plan(item, TODAY)
    assert chosen[0].option == "Dear Cam"
    assert cost == pytest.approx(249.0)
    assert not has_live_offer(item, TODAY)


def test_a_quote_made_to_us_still_badges_beside_a_public_code():
    """One of each on the same part. The badge follows the one somebody decided to give."""
    item = _item(
        {"name": "Dear Cam", "price": "$329", "offers": [_offer("Code Co", kind="public_discount",
                                                                price=249.0)]},
        {"name": "Cheap Cam", "price": "$264", "offers": [_offer("Second Source Co", price=241.0)]},
    )
    dear, cheap = item["options"]
    assert not option_has_live_offer(dear, "Depth camera", TODAY)
    assert option_has_live_offer(cheap, "Depth camera", TODAY)
    assert has_live_offer(item, TODAY)


def test_the_plan_marks_a_public_code_so_nobody_thanks_a_vendor_for_a_sale():
    data = {"groups": [{"items": [_item(
        {"name": "Dear Cam", "price": "$329",
         "offers": [_offer("Code Co", kind="public_discount", price=249.0)]},
    )]}]}
    table = plan_text(data, TODAY)
    assert "public code - badges nothing" in table
    # And the saving is not called an offer, because it is not one.
    assert "Offers are saving us" not in table


def test_a_public_code_can_never_become_a_sponsor_credit():
    """Buying at a public price is `purchased`, which names nobody.

    The offer kinds and the contribution kinds line up everywhere else - an accepted
    offer becomes the contribution of the same name - and this is the one that must not,
    or a sale would walk itself onto the thank-you list.
    """
    assert PUBLIC_KINDS.isdisjoint(KNOWN_CONTRIBUTIONS)
    assert PUBLIC_KINDS.isdisjoint(GIFT_KINDS)
    assert (KNOWN_OFFER_KINDS - PUBLIC_KINDS) <= KNOWN_CONTRIBUTIONS
