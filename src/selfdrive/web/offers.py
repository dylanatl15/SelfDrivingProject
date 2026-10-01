"""Vendor offers, and which path to a part is actually cheapest.

This module is deliberately **not** part of the public page. It answers the team's
question - if nobody donates the camera, what is the cheapest way to get one? - and the
answer is full of figures that are not ours to publish. `sponsor.py` uses exactly one
thing from here: whether a part has a live offer, which drives a badge that names nobody.

Two rules are worth stating because both are easy to get wrong:

**Rank by landed price, never by percent off.** 20 % off a $334 camera is $267; 10 % off a
$264 camera is $238. The bigger discount is the worse deal. Percent is also the one number
a vendor controls both halves of: quote list high, quote the discount high, change nothing.

**Compare build paths, not parts.** The options under one part are different products, and
some of them change the cost of a *different* part. An OAK-D runs the network on-camera,
so it lets the computer below be the small one; a camera that is $30 cheaper and forces the
bigger computer is not cheaper. `knock_on` carries that, in dollars, and is 0 almost
everywhere.

A gift wins on this ranking with no special case, because a donated part lands at $0.

**The figures are not in `parts.yaml`.** That file is committed to a public repository, so
writing a quote into it would publish the thing the page is careful not to publish, just
round the back. Offers live in `web/sponsor/offers.local.yaml`, which is gitignored, and
are merged onto the parts data at build time. A vendor's discount code, their price and the
name of the person who sent it stay on the machine that builds the page.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import yaml

# A gift and a discount are both offers; a part we simply buy is not one. These mirror the
# contribution names in `sponsor.py`, because an offer that is accepted becomes one.
GIFT_KINDS = frozenset({"donated", "loaned"})

# A standing public code - a student discount, an education price, a sale anyone can use -
# is NOT an offer to us: it is a price that was already there. It still changes what we
# pay, so it ranks like any other price, and it badges nothing, because the badge is for
# something a company decided to do for this team.
PUBLIC_KINDS = frozenset({"public_discount"})
KNOWN_OFFER_KINDS = GIFT_KINDS | PUBLIC_KINDS | {"discounted"}

# Most parts list alternatives; a few list a kit where we need all of them.
ALTERNATIVES = "alternatives"
ALL_NEEDED = "all_needed"
KNOWN_OPTION_MODES = frozenset({ALTERNATIVES, ALL_NEEDED})

MONEY = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")
# Phrases that mean a price covers the whole quantity rather than one unit.
WHOLE_LOT = ("for the set", "for the pair", "total", "for all")

# Where the figures live. Gitignored, and read only if it is there, so a fresh clone
# builds the same page without it.
OVERLAY = pathlib.Path(__file__).resolve().parents[3] / "web" / "sponsor" / "offers.local.yaml"
OVERLAY_KEYS = frozenset({"offers", "leads"})


def parse_money(text: Any) -> tuple[float | None, bool]:
    """First dollar figure in a human price string, and whether it is exact.

    The prices in `parts.yaml` are written for a reader, not a parser: `$329`, `$14.95
    each`, `~$75 for the set`, `$69-81`. Anything hedged, ranged or approximate comes back
    flagged inexact, and the planner prints those with a `~` instead of quietly treating a
    guess as a quote.
    """
    if text is None:
        return None, False
    if isinstance(text, (int, float)):
        return float(text), True
    raw = str(text)
    found = MONEY.search(raw)
    if not found:
        return None, False
    value = float(found.group(1).replace(",", ""))
    rest = MONEY.sub("", raw, count=1).strip().lower()
    # Exact only if nothing but "each" is left over. "~", a range, or any other words mean
    # the figure is a sketch of a price rather than a price.
    exact = "~" not in raw and rest in ("", "each") and len(MONEY.findall(raw)) == 1
    return value, exact


def parse_qty(qty: Any) -> int:
    """How many we need. `qty` is sometimes prose ('3 packs, 1 charger'); take the first
    number and move on, because this only scales a price estimate."""
    if isinstance(qty, int):
        return max(1, qty)
    found = re.search(r"\d+", str(qty or ""))
    return max(1, int(found.group())) if found else 1


def _as_date(value: Any, field: str, where: str) -> date:
    if isinstance(value, date):
        return value
    raise ValueError(f"offer on {where} has {field}={value!r}; expected a YYYY-MM-DD date")


@dataclass(frozen=True)
class Offer:
    """A price we could get on one product, better than its list price.

    Mostly this is what a vendor has told us they would do, before anything is committed.
    A `public_discount` is the exception: a price anyone qualifying can already have. It
    is here so the plan ranks what we would really pay, and `to_us` is false on it so the
    page does not claim a sponsor where there is only a sale.
    """

    by: str
    kind: str
    unit_price: float
    shipping: float
    expires: date
    checked: date
    note: str = ""

    @classmethod
    def parse(cls, raw: dict, where: str) -> Offer:
        by = raw.get("by")
        kind = raw.get("kind")
        if not by:
            raise ValueError(f"offer on {where} has no 'by'; an anonymous offer cannot be chased")
        if kind not in KNOWN_OFFER_KINDS:
            raise ValueError(
                f"offer from {by} on {where} has kind {kind!r}; "
                f"expected one of {sorted(KNOWN_OFFER_KINDS)}"
            )
        price = float(raw.get("unit_price", 0.0))
        if kind in GIFT_KINDS and price != 0.0:
            raise ValueError(
                f"offer from {by} on {where} is {kind} but costs {price}; "
                "a gift lands at 0 - use 'discounted' if we are paying for it"
            )
        if price < 0:
            raise ValueError(f"offer from {by} on {where} has a negative unit_price")
        # Both dates are required. Without `expires` an offer never leaves the page, and
        # the badge keeps claiming a lead that went cold months ago; without `checked` the
        # figure breaks the rule every other price on this page follows.
        return cls(
            by=by,
            kind=kind,
            unit_price=price,
            shipping=float(raw.get("shipping", 0.0)),
            expires=_as_date(raw.get("expires"), "expires", where),
            checked=_as_date(raw.get("checked"), "checked", where),
            note=raw.get("note", ""),
        )

    @property
    def to_us(self) -> bool:
        """Whether somebody chose to give this to this team, as opposed to a standing sale.

        The one thing the badge depends on. Badging a public code would say this project
        has a sponsor when it does not, which is unfair to the companies that do give.
        """
        return self.kind not in PUBLIC_KINDS

    @property
    def kind_text(self) -> str:
        return self.kind.replace("_", " ")

    def live(self, today: date) -> bool:
        return self.expires >= today

    def landed(self, qty: int) -> float:
        """What leaves the bank: per-unit price for every unit, plus one lot of shipping."""
        return self.unit_price * qty + self.shipping


@dataclass
class Path:
    """One way to get one part: a product, optionally with an offer attached to it."""

    option: str
    offer: Offer | None
    cost: float
    exact: bool
    qty: int = 1
    knock_on: float = 0.0
    knock_on_why: str = ""

    @property
    def label(self) -> str:
        if self.offer is None:
            return "list price"
        return f"{self.offer.by} ({self.offer.kind_text})"


def option_offers(opt: dict, where: str, today: date) -> list[Offer]:
    """Live offers on one product, best first. Expired ones are simply gone."""
    parsed = [Offer.parse(o, where) for o in opt.get("offers", [])]
    live = [o for o in parsed if o.live(today)]
    return sorted(live, key=lambda o: (o.landed(1), o.by))


def option_qty(opt: dict, item: dict) -> int:
    """How many of *this product* we need.

    Usually the item's quantity, but a composite line needs per-option counts: "3 packs,
    1 charger" is one item, and pricing the charger three times is how a plan table starts
    lying. An option may override with its own `qty`.
    """
    if "qty" in opt:
        return parse_qty(opt["qty"])
    return parse_qty(item.get("qty", 1))


def option_paths(opt: dict, item: dict, today: date) -> list[Path]:
    """Every way to get one product, cheapest first: its list price, and each live offer.

    The list price stays in the running even when an offer beats it, because the offer may
    expire before we order and then list is what we actually pay.
    """
    qty = option_qty(opt, item)
    name = str(opt.get("name", "?"))
    where = f"{item.get('part', '?')} / {name}"
    knock_on = float(opt.get("knock_on", 0.0) or 0.0)
    why = str(opt.get("knock_on_why", ""))
    paths: list[Path] = []
    listed, exact = parse_money(opt.get("price"))
    if listed is not None:
        per_unit = not any(p in str(opt.get("price", "")).lower() for p in WHOLE_LOT)
        base = listed * qty if per_unit else listed
        paths.append(Path(name, None, base + knock_on, exact, qty, knock_on, why))
    for offer in option_offers(opt, where, today):
        paths.append(Path(name, offer, offer.landed(qty) + knock_on, True, qty, knock_on, why))
    # An inexact figure loses a tie, so a guess never displaces a quote we actually hold.
    return sorted(paths, key=lambda p: (p.cost, not p.exact))


def item_paths(item: dict, today: date) -> list[Path]:
    """Every path to every option of a part, cheapest first."""
    paths = [p for opt in item.get("options", []) for p in option_paths(opt, item, today)]
    return sorted(paths, key=lambda p: (p.cost, not p.exact, p.option))


def options_mode(item: dict) -> str:
    """Whether the options under a part are choices or a shopping list.

    Default `alternatives` - pick one. A few lines are a kit: the radio line is a
    transmitter *and* a kill switch, and taking the cheaper of the two as "the part" would
    report an $8 solution to a $61 problem.
    """
    mode = item.get("options_are", ALTERNATIVES)
    if mode not in KNOWN_OPTION_MODES:
        raise ValueError(
            f"{item.get('part')!r} has options_are={mode!r}; "
            f"expected one of {sorted(KNOWN_OPTION_MODES)}"
        )
    return mode


def item_plan(item: dict, today: date, use_offers: bool = True) -> tuple[list[Path], float, bool]:
    """What this part costs on its best path, and which path(s) that is.

    With `use_offers` off it returns the same thing ignoring every offer, which is the
    baseline the offers are saving us against.
    """
    chosen: list[Path] = []
    for opt in item.get("options", []):
        paths = option_paths(opt, item, today)
        if not use_offers:
            paths = [p for p in paths if p.offer is None]
        if paths:
            chosen.append(paths[0])
    if not chosen:
        return [], 0.0, True
    if options_mode(item) == ALTERNATIVES:
        chosen = [min(chosen, key=lambda p: (p.cost, not p.exact, p.option))]
    return chosen, sum(p.cost for p in chosen), all(p.exact for p in chosen)


def option_has_live_offer(opt: dict, where: str, today: date) -> bool:
    """The one thing the public page is allowed to know, asked of one product.

    Per *option* on purpose. An offer on one camera tells you nothing about the camera
    beside it, and answering this at the part level would badge an option nobody has
    offered anything on, which is both untrue and unfair to the rest of the list.

    A standing public code is excluded. It is a price, not an answer to our ask.
    """
    live = option_offers(opt, f"{where} / {opt.get('name', '?')}", today)
    return any(offer.to_us for offer in live)


def has_live_offer(item: dict, today: date) -> bool:
    """Whether any option of this part has one. Used for counting, not for the badge."""
    where = str(item.get("part", "?"))
    return any(option_has_live_offer(opt, where, today) for opt in item.get("options", []))


def load_overlay(path: pathlib.Path = OVERLAY) -> dict:
    """The gitignored offers file, or an empty overlay if it is not there.

    Absent is the normal state for anyone who clones the repository, so this is not an
    error. The page they build is the page we publish, minus a badge or two.
    """
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} should be a mapping, not {type(loaded).__name__}")
    unknown = sorted(set(loaded) - OVERLAY_KEYS)
    if unknown:
        raise ValueError(f"{path} has unknown top-level key(s) {unknown}; expected offers, leads")
    return loaded


def attach_offers(data: dict, overlay: dict, path: pathlib.Path = OVERLAY) -> dict:
    """Hang the overlay's offers on the matching options of `data`, in place.

    Every key must match something. A part renamed in `parts.yaml` would otherwise take
    its offers with it in silence, and a silently dropped offer is worse than no file at
    all: the plan table would go on recommending a list price we are not paying, and the
    badge would vanish from a line that is live.
    """
    by_part = {
        str(item.get("part", "?")): item
        for group in data.get("groups", [])
        for item in group.get("items", [])
    }
    for part_name, options in (overlay.get("offers") or {}).items():
        item = by_part.get(part_name)
        if item is None:
            raise ValueError(
                f"{path}: no part named {part_name!r} in the parts file "
                f"(have: {', '.join(sorted(by_part))})"
            )
        by_option = {str(o.get("name", "?")): o for o in item.get("options", [])}
        for option_name, offers in (options or {}).items():
            opt = by_option.get(option_name)
            if opt is None:
                raise ValueError(
                    f"{path}: {part_name!r} has no option named {option_name!r} "
                    f"(have: {', '.join(sorted(by_option))})"
                )
            # Validate here rather than at render time, so a typo in a date fails the
            # build instead of quietly un-badging a live offer.
            where = f"{part_name} / {option_name}"
            for raw in offers or []:
                Offer.parse(raw, where)
            opt.setdefault("offers", []).extend(offers or [])
    return data


def load_parts(data_path: pathlib.Path, overlay_path: pathlib.Path = OVERLAY) -> dict:
    """The parts file with the offers merged on: what both the page and the plan read."""
    data = yaml.safe_load(data_path.read_text())
    return attach_offers(data, load_overlay(overlay_path), overlay_path)


def _money(value: float, exact: bool = True) -> str:
    return f"{'' if exact else '~'}${value:,.2f}".replace(".00", "")


def plan_text(data: dict, today: date, runners_up: int = 2, leads: Any = ()) -> str:
    """The team's decision table: cheapest path to each part, and what that totals.

    Internal by construction. It names vendors and prints figures, both of which the page
    refuses to do, so it is printed to a terminal and never written into `dist/`.

    `leads` are vendors who have offered something that is not yet attached to a product -
    "50 % off what you need" with no SKU named. They are listed and never costed. An offer
    we cannot point at an option cannot be ranked against one, and writing down a figure
    the vendor never quoted against a product they never named puts words in their mouth.
    """
    lines = [
        f"Build plan - cheapest path to each part, as of {today}",
        "",
        "Landed cost: unit price x qty, plus shipping, plus knock-on cost to the rest of the",
        "build. A leading ~ means the figure is approximate. Nothing here reaches the page.",
        "",
    ]
    best_total = base_total = 0.0
    best_exact = base_exact = True
    live: list[tuple[date, str, Offer]] = []
    unpriced: list[str] = []

    for group in data["groups"]:
        for item in group["items"]:
            part = str(item.get("part", "?"))
            chosen, cost, exact = item_plan(item, today)
            if not chosen:
                unpriced.append(part)
                continue
            kit = options_mode(item) == ALL_NEEDED
            head = f"{part}  [all of these]" if kit else part
            lines.append(head)
            taken = {(c.option, c.offer) for c in chosen}
            shown = chosen + [
                p for p in item_paths(item, today) if (p.option, p.offer) not in taken
            ][: 0 if kit else runners_up]
            for path in shown:
                mark = "+" if (path.option, path.offer) in taken and kit else (
                    "->" if (path.option, path.offer) in taken else "  "
                )
                qty = f" x{path.qty}" if path.qty > 1 else ""
                extra = (
                    f"  [+{_money(path.knock_on)} {path.knock_on_why}]" if path.knock_on else ""
                )
                lines.append(
                    f"  {mark:<2} {(path.option + qty)[:34]:<34} {path.label[:26]:<26}"
                    f" {_money(path.cost, path.exact):>10}{extra}"
                )
            if kit:
                lines.append(f"  {'=':<2} {'':<34} {'':<26} {_money(cost, exact):>10}")
            best_total += cost
            best_exact &= exact

            _, base_cost, base_ok = item_plan(item, today, use_offers=False)
            base_total += base_cost
            base_exact &= base_ok
            for opt in item.get("options", []):
                for offer in option_offers(opt, f"{part} / {opt.get('name', '?')}", today):
                    live.append((offer.expires, f"{part} / {opt.get('name', '?')}", offer))
            lines.append("")

    saved = _money(base_total - best_total, best_exact and base_exact)
    lines += [
        "Totals",
        f"  Best path, every part    {_money(best_total, best_exact):>12}",
        f"  At list price only       {_money(base_total, base_exact):>12}",
        # Not "offers are saving us": a public student code saves money and is not an
        # offer, and the difference matters when deciding who gets thanked.
        f"  Below list, in total     {saved:>12}",
        "",
    ]
    if unpriced:
        lines += [f"No priced option: {', '.join(unpriced)}", ""]
    if live:
        lines.append("Live prices below list, soonest to expire")
        for expires, where, offer in sorted(live, key=lambda r: (r[0], r[1], r[2].by)):
            days = (expires - today).days
            warn = "  <- expiring" if days <= 14 else ""
            # A public code is ranked but never badged: it is a price, not a gift, and
            # the thank-you tiers are for companies who decided to help this team.
            warn += "" if offer.to_us else "  (public code - badges nothing)"
            lines.append(
                f"  {expires}  ({days:>3}d)  {offer.by[:22]:<22} {offer.kind_text:<16}"
                f" {where[:38]:<38} checked {offer.checked}{warn}"
            )
    else:
        lines.append("Nothing live below list. Every figure above is a list price.")

    if leads:
        lines += ["", "Leads - not priced, not badged, not on the page"]
        for lead in leads:
            who = str(lead.get("by", "?"))
            heard = lead.get("heard", "?")
            contact = f"  {lead['contact']}" if lead.get("contact") else ""
            lines.append(f"  {who[:28]:<28} heard {heard}{contact}")
            if lead.get("said"):
                lines.append(f'      "{lead["said"]}"')
            if lead.get("next"):
                lines.append(f"      next: {lead['next']}")
    return "\n".join(lines)
