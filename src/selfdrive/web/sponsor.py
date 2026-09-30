"""Generate the public sponsor page from `web/sponsor/parts.yaml`.

    python -m selfdrive.web.sponsor            # -> web/sponsor/dist/

Four rules are enforced here rather than left to whoever edits the page next, because
each one is a way this kind of page does damage:

1. **No money total.** Counting dollars turns a parts list into a fundraiser, and a
   fundraiser invites a different and much worse conversation. The page counts *parts*.
2. **No vendor is named as having declined or ignored us.** `asked`, `declined` and
   `no_reply` all render identically to `needed`, so the team can track cold asks in the
   YAML without publishing a wall of shame. Only `pledged` and `received` name anyone.
3. **`team_note` never renders.** The YAML lives in a public repository, so this is not a
   secret - it is a place for per-vendor status that does not belong in front of sponsors.
4. **Every price carries the date it was checked**, and an unverified figure says so. A
   stale price in front of a sponsor is worse than no price.

`tests/test_sponsor_page.py` pins all four.
"""

from __future__ import annotations

import html
import shutil
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# True states that the page must not distinguish between. A vendor who said no, and a
# vendor nobody has emailed yet, look the same to a reader: the part is still needed.
STILL_NEEDED = frozenset({"needed", "asked", "declined", "no_reply"})
PUBLIC_NAMED = frozenset({"pledged", "received"})
KNOWN_STATES = STILL_NEEDED | PUBLIC_NAMED | {"covered"}

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA = REPO_ROOT / "web" / "sponsor" / "parts.yaml"
DIST = REPO_ROOT / "web" / "sponsor" / "dist"
DEMO_VIDEO = REPO_ROOT / "media" / "demo-loop.mp4"
DEMO_POSTER = REPO_ROOT / "media" / "demo-poster.jpg"
DEMO_REEL = REPO_ROOT / "media" / "demo.mp4"
MEDIA = (DEMO_VIDEO, DEMO_POSTER, DEMO_REEL)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


@dataclass
class Counts:
    needed: int = 0
    pledged: int = 0
    received: int = 0

    def __str__(self) -> str:
        return f"{self.needed} still needed, {self.pledged} pledged, {self.received} received"


def public_state(state: str) -> str:
    """Collapse the team's true state into what a visitor is allowed to see."""
    if state not in KNOWN_STATES:
        raise ValueError(f"unknown state {state!r}; expected one of {sorted(KNOWN_STATES)}")
    return "needed" if state in STILL_NEEDED else state


def _option(opt: dict) -> str:
    name = esc(opt["name"])
    if opt.get("url"):
        name = f'<a href="{esc(opt["url"])}" rel="noopener">{name}</a>'
    price = esc(opt.get("price", ""))
    tag = "" if opt.get("verified", True) else '<span class="unver">price not verified</span>'
    note = f'<p class="onote">{esc(opt["note"])}</p>' if opt.get("note") else ""
    return (
        '<li class="opt">'
        f'<div class="oline"><span class="oname">{name}</span>'
        f'<span class="oprice">{price}</span></div>{tag}{note}</li>'
    )


def _item(item: dict, counts: Counts) -> str:
    state = public_state(item.get("state", "needed"))
    if state == "needed":
        counts.needed += 1
        badge = '<span class="badge need">Still needed</span>'
    elif state == "pledged":
        counts.pledged += 1
        by = esc(item.get("by", "a sponsor"))
        badge = f'<span class="badge pledge">Pledged by {by}</span>'
    else:
        counts.received += 1
        by = esc(item.get("by", "a sponsor"))
        badge = f'<span class="badge got">Received - thank you, {by}</span>'

    qty = esc(item.get("qty", 1))
    qty_html = f'<span class="qty">{qty}</span>' if str(qty) not in ("1", "") else ""
    spec = ""
    if item.get("spec"):
        spec = f'<p class="spec"><strong>Has to:</strong> {esc(item["spec"])}</p>'
    note = f'<p class="inote">{esc(item["note"])}</p>' if item.get("note") else ""
    opts = "".join(_option(o) for o in item.get("options", []))
    return (
        '<article class="item">'
        f'<header><h3>{esc(item["part"])}{qty_html}</h3>{badge}</header>'
        f'<p class="why">{esc(item["why"])}</p>{spec}{note}'
        f'<ul class="opts">{opts}</ul>'
        "</article>"
    )


def render(data: dict) -> tuple[str, Counts]:
    meta = data["meta"]
    counts = Counts()

    groups = []
    for group in data["groups"]:
        items = "".join(_item(i, counts) for i in group["items"])
        blurb = f'<p class="gblurb">{esc(group["blurb"])}</p>' if group.get("blurb") else ""
        groups.append(
            f'<section class="group"><h2>{esc(group["name"])}</h2>{blurb}'
            f'<div class="items">{items}</div></section>'
        )

    status = "".join(f"<li>{esc(s)}</li>" for s in data.get("status", []))
    covered = "".join(
        f'<li><strong>{esc(c["part"])}</strong> - {esc(c["note"])}</li>'
        for c in data.get("covered", [])
    )
    gets = "".join(f"<li>{esc(g)}</li>" for g in data.get("sponsor_gets", []))

    # Parts, never dollars. See rule 1 in the module docstring.
    tally = (
        f'<strong>{counts.needed}</strong> parts still needed'
        + (f' &middot; <strong>{counts.pledged}</strong> pledged' if counts.pledged else "")
        + (f' &middot; <strong>{counts.received}</strong> received' if counts.received else "")
    )

    # Both addresses go in the To: field, not one in To and one in Cc. A vendor replying
    # from their phone hits "reply all" or they do not, and either way the reply has to land
    # somewhere that is read - and it has to show the address they were written from, so the
    # exchange does not look like two different people.
    # Two addresses, both the same person: one university, one professional. A vendor is
    # written to from the first and may reply to either, so both go in To:.
    #
    # Cloudflare's Email Obfuscation is on for this zone and rewrites these hrefs into
    # /cdn-cgi/l/email-protection. That is free anti-scraping and the decoded mailto is
    # byte-identical, so the buttons still open the same draft. Nothing here is meant to be
    # read as text, so no `email_off` directive is needed.
    addresses = meta["contact"]
    if isinstance(addresses, str):
        addresses = [addresses]
    mailto = esc(",".join(addresses))
    # Percent-encoded: a raw space in a mailto query is tolerated by most clients and
    # silently truncates the subject in a few.
    subject = urllib.parse.quote("Sponsoring the self-driving scale car")
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(meta["title"])} &middot; parts we still need</title>
<meta name="description" content="{esc(meta["tagline"])}">
<meta property="og:title" content="{esc(meta["title"])} - parts we still need">
<meta property="og:description" content="{esc(meta["tagline"])}">
<meta property="og:type" content="website">
<style>
  :root {{
    --bg: #000020; --panel: rgba(0,0,0,.40); --line: rgba(255,255,255,.10);
    --line2: rgba(255,255,255,.20); --ink: #fff; --dim: rgba(255,255,255,.62);
    --dimmer: rgba(255,255,255,.42); --cyan: #22d3ee; --violet: #a855f7; --pink: #e879f9;
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    margin: 0; background: var(--bg); color: var(--ink);
    font: 16px/1.65 Geist, ui-sans-serif, system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased; overflow-x: hidden;
  }}
  body::before {{
    content: ""; position: fixed; inset: 0; z-index: -1; pointer-events: none;
    background:
      radial-gradient(900px 560px at 12% -8%, rgba(34,211,238,.16), transparent 60%),
      radial-gradient(760px 520px at 88% 4%, rgba(168,85,247,.16), transparent 62%);
  }}
  .wrap {{ max-width: 1040px; margin: 0 auto; padding: 0 16px; }}
  a {{ color: var(--cyan); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  nav {{ padding: 20px 0 0; font-size: 14px; }}
  nav a {{ color: var(--dim); }}
  header.hero {{ padding: 40px 0 8px; }}
  .kicker {{
    font-size: 12px; letter-spacing: .14em; text-transform: uppercase; color: var(--dimmer);
    margin: 0 0 14px;
  }}
  h1 {{
    font-size: clamp(30px, 6vw, 52px); line-height: 1.08; margin: 0 0 16px;
    letter-spacing: -.02em; font-weight: 650;
  }}
  h1 .grad {{
    background: linear-gradient(100deg, var(--cyan), var(--violet) 55%, var(--pink));
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }}
  .lede {{
    font-size: clamp(17px, 2.4vw, 20px); color: var(--dim); max-width: 64ch; margin: 0 0 24px;
  }}
  .cta {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 8px; }}
  .btn {{
    display: inline-block; padding: 11px 18px; border-radius: 999px; font-size: 14.5px;
    font-weight: 550; border: 1px solid var(--line2); color: var(--ink);
  }}
  .btn:hover {{ text-decoration: none; border-color: rgba(255,255,255,.45); }}
  .btn.primary {{
    background: linear-gradient(100deg, var(--cyan), var(--violet)); border-color: transparent;
    color: #05050f;
  }}
  .tally {{ margin: 22px 0 0; font-size: 15px; color: var(--dim); }}
  .tally strong {{ color: var(--ink); font-variant-numeric: tabular-nums; }}
  figure.demo {{ margin: 34px 0 0; }}
  figure.demo video {{
    width: 100%; height: auto; display: block; border-radius: 14px; border: 1px solid var(--line);
    background: #000;
  }}
  figcaption {{ font-size: 13.5px; color: var(--dimmer); margin-top: 10px; }}
  .panel {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 16px;
    padding: 22px 24px; backdrop-filter: blur(14px);
    box-shadow: inset 0 1px 0 0 rgba(255,255,255,.06), 0 20px 40px -24px rgba(0,0,0,.8);
  }}
  section {{ margin: 48px 0; }}
  h2 {{
    font-size: clamp(21px, 3.2vw, 27px); margin: 0 0 10px; letter-spacing: -.01em; font-weight: 620;
  }}
  .gblurb {{ color: var(--dim); max-width: 72ch; margin: 0 0 20px; }}
  ul.plain {{ margin: 0; padding-left: 20px; color: var(--dim); }}
  ul.plain li {{ margin: 7px 0; }}
  ul.plain strong {{ color: var(--ink); font-weight: 550; }}
  .items {{ display: grid; gap: 16px; align-items: start; }}
  .item {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 16px;
    padding: 20px 22px;
    backdrop-filter: blur(14px);
  }}
  .item:hover {{ border-color: var(--line2); }}
  .item header {{
    display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px 14px; margin-bottom: 10px;
  }}
  .item h3 {{ font-size: 18.5px; margin: 0; font-weight: 600; }}
  .qty {{
    font-size: 12px; color: var(--dimmer); border: 1px solid var(--line); border-radius: 999px;
    padding: 2px 9px; margin-left: 10px; white-space: nowrap; font-weight: 400;
  }}
  .badge {{
    font-size: 11.5px; letter-spacing: .08em; text-transform: uppercase; font-weight: 600;
    border-radius: 999px; padding: 4px 11px; margin-left: auto; white-space: nowrap;
  }}
  .badge.need {{
    color: #fbbf24; background: rgba(251,191,36,.10); border: 1px solid rgba(251,191,36,.34);
  }}
  .badge.pledge {{
    color: var(--cyan); background: rgba(34,211,238,.10); border: 1px solid rgba(34,211,238,.34);
  }}
  .badge.got {{
    color: #34d399; background: rgba(52,211,153,.10); border: 1px solid rgba(52,211,153,.34);
  }}
  .why {{ margin: 0 0 10px; color: var(--dim); }}
  .spec, .inote {{ font-size: 14.5px; color: var(--dimmer); margin: 0 0 10px; }}
  .spec strong {{ color: var(--dim); font-weight: 600; }}
  ul.opts {{ list-style: none; margin: 14px 0 0; padding: 0; border-top: 1px solid var(--line); }}
  .opt {{ padding: 11px 0; border-bottom: 1px solid rgba(255,255,255,.055); }}
  .opt:last-child {{ border-bottom: 0; padding-bottom: 0; }}
  .oline {{ display: flex; gap: 14px; align-items: baseline; justify-content: space-between; }}
  .oname {{ font-weight: 520; }}
  .oprice {{
    font-family: "Geist Mono", ui-monospace, SFMono-Regular, monospace; font-size: 14px;
    color: var(--dim); white-space: nowrap; font-variant-numeric: tabular-nums;
  }}
  .onote {{ margin: 5px 0 0; font-size: 14px; color: var(--dimmer); max-width: 78ch; }}
  .unver {{ font-size: 12px; color: #fbbf24; }}
  footer {{
    margin: 64px 0 40px; padding-top: 22px; border-top: 1px solid var(--line);
    font-size: 13.5px; color: var(--dimmer);
  }}
  footer p {{ margin: 6px 0; }}
  @media (min-width: 760px) {{ .items {{ grid-template-columns: 1fr 1fr; }} }}
  @media (prefers-reduced-motion: reduce) {{ html {{ scroll-behavior: auto; }} }}
</style>
</head>
<body>
<div class="wrap">

<nav><a href="{esc(meta["portfolio"])}">&larr; dylantamayo.dev</a></nav>

<header class="hero">
  <p class="kicker">{esc(meta["course"])} &middot; due {esc(meta["deadline"])}</p>
  <h1>{esc(meta["title"])}<br><span class="grad">parts we still need</span></h1>
  <p class="lede">{esc(meta["tagline"])}</p>
  <div class="cta">
    <a class="btn primary" href="mailto:{mailto}?subject={subject}">Offer a part</a>
    <a class="btn" href="{esc(meta["repo"])}" rel="noopener">Read the code</a>
    <a class="btn" href="#parts">See the list</a>
  </div>
  <p class="tally">{tally}. We are not collecting money - only parts, and only the ones we
     have decided we would actually fit to the car.</p>
  <figure class="demo">
    <video src="demo-loop.mp4" poster="demo-poster.jpg" width="960" height="720"
           autoplay muted loop playsinline preload="metadata"
           aria-label="A trained policy driving a simulated car through a cluttered arena
                       to the goal it was given.">
    </video>
    <figcaption>
      A trained policy driving in our simulator, recorded straight out of the evaluation loop -
      the readout in the corner is live, not a caption. The gold ring is the goal it is aiming
      for, and the trail behind it is painted by speed.
      <a href="demo.mp4">The full two-minute reel</a> runs the same policy through all six of our
      test arenas.
    </figcaption>
  </figure>
</header>

<section id="status">
  <h2>Where the project actually is</h2>
  <div class="panel"><ul class="plain">{status}</ul></div>
</section>

<section id="parts">
  <h2>What we still need</h2>
  <p class="gblurb">
    Most parts below list a few options, because we would much rather have whatever is already
    on your shelf than the exact thing we named first. Every option is one we have checked and
    would genuinely fit to the car - if something was a step backwards from what we already have,
    we left it off rather than let anyone spend money on it. Prices are what the vendor's own
    store said on {esc(meta["prices_checked"])}, and each links to where we read it.
  </p>
</section>

{"".join(groups)}

<section id="covered">
  <h2>Already covered</h2>
  <p class="gblurb">
    We are listing these so you do not spend anything on our behalf that we do not need. They are
    genuinely handled, and we would rather tell you now than have you find out after posting.
  </p>
  <div class="panel"><ul class="plain">{covered}</ul></div>
</section>

<section id="gets">
  <h2>What a sponsor gets</h2>
  <div class="panel"><ul class="plain">{gets}</ul></div>
</section>

<section id="else">
  <h2>Have something close, but not exactly this?</h2>
  <div class="panel">
    <p style="margin:0 0 12px">
      We would still love to hear about it. The list above is what our current design asks for,
      which is not the same as the limit of what we can use - a different sensor, another motor, a
      spare battery, or something we simply did not think to ask for are all worth an email. We
      will read it properly and tell you honestly whether it fits, and if it changes the design we
      will write up how, publicly, in the repository.
    </p>
    <p style="margin:0 0 12px">
      And if the answer is no, that is completely fine. We know parts cost money and shipping
      costs time, and we are grateful for the look either way.
    </p>
    <p style="margin:0">
      <a class="btn primary" href="mailto:{mailto}?subject={subject}">Email us</a>
    </p>
  </div>
</section>

<footer>
  <p>Prices checked against vendor stores on {esc(meta["prices_checked"])}. Figures marked
     <span class="unver">price not verified</span> are estimates nobody has confirmed - ask us
     before relying on one.</p>
  <p>The full engineering bill of materials, with the reasoning behind every choice, is
     <a href="{esc(meta["bom"])}" rel="noopener">in the repository</a>. So is the page you are
     reading - it is generated from one file, so it cannot drift from what we tell you by email.</p>
  <p>Contact buttons direct to <a href="{esc(meta["portfolio"])}">Dylan Tamayo</a>, Software
     &amp; Autonomy Engineer for the project.</p>
</footer>

</div>
</body>
</html>
"""
    return page, counts


def build(data_path: Path = DATA, out_dir: Path = DIST) -> Counts:
    data = yaml.safe_load(data_path.read_text())
    page, counts = render(data)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(page)
    for src in MEDIA:
        if src.exists():
            shutil.copy2(src, out_dir / src.name)
    return counts


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--out", type=Path, default=DIST)
    args = p.parse_args(argv)

    counts = build(args.data, args.out)
    page = args.out / "index.html"
    print(f"wrote {page} ({page.stat().st_size / 1024:.0f} kB) - {counts}")
    for src in MEDIA:
        if not src.exists():
            print(f"warning: {src} missing, the page will link to nothing")


if __name__ == "__main__":
    main()
