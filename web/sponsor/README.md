# The public sponsor page

A single static page that lists the parts this project still needs, hosted at
**sponsor.dylantamayo.dev** alongside the rest of the portfolio stack.

It exists because a private spreadsheet cannot do two things this page does: a vendor who
was not asked, or who answered late, can still see what is open and offer something else;
and anyone who wonders whether the ask is honest can read the list, the prices, the dates,
and the code behind them in one sitting.

## How it is put together

```
web/sponsor/parts.yaml        the only thing you edit - committed, so public
web/sponsor/offers.local.yaml what vendors have quoted - GITIGNORED, never committed
src/selfdrive/web/sponsor.py  generator - renders the page, enforces the publishing rules
src/selfdrive/web/offers.py   ranks offers by landed price; merges the overlay above
web/sponsor/nginx.conf        serving config (behind the Cloudflare tunnel)
web/sponsor/Dockerfile        multi-stage: generates the page, then serves it
tests/test_sponsor_page.py    pins the publishing rules
tests/test_offers.py          pins the ranking and the overlay
```

The page is generated at **image build time** from `parts.yaml` and the clips in `media/`,
so it cannot be stale relative to its data. The hero is `media/demo-loop.mp4` played inline;
`media/demo.mp4` is linked from the caption as the full reel. Both are copied into `dist/`
by `build()`, and `.dockerignore` has to allow each one through or the page ships a broken
player. `web/sponsor/dist/` is generated output and is
gitignored.

Preview locally without Docker:

```bash
python -m selfdrive.web.sponsor && xdg-open web/sponsor/dist/index.html
```

## The five publishing rules

These are enforced in the generator and pinned by tests, not left to whoever edits next.

1. **The page counts parts, never money.** This is a list of parts we need, not a
   fundraiser, and a dollar total would reframe it as one.
2. **No vendor is ever published as having declined or ignored us.** `state` in the YAML can
   be `needed`, `asked`, `declined` or `no_reply`, and all four render identically as
   "Still needed". Only `pledged` and `received` name anyone, and only then with `by:`.
3. **`team_note:` never renders.** It is a place for per-vendor status that does not
   belong in front of sponsors. Keep anything a vendor should not read out of this
   repository entirely - the repository is public, and the page invites them to read it.
4. **Every price carries the date it was checked**, and `verified: false` prints "price not
   verified" next to a figure nobody has confirmed.
5. **An offer is published as a fact, never as a figure**, and on the option rather than
   the part. The badge reads "Offer received - still open" and nothing else - no name, no
   amount, no kind - and the part above it still reads "Still needed". What a company
   offered us is theirs to share rather than ours, and some of it is sent under terms that
   forbid passing it on. A standing public code is not an offer and does not badge: a
   price anyone can get is not sponsorship.

How we word things, and our working notes on particular vendors, are in
`web/sponsor/playbook.local.md`, which is gitignored for the reason in rule 3.

## Offers, and which path is actually cheapest

**Quotes do not go in `parts.yaml`.** That file is committed, and this repository is
public, so a figure written there is published on github.com - the thing the page is
careful not to do, just round the back. They go in `web/sponsor/offers.local.yaml`, which
is gitignored and merged onto the options at build time by
[`offers.py`](../../src/selfdrive/web/offers.py).

```yaml
# web/sponsor/offers.local.yaml - part name, then option name, both exactly as written
# in parts.yaml. A key that matches nothing is a hard error: a renamed part would
# otherwise take its offers with it in silence.
offers:
  Depth camera:
    Orbbec Gemini 335:
      - by: Orbbec
        kind: discounted      # donated | loaned | discounted; a gift is unit_price 0
        unit_price: 237.60    # per unit, what we would actually pay
        shipping: 0           # per order
        expires: 2026-12-01   # required
        checked: 2026-09-30   # required
```

Copy `offers.local.example.yaml` to start. A clone without the overlay builds the same
page minus a badge, which is the point: nobody needs the figures to work on this.

**A standing public code is not an offer.** A student discount, an education price or a
sale anyone can use gets `kind: public_discount`. It still changes what we pay, so it
ranks in the plan like any other price, and it badges nothing, because a price anyone can
get is not sponsorship and listing it as such would be unfair to the companies that gave
us something. Buying at a public price is `purchased`, which names nobody, so a public
code cannot become a sponsor credit by any route; `tests/test_offers.py` pins that the
offer kinds and the contribution kinds line up everywhere except there.

A `leads:` list in the same file holds vendors who have offered something with no product
named - "50 % off MSRP on what you need". `--plan` lists them and costs them nowhere. An
offer that is not attached to an option cannot be ranked against one, and writing down a
figure against a product nobody named would be putting words in their mouth. Ask which
products it covers first, then it becomes an offer.

Both dates are required and the generator raises without them. `expires` is the one that
matters on the public side: the badge disappears on the first build after it passes, so a
lead that went cold in October is not still advertised in December.

**The badge goes on the option, not on the part.** An offer on the Orbbec says nothing
about the OAK-D listed beside it, so every other option on that part stays plainly open.

**Rank by landed price, never by percent off.** 20 % off a $334 camera is $267; 10 % off a
$264 camera is $238, so the bigger discount is the worse deal. Landed price is also the
only figure that is comparable across sellers, since each sets its own list price.
`tests/test_offers.py` pins that exact case.

```bash
python -m selfdrive.web.sponsor --plan
```

prints the team's decision table: the cheapest path to each part, what the whole build
costs on those paths, what it costs with no offers at all, every live offer sorted by
expiry, and the unpinned leads. It names vendors and prints money, so it goes to a
terminal and never into `dist/`.

Two fields exist only for that table. `knock_on:` on an option is the dollars a cheaper
choice adds *elsewhere* - a camera that cannot run the network on board forces the bigger
computer, and a part $30 cheaper that costs $65 downstream is not cheaper.
`options_are: all_needed` on an item marks a line that is a kit rather than a choice, so
its options are summed instead of minimised; an option may then carry its own `qty:`. The
radio line is a transmitter *and* a kill switch, and without the flag the plan reported an
$8 answer to a $61 problem.

**Why no figure on the page.** What a company offered us is theirs to share rather than
ours, and quotes of this kind routinely carry a line asking that they not be passed on.
The badge still does the useful half of the job: it says the part is spoken for by nobody
yet, and that a gift always outranks a discount. The fuller reasoning is in the playbook,
which is not committed.

## Updating it when a sponsor answers

```yaml
- part: Depth camera
  state: received          # was: needed
  by: Luxonis              # shown publicly, with thanks
  contribution: donated    # donated | loaned | discounted | purchased
```

When an offer is accepted it stops being an offer: set `state` and `contribution` on the
item, and delete or let the `offers:` entry lapse.

`contribution` is required the moment `state` becomes `pledged` or `received`, and the
generator raises if it is missing. That is deliberate: without it a forgotten field would
fall back to the most generous reading and credit a discount as a donation.

**A discount does not buy a logo.** `donated` and `loaned` earn the logo tier - page, repo
README, poster, report, demo video. `discounted` earns a named thank-you and nothing else.
`purchased` means we paid the ordinary price, which is not sponsorship, so it names nobody.
The page publishes both tiers, so no vendor can be surprised by which one they are in.

The tier belongs to the **vendor**, not the part, and their best contribution wins. A
company that donates one part and discounts another is a donor outright - the page will
not split them across both lists, and `tests/test_sponsor_page.py` pins that.

Then rebuild the container. The git history of `parts.yaml` is the audit trail of who
offered what and when, which is the reason this is one file and not a page of HTML.

**Then add them to the repository README as well**, in the tier they earned. What the page
promises a sponsor is deliberately split: this page carries their name only while the ask is
open and comes down when the car is built, so the durable half of the promise is the
project's own README. A sponsor recorded only here would lose their credit the day this
container is stopped.

## Adding it to the portfolio stack

Add this service to `/home/dylan/my-server/dylantamayo/docker-compose.yml`. It follows the
same shape as the other project services there - no `networks:` block, because only
`cloudflared` declares networking and every service shares the default network with it.

```yaml
  sponsorpage:
    build:
      context: ../SelfDrivingProject
      dockerfile: web/sponsor/Dockerfile
    container_name: sponsorpage
    restart: always
    mem_limit: 256m
```

Then:

```bash
docker compose -f /home/dylan/my-server/dylantamayo/docker-compose.yml up -d --build sponsorpage
```

`up -d sponsorpage` names the one service, so nothing else in the stack is touched.

**The last step has to be done in the Cloudflare dashboard, not here.** That tunnel is run
with `--token`, which means its routing lives in Cloudflare and there is no local ingress
file to edit:

1. Zero Trust → Networks → Tunnels → the tunnel this machine runs.
2. Public Hostname → Add.
3. Subdomain `sponsor`, domain `dylantamayo.dev`.
4. Service type `HTTP`, URL `sponsorpage:80`.

Rebuilding the page later never touches that route.

## Keeping it honest

The page states that the car crashes on 17 % of held-out runs. That number belongs there.
A sponsor who reads the repository - which the page invites them to do - will find it
either way, and finding it after being told the opposite is the only version of this that
costs the team anything.
