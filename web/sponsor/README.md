# The public sponsor page

A single static page that lists the parts this project still needs, hosted at
**sponsor.dylantamayo.dev** alongside the rest of the portfolio stack.

It exists because a private spreadsheet cannot do two things this page does: a vendor who
was not asked, or who answered late, can still see what is open and offer something else;
and anyone who wonders whether the ask is honest can read the list, the prices, the dates,
and the code behind them in one sitting.

## How it is put together

```
web/sponsor/parts.yaml        the only thing you edit
src/selfdrive/web/sponsor.py  generator - renders the page, enforces the publishing rules
web/sponsor/nginx.conf        serving config (behind the Cloudflare tunnel)
web/sponsor/Dockerfile        multi-stage: generates the page, then serves it
tests/test_sponsor_page.py    pins the publishing rules
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

## The four publishing rules

These are enforced in the generator and pinned by tests, not left to whoever edits next.

1. **The page counts parts, never money.** A dollar total reframes a parts list as a
   fundraiser, which is a different and much worse conversation to have with a vendor.
2. **No vendor is ever published as having declined or ignored us.** `state` in the YAML can
   be `needed`, `asked`, `declined` or `no_reply`, and all four render identically as
   "Still needed". Only `pledged` and `received` name anyone, and only then with `by:`.
3. **`team_note:` never renders.** The repository is public, so this is not a secret - it is
   a place for per-vendor status that does not belong in front of sponsors.
4. **Every price carries the date it was checked**, and `verified: false` prints "price not
   verified" next to a figure nobody has confirmed.

## Updating it when a sponsor answers

```yaml
- part: Depth camera
  state: received          # was: needed
  by: Luxonis              # shown publicly, with thanks
```

Then rebuild the container. The git history of `parts.yaml` is the audit trail of who
offered what and when, which is the reason this is one file and not a page of HTML.

**Then add them to the repository README as well.** What the page promises a sponsor is
deliberately split: this page carries their name only while the ask is open and comes down
when the car is built, so the durable half of the promise is the project's own README. A
sponsor recorded only here would lose their credit the day this container is stopped.

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
