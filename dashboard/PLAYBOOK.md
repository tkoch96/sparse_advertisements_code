# Dashboard updating playbook

Canonical reference for how the SCULPTOR dash gets built, published, and
debugged. Written 2026-08-26 after a day of "it's stale" incidents; every
rule below was paid for. Update this file when the pipeline changes.

## The pipeline (one direction, five stages)

    VM run dirs ──(expctl watch / cluster.harvest_all)──> cache/cluster_runs/<run_id>/
        │                                                    logs/run.log, results/**, progress.json
        └── cell logs land under results/cache__* (names are path-mangled)

    cache/* ──(dashboard/plot_*.py, one module per figure family)──> figures/dashboards/<dash>/*.png
                                                                     figures/cluster/<run_id>/*.pdf
    figures ──(dashboard.generate)──> dashboard_site/index.html   (mtime cache-busted <img> tags)
    dashboard_site/ ──(rsync)──> ubuntu@<host>:/var/www/dash/  on BOTH dash hosts
    Apache (basic-auth vhost `dash`, docroot /var/www/dash) serves it.

**The single entry point is `dashboard/update_dashboard.sh`.** One shot =
harvest live runs -> per-run timing plots -> site build -> push both hosts.
`--loop 30` runs forever: FAST path (site build + push, ~15-70s) every
cycle; SLOW path (harvest + all figure scripts, ~9 min) at most every
`SLOW_EVERY` (default 300s). Portable mkdir lock, no overlapping cycles.
Log: `~/sculptor_dashboard/update_dashboard.log`.

Never build a second publisher; never hand-roll pieces of this chain in a
one-off shell loop — extend the script.

## Processes that must be alive (and how to check)

| process | purpose | check |
|---|---|---|
| `update_dashboard.sh --loop 30` | THE pipeline | `ps aux \| grep update_dashboard` |
| `expctl watch <run_id>` (one per live run) | pulls run data + verdict stamping | `ps aux \| grep "expctl watch"` |
| `dashboard/refresh.py --loop 300` | heavy non-cluster tabs (staleness-gated steps) | `ps aux \| grep refresh.py` |
| `dash_public_sync.sh` (legacy) | redundant dual-host push; harmless | optional |
| disk guard (rides the 3-min autocheck cron) | reclaims at >=97% -- raylet-safe | `~/sculptor_dashboard/disk_guard.log` |

Duplicate/stale watches happen after kills+resumes: kill all, start one per
live run. A watch pinned to a dead segment stamps stale verdicts.

## Contracts (violate these and the dash lies)

1. **Steps contract**: every figure a tab embeds must be the output of a
   registered refresh step (EXPERIMENTS entry or update_dashboard step) —
   never a one-off manual render (the ladder_*.png set froze for 3 days
   this way).
2. **Cache-busting**: every `<img>` goes through `_img()` (mtime `?v=`).
   A raw same-named src lets the browser serve yesterday's PNG forever.
3. **figpaths**: figure basenames map to dash dirs via `_RULES` prefixes in
   `helpers/figpaths.py`; an unregistered prefix lands in `misc/` and no
   tab shows it.
4. **Resumed runs**: segments append to run.log and REUSE iteration
   numbers. Any per-iteration parser must keep the LAST occurrence /
   latest attempt (see `_latest_attempt`, the iter-timing dedupe) and
   scope counts to the current `[sweep] === dpsize=` banner.
5. **Banner/format drift**: parsers regex production log lines
   ([mem], [wt], sweep banners). When a log format changes (new fields,
   dpsize=actual-32 vs dpsize=32), grep every dashboard/*.py for the old
   pattern in the same commit. MEM_RE-style positional field coupling has
   silently zeroed a plot twice.
6. **Dual host**: pushes go to BOTH instances (`DASH_INSTANCES`). Tom's
   bookmark is 107.22.173.189 (i-0428c); the pinned box is i-09a6. A
   freshly-restarted box serves whatever stale content it last had.

## Debug order for "the dash is stale" (do IN ORDER, no skipping)

1. **Which tab / which table?** Ask or infer precisely. Two tables are
   both called "progress" (dpsweep + papertable). Fixing the wrong one
   wastes a round-trip.
2. Loops alive? (table above). Restart what's dead.
3. Harvest actually fresh? Check mtimes of the FILES the parser reads
   (run.log, results/cell logs) — `harvest.json`'s timestamp updates even
   when rsync moved one file.
4. Figure rebuilt AFTER the fresh pull? (A rebuild 3 min before the pull
   looks fresh by mtime and is stale by content.)
5. Page rebuilt after the figure, pushed to the host THE USER VIEWS?
6. **Verify by rendering, not grepping**: Read the actual PNG/PDF (or the
   served index section) and look at the axes/values. String-presence
   checks pass on the wrong tab.
7. Browser cache last — only after 1-6 are proven good.

## Adding a new figure/tab checklist

1. Plot module `dashboard/plot_<name>.py` writing via
   `helpers.figpaths.fig_path` (+ add a `_RULES` prefix).
2. Register: EXPERIMENTS entry in `dashboard/generate.py` (`figures` /
   `figures_glob`) + a refresh step, or add to update_dashboard's plot
   pass for per-run figures.
3. Run the module once, `dashboard.generate`, then **Read the rendered
   output** before telling anyone it's live.
4. Push happens via the loop within ~30s; force with
   `bash dashboard/update_dashboard.sh` (one shot).

## Known traps (each cost real time)

- zsh eats bare `[5,10]` — quote bracket args.
- `crontab` WRITES hang headlessly (macOS TCC prompt) — ride the existing
  autocheck cron instead.
- The disk guard must never `rm -rf /tmp/ray` while a raylet runs.
- `sips` converts the sweep PDFs to dash PNGs (pdftoppm fallback).
- Result-JSON field names drift when forks merge to mainline — the queue
  audit's "stale code" flag is usually RIGHT.
- ENOSPC kills the local Ray GCS first; the dash pipeline dies with it.
  Watch `df` before long local campaigns; APFS purgeable hides usage.

## Postscript (same day): the HTML itself was browser-cached

Step 7 fired within the hour of writing this. Images were cache-busted;
`index.html` was not — the browser happily kept an old page over a fresh
server copy. Fixed at the source: both dash vhosts now send
`Cache-Control: no-store, must-revalidate` for `*.html` (mod_headers,
FilesMatch in `sites-available/dash.conf`). One final hard-refresh
escapes an already-cached copy; after that plain reloads are always
fresh. If a THIRD cache layer ever appears, the debug order still holds:
prove server content first (grep the served file over ssh), then blame
the client.

## THE definitive test: render it and look (Tom 2026-08-26)

Greps lie by omission (right string, wrong tab) and there are ~8 cache
layers to misdirect you. The only test that closes the loop:

1. Pull the exact served bytes:
   `vmctl ssh <dash-instance> -- 'cat /var/www/dash/index.html' > /tmp/served.html`
2. Serve locally (`.claude/launch.json` entry `served-dash-copy`,
   python http.server) and open in a REAL browser.
3. Click to the exact tab/subtab in question, and LOOK AT THE PIXELS.
   Screenshot it. If you can't show a picture of the thing, it is not
   verified.

Browser-pane quirks learned doing this: screenshots capture the top of
long pages regardless of scroll -- zoom out (body.style.zoom≈0.4) to get
everything in frame; plots/ images 404 in a local copy (fine -- tables
and text are what you're verifying).

Client side is pre-solved: both vhosts send `Cache-Control: no-store`
for HTML, so after ONE hard refresh a viewer can never be HTML-stale
again. If someone reports staleness anyway, run this test FIRST and
show them the screenshot -- it either exposes a real pipeline bug
(mis-parsed section, clipped panel) or proves the content and isolates
the client in one step.
