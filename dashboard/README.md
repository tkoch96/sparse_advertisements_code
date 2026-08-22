# SCULPTOR experiments dashboard

Localhost dashboard for watching experiment grids fill in while the VM
trains: left sidebar = one tab per experiment; each tab shows the
experiment's ladder-over-N figure plus an arm x N table of per-seed
own-objective scores, with every value click-through to that run's
convergence-over-iterations PDF.

## Refresh cycle (Mac-side, VM undisturbed)

```bash
# 1. pull new results + harvested convergence figures from the head
rsync -az -e "ssh -i ~/.ssh/ray-autoscaler_us-east-1.pem" \
  ubuntu@<HEAD_IP>:sparse_advertisements_code/cache/ablation/hardB3/ \
  cache/ablation/hardB3/
rsync -az -e "ssh -i ~/.ssh/ray-autoscaler_us-east-1.pem" \
  ubuntu@<HEAD_IP>:sparse_advertisements_code/cache/ablation/hardB3_artifacts/figs/ \
  cache/ablation/hardB3_artifacts/figs/

# 2. score new cells on their OWN objectives (incremental: cached by
#    file mtime in cache/model_error/hardB3_scores.json; parallel child
#    per (world, seed))
python dashboard/score_hardb3.py --jobs 4

# 3. regenerate figures (combined + per-objective panels)
python dashboard/plot_hardb3.py

# 4. regenerate the site (dashboard_site/index.html)
python -m dashboard.generate
```

Serve once (persists): `python3 -m http.server 8643 --directory
dashboard_site` (or the `.claude/launch.json` entry `hardb3-dash`),
then open http://localhost:8643 and reload after each cycle.

## Architecture / adding an experiment

`generate.py` holds an `EXPERIMENTS` registry: one entry per sidebar
tab. Each entry selects a renderer (`kind`) and carries presentation
overrides:

- `kind: 'objective_ladder'` — the hardB3 shape: scores store + arm x N
  table + convergence links + painter row. Overrides: `fmt` (value
  format), `intro` (HTML blurb), `figures` (PNGs above the table),
  `world`, `figs_dir`.
- `kind: 'static'` — figures + text only (the overview tab).

Adding a new experiment = appending a registry entry; a new result
shape = adding a renderer function to `RENDERERS`. Scores stores are
produced by per-experiment scoring scripts (`score_hardb3.py` is the
template: incremental, per-cell cache keyed by path+mtime, child
process per (world, seed) so deployments build once).

## Caveats

- `dashboard_site/` is generated output (symlinks into cache/ and
  figures/) — do not commit it.
- Convergence links appear only after the head-side harvest gzips a
  run's workspace (in-flight batches show unlinked values).
- Scoring uses the same env-knob worlds as training; site_failure
  (popfail) evals run on the stock world, the georand objectives under
  SCULPTOR_LAT_MODEL=geo SCULPTOR_PREF_MODEL=random.

## Data-format contracts (for future agents)

**Scores store** (`cache/model_error/hardB3_scores.json`): one flat JSON
object; key = repo-relative result path (`cache/ablation/hardB3/<obj>/
<pmode>/N<k>/seed_<s>_<rung>.json`) or `painter:<obj>:<path>` for
painter refs cross-scored under objective `<obj>`; value =
`{"key", "mtime", "obj_val", "opp_val"}`. `obj_val` is the LITERAL
generic-LP objective scalar (benefit convention, higher = better;
null = eval failed). `mtime` is the result file's mtime at scoring time
-- rescoring happens automatically when the file changes.

**Convergence-figure filenames** (what `conv_grid` expects):
- hardB3 harvest:  `<obj>_<pmode>_<rung>-dep<seed>-N<n>-<probemode>.pdf`
  (e.g. `fracb_sched_no_mc-dep2-N5-scheduled.pdf`)
- fixed-mode runs drop the N part: `<obj>_fixed_no_mc-dep<seed>-fixed.pdf`
- policy-ladder harvest (no obj prefix): `<rung>-dep<seed>-N<n>-<pmode>.pdf`
  / `no_mc-dep<seed>-fixed.pdf`
- PAINTER RUNS PRODUCE NO CONVERGENCE FIGURE (not an iterative solver);
  painter rows in link grids stay unlinked permanently.

**Out-root layouts** consumed here:
- hardB3 style: `cache/ablation/hardB3/<obj>/{fixed,sched,smart}/N<k>/
  seed_<s>_<rung>.json` (+ `painter_georand/`, `painter_stock/` at
  `N1/seed_<s>_painter.json`).
- policy-ladder style: `cache/ablation/policy_ladder_fixed/<arm>/N<k>/
  seed_<s>_<rung>.json`.

Queues complete cells in their own order (rungs within a --rungs list
are NOT interleaved) -- an empty N-dir or missing rung means NOT RUN
YET, not lost; check the queue log before assuming loss.

## Refreshing (the ONLY supported mechanism)

`python -m dashboard.refresh [--loop 180] [--heavy-every 4]
[--host <ip>] [--experiments id1,id2]`

One generic, registry-driven cycle: for every experiment/section whose
registry entry carries a `refresh` spec, it (1) runs the spec's
`remote_harvest` shell on the head, (2) mirrors each `pull` pair with
rsync --delete (HEAD IS AUTHORITATIVE -- local stale files are
removed), (3) runs the spec's `steps` pipeline (below), (4) runs any
legacy `evals`/`heavy` argv lists, then regenerates the site once.
Head IP resolves from --host, SCULPTOR_HEAD_IP, or the cluster alert
JSON. Do NOT write per-request refresh scripts -- add/extend a
registry `refresh` spec instead.

### `steps`: staleness-gated eval/figure pipeline (2026-08-14)

Each step declares its dependencies explicitly and runs ONLY when
stale -- make-style:

```python
'steps': [
  {'in':  ['cache/ablation/<exp>/*/N*/seed_*_*.json'],  # source globs
   'out': ['cache/model_error/steady/<tag>.json'],      # what it makes
   'argv': ['{py}', '-m', '...'],
   'world': 'georand',                  # REQUIRED for eval steps: env
                                        #   knobs resolved from
                                        #   old_handoffs/MODEL_UNCERTAINTY_DIMENSIONS.md/
                                        #   worlds.py (single source of
                                        #   truth — never retype knobs)
   'env': {'POLICY_PLOT_OUT': '...'},   # optional extra env
   'every': 4},                          # optional cost cap (Nth cycle)
]
```

An eval step WITHOUT `world` builds deployments in the stock world and
produces garbage silently (2026-08-14: every v2 arm showed ~12 ms,
below the georand opp — impossible; wrong-world stores quarantined in
`cache/model_error/WRONGWORLD_STOCK_EVAL/`). Sanity gate: per-seed arm
means must sit ABOVE the per-seed opp ref (STEADY only — the
failure-composite can legitimately beat opp).

### Self-healing (2026-08-14/15 — all three learned the hard way)

The loop re-resolves the head IP from
`~/.sculptor_cluster_alert/active_cluster.json` EVERY cycle (instance
restarts change it), reloads the registry module every cycle (edits
to generate.py take effect without restarting the loop), and uses
`StrictHostKeyChecking=accept-new` on both the harvest ssh and pull
rsync (new IPs otherwise fail host-key verification silently —
stderr is discarded). Staleness additionally fingerprints the INPUT
FILE SET (stamps in `cache/model_error/.refresh_stamps/`), so
deletions/quarantines re-trigger evals — mtime alone cannot see a
removed file, and quarantined arms used to linger in stores/figures.

Site symlinks (created by generate.main): `figs` (hardB3 legacy figs),
`figs_hb3v2` (hard-objectives v2), `figs_ladder2` (policy ladder v2),
`figs_ladder` (bad-grads era), `plots` (repo figures/). Current tabs:
'Ablation: policy ladder' + 'Ablation: hard objectives' (legacy tabs
removed 2026-08-15; data remains on disk, quarantined).

A step runs iff an `out` is missing (while inputs exist) or the newest
`in` file is newer than the oldest `out`. Chain data -> eval store ->
figure by listing the store in the figure step's `in`: any new result
propagates to the figure within one cycle, and idle experiments cost
zero compute.

`'always': True` (Tom, 2026-08-14): CHEAP steps — every plot/render —
must set this and run every cycle unconditionally. The dash's contract
is full recomputation of all available stats all the time; staleness
gating is a COST optimization reserved for expensive evals (which save
to the standard cache/model_error stores the plots read). A plot gated
on store mtimes goes stale whenever plotting CODE changes — that class
of "forgot to re-render" must not exist. Every figure a tab shows MUST be the `out` of some step
whose env/argv pins the output filename -- never rely on a plot
script's default output name (that is how the v2 figure went stale
while silently overwriting the bad-grads figure, 2026-08-14). Prefer
`steps` for all new entries; `evals`/`heavy` are legacy.
