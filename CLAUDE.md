# CLAUDE.md

Guidance for Claude Code in this repo. Ask me about anything unclear before implementing.
Keep this file short (under 150 lines): rules most tasks need, pointers for the rest.

## What this is

Thesis research on **constraint learning under label noise**: a model fitted to noisy
labels is embedded as a constraint `f(x) <= b`, and the optimizer's decision violates
the true constraint. The standing thesis plan (Papers A–C) is `notes/research_plan.md`.

**Current work: Paper A** (target *Operations Research*, with DB = Dimitris Bertsimas).
**Still exploratory: edit the outline and plan, not `paper.tex`** (DB, 2026-11-16).

## Paper A files (`paper/`): read before working on the paper

| file | role |
|---|---|
| `paper_outline.md` | **the method reference**: problem, methods, guarantee, experiments (v4) |
| `paper_plan.md` | decisions, probes (F−1–F2, P1, P2, R*), theory tasks (T*), implementation tasks (I*), risks, questions for DB. Organized by topic, decisions first |
| `paper_a_guarantee.md` | proof notes (its Sections 5–6 are superseded by the plan) |
| `paper.tex` + `paper_a_guarantee.tex`, `paper_a_proofs.tex` | the draft; `paper_a_guarantee_held.tex` is NOT input, waiting on probe P1 |
| `paper_a_literature.md`, `paper_a.bib` | literature check and bib (also in Zotero) |

**When a decision changes, update the outline and the plan in the same change**, and
add a dated entry to `notes/progress.md` when a result lands (its header says how).
Status tags in the outline: [proved] / [to prove] / [open] / [to run]. Nothing in
outline Section 6 is a result until a probe says so.

## Methods

- **RC (the paper's method)**: `f(x; θ̂) + R‖s(x)‖₂ <= b` for models affine in the
  labels, `f(x; θ(y)) = s(x)ᵀy + c(x)` (ridge on a feature map, last-layer network,
  trees with fixed splits). Theorem 1 holds at any RC-feasible `x*`. The current
  implementation is `experiments/probe_rc_ridge.py`; there is no `src/methods/rc.py` yet.
- **FO-CP (new, not built)**: cutting planes whose separation retrains the model on
  `ȳ + Δy` and ascends by first-order methods. On affine models its cut is the
  RC's (probe F0 checks this); the RC's radius does not carry over, and F−1
  tests the model-free one. Guarantee open. sklearn + numpy first, PyTorch only
  if too slow (I11).
- **Baselines**: constant margin, split conformal, the wrapper, C-MICL (as published,
  and on the same model). `src/methods/` still holds the legacy implementations.
- **Not in the paper**: random-bank CP (`src/methods/cp.py`) and the dial/rho sweeps.
  `presentations/method.tex` describes that old pipeline and is frozen. **Do not use
  it as the method reference.** Details: `notes/ops/legacy-pipeline.md`.

## Environment & commands

Python **>=3.14** via **uv** (`uv run ...`). **Gurobi license required.** No test
suite, no linter. Entry points are in `experiments/`. Prefer Bash.

**Probes** (`experiments/probe_*.py`, one `submit_*.sh` each) import no runner and
write only under `results/paper_a/prelim/` (or `results/reactor/diagnostics/`).
Commit outputs with the job ids in the message. Anything over a few minutes runs on
the cluster: `mit_quicktest` for 15-minute pieces, `mit_preemptable` with
`--requeue` beyond that. Submit scripts email on `END,FAIL`; wait for the email.

```bash
uv run python experiments/probe_rc_ridge.py --stage {solve,radius,redraw,frontier,conformal,...} \
    --model {cubic,net_frozen,net_standard,net_honest} --costs paper --design {full,sub,thin}
sbatch experiments/submit_rc_probe.sh        # go/no-go probe (job 29141177)
STAGE=conformal DESIGN=sub sbatch experiments/submit_rc_frontier.sh   # refuses the committed full-design frontier
sbatch experiments/submit_cmicl_family.sh    # C-MICL Assumption 4.1 by surrogate family
uv run python experiments/verify_embedding.py [--problem reactor]   # MIP vs sklearn agreement
```

**Never run the legacy sweeps locally** (`run_dial_sweep.py`, `run_rho_sweep.py`,
`run_dial_test.py`, full `run_chemo_robust.py`): they write in place into committed
`results/rho_sweep/` files. Their rules are in `notes/ops/legacy-pipeline.md`.

## Instances

- **Reactor (lead, SS1)**: Ovalle et al.'s exact DMA-MR instance, ODE judge, σ = 2,
  their 100 cost vectors (`--costs paper`) with noise redrawn per run. The repo's
  fixed `balanced` cost is legacy. **Never quote a C-MICL reactor feasibility without
  naming the cost distribution.** Before calling a `cmicl` number theirs, read the
  DIFFERENCES section of `src/methods/cmicl.py`. Facts: `notes/ops/reactor-and-cmicl.md`.
- **Chemotherapy**: 416 real arms, six outcomes, Maragno et al.'s protocol (96
  held-out arms, full-cohort ensemble judge). SS2 (semi-synthetic, binomial σ per
  arm) needs a generator (I14).
- **WFP diet**: not in the repo.
- The analytic synthetic instance is dropped.

## Code conventions that still bind

- `src/models/train.py` / `embed.py`: a new model type needs both files. **Embedding
  tolerances are load-bearing** (`SPLIT_EPS`, float32 routing, `IntFeasTol=1e-9`):
  run `verify_embedding.py` after touching them.
- `src/evaluation/metrics.py` is the only place `f_true` is allowed; gastric outcomes
  are judged by the GT ensemble, never the embedded models.
- `run_cv.py` overwrites `results/cv/*_selected_configs.json`; `git checkout` them
  after an inspection run.
- Gurobi and library parameters stay at their defaults unless the deviation is
  documented and load-bearing.
- The full repo map is in `notes/ops/legacy-pipeline.md` ("Repo map").

## Writing

- Paper prose uses robust-optimization vocabulary, except when naming cited
  statistics literature.
- Build with **`latexmk`, never bare `pdflatex`** (`paper/` and `presentations/` each
  have a `.latexmkrc`; output beside the `.tex`, aux in `build/`).
- Weekly advisor decks: `../research-updates/` via `/weekly-update`, fed by
  `notes/progress.md`. Old dated decks in `presentations/` are frozen.
