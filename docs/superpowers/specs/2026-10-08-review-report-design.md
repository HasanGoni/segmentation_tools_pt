# Design: shared review report (one HTML report + click-to-comment feedback)

Status: spec for map #48, ticket #49. Source: the "Ticket Review Report" spec (T1–T6),
adapted to this repo. Decisions below use expert defaults; change any of them by editing
this file and commenting on #49.

## 1. Purpose

After a ticket (or an evaluation run) the agent calls one function. It writes one
self-contained `report.html`. The reviewer opens it without starting a kernel, checks
images, metrics and gates, clicks on a problem, and writes a short comment. Comments are
saved as JSON. The agent reads them and fixes exactly those samples.

This is the **only** HTML renderer in the library. Evaluation (#28) and embeddings (#39)
produce *sections*; this module *displays* them. It is not a second evaluation engine.

## 2. Repo assumptions (checked 2026-10-08)

| Question | Finding | Consequence |
|---|---|---|
| Image format / bit depth | Dataset loaders read channels with `cv2.IMREAD_GRAYSCALE` (8-bit) and some branches with `IMREAD_UNCHANGED`; tensors are normalised floats. | A 16-bit file loses range before it reaches a `Sample`. The report adapter re-reads the **raw file** by `sample_id` with `IMREAD_UNCHANGED`. |
| Masks | PNG. `mask_mode`: `single` (binary or integer labels), `stacked`, `per_class_files`. Multilabel masks are `(K,H,W)` 0/1; multiclass masks are `(H,W)` integer labels. Class names come from `data.class_names`. | `Sample.gt`/`pred` accept both layouts. Class names and colours are config, never hard-coded. |
| Where validation data lives | Not in the repo. `data.raw_dir` in config; val ids come from `split_sample_ids` / `metadata.json`. | Real-data verification reads paths from config. Real data is never committed. |
| Metric code to reuse | `metrics.SegMetrics` (pixel IoU/Dice/recall/precision, all three task types). | Per-class Dice/IoU use `SegMetrics`. No second implementation. |
| Ranking / poster code | `visualize.rank_samples` and `save_error_poster` need a live model **and** a dataset; the poster is matplotlib + JPEG. | Not callable by the renderer. Reuse the **record format** `{'idx','sample_id','score'}`, the colours (FN green, FP red, overlap magenta) and the `_poster_masks` rules. The renderer scores stored predictions with `SegMetrics`. |
| Dependencies | numpy, cv2, matplotlib, torch available. **No Jinja2, no scikit-learn.** PyYAML only arrives through `hydra-core`. | No new heavy dependency. Plain Python string templates. Add `pyyaml` explicitly to `pyproject.toml`. |
| Tests / typechecker | No `tests/` folder. Tests are footnote cells in notebooks, run by `nbdev-test`. No typechecker configured. | Tests live in the new notebook. The "typecheck often" step from the pasted spec does not apply; we do not add a typechecker here. |
| Issue tracker | GitHub (`gh`). | — |
| Vocabulary | `CONTEXT.md` has a *Language* section. | Four terms added (see §10). |

## 3. Decisions

1. **Placement.** New notebook `nbs/16_review_report.ipynb` → module
   `segmentation_tools_pt.review_report`. (Evaluation should take `17_evaluation`,
   embeddings `18_embedding`; numbers only set the order in the docs.) The pasted spec's
   `ticket_report` package name is dropped.
2. **Templates.** Plain Python string templates and small helper functions; no Jinja2.
   JS and CSS are inline, vanilla, no CDN.
3. **Sections are plug-ins** (§4). Evaluation and embeddings add sections without
   touching the renderer.
4. **`Sample` holds stored predictions**, not a model (§5). Adapters turn repo data into
   `Sample`s.
5. **Gates file** is shared with the regression gate in #37 (§7).
6. **No LLM calls** in the library. The summary text comes from a template and real numbers.
7. **CLI.** `segpt-review serve <out_dir>` (new console script, added in `nbs/14_cli`).
   Standard-library server, bound to `127.0.0.1` only.
8. **Config-only knobs** (Hydra YAML `configs/review.yaml`): sections, N per group, seed,
   gates file, class names and colours, thumbnail edge, window percentiles.

## 4. Section interface

```python
@dataclass
class SectionOutput:
    title: str
    html: str                     # body fragment; no external URLs
    status: str = 'info'          # 'info' | 'pass' | 'fail'
    metrics: dict = field(default_factory=dict)   # flat dotted keys, e.g. {'dice.void': 0.83}

Section = Callable[[ReportContext], SectionOutput]
SECTIONS: dict[str, Section]      # registry
def register_section(name): ...   # decorator
```

- `ReportContext` carries `ticket_id`, `out_dir`, `samples`, `seed`, `cfg`, and an
  `assets` helper (writes files under `assets/` and returns relative paths).
- `build_report(..., sections=[...])` accepts built-in names or callables, so a client map
  can pass its own section function directly.
- Built-in sections: `summary`, `gates`, `overlay`, `preprocessing`, `synthetic`.
  Planned client sections: evaluation tables/failures/findings (#35, #36), novelty and
  drift (#44).
- Section `metrics` are merged into one flat dict. Gates read from that dict.

## 5. `Sample` and adapters

```python
@dataclass
class Sample:
    sample_id: str
    image: np.ndarray             # raw dtype kept (uint8 or uint16), (H,W) or (H,W,C)
    gt: np.ndarray                # (H,W) int labels, or (K,H,W) 0/1
    pred: np.ndarray              # same layout as gt
    prob: np.ndarray | None = None
    class_names: list[str] | None = None
    score: float | None = None    # optional precomputed ranking score
    meta: dict = field(default_factory=dict)      # source, lot, camera, ...
```

- Adapters live next to the renderer: `samples_from_dataset(model, dataset, raw_dir, ...)`
  (runs the model once, re-reads raw images) and a second one for the evaluation
  prediction cache (#31) when it exists. Any iterable of `Sample` works.
- Sampling: groups *worst N*, *random N*, *best N* by `score` (computed with `SegMetrics`
  when absent; `rank_samples` records can be passed in as precomputed scores).
  Fixed seed; each sample is tagged with its group.

## 6. Output layout and size

```
reports/<ticket_id>/
  report.html                    one file, CSS+JS inline, no network at view time
  report.json                    machine-readable: metrics, gate results, sample groups
  assets/<sample_id>/full.png    raw-value crop (16-bit preserved), loaded on demand
  feedback/<ticket_id>.json      written by the server or the download button
```

- Thumbnails: JPEG base64, quality about 85, long edge about 512 px.
- Masks and overlays: SVG contours (from `cv2.findContours`) or palette PNG.
  **Never JPEG for masks.**
- 16-bit images: display window by percentile (default 1–99.5), adjustable in the page.
  The `full.png` link keeps the original values.
- Budget: 60 samples below about 25 MB (test builds a 60-sample dummy report and checks it).
- One folder per ticket; no shared mutable files; one git worktree per parallel ticket.
  The library uses no GPU (note GPU contention for the *model run* in the docs).

## 7. Gates (shared with #37)

Compact form, kept from the pasted spec:

```yaml
dice_min: {void: 0.85}        # per class; metric key "dice.void" must be >= 0.85
max_clip_percent: 0.5         # scalar; metric key "clip_percent" must be <= 0.5
```

- Key suffix `_min` means "at least", `max_` prefix means "at most".
  `parse_gates(path_or_dict) -> list[Gate]` and
  `evaluate_gates(gates, metrics) -> list[GateResult]` are pure functions.
- A gate whose metric is missing **fails** with the reason "metric not found". It never
  passes silently.
- The overall banner is FAIL if any gate fails. Failed gates are shown first.
- Both functions import only numpy/PyYAML, so `compare` (#37) can use them without the
  renderer.

## 8. Feedback

```json
{
  "schema_version": 1,
  "ticket_id": "T-123",
  "sample_id": "sample_0042",
  "x": 312, "y": 148,
  "layer": "prediction",
  "comment": "Void is missed at the edge.",
  "created_at": "2026-10-08T21:00:00Z"
}
```

- `x`, `y` are in **original image pixels**. Each image element carries its original width
  and height; the page maps the click:
  `x_orig = round(x_thumb * W / w_thumb)`. Test with a scaled thumbnail.
- `layer` is one of `image | gt | prediction | errors`.
- `load_feedback(ticket_id, out_dir) -> list[FeedbackItem]` validates every item and
  raises an error that names the item and the bad field.
- Server (`segpt-review serve`): serves the folder read-only, accepts `POST /feedback`
  only, validates before writing, appends atomically (temp file + rename), rejects paths
  that leave the folder. Binds `127.0.0.1`. Fallback without a server: a
  "Download feedback.json" button.

## 9. Summary text

Built from a template and the real numbers. Rules: short sentences, one word per
meaning, active voice, at most 20 words per procedural sentence and 25 per descriptive
sentence (ASD-STE100 style). A test checks the limits for every generated sentence.

## 10. Vocabulary added to `CONTEXT.md`

- **Review report** — the single HTML page for checking a ticket's or run's visual result.
- **Gate** — a pass/fail rule on one named metric, read from a gates file.
- **Section** — one block of a review report, produced by a registered function.
- **Feedback item** — one click-and-comment on a report, saved with original-pixel coordinates.

## 11. Impact on other tickets

| Ticket | Change |
|---|---|
| #50 core renderer | Implements §4–§6, built-in `overlay` section, dummy-data generator. |
| #51 feedback loop | Implements §8. |
| #52 gates + summary | Implements §7 and §9. |
| #53 preprocessing | `preprocessing` section; reads raw files per §2. |
| #54 synthetic vs real | No scikit-learn: realism classifier is a small numpy/torch logistic regression with cross-validation and a rank-based AUC. |
| #31, #36 (evaluation) | `#31` should provide a `Sample` adapter for its prediction cache; `#36` passes evaluation sections to `build_report`. |
| #55 skill | Teaches the workflow in the pasted spec's §6, using `segpt-review serve` and `load_feedback`. |

## 12. Out of scope (kept in the map's Fog)

FiftyOne adapter, report index page, mask editing, tiled viewer for very large images.
