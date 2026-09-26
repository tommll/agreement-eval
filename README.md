# Can you trust an LLM's receipt extraction without checking it?

LLMs can pull the merchant, date, address and total off a scanned receipt in one call. The hard
part is knowing **which answers to trust without a human reviewing them**. The common shortcut is
to cross-check: run two or three models or prompts over the same document, and auto-accept any field
where they agree. That only works if the raters' errors are independent. Configs that share a model,
a prompt style or the same flawed input tend to make the *same* mistakes, and then agree while
being wrong.

This project measures when that shortcut holds, field by field, against labelled receipts, running
entirely on local models on a laptop CPU. It asks three questions:

| Question | How it's measured |
|---|---|
| **What limits accuracy: the model or its input?** | The same models on OCR text vs. the receipt image, and a 4B model vs. a 27B model, scored by exact match and by Jaccard similarity |
| **When does agreement mean correctness?** | AUC of agreement for predicting a correct answer, calibration, and the cost of auto-accepting unanimous fields (coverage, errors per 1,000 accepted) |
| **How should several raters' answers be combined?** | Majority vote vs. Dawid-Skene vs. each config alone, with the "any config was right" ceiling |

## Results

Local models only, on a laptop CPU (Intel i7-1365U, no GPU), served by llama.cpp. 12 SROIE test
receipts x 4 fields = 48 fields per config. "Exact" is a match after type-aware normalization;
"Jaccard" also counts a free-text field correct at token-set Jaccard >= 0.8 to the label (dates,
amounts and ids stay exact). Runs: `sroie-local-v1` (text) and `sroie-local-vision-v1` (image).

| Config | Input | Exact | Jaccard | Address (exact) | Time / receipt |
|---|---|---|---|---|---|
| Qwen3.5-27B IQ4_XS | image | **93.8%** | **97.9%** | 10/12 | 8.5 min |
| Qwen3.5-4B Q4_K_M | image + OCR text | 89.6% | 91.7% | 9/12 | 83 s |
| Qwen3.5-4B Q4_K_M | image | 87.5% | 91.7% | 10/12 | 81 s |
| Gemma 4 E4B Q4_0 | image | 58.3% | 66.7% | 0/12 | 56 s |
| Qwen3.5-4B Q4_K_M | OCR text | 56.3% | 58.3% | 1/12 | 68 s |

**1. The input mattered more than the model.** The same 4B model went from 56% to 88% when it
read the receipt image instead of the dataset's OCR text. The OCR text was missing lines the labels
contain: the full gold address was present in it for only 1 of 12 receipts. Scaling 4B to 27B
added 6 more points at 6x the time and ~5x the memory.

**2. Agreement between strong raters was a reliable trust signal; between weak raters it was not.**

| | Text run (3 x Qwen 4B) | Vision run (4 configs, 2 families) |
|---|---|---|
| Share of fields where every models give the same answer| (4%) 2/48 | (54%) 26/48 |
| Precision | 100% (2/2) | 100% (26/26) |
| Discrimination (AUC) | 0.56 | **0.97** |
| Accuracy: Majority vote / Dawid-Skene / best single config | 31% / 56% / 56% | 94% / 94% / 94% |


**Caveats.** 12 receipts is a small sample: read these as directions, not precise rates. The AUC of
0.97 rests on only 3 wrong consensus answers, one of them the label typo. The two Qwen 4B configs
share weights, so their agreement is not independent; Gemma is the cross-family rater.

## Quickstart (no API key, no downloads)

```bash
make install     # uv venv + editable install
make demo        # Postgres via docker compose, synthetic receipts, mock models, full report
```

`make demo` writes `reports/demo-v1/report.md`, a CSV per table, and four figures. Add `--open` (or
`make html`) for a single self-contained `report.html` - figures embedded, no network, opens from
disk and prints:

```bash
.venv/bin/agreement-eval evaluate --experiment experiments/sroie-local.yaml --html --open
```
 The mock
provider is rigged with two failure modes that matter: a pair of configs whose errors are
*correlated*, and a slice of "ambiguous" fields that every config reads the same wrong way - so
some unanimous items are unanimously wrong, exactly as they are in real runs.

## Real run on SROIE

Receipts come from [`jsdnrs/ICDAR2019-SROIE`](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE),
which ships gold `company / date / address / total` plus OCR words per receipt.

```bash
make db-up && .venv/bin/agreement-eval initdb
.venv/bin/agreement-eval ingest   --dataset sroie --split test --limit 100
.venv/bin/agreement-eval extract  --experiment experiments/sroie.yaml
.venv/bin/agreement-eval judge    --experiment experiments/sroie.yaml
.venv/bin/agreement-eval evaluate --experiment experiments/sroie.yaml
```

`agreement-eval all --experiment experiments/sroie.yaml` runs all four. Ingestion pages the HF
datasets-server rows API (no bulk download); `--source parquet` pulls the full split instead, and
`--download-images` fetches the scans so vision configs can run.

The shipped experiment has five raters that vary along the two axes that matter: three share a
model and differ only in prompt, three share a prompt and differ only in model. That is what lets
the analysis separate *agreement because both are right* from *agreement because they are the same
model making the same mistake*. Cost scales as configs x documents; 100 receipts x 5 configs is a
few dollars. Every call's token usage and estimated cost is stored per row.

## Running it on local models (no API key, no data leaving the machine)

The same pipeline runs against a local [llama.cpp](https://github.com/ggml-org/llama.cpp) server
speaking the OpenAI-compatible protocol, so GGUF weights become raters alongside (or instead of)
API models. A rater is a `(provider, model, prompt)` triple; nothing downstream knows the difference.

```bash
scripts/serve_local.sh qwen              # port 8080, Qwen3.5-4B-Q4_K_M
.venv/bin/agreement-eval ingest   --dataset sroie --split test --limit 12
.venv/bin/agreement-eval all --experiment experiments/sroie-local.yaml --skip-ingest
```

`experiments/sroie-local.yaml` defines three raters on one 4B model - two prompts at temperature 0 and
a third at temperature 0.8 - and points the judge at the same server. Two implementation details
are load-bearing, both pinned by tests:

- **`response_format: {type: json_schema}`** compiles the schema to a GBNF grammar and constrains
  decoding. Without it a meaningful share of a 4B model's responses are unparseable, and the
  "errors" you would then measure are the harness's, not the model's.
- **`chat_template_kwargs: {enable_thinking: false}`**. Qwen3-family templates default to thinking
  on; the reasoning lands in `reasoning_content`, the answer never arrives inside `max_tokens`, and
  `content` comes back empty. At ~4.5 tok/s decode it is also the difference between 30 seconds and
  several minutes per document.

Measured on a 15 W laptop (i7-1365U, Qwen3.5-4B Q4_K_M), one SROIE receipt with the `detailed`
prompt - 563 prompt tokens in, ~75 JSON tokens out:

| serving mode | prefill | decode | wall clock per rater |
|---|---|---|---|
| CPU, 8 threads | ~23 tok/s | ~5 tok/s | ~40 s |
| Vulkan (iGPU, `-ngl 99`) | ~27 tok/s | ~3.6 tok/s | ~42 s |

Extraction is prefill-heavy (7.5 prompt tokens per generated token), which is the regime where the
iGPU should win - but the Vulkan prefill measured here is well short of its 38 tok/s benchmark
because a browser was sharing the same iGPU. On an idle machine it should pull ahead; the point is
that this is measurable per run, since llama.cpp's own `timings` are stored on every call and the
report carries a `local_speed` table with prefill and decode rates per config.

**Concurrency does not help.** The server offers four slots, but three concurrent decodes on this
CPU dropped per-stream decode from 4.5 to 0.8 tok/s - worse in aggregate than running them one at a
time. Local serving is memory-bandwidth-bound, so `concurrency: 1` is the right setting and the
shipped config says so.

### What the local run found (12 SROIE receipts, 3 raters, 36 calls)

Full report: `reports/sroie-local-v1/report.html`. A proof of concept on 48 items, so the intervals
are wide - but several effects are large enough to act on.

- **Exact-match accuracy 29%** overall. The best rater (`detailed`, temperature 0) reached 56%, the
  `terse` rater 11%. Most of the gap is *misses*: the terse prompt returned null on 25 of 44 fields,
  the detailed prompt on 2 of 48. Prompt wording, not model capacity, decided whether a field came
  back at all.
- **Temperature is not a second opinion.** Same weights, same prompt, temperature 0 vs 0.8: identical
  answers on 39% of items where both returned anything, and where they differed the hotter rater was
  usually degenerating rather than disagreeing. It is the cheapest way to manufacture agreement and
  the least informative.
- **A grammar guarantees shape, not sense.** 8 of the first 36 calls returned structurally valid JSON
  cut off mid-string: inside a string literal the model degenerated - repeating its own system
  prompt, emitting `<think>`, runs of zeros - until it hit the token cap. None at temperature 0 with
  the detailed prompt; a resume retry recovered 5, leaving 3.
- **Majority vote was actively harmful**: 0.31, well below the best single rater's 0.56. Two raters
  that mostly answer "absent" agree with each other constantly - 82% of their pairwise agreement was
  *both saying null* - and they out-voted the one rater doing the work.
- **Dawid-Skene recovered from that, but only once null was modelled properly.** The textbook
  one-coin model scored 0.125 and ranked the configs exactly backwards, handing the worst rater a
  reliability of 0.999. Pricing null as the cheap shared answer it is (see below) moved it to 0.56 -
  matching the best single config without being told which one that was, and estimating
  reliabilities of 1.00 / 0.18 / 0.14 against measured accuracies of 0.56 / 0.15 / 0.11.
- **The local judge failed its own controls**, so these numbers are exact-match only.

### Validate the judge before believing it

`agreement-eval judge --check` scores the judge on comparisons whose answer is known: identical
strings, cosmetic rewrites, and values lifted from a different document.

| control | n | Qwen3.5-4B judge |
|---|---|---|
| identical strings | 6 | 0.50 |
| cosmetic rewrites | 6 | 1.00 |
| unrelated values | 4 | 0.50 |

A judge that calls two byte-identical strings different half the time, and merges unrelated values
half the time, would manufacture exactly the agreement this study measures. So the shipped local
config has `judge.enabled: false` and the report prints the control scores next to that decision.
Run the check against any judge - local or API - before turning it on.

A second model is a second server and four lines of YAML:

```bash
scripts/serve_local.sh deepseek 8081
# uncomment the deepseek rater in experiments/sroie-local.yaml, then:
.venv/bin/agreement-eval extract --experiment experiments/sroie-local.yaml
```

`extract` resumes, so adding a rater later runs only that rater's calls against the existing run.

## How it works

```
ingest ──► documents ──► extract ──► extractions ──► field_outputs ──► evaluate ──► report.md
           (Postgres)    (N configs)                 (one row per      + judge      + CSVs
                                                      doc x field      (judgments)  + figures
                                                      x config)
```

Everything downstream reads `field_outputs`, so a run can be re-judged or re-analyzed months later
without calling a model again. `extract` is resumable: completed `(run, config, document)` triples
are skipped, and a re-run after a transient failure replaces exactly one row.

### Tables

| Table | Grain |
|---|---|
| `documents` | one scanned document: gold JSONB, OCR text, image URI/path |
| `runs`, `configs` | the experiment snapshot and each rater's model/prompt/effort + prompt hash |
| `extractions` | one model call: raw JSON, status, latency, tokens, cost |
| `field_outputs` | one field of one call: raw value, normalized value, is_null, evidence |
| `judgments` | one judge verdict: `pair` (two configs) or `gold` (config vs. label) |
| `analysis_results` | headline numbers and key tables, keyed by run |

## Method decisions worth knowing

**One normalizer, both metrics.** Accuracy and agreement compare values through the same
field-type-aware normalizer (`money`, `date`, `org`, `address`, `phone`, `integer`, `text`). If
agreement were measured more leniently than accuracy, the headline question would be answered by
the metric rather than by the models.

**Jaccard is a graded companion, not a replacement.** Every prediction also gets a token-set
Jaccard similarity to its label, and every item a mean pairwise Jaccard between configs
(`jaccard_agreement`, `jaccard_matrix`). Free-text fields (`text`, `org`, `address`) get partial
credit; dates, amounts, integers and phones stay all-or-nothing, since one wrong digit is a wrong
value. `accuracy_jaccard` counts a field correct at Jaccard >= 0.8 (`evaluate --jaccard-threshold`).
Unanimity, majority vote and Dawid-Skene stay on exact match: 1 - Jaccard is a metric, but a
threshold on it is not transitive, so it cannot define the answer clusters those methods vote over.

**Null means normalized-absent.** `"N/A"`, `"-"`, `""` and a money field that parses to no number
all collapse to null, on the prediction side and the label side alike. A field that is absent from
the document is a real answer a model can get right or wrong, so both-null counts as correct.

**Items are (document, field).** Agreement is computed per item across configs; correctness of the
*consensus* is the outcome variable. A second, prediction-level view asks the other question a
reviewer asks - "do the other configs back *this* answer?" - and reports it per config.

**The judge is bounded on purpose.** It only sees pairs that exact match already rejected, and only
when both sides are populated; null-vs-populated is a disagreement about whether the field exists
and no judge is allowed to wave it away. Judge verdicts merge values into equivalence clusters, so
unanimity, majority, entropy and the Dawid-Skene input all stay consistent with them. On the gold
queue the judge can only *upgrade* an exact-match miss, never overturn a match, which keeps
judge-lenient accuracy an upper bound.

**Dawid-Skene, two variants, two corrections.** Extraction answers are open-vocabulary, so a full
per-class confusion matrix is not identifiable; the value-level model is the one-coin variant (a
single reliability per config). Two parameters in it are load-bearing, and both were found by a run
producing obviously wrong answers:

- `vocabulary_size` - how many values a wrong config could have produced. Set to just the candidates
  present in an item, coincidental agreement becomes free and EM runs away to a degenerate solution
  declaring one config infallible. It is estimated per field from the distinct answers observed.
- `null_aware` (default on) - "field is absent" is not one of V arbitrary strings. Every config can
  reach it, and one that gives up often emits it constantly, so two such configs agree on null
  without that agreement carrying information. The textbook model treats shared silence as a
  coincidence too unlikely to ignore and crowns the laziest raters; each config therefore gets a
  second estimated parameter, how often it answers null *given that it is wrong*. On the local run
  this was the difference between 0.125 with the ranking inverted and 0.56 with it correct.

The populated/null decision *is* a fixed two-class problem, so that additionally gets the full 2x2
confusion-matrix model, which is where a config's asymmetry (eager to fill vs. eager to say null)
shows up.

**Dawid-Skene inherits the correlated-error blind spot.** It assumes raters err independently given
the truth. Configs sharing a base model break that assumption, and DS then ratifies their agreement
exactly as majority vote does - `tests/test_dawid_skene.py` pins both the win case and this one.
That is why the oracle ceiling is printed next to every aggregator.

## Limitations

- **SROIE labels every field on every receipt**, so its gold-null arm is empty. The null half of
  the question is answered there by the consensus-state split (items the configs *decided* were
  absent), and properly by running the same study on **FUNSD** (`experiments/funsd.yaml`), where most
  schema fields genuinely are absent on most forms.
- Exact match after normalization is strict; the judge measures how much of the residual
  disagreement was only ever formatting, but the judge is itself a model. Jaccard accuracy is the
  deterministic lenient bound, and its threshold is a choice that moves the number.
- All shipped configs are Claude models, which is the realistic setting for most teams and also the
  setting in which errors are *most* likely to correlate. Adding a rater from a different family
  would strengthen the design; the provider interface is one class.
- Agreement statistics over 3-6 raters are coarse by construction - the pairwise score only takes a
  handful of distinct values, which is why buckets are the observed values rather than fixed bins.

## Layout

```
src/agreement_eval/
  schemas.py        field schemas + the JSON schema sent to the model
  normalize.py      field-type-aware normalization (the definition of "equal")
  db.py, sql/       Postgres access layer and DDL
  ingest/           sroie (HF), funsd (local), synthetic (offline)
  extract/          prompts (terse | detailed | evidence), providers, runner
                    providers: anthropic | llamacpp (local GGUF) | mock
  judge.py          LLM judge for semantic agreement and semantic correctness
  metrics/          accuracy, agreement, predictive, dawid_skene
  analysis.py       builds every table; report.py / plots.py render them
experiments/        experiment definitions (demo, sroie, sroie-local, funsd)
scripts/            serve_local.sh - start a llama.cpp server for a local model
tests/              unit tests + a Postgres integration test (skipped if no DB)
```

Adding a dataset is a schema in `schemas.py`, a loader returning `Document`s, and one entry in
`ingest.LOADERS`. Adding a rater is four lines of YAML.

## Tests

```bash
make test
```

80 tests: normalization and Jaccard edge cases, agreement and judge-merge semantics, the Dawid-Skene win case
and its blind spot, AUC/Wilson/calibration against hand-computed values, provider request shape for
both backends (that sampling parameters are never sent to Claude, which rejects them; that the
local backend always sends the JSON grammar and never enables thinking), local-server failure
handling, both Dawid-Skene failure modes (shared nulls, correlated errors), and a full
extract -> analyze round trip against Postgres.
