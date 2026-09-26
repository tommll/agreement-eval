# When models agree, are they right? Receipt extraction on a laptop

*Measuring whether cross-checking LLM extractions can be trusted, and what changed when a 4B local model started reading the receipt image itself.*

## Table of contents

- [Problem](#problem)
- [High-level solution](#high-level-solution)
- [System design](#system-design)
- [Benchmark results](#benchmark-results)

---

## Problem

Turning a scanned receipt into structured data (merchant, date, address, total) used to need a bespoke OCR-and-rules pipeline. Now a single LLM call does it. The hard part is no longer extraction; it's knowing **which extractions you can trust without a human looking at them**.

The common answer is to cross-check. Run two or three models, or one model with different prompts, over the same document, and auto-accept any field where they agree. It's cheap, it needs no labels, and it feels safe.

It rests on an assumption that rarely gets tested: that **agreement predicts correctness**. That only holds if the raters' errors are independent. Configs that share a base model, a prompt style, or the same flawed input tend to make the *same* mistakes. When they do, they agree confidently while being wrong.

So this project asks three concrete questions, field by field:

1. **Discrimination:** does agreement separate correct answers from wrong ones?
2. **Calibration:** if three configs agree 75% of the time, are they right 75% of the time?
3. **Cost of a trust rule:** if I auto-accept every unanimous field, how many errors slip through per 1,000 accepted?

A second constraint shaped the work: receipts carry personal and financial data. I wanted the whole pipeline to run **locally, on a 15 W laptop CPU, with no data leaving the machine**, and to see how far a small open model could get.

## High-level solution

Treat every extraction configuration as a **rater**, like annotators in a labeling study. A rater is a `(provider, model, prompt, input)` combination. Run every rater over every document, keep every answer at field level, compare against gold labels, and measure agreement and accuracy side by side.

```
                 ┌── rater A: Qwen3.5-4B, detailed prompt, temp 0 ──┐
receipt ─────────┼── rater B: Qwen3.5-4B, terse prompt,    temp 0 ──┼──► per-field answers ──► agreement ──┐
                 └── rater C: Qwen3.5-4B, detailed prompt, temp 0.8 ┘                                     ├──► does agreement
                                                                              gold labels ──► accuracy ───┘    predict accuracy?
```

The analysis then reports:

- **AUC** of several agreement scores (pairwise agreement, consensus fraction, entropy) for predicting whether the consensus answer is correct.
- **Calibration**: accuracy per agreement bucket, and the expected calibration error of reading an agreement rate as an accuracy.
- **Trust-rule economics**: coverage, precision, and errors per 1,000 auto-accepted fields.
- **Aggregators**: each config alone vs. majority vote vs. **Dawid-Skene** (which estimates each rater's reliability without labels), with an oracle ceiling ("did *any* config get it right?") so the headroom is visible.

The raters are deliberately **not** independent. In the local run, all three share one model. One of them, the temperature-0.8 rater, is a control: same weights and same prompt, just a different sample. Agreement between it and its temperature-0 twin looks like confirmation but isn't.

The project went through two phases:

1. **Text input.** Models read the OCR text that ships with the dataset. This is the setup most extraction pipelines use.
2. **Image input.** The same local model reads the **receipt scan** directly, using its vision encoder. This phase started after digging into the text results showed that the input, not the model, was the main source of error.

## System design

### Pipeline

```
Hugging Face (SROIE)          experiment YAML
        │                     (run + list of raters)
        ▼                            │
   ingest ──► documents ◄────────────┤
                  │                  ▼
                  └──────────► extract ──► extractions ──► field_outputs
                                (configs × documents,       (one row per field,
                                 resumable)                  raw + normalized)
                                                                  │
                                            judge (optional) ◄────┤
                                                                  ▼
                                                              evaluate ──► analysis_results
                                                                           + Markdown/HTML report
```

Everything lives in **Postgres**. The CLI has one command per stage (`ingest`, `extract`, `judge`, `evaluate`, or `all`), and each stage reads its inputs from the database, so any stage can be re-run on its own.

### Data model

| Table | Grain | Purpose |
|---|---|---|
| `documents` | one per receipt | gold labels (JSONB), OCR text, image path |
| `runs` | one per experiment | dataset, schema, the full experiment YAML |
| `configs` | one per rater in a run | provider, model, prompt variant, **input mode**, sampling params, prompt hash |
| `extractions` | one per (run, config, document) | raw model output, status, latency, tokens, cost |
| `field_outputs` | one per (extraction, field) | raw value, normalized value, null flag, evidence |
| `judgments` | one per judged pair | optional LLM-judge verdicts on semantic equivalence |
| `analysis_results` | one per (run, table) | every analysis table as JSONB, including per-field verdicts |

A few design choices matter more than they look:

- **Configs are scoped to their run** (primary key `(run_id, config_id)`). Each run keeps its own copy of the settings it used, so editing a config for a new run never rewrites the history of an old one. Composite foreign keys from `extractions` and `field_outputs` make sure every answer belongs to a config that exists in that run.
- **Extraction is resumable.** A unique key on `(run_id, config_id, doc_id)` means re-running `extract` only makes the calls that are missing or failed. On a laptop where one call takes a minute, that matters.
- **Prompts are fingerprinted.** `prompt_version` is a short SHA-1 of the rendered prompt text, so two configs that share a prompt name but sent different instructions can't be silently mixed.
- **Normalization is the definition of "equal".** Each field has a type (`org`, `date`, `address`, `money`, …), and agreement and accuracy both use the same normalizer. `15/01/2019 11:05 AM` and `2019-01-15` compare equal as dates; `JALAN` and `JLN` compare equal in addresses.

### Model serving

Raters come from one of three providers behind a single interface: the Claude API, a **local llama.cpp server** (OpenAI-compatible), or a deterministic mock that makes the pipeline testable offline. The local model is **Qwen3.5-4B, quantized to Q4_K_M (2.7 GB)**, on an Intel i7-1365U with no discrete GPU.

Two settings were essential for a 4B model to work as a rater:

- **Grammar-constrained output.** `response_format: json_schema` compiles the schema into a grammar that restricts decoding, so every response that finishes is valid JSON with every field present. Without it, many "errors" would be the harness failing to parse, not the model failing to read.
- **Thinking disabled.** Qwen3-family chat templates reason by default. At ~5 tokens/s the answer often never arrives inside the token budget, so thinking is switched off explicitly.

### Adding vision

Qwen3.5-4B is natively multimodal, so image input needed no new model. The pipeline change was small: a config gains `input_mode: text | image | both`. In `image` mode the provider attaches the receipt as a base64 image and **removes the OCR text from the prompt**, so every value has to come from the scan. `both` sends the image and the OCR text together.

Getting it to run reliably on a laptop took four fixes:

1. **Load the vision projector.** The GGUF weights alone are text-only. The model's `mmproj` file (672 MB) maps image patches into model tokens and is passed to `llama-server` with `--mmproj`.
2. **Store images locally.** The dataset's image links are signed URLs that expire soon after they're fetched, so ingest now downloads each scan and records its local path.
3. **Keep each image in one batch.** A receipt becomes 550–1,024 image tokens, more than the server's default 512-token micro-batch. The micro-batch was raised to 2,048 so an image is never split.
4. **Keep the vision encoder off the integrated GPU.** By default llama.cpp ran the image encoder on the laptop's Iris Xe through Vulkan, even with all model layers pinned to the CPU. There we saw garbage output, then a crash with `ErrorDeviceLost`. With `--no-mmproj-offload`, everything runs on the CPU and the output is stable.

Image tokens are capped at 1,024 per receipt. At ~15 tokens/s CPU prefill, every extra 1,000 tokens costs over a minute.

## Benchmark results

**Setup.** 12 receipts from the SROIE test set (ICDAR 2019), 4 fields each (48 fields per config). Local Qwen3.5-4B Q4_K_M, detailed prompt unless stated. Scoring is exact match after normalization; the LLM judge was off. Twelve receipts is a small sample, so read the counts as indicative.

### Accuracy by configuration

| Run | Config | Input | Correct | Accuracy |
|---|---|---|---|---|
| text | `qwen4b-detailed-t0` | OCR text | 27/48 | 56.3% |
| text | `qwen4b-detailed-t08` | OCR text, temp 0.8 | 6/40 | 15.0% |
| text | `qwen4b-terse-t0` | OCR text, terse prompt | 5/44 | 11.4% |
| vision | `qwen4b-detailed-image` | image only | **42/48** | **87.5%** |
| vision | `qwen4b-detailed-both` | image + OCR text | **43/48** | **89.6%** |

The two weaker text configs have fewer than 48 fields because 3 of their 24 calls failed. The same weights went from 56% to 88–90% by changing only the input.

### Accuracy by field

| Field | OCR text (`detailed-t0`) | Image only | Image + OCR |
|---|---|---|---|
| company | 8/12 | 10/12 | 10/12 |
| date | 9/12 | **12/12** | **12/12** |
| address | **1/12** | **10/12** | 9/12 |
| total | 9/12 | 10/12 | **12/12** |

### Why text input was capped

The text results looked like a weak model. Tracing every field back to its input showed something else:

- **The OCR text was missing lines the labels contain.** The gold address was fully present in the OCR text for only **1 of 12** receipts, and that was the only address the model got right. Street lines ("JALAN BAYU 4") and state names were usually absent, so no model could have produced them from text.
- **Three receipts collapsed completely.** The model filled in placeholders: `Merchant`, `2023-10-15`, `123 Main St, City, State 12345`. One address field even leaked a fragment of reasoning. These three receipts account for 11 of the 21 text-mode errors.
- **Only 2 of the 21 errors were genuine misreadings** of information the model actually had.

Reading the scan removed both problems. The address lines were there, and the placeholder outputs disappeared.

### What the image-only misses look like

Six fields were scored wrong. Checked against each receipt:

| Cause | Count | Example |
|---|---|---|
| Label error | 1 | Model read the postcode as `81750`, which is correct; the gold label has the OCR typo `B1750` |
| Extra text | 2 | Company name joined with the operator printed on the next line; a registration number kept in brackets |
| Wrong line | 1 | Took `PAID AMOUNT 10.00` (cash handed over) instead of `TOTAL AMT PAYABLE 7.95` |
| Misread | 2 | `438.20` for `436.20`; "Sekyen" for "SEKYSEN" |

Counting the label error as correct, image-only accuracy is 43/48.

### Does agreement predict correctness?

| | Text run (3 configs) | Vision run (2 configs) |
|---|---|---|
| Fields where all configs agreed | 2/48 (4%) | 40/48 (83%) |
| Accuracy when unanimous | 2/2 | **40/40** |
| Accuracy when split | 28% | 50% |
| Agreement AUC (0.5 = no signal) | 0.56 | **0.955** |
| Calibration error of agreement-as-accuracy | 0.19 | 0.08 |
| Best single config | 56.3% | 89.6% |
| Majority vote | **31.3%** | 91.7% |
| Dawid-Skene | 56.3% | 87.5% |
| Oracle (any config correct) | 60.4% | 93.8% |

Two different stories:

- **With weak, unequal raters, majority vote made things worse.** In the text run, the two poor configs outvoted the good one: 31% against 56% for the best single config. Dawid-Skene recovered the 56% without labels. Its estimated reliabilities (0.999, 0.175, 0.136) tracked the true accuracies (0.56, 0.15, 0.11) closely enough to trust the right rater.
- **With strong raters, agreement became a useful signal.** In the vision run, all 40 unanimous fields were correct, and the 8 disagreements held all 4 remaining errors of the consensus. An "auto-accept when unanimous, review the rest" rule would have covered 83% of fields with zero accepted errors on this sample (95% confidence lower bound: 91%).

The caveat is the one the project exists to measure: both vision configs share one model, so their errors are correlated. A 100% unanimous-correct rate on 40 fields from same-model raters is encouraging, not proof. The next step is adding a rater from a different model family and a larger sample.

### Speed

| Input | Mean time per receipt | Prompt tokens |
|---|---|---|
| OCR text | 58–68 s (by config) | 440–870 |
| Image only | 81 s (37–132 s) | 570–1,240 |
| Image + OCR | 83 s | ~1,140 |

On a 15 W laptop CPU, reading the image took about 20% longer per receipt than the same prompt on OCR text (81 s against 68 s), for a 31-point accuracy gain. Output is short (about 90 JSON tokens), so time is dominated by processing the image, not by generation.

### Takeaways

1. **Check the input before blaming the model.** The biggest accuracy gain came from giving the model the information the labels were based on.
2. **Agreement is only as good as the raters' independence.** Among weak same-model raters, agreement was nearly useless and majority vote hurt. Among strong raters it was highly predictive, but same-model raters can still share blind spots.
3. **Use a reliability-weighted aggregator when raters differ in quality.** Dawid-Skene picked out the good rater without any labels; majority vote couldn't.
4. **Small local models are viable for receipt extraction**, with the right serving setup: constrained JSON output, thinking off, and the vision encoder kept on the CPU on integrated-GPU laptops.
5. **Look at the errors.** Of six "misses", one was a label typo and two were formatting choices; only three were real misreadings.
