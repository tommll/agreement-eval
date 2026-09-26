# Agreement vs. accuracy - run `sroie-local-v1`

Dataset schema: `sroie`. Configs: 3. Items (document x field): 48.

## Verdict


Across **48 items**, the configs were unanimous on **0.042** of them. Unanimous items were correct **1.000** of the time against a base rate of **0.312**; split items were correct **0.283** of the time.

Restricted to fields the label says are **populated**, unanimity is worth 1.000 (n=48); on **null** fields it is - (n=0). Pairwise agreement discriminates correct from incorrect consensus at AUC 0.563 overall and 0.563 on populated fields only.

> Every labelled field in this dataset is populated (SROIE labels all four fields on every receipt), so there is no gold-null arm to split on. The null side of the question is answered here by `trust_rule_by_consensus_state` - items where the configs *decided* the field was absent - and by running the same study on FUNSD, where most fields genuinely are absent.

Reading an agreement rate *as if it were* an accuracy estimate is off by 0.187 on average (expected calibration error).

Dawid-Skene beats majority vote (0.562 vs 0.312); the best single config scores 0.562 and the ceiling (any config correct) is 0.604.

## Judge validation

Before trusting the judge to merge values, it was run on 16 controls with known answers (llamacpp/Qwen3.5-4B-Q4_K_M): identical strings, cosmetic rewrites, and values taken from different documents.

| control | n | judge accuracy |
| --- | --- | --- |
| identical | 6 | 0.500 |
| mismatched | 4 | 0.500 |
| reformatted | 6 | 1.000 |


Overall 0.688; false-equivalence rate on values that are genuinely different: 0.500.

> This judge does not pass its own controls, so its verdicts are not used in the numbers below. A judge that cannot recognise two identical strings - or that merges unrelated values - would manufacture exactly the agreement this study is trying to measure.

## Does agreement predict correctness?

**trust_rule_unanimous**

_Auto-accept unanimous items, review the rest. `errors_per_1000_accepted` is the error budget that policy spends._

| split | rule | n_items | coverage | p_correct_when_rule_fires | ci_low | ci_high | p_correct_when_rule_does_not_fire | base_rate | lift_over_base | errors_per_1000_accepted | errors_missed | errors_caught |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | unanimous | 48 | 0.042 | 1.000 | 0.342 | 1.000 | 0.283 | 0.312 | 3.200 | 0.000 | 0 | 33 |
| gold_populated=True | unanimous | 48 | 0.042 | 1.000 | 0.342 | 1.000 | 0.283 | 0.312 | 3.200 | 0.000 | 0 | 33 |


**agreement_buckets**

_P(consensus correct) by agreement bucket, split by whether the label is populated._

| score | split | agreement_bucket | n | n_correct | p_correct | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pairwise_agreement | all | 0% | 28 | 8 | 0.286 | 0.153 | 0.471 |
| pairwise_agreement | all | 33% | 18 | 5 | 0.278 | 0.125 | 0.509 |
| pairwise_agreement | all | 100% | 2 | 2 | 1.000 | 0.342 | 1.000 |
| pairwise_agreement | gold_populated=True | 0% | 28 | 8 | 0.286 | 0.153 | 0.471 |
| pairwise_agreement | gold_populated=True | 33% | 18 | 5 | 0.278 | 0.125 | 0.509 |
| pairwise_agreement | gold_populated=True | 100% | 2 | 2 | 1.000 | 0.342 | 1.000 |


**discrimination**

_AUC of each agreement score for predicting consensus correctness. 0.5 = no signal._

| score | split | n | base_rate | auc |
| --- | --- | --- | --- | --- |
| pairwise_agreement | all | 48 | 0.312 | 0.563 |
| pairwise_agreement | gold_populated=True | 48 | 0.312 | 0.563 |
| consensus_frac | all | 48 | 0.312 | 0.692 |
| consensus_frac | gold_populated=True | 48 | 0.312 | 0.692 |
| entropy (inverted) | all | 48 | 0.312 | 0.433 |
| entropy (inverted) | gold_populated=True | 48 | 0.312 | 0.433 |
| jaccard_agreement | all | 48 | 0.312 | 0.555 |
| jaccard_agreement | gold_populated=True | 48 | 0.312 | 0.555 |


**discrimination_jaccard**

_Partial credit: the outcome is a consensus within the Jaccard threshold of the label, and `jaccard_agreement` is the mean pairwise token-set Jaccard._

| score | split | n | base_rate | auc |
| --- | --- | --- | --- | --- |
| jaccard_agreement | all | 48 | 0.333 | 0.532 |
| jaccard_agreement | gold_populated=True | 48 | 0.333 | 0.532 |
| pairwise_agreement | all | 48 | 0.333 | 0.541 |
| pairwise_agreement | gold_populated=True | 48 | 0.333 | 0.541 |


## Calibration

**calibration**

_If agreement were an accuracy estimate, `gap` would be zero._

| bin | n | mean_agreement | observed_accuracy | gap |
| --- | --- | --- | --- | --- |
| [0.00, 0.20] | 28 | 0.000 | 0.286 | 0.286 |
| [0.20, 0.40] | 18 | 0.333 | 0.278 | -0.056 |
| [0.80, 1.00] | 2 | 1.000 | 1.000 | 0.000 |


## Per field

**per_field_trust**

_The same trust rule per field - averages hide fields where agreement is useless._

| field | n_items | gold_populated_rate | unanimity_rate | p_correct_when_unanimous | p_correct_when_split | base_rate | auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| address | 12 | 1.000 | 0.000 | - | 0.000 | 0.000 | - |
| company | 12 | 1.000 | 0.000 | - | 0.417 | 0.417 | 0.629 |
| date | 12 | 1.000 | 0.083 | 1.000 | 0.364 | 0.417 | 0.300 |
| total | 12 | 1.000 | 0.083 | 1.000 | 0.364 | 0.417 | 0.571 |


**accuracy_per_field**

| field | n | accuracy | accuracy_jaccard | mean_jaccard | gold_populated_rate | acc_populated | acc_null |
| --- | --- | --- | --- | --- | --- | --- | --- |
| address | 33 | 0.030 | 0.061 | 0.206 | 1.000 | 0.030 | - |
| company | 33 | 0.333 | 0.333 | 0.333 | 1.000 | 0.333 | - |
| date | 33 | 0.394 | 0.394 | 0.394 | 1.000 | 0.394 | - |
| total | 33 | 0.394 | 0.394 | 0.394 | 1.000 | 0.394 | - |


## Per prediction (is *my* answer backed by the others?)

**prediction_level_support**

| config_id | n | accuracy | auc_support_frac | p_correct_when_all_others_agree | p_correct_when_none_agree |
| --- | --- | --- | --- | --- | --- |
| qwen4b-detailed-t0 | 48 | 0.562 | 0.541 | 1.000 | 0.541 |
| qwen4b-detailed-t08 | 40 | 0.150 | 0.716 | 1.000 | 0.083 |
| qwen4b-terse-t0 | 44 | 0.114 | 0.923 | 1.000 | 0.000 |


**support_buckets**

_Prediction-level: how often a single config's answer is right given how many other configs back it._

| support_bucket | gold_populated | n | n_correct | p_correct |
| --- | --- | --- | --- | --- |
| 0% | yes | 90 | 22 | 0.244 |
| 50% | yes | 36 | 10 | 0.278 |
| 100% | yes | 6 | 6 | 1.000 |


## Aggregation: majority vote vs. Dawid-Skene

**aggregator_comparison**

_Same items, different ways of combining the configs._

| method | n | accuracy | acc_populated | acc_null |
| --- | --- | --- | --- | --- |
| dawid_skene_binary (null/populated only) | 48 | 1.000 | 1.000 | - |
| majority_vote (null/populated only) | 48 | 0.792 | 0.792 | - |
| oracle: any config correct | 48 | 0.604 | 0.604 | - |
| single:qwen4b-detailed-t0 | 48 | 0.562 | 0.562 | - |
| dawid_skene_one_coin | 48 | 0.562 | 0.562 | - |
| majority_vote | 48 | 0.312 | 0.312 | - |
| single:qwen4b-detailed-t08 | 40 | 0.150 | 0.150 | - |
| single:qwen4b-terse-t0 | 44 | 0.114 | 0.114 | - |


**ds_worker_quality**

_Dawid-Skene estimates reliability without labels; compare to measured accuracy._

| config_id | ds_estimated_reliability | true_accuracy |
| --- | --- | --- |
| qwen4b-detailed-t0 | 0.999 | 0.562 |
| qwen4b-detailed-t08 | 0.175 | 0.150 |
| qwen4b-terse-t0 | 0.136 | 0.114 |


**ds_signal**

_Is the Dawid-Skene posterior a better trust signal than raw agreement?_

| score | split | n | base_rate | auc |
| --- | --- | --- | --- | --- |
| pairwise_agreement | all | 48 | 0.562 | 0.586 |
| pairwise_agreement | gold_populated=True | 48 | 0.562 | 0.586 |
| ds_confidence | all | 48 | 0.562 | 0.635 |
| ds_confidence | gold_populated=True | 48 | 0.562 | 0.635 |


**ds_binary_quality**

| parameter | value |
| --- | --- |
| qwen4b-detailed-t08|p(null->null) | 0.500 |
| qwen4b-detailed-t08|p(pop->pop) | 0.500 |
| qwen4b-detailed-t0|p(null->null) | 0.500 |
| qwen4b-detailed-t0|p(pop->pop) | 0.940 |
| qwen4b-terse-t0|p(null->null) | 0.500 |
| qwen4b-terse-t0|p(pop->pop) | 0.435 |


## Per config accuracy

**accuracy_per_config**

| config_id | n | accuracy | accuracy_jaccard | mean_jaccard | acc_populated | acc_null | hallucination_rate | miss_rate | wrong_value_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen4b-detailed-t0 | 48 | 0.562 | 0.583 | 0.659 | 0.562 | - | 0.000 | 0.042 | 0.396 |
| qwen4b-detailed-t08 | 40 | 0.150 | 0.150 | 0.171 | 0.150 | - | 0.000 | 0.500 | 0.350 |
| qwen4b-terse-t0 | 44 | 0.114 | 0.114 | 0.121 | 0.114 | - | 0.000 | 0.568 | 0.318 |


**accuracy_per_config_field**

| config_id | field | n | accuracy | accuracy_jaccard | mean_jaccard | n_gold_populated | n_pred_populated | hallucination_rate | miss_rate | wrong_value_rate | acc_populated | acc_null |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen4b-detailed-t0 | address | 12 | 0.083 | 0.167 | 0.469 | 12 | 12 | 0.000 | 0.000 | 0.917 | 0.083 | - |
| qwen4b-detailed-t08 | address | 10 | 0.000 | 0.000 | 0.083 | 10 | 6 | 0.000 | 0.400 | 0.600 | 0.000 | - |
| qwen4b-terse-t0 | address | 11 | 0.000 | 0.000 | 0.030 | 11 | 7 | 0.000 | 0.364 | 0.636 | 0.000 | - |
| qwen4b-detailed-t0 | company | 12 | 0.667 | 0.667 | 0.667 | 12 | 11 | 0.000 | 0.083 | 0.250 | 0.667 | - |
| qwen4b-detailed-t08 | company | 10 | 0.200 | 0.200 | 0.200 | 10 | 5 | 0.000 | 0.500 | 0.300 | 0.200 | - |
| qwen4b-terse-t0 | company | 11 | 0.091 | 0.091 | 0.091 | 11 | 8 | 0.000 | 0.273 | 0.636 | 0.091 | - |
| qwen4b-detailed-t0 | date | 12 | 0.750 | 0.750 | 0.750 | 12 | 11 | 0.000 | 0.083 | 0.167 | 0.750 | - |
| qwen4b-detailed-t08 | date | 10 | 0.200 | 0.200 | 0.200 | 10 | 4 | 0.000 | 0.600 | 0.200 | 0.200 | - |
| qwen4b-terse-t0 | date | 11 | 0.182 | 0.182 | 0.182 | 11 | 2 | 0.000 | 0.818 | 0.000 | 0.182 | - |
| qwen4b-detailed-t0 | total | 12 | 0.750 | 0.750 | 0.750 | 12 | 12 | 0.000 | 0.000 | 0.250 | 0.750 | - |
| qwen4b-detailed-t08 | total | 10 | 0.200 | 0.200 | 0.200 | 10 | 5 | 0.000 | 0.500 | 0.300 | 0.200 | - |
| qwen4b-terse-t0 | total | 11 | 0.182 | 0.182 | 0.182 | 11 | 2 | 0.000 | 0.818 | 0.000 | 0.182 | - |


## Raw agreement between configs

**agreement_matrix**

_Pairwise exact-match agreement rate between configs._

| index | qwen4b-detailed-t0 | qwen4b-detailed-t08 | qwen4b-terse-t0 |
| --- | --- | --- | --- |
| qwen4b-detailed-t0 | 1.000 | 0.175 | 0.136 |
| qwen4b-detailed-t08 | 0.175 | 1.000 | 0.306 |
| qwen4b-terse-t0 | 0.136 | 0.306 | 1.000 |


**jaccard_matrix**

_Mean pairwise token-set Jaccard between configs (atomic fields score 1 or 0)._

| index | qwen4b-detailed-t0 | qwen4b-detailed-t08 | qwen4b-terse-t0 |
| --- | --- | --- | --- |
| qwen4b-detailed-t0 | 1.000 | 0.175 | 0.152 |
| qwen4b-detailed-t08 | 0.175 | 1.000 | 0.306 |
| qwen4b-terse-t0 | 0.152 | 0.306 | 1.000 |


## Run cost and reliability

**extraction_stats**

| config_id | calls | errors | mean_latency_ms | input_tokens | output_tokens | cost_usd |
| --- | --- | --- | --- | --- | --- | --- |
| qwen4b-detailed-t0 | 12 | 0 | 67717.166666666667 | 7048 | 983 | 0.000000 |
| qwen4b-detailed-t08 | 12 | 2 | 60388.833333333333 | 7048 | 2154 | 0.000000 |
| qwen4b-terse-t0 | 12 | 1 | 58315.000000000000 | 5128 | 1796 | 0.000000 |


**local_speed**

_llama.cpp's own timings for locally served configs. Prefill is compute-bound and decode is memory-bandwidth-bound, so they move independently._

| config_id | calls | mean_prompt_tokens | prefill_tok_s | mean_output_tokens | decode_tok_s | mean_latency_s |
| --- | --- | --- | --- | --- | --- | --- |
| qwen4b-detailed-t0 | 12 | 389.083 | 16.727 | 81.917 | 2.988 | 67.7171666666666670 |
| qwen4b-detailed-t08 | 10 | 245.000 | 14.495 | 135.400 | 3.521 | 53.0986000000000000 |
| qwen4b-terse-t0 | 11 | 409.273 | 30.756 | 126.909 | 3.427 | 54.9932727272727270 |


## Method notes

- Agreement and accuracy use the *same* normalizer per field type, so neither is measured more leniently than the other.
- `null` means normalized-absent: 'N/A', '-' and '' all count as null, on both sides.
- Items are (document, field) pairs; a config that errored on a document contributes no rows for it, so item rater counts can differ.
- Dawid-Skene over open-vocabulary values uses the one-coin model (one reliability per config); the populated/null decision additionally gets a full 2x2 confusion-matrix model.

<details><summary>Experiment config</summary>

```yaml
run_id: sroie-local-v1
dataset: sroie
schema: sroie
split: test
limit: 12
concurrency: 1
notes: Local llama.cpp raters (Qwen3.5-4B Q4_K_M), OCR-text input
ingest:
  source: api
  download_images: False
configs:
  -
    id: qwen4b-detailed-t0
    provider: llamacpp
    model: Qwen3.5-4B-Q4_K_M
    prompt: detailed
    input_mode: text
    effort: None
    max_tokens: 400
    max_ocr_chars: 8000
    params:
      base_url: http://localhost:8080/v1
      temperature: 0.0
      seed: 1
  -
    id: qwen4b-terse-t0
    provider: llamacpp
    model: Qwen3.5-4B-Q4_K_M
    prompt: terse
    input_mode: text
    effort: None
    max_tokens: 400
    max_ocr_chars: 8000
    params:
      base_url: http://localhost:8080/v1
      temperature: 0.0
      seed: 1
  -
    id: qwen4b-detailed-t08
    provider: llamacpp
    model: Qwen3.5-4B-Q4_K_M
    prompt: detailed
    input_mode: text
    effort: None
    max_tokens: 400
    max_ocr_chars: 8000
    params:
      base_url: http://localhost:8080/v1
      temperature: 0.8
      top_p: 0.95
      seed: 7
judge:
  enabled: False
  provider: llamacpp
  model: Qwen3.5-4B-Q4_K_M
  effort: None
  max_tokens: 600
  batch_size: 4
  only_disagreements: True
  judge_gold: True
  include_rationale: False
  params:
    base_url: http://localhost:8080/v1
    temperature: 0.0
    seed: 1
```

</details>
