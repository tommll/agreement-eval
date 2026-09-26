# Agreement vs. accuracy - run `sroie-local-vision-v1`

Dataset schema: `sroie`. Configs: 4. Items (document x field): 48.

## Verdict


Across **48 items**, the configs were unanimous on **0.542** of them. Unanimous items were correct **1.000** of the time against a base rate of **0.938**; split items were correct **0.864** of the time.

Restricted to fields the label says are **populated**, unanimity is worth 1.000 (n=48); on **null** fields it is - (n=0). Pairwise agreement discriminates correct from incorrect consensus at AUC 0.967 overall and 0.967 on populated fields only.

> Every labelled field in this dataset is populated (SROIE labels all four fields on every receipt), so there is no gold-null arm to split on. The null side of the question is answered here by `trust_rule_by_consensus_state` - items where the configs *decided* the field was absent - and by running the same study on FUNSD, where most fields genuinely are absent.

Reading an agreement rate *as if it were* an accuracy estimate is off by 0.201 on average (expected calibration error).

Dawid-Skene does not beat majority vote (0.938 vs 0.938); the best single config scores 0.938 and the ceiling (any config correct) is 0.938.

## Does agreement predict correctness?

**trust_rule_unanimous**

_Auto-accept unanimous items, review the rest. `errors_per_1000_accepted` is the error budget that policy spends._

| split | rule | n_items | coverage | p_correct_when_rule_fires | ci_low | ci_high | p_correct_when_rule_does_not_fire | base_rate | lift_over_base | errors_per_1000_accepted | errors_missed | errors_caught |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | unanimous | 48 | 0.542 | 1.000 | 0.871 | 1.000 | 0.864 | 0.938 | 1.067 | 0.000 | 0 | 3 |
| gold_populated=True | unanimous | 48 | 0.542 | 1.000 | 0.871 | 1.000 | 0.864 | 0.938 | 1.067 | 0.000 | 0 | 3 |


**agreement_buckets**

_P(consensus correct) by agreement bucket, split by whether the label is populated._

| score | split | agreement_bucket | n | n_correct | p_correct | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pairwise_agreement | all | 17% | 4 | 2 | 0.500 | 0.150 | 0.850 |
| pairwise_agreement | all | 33% | 2 | 1 | 0.500 | 0.095 | 0.905 |
| pairwise_agreement | all | 50% | 16 | 16 | 1.000 | 0.806 | 1.000 |
| pairwise_agreement | all | 100% | 26 | 26 | 1.000 | 0.871 | 1.000 |
| pairwise_agreement | gold_populated=True | 17% | 4 | 2 | 0.500 | 0.150 | 0.850 |
| pairwise_agreement | gold_populated=True | 33% | 2 | 1 | 0.500 | 0.095 | 0.905 |
| pairwise_agreement | gold_populated=True | 50% | 16 | 16 | 1.000 | 0.806 | 1.000 |
| pairwise_agreement | gold_populated=True | 100% | 26 | 26 | 1.000 | 0.871 | 1.000 |


**discrimination**

_AUC of each agreement score for predicting consensus correctness. 0.5 = no signal._

| score | split | n | base_rate | auc |
| --- | --- | --- | --- | --- |
| pairwise_agreement | all | 48 | 0.938 | 0.967 |
| pairwise_agreement | gold_populated=True | 48 | 0.938 | 0.967 |
| consensus_frac | all | 48 | 0.938 | 0.967 |
| consensus_frac | gold_populated=True | 48 | 0.938 | 0.967 |
| entropy (inverted) | all | 48 | 0.938 | 0.967 |
| entropy (inverted) | gold_populated=True | 48 | 0.938 | 0.967 |
| jaccard_agreement | all | 48 | 0.938 | 0.759 |
| jaccard_agreement | gold_populated=True | 48 | 0.938 | 0.759 |


**discrimination_jaccard**

_Partial credit: the outcome is a consensus within the Jaccard threshold of the label, and `jaccard_agreement` is the mean pairwise token-set Jaccard._

| score | split | n | base_rate | auc |
| --- | --- | --- | --- | --- |
| jaccard_agreement | all | 48 | 0.979 | 0.670 |
| jaccard_agreement | gold_populated=True | 48 | 0.979 | 0.670 |
| pairwise_agreement | all | 48 | 0.979 | 0.904 |
| pairwise_agreement | gold_populated=True | 48 | 0.979 | 0.904 |


## Calibration

**calibration**

_If agreement were an accuracy estimate, `gap` would be zero._

| bin | n | mean_agreement | observed_accuracy | gap |
| --- | --- | --- | --- | --- |
| [0.00, 0.20] | 4 | 0.167 | 0.500 | 0.333 |
| [0.20, 0.40] | 2 | 0.333 | 0.500 | 0.167 |
| [0.40, 0.60] | 16 | 0.500 | 1.000 | 0.500 |
| [0.80, 1.00] | 26 | 1.000 | 1.000 | 0.000 |


## Per field

**per_field_trust**

_The same trust rule per field - averages hide fields where agreement is useless._

| field | n_items | gold_populated_rate | unanimity_rate | p_correct_when_unanimous | p_correct_when_split | base_rate | auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| address | 12 | 1.000 | 0.000 | - | 0.833 | 0.833 | 0.950 |
| company | 12 | 1.000 | 0.667 | 1.000 | 0.750 | 0.917 | 0.955 |
| date | 12 | 1.000 | 1.000 | 1.000 | - | 1.000 | - |
| total | 12 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | - |


**accuracy_per_field**

| field | n | accuracy | accuracy_jaccard | mean_jaccard | gold_populated_rate | acc_populated | acc_null |
| --- | --- | --- | --- | --- | --- | --- | --- |
| address | 48 | 0.604 | 0.792 | 0.891 | 1.000 | 0.604 | - |
| company | 48 | 0.833 | 0.833 | 0.914 | 1.000 | 0.833 | - |
| date | 48 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| total | 48 | 0.854 | 0.854 | 0.854 | 1.000 | 0.854 | - |


## Per prediction (is *my* answer backed by the others?)

**prediction_level_support**

| config_id | n | accuracy | auc_support_frac | p_correct_when_all_others_agree | p_correct_when_none_agree |
| --- | --- | --- | --- | --- | --- |
| gemma4-e4b-detailed-image | 48 | 0.583 | 1.000 | 1.000 | 0.000 |
| qwen27b-detailed-image | 48 | 0.938 | 0.967 | 1.000 | - |
| qwen4b-detailed-both | 48 | 0.896 | 0.995 | 1.000 | 0.000 |
| qwen4b-detailed-image | 48 | 0.875 | 0.992 | 1.000 | 0.000 |


**support_buckets**

_Prediction-level: how often a single config's answer is right given how many other configs back it._

| support_bucket | gold_populated | n | n_correct | p_correct |
| --- | --- | --- | --- | --- |
| 0% | yes | 24 | 0 | 0.000 |
| 33% | yes | 16 | 6 | 0.375 |
| 67% | yes | 48 | 48 | 1.000 |
| 100% | yes | 104 | 104 | 1.000 |


## Aggregation: majority vote vs. Dawid-Skene

**aggregator_comparison**

_Same items, different ways of combining the configs._

| method | n | accuracy | acc_populated | acc_null |
| --- | --- | --- | --- | --- |
| dawid_skene_binary (null/populated only) | 48 | 1.000 | 1.000 | - |
| majority_vote (null/populated only) | 48 | 1.000 | 1.000 | - |
| dawid_skene_one_coin | 48 | 0.938 | 0.938 | - |
| single:qwen27b-detailed-image | 48 | 0.938 | 0.938 | - |
| majority_vote | 48 | 0.938 | 0.938 | - |
| oracle: any config correct | 48 | 0.938 | 0.938 | - |
| single:qwen4b-detailed-both | 48 | 0.896 | 0.896 | - |
| single:qwen4b-detailed-image | 48 | 0.875 | 0.875 | - |
| single:gemma4-e4b-detailed-image | 48 | 0.583 | 0.583 | - |


**ds_worker_quality**

_Dawid-Skene estimates reliability without labels; compare to measured accuracy._

| config_id | ds_estimated_reliability | true_accuracy |
| --- | --- | --- |
| gemma4-e4b-detailed-image | 0.604 | 0.583 |
| qwen27b-detailed-image | 0.999 | 0.938 |
| qwen4b-detailed-both | 0.896 | 0.896 |
| qwen4b-detailed-image | 0.917 | 0.875 |


**ds_signal**

_Is the Dawid-Skene posterior a better trust signal than raw agreement?_

| score | split | n | base_rate | auc |
| --- | --- | --- | --- | --- |
| pairwise_agreement | all | 48 | 0.938 | 0.967 |
| pairwise_agreement | gold_populated=True | 48 | 0.938 | 0.967 |
| ds_confidence | all | 48 | 0.938 | 0.967 |
| ds_confidence | gold_populated=True | 48 | 0.938 | 0.967 |


**ds_binary_quality**

| parameter | value |
| --- | --- |
| gemma4-e4b-detailed-image|p(null->null) | 0.500 |
| gemma4-e4b-detailed-image|p(pop->pop) | 0.980 |
| qwen27b-detailed-image|p(null->null) | 0.500 |
| qwen27b-detailed-image|p(pop->pop) | 0.980 |
| qwen4b-detailed-both|p(null->null) | 0.500 |
| qwen4b-detailed-both|p(pop->pop) | 0.980 |
| qwen4b-detailed-image|p(null->null) | 0.500 |
| qwen4b-detailed-image|p(pop->pop) | 0.980 |


## Per config accuracy

**accuracy_per_config**

| config_id | n | accuracy | accuracy_jaccard | mean_jaccard | acc_populated | acc_null | hallucination_rate | miss_rate | wrong_value_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen27b-detailed-image | 48 | 0.938 | 0.979 | 0.988 | 0.938 | - | 0.000 | 0.000 | 0.062 |
| qwen4b-detailed-both | 48 | 0.896 | 0.917 | 0.954 | 0.896 | - | 0.000 | 0.000 | 0.104 |
| qwen4b-detailed-image | 48 | 0.875 | 0.917 | 0.939 | 0.875 | - | 0.000 | 0.000 | 0.125 |
| gemma4-e4b-detailed-image | 48 | 0.583 | 0.667 | 0.778 | 0.583 | - | 0.000 | 0.000 | 0.417 |


**accuracy_per_config_field**

| config_id | field | n | accuracy | accuracy_jaccard | mean_jaccard | n_gold_populated | n_pred_populated | hallucination_rate | miss_rate | wrong_value_rate | acc_populated | acc_null |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma4-e4b-detailed-image | address | 12 | 0.000 | 0.333 | 0.692 | 12 | 12 | 0.000 | 0.000 | 1.000 | 0.000 | - |
| qwen27b-detailed-image | address | 12 | 0.833 | 1.000 | 0.976 | 12 | 12 | 0.000 | 0.000 | 0.167 | 0.833 | - |
| qwen4b-detailed-both | address | 12 | 0.750 | 0.833 | 0.922 | 12 | 12 | 0.000 | 0.000 | 0.250 | 0.750 | - |
| qwen4b-detailed-image | address | 12 | 0.833 | 1.000 | 0.976 | 12 | 12 | 0.000 | 0.000 | 0.167 | 0.833 | - |
| gemma4-e4b-detailed-image | company | 12 | 0.750 | 0.750 | 0.837 | 12 | 12 | 0.000 | 0.000 | 0.250 | 0.750 | - |
| qwen27b-detailed-image | company | 12 | 0.917 | 0.917 | 0.976 | 12 | 12 | 0.000 | 0.000 | 0.083 | 0.917 | - |
| qwen4b-detailed-both | company | 12 | 0.833 | 0.833 | 0.893 | 12 | 12 | 0.000 | 0.000 | 0.167 | 0.833 | - |
| qwen4b-detailed-image | company | 12 | 0.833 | 0.833 | 0.948 | 12 | 12 | 0.000 | 0.000 | 0.167 | 0.833 | - |
| gemma4-e4b-detailed-image | date | 12 | 1.000 | 1.000 | 1.000 | 12 | 12 | 0.000 | 0.000 | 0.000 | 1.000 | - |
| qwen27b-detailed-image | date | 12 | 1.000 | 1.000 | 1.000 | 12 | 12 | 0.000 | 0.000 | 0.000 | 1.000 | - |
| qwen4b-detailed-both | date | 12 | 1.000 | 1.000 | 1.000 | 12 | 12 | 0.000 | 0.000 | 0.000 | 1.000 | - |
| qwen4b-detailed-image | date | 12 | 1.000 | 1.000 | 1.000 | 12 | 12 | 0.000 | 0.000 | 0.000 | 1.000 | - |
| gemma4-e4b-detailed-image | total | 12 | 0.583 | 0.583 | 0.583 | 12 | 12 | 0.000 | 0.000 | 0.417 | 0.583 | - |
| qwen27b-detailed-image | total | 12 | 1.000 | 1.000 | 1.000 | 12 | 12 | 0.000 | 0.000 | 0.000 | 1.000 | - |
| qwen4b-detailed-both | total | 12 | 1.000 | 1.000 | 1.000 | 12 | 12 | 0.000 | 0.000 | 0.000 | 1.000 | - |
| qwen4b-detailed-image | total | 12 | 0.833 | 0.833 | 0.833 | 12 | 12 | 0.000 | 0.000 | 0.167 | 0.833 | - |


## Raw agreement between configs

**agreement_matrix**

_Pairwise exact-match agreement rate between configs._

| index | gemma4-e4b-detailed-image | qwen27b-detailed-image | qwen4b-detailed-both | qwen4b-detailed-image |
| --- | --- | --- | --- | --- |
| gemma4-e4b-detailed-image | 1.000 | 0.604 | 0.625 | 0.542 |
| qwen27b-detailed-image | 0.604 | 1.000 | 0.896 | 0.917 |
| qwen4b-detailed-both | 0.625 | 0.896 | 1.000 | 0.833 |
| qwen4b-detailed-image | 0.542 | 0.917 | 0.833 | 1.000 |


**jaccard_matrix**

_Mean pairwise token-set Jaccard between configs (atomic fields score 1 or 0)._

| index | gemma4-e4b-detailed-image | qwen27b-detailed-image | qwen4b-detailed-both | qwen4b-detailed-image |
| --- | --- | --- | --- | --- |
| gemma4-e4b-detailed-image | 1.000 | 0.784 | 0.792 | 0.754 |
| qwen27b-detailed-image | 0.784 | 1.000 | 0.955 | 0.949 |
| qwen4b-detailed-both | 0.792 | 0.955 | 1.000 | 0.907 |
| qwen4b-detailed-image | 0.754 | 0.949 | 0.907 | 1.000 |


## Run cost and reliability

**extraction_stats**

| config_id | calls | errors | mean_latency_ms | input_tokens | output_tokens | cost_usd |
| --- | --- | --- | --- | --- | --- | --- |
| gemma4-e4b-detailed-image | 12 | 0 | 56177.416666666667 | 7428 | 1062 | 0.000000 |
| qwen27b-detailed-image | 12 | 0 | 507447.750000000000 | 11510 | 1077 | 0.000000 |
| qwen4b-detailed-both | 12 | 0 | 83192.250000000000 | 15750 | 1077 | 0.000000 |
| qwen4b-detailed-image | 12 | 0 | 80770.416666666667 | 11510 | 1115 | 0.000000 |


**local_speed**

_llama.cpp's own timings for locally served configs. Prefill is compute-bound and decode is memory-bandwidth-bound, so they move independently._

| config_id | calls | mean_prompt_tokens | prefill_tok_s | mean_output_tokens | decode_tok_s | mean_latency_s |
| --- | --- | --- | --- | --- | --- | --- |
| gemma4-e4b-detailed-image | 12 | 415.500 | 10.973 | 88.500 | 6.089 | 56.1774166666666670 |
| qwen27b-detailed-image | 12 | 731.250 | 1.659 | 89.750 | 1.274 | 507.4477500000000000 |
| qwen4b-detailed-both | 12 | 1138.5 | 17.141 | 89.750 | 6.459 | 83.1922500000000000 |
| qwen4b-detailed-image | 12 | 814.167 | 14.048 | 92.917 | 5.763 | 80.7704166666666670 |


## Method notes

- Agreement and accuracy use the *same* normalizer per field type, so neither is measured more leniently than the other.
- `null` means normalized-absent: 'N/A', '-' and '' all count as null, on both sides.
- Items are (document, field) pairs; a config that errored on a document contributes no rows for it, so item rater counts can differ.
- Dawid-Skene over open-vocabulary values uses the one-coin model (one reliability per config); the populated/null decision additionally gets a full 2x2 confusion-matrix model.

<details><summary>Experiment config</summary>

```yaml
run_id: sroie-local-vision-v1
dataset: sroie
schema: sroie
split: test
limit: 12
concurrency: 1
notes: Local llama.cpp vision raters (Qwen3.5-4B Q4_K_M + mmproj), image input
ingest:
  source: api
  download_images: True
configs:
  -
    id: qwen4b-detailed-image
    provider: llamacpp
    model: Qwen3.5-4B-Q4_K_M
    prompt: detailed
    input_mode: image
    effort: None
    max_tokens: 400
    max_ocr_chars: 8000
    params:
      base_url: http://localhost:8082/v1
      temperature: 0.0
      seed: 1
  -
    id: qwen4b-detailed-both
    provider: llamacpp
    model: Qwen3.5-4B-Q4_K_M
    prompt: detailed
    input_mode: both
    effort: None
    max_tokens: 400
    max_ocr_chars: 8000
    params:
      base_url: http://localhost:8082/v1
      temperature: 0.0
      seed: 1
  -
    id: gemma4-e4b-detailed-image
    provider: llamacpp
    model: gemma-4-E4B-it-Q4_0
    prompt: detailed
    input_mode: image
    effort: None
    max_tokens: 400
    max_ocr_chars: 8000
    params:
      base_url: http://localhost:8083/v1
      temperature: 0.0
      seed: 1
  -
    id: qwen27b-detailed-image
    provider: llamacpp
    model: Qwen3.5-27B-IQ4_XS
    prompt: detailed
    input_mode: image
    effort: None
    max_tokens: 400
    max_ocr_chars: 8000
    params:
      base_url: http://localhost:8084/v1
      temperature: 0.0
      seed: 1
      timeout: 2400
judge:
  enabled: False
  provider: anthropic
  model: claude-opus-5
  effort: low
  max_tokens: 4000
  batch_size: 20
  only_disagreements: True
  judge_gold: True
  include_rationale: True
  params:

```

</details>
