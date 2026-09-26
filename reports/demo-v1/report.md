# Agreement vs. accuracy - run `demo-v1`

Dataset schema: `synthetic`. Configs: 4. Items (document x field): 1200.

## Verdict

Across **1200 items**, the configs were unanimous on **0.543** of them. Unanimous items were correct **0.943** of the time against a base rate of **0.928**; split items were correct **0.911** of the time.

Restricted to fields the label says are **populated**, unanimity is worth 0.943 (n=1033); on **null** fields it is 0.946 (n=167). Pairwise agreement discriminates correct from incorrect consensus at AUC 0.646 overall and 0.655 on populated fields only.

Reading an agreement rate *as if it were* an accuracy estimate is off by 0.250 on average (expected calibration error).

Dawid-Skene beats majority vote (0.939 vs 0.928); the best single config scores 0.857 and the ceiling (any config correct) is 0.969.

## Does agreement predict correctness?

**trust_rule_unanimous**

_Auto-accept unanimous items, review the rest. `errors_per_1000_accepted` is the error budget that policy spends._

| split | rule | n_items | coverage | p_correct_when_rule_fires | ci_low | ci_high | p_correct_when_rule_does_not_fire | base_rate | lift_over_base | errors_per_1000_accepted | errors_missed | errors_caught |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | unanimous | 1200 | 0.543 | 0.943 | 0.923 | 0.959 | 0.911 | 0.928 | 1.016 | 56.748 | 37 | 49 |
| gold_populated=True | unanimous | 1033 | 0.505 | 0.943 | 0.919 | 0.959 | 0.912 | 0.927 | 1.016 | 57.471 | 30 | 45 |
| gold_populated=False | unanimous | 167 | 0.778 | 0.946 | 0.893 | 0.974 | 0.892 | 0.934 | 1.013 | 53.846 | 7 | 4 |


**agreement_buckets**

_P(consensus correct) by agreement bucket, split by whether the label is populated._

| score | split | agreement_bucket | n | n_correct | p_correct | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pairwise_agreement | all | 0% | 3 | 0 | 0.000 | 0.000 | 0.562 |
| pairwise_agreement | all | 17% | 82 | 69 | 0.841 | 0.747 | 0.905 |
| pairwise_agreement | all | 33% | 56 | 27 | 0.482 | 0.357 | 0.610 |
| pairwise_agreement | all | 50% | 407 | 403 | 0.990 | 0.975 | 0.996 |
| pairwise_agreement | all | 100% | 652 | 615 | 0.943 | 0.923 | 0.959 |
| pairwise_agreement | gold_populated=True | 0% | 3 | 0 | 0.000 | 0.000 | 0.562 |
| pairwise_agreement | gold_populated=True | 17% | 82 | 69 | 0.841 | 0.747 | 0.905 |
| pairwise_agreement | gold_populated=True | 33% | 53 | 27 | 0.509 | 0.379 | 0.639 |
| pairwise_agreement | gold_populated=True | 50% | 373 | 370 | 0.992 | 0.977 | 0.997 |
| pairwise_agreement | gold_populated=True | 100% | 522 | 492 | 0.943 | 0.919 | 0.959 |
| pairwise_agreement | gold_populated=False | 0% | 0 | 0 | - | - | - |
| pairwise_agreement | gold_populated=False | 17% | 0 | 0 | - | - | - |
| pairwise_agreement | gold_populated=False | 33% | 3 | 0 | 0.000 | 0.000 | 0.562 |
| pairwise_agreement | gold_populated=False | 50% | 34 | 33 | 0.971 | 0.851 | 0.995 |
| pairwise_agreement | gold_populated=False | 100% | 130 | 123 | 0.946 | 0.893 | 0.974 |


**discrimination**

_AUC of each agreement score for predicting consensus correctness. 0.5 = no signal._

| score | split | n | base_rate | auc |
| --- | --- | --- | --- | --- |
| pairwise_agreement | all | 1200 | 0.928 | 0.646 |
| pairwise_agreement | gold_populated=True | 1033 | 0.927 | 0.655 |
| pairwise_agreement | gold_populated=False | 167 | 0.934 | 0.605 |
| consensus_frac | all | 1200 | 0.928 | 0.655 |
| consensus_frac | gold_populated=True | 1033 | 0.927 | 0.665 |
| consensus_frac | gold_populated=False | 167 | 0.934 | 0.605 |
| entropy (inverted) | all | 1200 | 0.928 | 0.663 |
| entropy (inverted) | gold_populated=True | 1033 | 0.927 | 0.674 |
| entropy (inverted) | gold_populated=False | 167 | 0.934 | 0.605 |


## Calibration

**calibration**

_If agreement were an accuracy estimate, `gap` would be zero._

| bin | n | mean_agreement | observed_accuracy | gap |
| --- | --- | --- | --- | --- |
| [0.00, 0.20] | 85 | 0.161 | 0.812 | 0.651 |
| [0.20, 0.40] | 56 | 0.333 | 0.482 | 0.149 |
| [0.40, 0.60] | 407 | 0.500 | 0.990 | 0.490 |
| [0.80, 1.00] | 652 | 1.000 | 0.943 | -0.057 |


## Per field

**per_field_trust**

_The same trust rule per field - averages hide fields where agreement is useless._

| field | n_items | gold_populated_rate | unanimity_rate | p_correct_when_unanimous | p_correct_when_split | base_rate | auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| address | 200 | 1.000 | 0.440 | 0.943 | 0.848 | 0.890 | 0.769 |
| company | 200 | 1.000 | 0.445 | 0.933 | 0.919 | 0.925 | 0.646 |
| date | 200 | 1.000 | 0.645 | 0.984 | 0.944 | 0.970 | 0.725 |
| phone | 200 | 0.630 | 0.605 | 0.942 | 0.924 | 0.935 | 0.595 |
| tax | 200 | 0.535 | 0.655 | 0.924 | 0.913 | 0.920 | 0.549 |
| total | 200 | 1.000 | 0.470 | 0.926 | 0.934 | 0.930 | 0.576 |


**accuracy_per_field**

| field | n | accuracy | gold_populated_rate | acc_populated | acc_null |
| --- | --- | --- | --- | --- | --- |
| address | 800 | 0.795 | 1.000 | 0.795 | - |
| company | 800 | 0.790 | 1.000 | 0.790 | - |
| date | 800 | 0.875 | 1.000 | 0.875 | - |
| phone | 800 | 0.841 | 0.630 | 0.819 | 0.878 |
| tax | 800 | 0.834 | 0.535 | 0.769 | 0.909 |
| total | 800 | 0.789 | 1.000 | 0.789 | - |


## Per prediction (is *my* answer backed by the others?)

**prediction_level_support**

| config_id | n | accuracy | auc_support_frac | p_correct_when_all_others_agree | p_correct_when_none_agree |
| --- | --- | --- | --- | --- | --- |
| mock-a | 1200 | 0.825 | 0.868 | 0.943 | 0.023 |
| mock-b | 1200 | 0.768 | 0.903 | 0.943 | 0.016 |
| mock-c | 1200 | 0.857 | 0.838 | 0.943 | 0.044 |
| mock-d | 1200 | 0.833 | 0.862 | 0.943 | 0.061 |


**support_buckets**

_Prediction-level: how often a single config's answer is right given how many other configs back it._

| support_bucket | gold_populated | n | n_correct | p_correct |
| --- | --- | --- | --- | --- |
| 0% | no | 34 | 1 | 0.029 |
| 0% | yes | 549 | 19 | 0.035 |
| 33% | no | 12 | 6 | 0.500 |
| 33% | yes | 376 | 244 | 0.649 |
| 67% | no | 102 | 99 | 0.971 |
| 67% | yes | 1119 | 1110 | 0.992 |
| 100% | no | 520 | 492 | 0.946 |
| 100% | yes | 2088 | 1968 | 0.943 |


## Aggregation: majority vote vs. Dawid-Skene

**aggregator_comparison**

_Same items, different ways of combining the configs._

| method | n | accuracy | acc_populated | acc_null |
| --- | --- | --- | --- | --- |
| dawid_skene_binary (null/populated only) | 1200 | 0.991 | 1.000 | 0.934 |
| majority_vote (null/populated only) | 1200 | 0.985 | 0.993 | 0.934 |
| oracle: any config correct | 1200 | 0.969 | 0.971 | 0.958 |
| dawid_skene_one_coin | 1200 | 0.939 | 0.937 | 0.952 |
| majority_vote | 1200 | 0.928 | 0.927 | 0.934 |
| single:mock-c | 1200 | 0.857 | 0.846 | 0.922 |
| single:mock-d | 1200 | 0.833 | 0.818 | 0.922 |
| single:mock-a | 1200 | 0.825 | 0.816 | 0.880 |
| single:mock-b | 1200 | 0.768 | 0.754 | 0.856 |


**ds_worker_quality**

_Dawid-Skene estimates reliability without labels; compare to measured accuracy._

| config_id | ds_estimated_reliability | true_accuracy |
| --- | --- | --- |
| mock-a | 0.869 | 0.825 |
| mock-b | 0.813 | 0.768 |
| mock-c | 0.887 | 0.857 |
| mock-d | 0.854 | 0.833 |


**ds_signal**

_Is the Dawid-Skene posterior a better trust signal than raw agreement?_

| score | split | n | base_rate | auc |
| --- | --- | --- | --- | --- |
| pairwise_agreement | all | 1200 | 0.939 | 0.594 |
| pairwise_agreement | gold_populated=True | 1033 | 0.937 | 0.613 |
| pairwise_agreement | gold_populated=False | 167 | 0.952 | 0.448 |
| ds_confidence | all | 1200 | 0.939 | 0.600 |
| ds_confidence | gold_populated=True | 1033 | 0.937 | 0.622 |
| ds_confidence | gold_populated=False | 167 | 0.952 | 0.443 |


**ds_binary_quality**

| parameter | value |
| --- | --- |
| mock-a|p(null->null) | 0.924 |
| mock-a|p(pop->pop) | 0.946 |
| mock-b|p(null->null) | 0.897 |
| mock-b|p(pop->pop) | 0.919 |
| mock-c|p(null->null) | 0.955 |
| mock-c|p(pop->pop) | 0.968 |
| mock-d|p(null->null) | 0.946 |
| mock-d|p(pop->pop) | 0.912 |


## Per config accuracy

**accuracy_per_config**

| config_id | n | accuracy | acc_populated | acc_null | hallucination_rate | miss_rate | wrong_value_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mock-c | 1200 | 0.857 | 0.846 | 0.922 | 0.011 | 0.026 | 0.107 |
| mock-d | 1200 | 0.833 | 0.818 | 0.922 | 0.011 | 0.073 | 0.083 |
| mock-a | 1200 | 0.825 | 0.816 | 0.880 | 0.017 | 0.047 | 0.112 |
| mock-b | 1200 | 0.768 | 0.754 | 0.856 | 0.020 | 0.070 | 0.142 |


**accuracy_per_config_field**

| config_id | field | n | accuracy | n_gold_populated | n_pred_populated | hallucination_rate | miss_rate | wrong_value_rate | acc_populated | acc_null |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mock-a | address | 200 | 0.785 | 200 | 189 | 0.000 | 0.055 | 0.160 | 0.785 | - |
| mock-b | address | 200 | 0.725 | 200 | 184 | 0.000 | 0.080 | 0.195 | 0.725 | - |
| mock-c | address | 200 | 0.850 | 200 | 192 | 0.000 | 0.040 | 0.110 | 0.850 | - |
| mock-d | address | 200 | 0.820 | 200 | 184 | 0.000 | 0.080 | 0.100 | 0.820 | - |
| mock-a | company | 200 | 0.775 | 200 | 189 | 0.000 | 0.055 | 0.170 | 0.775 | - |
| mock-b | company | 200 | 0.755 | 200 | 189 | 0.000 | 0.055 | 0.190 | 0.755 | - |
| mock-c | company | 200 | 0.805 | 200 | 194 | 0.000 | 0.030 | 0.165 | 0.805 | - |
| mock-d | company | 200 | 0.825 | 200 | 182 | 0.000 | 0.090 | 0.085 | 0.825 | - |
| mock-a | date | 200 | 0.885 | 200 | 186 | 0.000 | 0.070 | 0.045 | 0.885 | - |
| mock-b | date | 200 | 0.845 | 200 | 186 | 0.000 | 0.070 | 0.085 | 0.845 | - |
| mock-c | date | 200 | 0.905 | 200 | 195 | 0.000 | 0.025 | 0.070 | 0.905 | - |
| mock-d | date | 200 | 0.865 | 200 | 184 | 0.000 | 0.080 | 0.055 | 0.865 | - |
| mock-a | phone | 200 | 0.870 | 126 | 131 | 0.050 | 0.025 | 0.055 | 0.873 | 0.865 |
| mock-b | phone | 200 | 0.775 | 126 | 126 | 0.065 | 0.065 | 0.095 | 0.746 | 0.824 |
| mock-c | phone | 200 | 0.885 | 126 | 129 | 0.030 | 0.015 | 0.070 | 0.865 | 0.919 |
| mock-d | phone | 200 | 0.835 | 126 | 122 | 0.035 | 0.055 | 0.075 | 0.794 | 0.905 |
| mock-a | tax | 200 | 0.815 | 107 | 110 | 0.050 | 0.035 | 0.100 | 0.748 | 0.892 |
| mock-b | tax | 200 | 0.815 | 107 | 112 | 0.055 | 0.030 | 0.100 | 0.757 | 0.882 |
| mock-c | tax | 200 | 0.870 | 107 | 113 | 0.035 | 0.005 | 0.090 | 0.822 | 0.925 |
| mock-d | tax | 200 | 0.835 | 107 | 100 | 0.030 | 0.065 | 0.070 | 0.748 | 0.935 |
| mock-a | total | 200 | 0.820 | 200 | 192 | 0.000 | 0.040 | 0.140 | 0.820 | - |
| mock-b | total | 200 | 0.695 | 200 | 176 | 0.000 | 0.120 | 0.185 | 0.695 | - |
| mock-c | total | 200 | 0.825 | 200 | 192 | 0.000 | 0.040 | 0.135 | 0.825 | - |
| mock-d | total | 200 | 0.815 | 200 | 186 | 0.000 | 0.070 | 0.115 | 0.815 | - |


## Raw agreement between configs

**agreement_matrix**

_Pairwise exact-match agreement rate between configs._

| index | mock-a | mock-b | mock-c | mock-d |
| --- | --- | --- | --- | --- |
| mock-a | 1.000 | 0.717 | 0.776 | 0.745 |
| mock-b | 0.717 | 1.000 | 0.726 | 0.710 |
| mock-c | 0.776 | 0.726 | 1.000 | 0.766 |
| mock-d | 0.745 | 0.710 | 0.766 | 1.000 |


## Run cost and reliability

**extraction_stats**

| config_id | calls | errors | mean_latency_ms | input_tokens | output_tokens | cost_usd |
| --- | --- | --- | --- | --- | --- | --- |
| mock-a | 200 | 0 | 5.0000000000000000 | 19161 | 8000 | 0.000000 |
| mock-b | 200 | 0 | 5.0000000000000000 | 19161 | 8000 | 0.000000 |
| mock-c | 200 | 0 | 5.0000000000000000 | 19161 | 8000 | 0.000000 |
| mock-d | 200 | 0 | 5.0000000000000000 | 19161 | 8000 | 0.000000 |


## Method notes

- Agreement and accuracy use the *same* normalizer per field type, so neither is measured more leniently than the other.
- `null` means normalized-absent: 'N/A', '-' and '' all count as null, on both sides.
- Items are (document, field) pairs; a config that errored on a document contributes no rows for it, so item rater counts can differ.
- Dawid-Skene over open-vocabulary values uses the one-coin model (one reliability per config); the populated/null decision additionally gets a full 2x2 confusion-matrix model.

<details><summary>Experiment config</summary>

```yaml
run_id: demo-v1
dataset: synthetic
schema: synthetic
split: demo
limit: 200
concurrency: 8
notes: Offline smoke run with correlated mock errors
ingest:
  seed: 7
configs:
  -
    id: mock-a
    provider: mock
    model: mock-1
    prompt: detailed
    input_mode: text
    effort: medium
    max_tokens: 4000
    max_ocr_chars: 8000
    params:
      error_rate: 0.14
      miss_rate: 0.05
      halluc_rate: 0.08
      cluster: shared
      systematic_rate: 0.03
      seed: 17
  -
    id: mock-b
    provider: mock
    model: mock-1
    prompt: terse
    input_mode: text
    effort: medium
    max_tokens: 4000
    max_ocr_chars: 8000
    params:
      error_rate: 0.16
      miss_rate: 0.06
      halluc_rate: 0.1
      cluster: shared
      systematic_rate: 0.03
      seed: 17
  -
    id: mock-c
    provider: mock
    model: mock-2
    prompt: detailed
    input_mode: text
    effort: medium
    max_tokens: 4000
    max_ocr_chars: 8000
    params:
      error_rate: 0.12
      miss_rate: 0.04
      halluc_rate: 0.05
      cluster: solo-c
      systematic_rate: 0.03
      seed: 17
  -
    id: mock-d
    provider: mock
    model: mock-3
    prompt: evidence
    input_mode: text
    effort: medium
    max_tokens: 4000
    max_ocr_chars: 8000
    params:
      error_rate: 0.09
      miss_rate: 0.08
      halluc_rate: 0.03
      cluster: solo-d
      systematic_rate: 0.03
      seed: 17
judge:
  enabled: False
  provider: anthropic
  model: claude-opus-5
  effort: low
  max_tokens: 4000
  batch_size: 20
  only_disagreements: True
  judge_gold: True
  params:

```

</details>
