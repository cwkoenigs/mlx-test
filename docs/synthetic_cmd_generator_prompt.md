# Prompt: Generate synthetic CMD training and one-year-forward arrays

> Transcribed on 2026-10-07 from photographs of `synthetic_cmd_generator_prompt.md` (16 screen photos).
> Every list and number below was cross-checked: the seven Appendix C lists count out to exactly
> 255 / 324 / 18 / 56 / 38 / 19 / 56 entries, and the Appendix A/B numbers reproduce the document's
> own totals (code-ID directory 34.205 / 33.632 GiB, split totals 154.07 / 151.34 GiB).
> One gap: the forward-test `outcome_months` list was cut off after `2024-09-01`; twelve months
> `2024-01-01 .. 2024-12-01` are assumed. Implemented by `scripts/generate_synthetic_cmd_arrays.py`
> (contract in `scripts/synthetic_cmd_spec.py`).

You are working in the `member-level-predict-deep-learning` repository on a Mac. Write Python code that produces entirely synthetic CMD arrays so I can experiment with model architecture, PyTorch loading, memory pressure, disk I/O, and Apple Silicon hardware optimization. This document is the complete input specification. You have no real array files, metadata files, external manifests, or attachments. Do not request or copy real member data.

## Deliverables

1. Implement `scripts/generate_synthetic_cmd_arrays.py` with `--output-root`, `--scale`, `--seed`, `--dry-run`, `--resume`, `--full`, `--pattern {fast-io,structured}`, and `--stage-chunks` options. Default to a small scaled run. Require `--full` before writing the exact populations below.
2. Write assembled output to `<output-root>/arrays/train__CMD/` and `<output-root>/arrays/one_year_forward_test__CMD/`. If the existing project is installed, `DATA_ROOT=<output-root>` should let `MemberDataset` find these files. Check whether runtime configuration overrides `DATA_ROOT` before testing.
3. Generate `metadata.json`, all 17 root `.npy` arrays in the table below, 50 code families as 100 individual `.npy` files under `code_ids/`, `partial_exposure_indices.json`, `member_strat_indices.json`, and synthetic `member_labels.parquet`. The latter has two `large_string` columns: `PERSON_ID` and `PRODUCT_LINE_OF_BUSINESS`. Invent IDs such as `SYN_TRAIN_000000001`; every LOB value is `CMD`. No real identifiers or values may appear in the output.
4. Add a validation subcommand or companion script. Run only a small scaled smoke test and report its results. Give me the exact full-run command, but do not start full generation without my explicit request.

## Full-scale contract

| Split directory | Members N | N x 105 code bags per family | Root `.npy` files | Code `.npy` files | Total files | Logical size, all files |
|---|---|---|---|---|---|---|
| `train__CMD` | 863,154 | 90,631,170 | 17 | 100 | 121 | 154.07 GiB |
| `one_year_forward_test__CMD` | 847,568 | 88,994,640 | 17 | 100 | 121 | 151.34 GiB |

The combined assembled dataset is approximately 305.41 GiB, before free-space headroom. `X_seq.npy` alone is 109.391 GiB for train and 107.416 GiB for forward test. Both are float32 `[N, 105, 324]`: 105 historical time bins and 324 sequential features. `X_static.npy` is float32 `[N, 255]`. Future cost targets are monthly allowed cost, not pre-cumulated cost: `y.npy` is float32 `[N, 12]`. Train and forward test are separate cohorts. The training loader may then partition train members into an internal 80/20 train/validation split; forward test must remain held out.

Use the exact root array shapes and dtypes in Appendix A. Use the exact 50 code-family names, vocabulary sizes, and full-scale flat-ID counts in Appendix B. Use the feature-name lists in Appendix C in metadata, preserving their order. At scaled sizes, change only leading member dimensions, bag counts, flat-ID counts, split indices, and sizes implied by those changes. Keep feature widths, time lengths, dtypes, family names, and vocabularies unchanged.

## Generation rules

- Use `numpy.lib.format.open_memmap` or an equivalent bounded-memory `.npy` writer. Write member blocks; never allocate an entire large array in memory. Keep generation peak RSS configurable, with a default below 1 GiB excluding OS page cache. Make block seeds deterministic per split, file, and block so `--resume` is repeatable.
- A full run must physically write representative values across every file. A sparse file filled only by truncation or a mostly untouched zero memmap does not simulate disk I/O. Report both logical bytes (`st_size`) and allocated bytes (`st_blocks x 512`) and flag a large discrepancy.
- `fast-io` may use cheap pseudo-random values to stress storage and the loader. `structured` should create mixed continuous, sparse, count-like, and categorical features; zero-inflated heavy-tailed nonnegative monthly cost; a small number of high-cost members; and rare but nonzero event and spend-archetype labels. Statistical resemblance only needs to support model and hardware experiments, not reproduce private distributions.
- `outcome_enrollment_mask.npy` is binary `[N,12]`. Set `y` and `event_y` to NaN where that mask is zero; enrolled zero-cost months must contain zero, not NaN. `event_y` has five channels. `spend_archetype_multilabel` has six channels. Make the 105-bin historical `enrollment_mask` plausible and binary. Mask arrays for future features must align with their corresponding feature arrays.
- Generate `partial_exposure_indices.json` from members with fewer than 12 enrolled outcome months. Target about 119,410 train members and 117,136 forward-test members at full size. `member_strat_indices.json` has `New` and `Non-New` lists partitioning all rows: train 96,081/767,073; forward test 65,237/782,331. Scale counts proportionally for smoke runs.
- For each code family, write `code_ids/<family>_flat_ids.npy` and `code_ids/<family>_offsets.npy` as int32. Offsets length is `N*105 + 1`; offsets start at zero, never decrease, and end at `len(flat_ids)`. Keep every `flat_ids` value in `[0, vocab_size)`, reserving zero as padding. Match Appendix B's `n_ids` exactly for full size and approximately proportionally for scaled size. Each individual family remains below the signed int32 offset ceiling. Stream both arrays in blocks and do not build an all-family bag-count matrix.
- Use the full contract's feature lists and vocabularies in `metadata.json`. Include `split`, `X_static_shape`, `X_seq_shape`, `y_shape`, `sequence_length=105`, `outcome_length=12`, `static_columns`, `seq_columns`, `code_families` (`n_ids`, `n_member_months`, `vocab_size`), the future-array shape/column keys in Appendix C, `plan_cat_columns`, `plan_cat_vocab_sizes`, and the split's `outcome_months`. Do not fabricate source member-order fingerprints or copy scaler values from real training.
- `member_labels.parquet` needs only synthetic IDs and `CMD`; write it in bounded-size row groups. For train, optionally write deterministic `train_val_split.json` with disjoint 80/20 member indices, seed, fraction, and `n_members`. It is absent from the measured 121-file source layout, so label it as an optional extra file in the report.
- Before writing, estimate output size and require enough free disk for output plus at least 15% headroom. If `--stage-chunks` would coexist with assembled `X_seq.npy`, account for the duplicate bytes. Write completion markers or temporary names so an interrupted run can resume without accepting a partial file as complete. Keep the two splits distinct.

## Optional fragmented stage layout

The local training loader expects a single `X_seq.npy` per split. The staging path fragments that tensor into about 5 GiB member-major files called `X_seq_chunk_000.npy`, `X_seq_chunk_001.npy`, and so on, plus `X_seq_manifest.json` containing `n_chunks`, `shape`, `dtype`, and `rows_per_chunk`. For these full CMD shapes, use `rows_per_chunk=39,452`, yielding 22 chunks per split. Implement this as an optional mode to benchmark transfer and reassembly. Provide a reassembly command that streams chunks into `X_seq.npy` and validates row order. Avoid holding both layouts on disk unless explicitly requested. The 100 code-ID `.npy` files per split are part of the assembled layout itself; preserve that file fragmentation in all modes.

## Validation and reporting

Check exact full-size shapes, dtypes, filenames, file counts, and logical bytes implied by the table and appendices. For scaled runs, check the scaled equivalents. Validate all code offsets, ID ranges, row-index lists, masks, target NaNs, nonnegative enrolled costs, and metadata consistency. Open every `.npy` with `np.load(..., mmap_mode='r')` rather than reading it into RAM. If the repository imports cleanly, construct `MemberDataset('train__CMD')` and `MemberDataset('one_year_forward_test__CMD')`, take a small batch from each, and check its tensor dimensions. Report generation elapsed time, peak RSS, logical/allocated GiB per split, and a per-file size summary. On macOS, make benchmark settings for DataLoader workers, batch size, and MPS optional rather than hardcoded.

## Appendix A: root array inventory

The table below is generated from the existing array headers and file sizes. It contains no array values.

| File | Train shape | Forward-test shape | dtype | Train GiB | Forward-test GiB |
|---|---|---|---|---|---|
| `X_cluster_future.npy` | `[863154, 12, 56]` | `[847568, 12, 56]` | float32 | 2.161 | 2.122 |
| `X_crude_future.npy` | `[863154, 12, 19]` | `[847568, 12, 19]` | float32 | 0.733 | 0.720 |
| `X_member_future.npy` | `[863154, 12, 38]` | `[847568, 12, 38]` | float32 | 1.466 | 1.440 |
| `X_plan_cat.npy` | `[863154, 18]` | `[847568, 18]` | int64 | 0.116 | 0.114 |
| `X_prospective_static.npy` | `[863154, 56]` | `[847568, 56]` | float32 | 0.180 | 0.177 |
| `X_seq.npy` | `[863154, 105, 324]` | `[847568, 105, 324]` | float32 | 109.391 | 107.416 |
| `X_static.npy` | `[863154, 255]` | `[847568, 255]` | float32 | 0.820 | 0.805 |
| `cluster_profile_mask.npy` | `[863154, 12, 56]` | `[847568, 12, 56]` | float32 | 2.161 | 2.122 |
| `crude_future_mask.npy` | `[863154, 12, 19]` | `[847568, 12, 19]` | float32 | 0.733 | 0.720 |
| `enrollment_mask.npy` | `[863154, 105]` | `[847568, 105]` | float32 | 0.338 | 0.332 |
| `event_y.npy` | `[863154, 12, 5]` | `[847568, 12, 5]` | float32 | 0.193 | 0.189 |
| `member_future_mask.npy` | `[863154, 12, 38]` | `[847568, 12, 38]` | float32 | 1.466 | 1.440 |
| `outcome_enrollment_mask.npy` | `[863154, 12]` | `[847568, 12]` | float32 | 0.039 | 0.038 |
| `prospective_static_mask.npy` | `[863154]` | `[847568]` | float32 | 0.003 | 0.003 |
| `spend_archetype.npy` | `[863154]` | `[847568]` | int8 | 0.001 | 0.001 |
| `spend_archetype_multilabel.npy` | `[863154, 6]` | `[847568, 6]` | float32 | 0.019 | 0.019 |
| `y.npy` | `[863154, 12]` | `[847568, 12]` | float32 | 0.039 | 0.038 |

## Appendix B: ragged code-family inventory

Every family has one flat-ID file and one offsets file per split. Offsets are int32 `[N*105+1]` and have the same size within a split: approximately 0.338 GiB for train and 0.332 GiB for forward test. Flat IDs are int32 `[n_ids]`. The full code-ID directory totals 34.205 GiB for train and 33.632 GiB for forward test.

| Family | Train flat IDs | Forward-test flat IDs | Train vocab | Forward-test vocab |
|---|---|---|---|---|
| `admit_type_cd` | 90,997,959 | 89,377,525 | 9 | 9 |
| `auth_type_name` | 90,631,779 | 88,995,395 | 9 | 9 |
| `billtype` | 90,726,872 | 89,093,125 | 37 | 37 |
| `cpt_ccs_group` | 98,432,142 | 97,347,981 | 91 | 91 |
| `diags_ip_ccsr` | 100,192,272 | 99,450,888 | 277 | 277 |
| `diags_ip_ccsr_acute` | 95,090,779 | 93,835,051 | 193 | 193 |
| `diags_ip_ccsr_chronic` | 93,924,391 | 92,628,896 | 167 | 167 |
| `diags_op_ccsr` | 103,496,160 | 102,786,422 | 278 | 278 |
| `diags_op_ccsr_acute` | 97,860,501 | 96,636,345 | 192 | 192 |
| `diags_op_ccsr_chronic` | 94,108,429 | 92,820,189 | 170 | 170 |
| `discharge_status` | 90,650,572 | 89,015,497 | 14 | 14 |
| `drug_class_cd` | 90,926,954 | 89,392,888 | 8 | 8 |
| `drug_compound_indc` | 90,641,081 | 89,007,659 | 5 | 5 |
| `drug_formulary_indc` | 90,660,970 | 89,024,975 | 5 | 5 |
| `drug_generic_cd` | 91,123,547 | 89,586,241 | 5 | 5 |
| `esi_contracted_specialty_indc` | 90,631,170 | 88,994,640 | 4 | 4 |
| `esi_drug_type` | 90,929,842 | 89,396,296 | 7 | 7 |
| `ip_notif_admission_reason` | 90,631,397 | 88,994,869 | 23 | 23 |
| `ip_notif_auth_type` | 90,631,268 | 88,994,731 | 7 | 7 |
| `ip_notif_diags_ccsr_ip` | 90,631,760 | 88,995,174 | 22 | 22 |
| `ip_notif_treatment_type` | 90,631,376 | 88,994,848 | 21 | 21 |
| `lifestyle_domain` | 90,927,432 | 89,345,023 | 38 | 38 |
| `mail_order_retail` | 90,650,422 | 89,016,998 | 5 | 5 |
| `medicare_part_code` | 90,631,170 | 88,994,640 | 5 | 5 |
| `milliman_hcg_drg` | 90,632,828 | 88,996,253 | 191 | 191 |
| `milliman_hcg_family1` | 100,333,706 | 99,060,412 | 133 | 133 |
| `milliman_hcg_family2` | 91,090,520 | 89,519,059 | 97 | 97 |
| `milliman_hcg_mr_line_case` | 103,175,307 | 102,869,324 | 109 | 109 |
| `milliman_hcg_mr_line_case_svc_type_catgry` | 91,824,872 | 90,274,062 | 6 | 6 |
| `milliman_hcg_spec_type` | 92,737,984 | 91,107,164 | 11 | 11 |
| `ndc_ahfs_cd` | 90,762,166 | 89,128,310 | 77 | 77 |
| `ndc_dosage_form_name` | 92,560,242 | 91,580,378 | 69 | 69 |
| `ndc_gpi_6_descr` | 94,049,642 | 93,245,786 | 217 | 217 |
| `ndc_route_name` | 91,340,848 | 89,901,922 | 36 | 36 |
| `ndc_substance_name` | 93,836,673 | 93,014,818 | 309 | 309 |
| `new_refill_cd` | 91,344,278 | 89,893,627 | 5 | 5 |
| `paid_drg_cd` | 90,632,265 | 88,995,731 | 215 | 215 |
| `paid_drg_mdc_cd` | 90,631,553 | 88,995,022 | 25 | 25 |
| `pharmacy_class_code` | 90,702,132 | 89,095,791 | 13 | 13 |
| `place_of_service_cd` | 92,789,784 | 91,057,245 | 25 | 25 |
| `pos_type` | 91,815,450 | 90,264,528 | 6 | 6 |
| `prior_auth_diag_ccsr_ip` | 90,631,240 | 88,994,957 | 98 | 98 |
| `prior_auth_diag_ccsr_op` | 90,631,253 | 88,995,031 | 105 | 105 |
| `proc_cd` | 108,195,489 | 107,292,039 | 642 | 642 |
| `proc_description` | 90,632,173 | 88,995,844 | 86 | 86 |
| `provider_servicing_taxonomy` | 94,678,550 | 93,137,992 | 101 | 100 |
| `provider_servicing_type_cd` | 94,605,854 | 93,057,254 | 79 | 79 |
| `revenue_cd` | 93,668,165 | 92,211,955 | 137 | 137 |
| `treatment_type_name` | 90,631,902 | 88,995,415 | 94 | 94 |
| `ubh_indc` | 90,631,170 | 88,994,640 | 4 | 4 |

## Appendix C: ordered metadata lists

The lists below are needed to preserve feature dimensions and channel routing. They are structural labels only. `plan_cat_vocab_sizes` differs in three fields between the two splits, so both maps are given.

### `static_columns` (255 entries)

```json
[
  "AGE_AS_OF_FIRST_OF_MNTH",
  "CONTINUOUS_ENROLLED_MONTHS",
  "COVERAGE_GAP_MONTHS_PRIOR_12M",
  "ENROLLED_MONTHS_PRIOR_12M",
  "ENROLLED_MONTHS_PRIOR_3M",
  "ENROLLED_MONTHS_PRIOR_6M",
  "ENROLLMENT_FRACTION",
  "LOB_TRANSITION_FLAG",
  "MONTHS_SINCE_FIRST_ENROLLMENT",
  "MONTHS_TO_PLAN_EXPIRY",
  "NETWORK_TRANSITION_FLAG",
  "NEW_MEMBER_FLAG",
  "PLAN_EXPIRING_WITHIN_3M_FLAG",
  "RETURNING_MEMBER_FLAG",
  "SMOKER_INDC",
  "STATE_TRANSITION_FLAG",
  "ZIP5_AREA_SUMMARY_AGE_30S",
  "ZIP5_AREA_SUMMARY_AGE_60S",
  "ZIP5_AREA_SUMMARY_AGE_70S",
  "ZIP5_AREA_SUMMARY_AGE_MEDIAN",
  "ZIP5_AREA_SUMMARY_AGE_OVER_80",
  "ZIP5_AREA_SUMMARY_AGE_UNDER_10",
  "ZIP5_AREA_SUMMARY_FAMILY_SIZE",
  "ZIP5_AREA_SUMMARY_FARMER",
  "ZIP5_AREA_SUMMARY_INCOME_HOUSEHOLD_20_TO_25",
  "ZIP5_AREA_SUMMARY_INCOME_HOUSEHOLD_25_TO_35",
  "ZIP5_AREA_SUMMARY_LABOR_FORCE_PARTICIPATION",
  "ZIP5_AREA_SUMMARY_RACE_MULTIPLE",
  "ZIP5_AREA_SUMMARY_SELF_EMPLOYED",
  "ZIP5_AREA_SUMMARY_UNEMPLOYMENT_RATE",
  "ZIP5_AREA_SUMMARY_VETERAN",
  "ZIP5_AREA_SUMMARY_WIDOWED",
  "TOTAL_SPEND_PM_LAST_12M_PMPM",
  "TOTAL_UTIL_PM_LAST_12M_PMPM",
  "TOTAL_SPEND_ER_PM_LAST_12M_PMPM",
  "TOTAL_UTIL_ER_PM_LAST_12M_PMPM",
  "TOTAL_SPEND_IP_PM_LAST_12M_PMPM",
  "TOTAL_UTIL_IP_PM_LAST_12M_PMPM",
  "TOTAL_SPEND_OP_PM_LAST_12M_PMPM",
  "TOTAL_SPEND_RX_PM_LAST_12M_PMPM",
  "TOTAL_UTIL_RX_PM_LAST_12M_PMPM",
  "TOTAL_SPEND_MENTAL_HEALTH_PM_LAST_12M_PMPM",
  "TOTAL_UTIL_MENTAL_HEALTH_PM_LAST_12M_PMPM",
  "TOTAL_SPEND_ONCOLOGY_CHEMO_PM_LAST_12M_PMPM",
  "TOTAL_SPEND_SNF_PM_LAST_12M_PMPM",
  "TOTAL_SPEND_SUD_PM_LAST_12M_PMPM",
  "TOTAL_SPEND_DIALYSIS_PM_LAST_12M_PMPM",
  "TOTAL_SPEND_TREND_6M_12M",
  "TOTAL_UTIL_TREND_6M_12M",
  "TOTAL_SPEND_ER_TREND_6M_12M",
  "TOTAL_SPEND_IP_TREND_6M_12M",
  "TOTAL_SPEND_OP_TREND_6M_12M",
  "TOTAL_SPEND_RX_TREND_6M_12M",
  "TOTAL_SPEND_MENTAL_HEALTH_TREND_6M_12M",
  "TOTAL_SPEND_ONCOLOGY_CHEMO_TREND_6M_12M",
  "TOTAL_SPEND_SNF_TREND_6M_12M",
  "TOTAL_SPEND_SUD_TREND_6M_12M",
  "TOTAL_SPEND_DIALYSIS_TREND_6M_12M",
  "TOTAL_UTIL_ER_TREND_6M_12M",
  "TOTAL_UTIL_IP_TREND_6M_12M",
  "TOTAL_UTIL_RX_TREND_6M_12M",
  "TOTAL_UTIL_MENTAL_HEALTH_TREND_6M_12M",
  "RX_SPEND_SPECIALTY_PM_LAST_12M",
  "RX_SPEND_GLP1_INCRETIN_PM_LAST_12M",
  "RX_SPEND_ANTINEOPLASTIC_PM_LAST_12M",
  "RX_SPEND_OPIOID_PM_LAST_12M",
  "RX_TOTAL_FILLS_PM_LAST_12M",
  "CM_ANY_PROGRAM_OPEN_AT_INDEX",
  "CM_TOTAL_PROGRAMS_EVER",
  "CM_MONTHS_IN_ANY_PROGRAM",
  "CM_HIGH_TIER_EVER",
  "CM_MEDIUM_TIER_EVER",
  "CM_HIGH_TIER_MONTHS",
  "CM_MEDIUM_TIER_MONTHS",
  "PRIOR_AUTH_TOTAL_12M",
  "PRIOR_AUTH_DENIED_RATE_12M",
  "IP_NOTIF_TOTAL_12M",
  "IP_NOTIF_DISTINCT_CCSR_12M",
  "PREMIUM_RATE",
  "BILLED_PREMIUM_AMT",
  "RX_CARVE_OUT_INDC",
  "TERM_MEMBER_INDC",
  "LIFESTYLE_TOBACCO_DEPENDENCE_SVC_COUNT",
  "LIFESTYLE_TOBACCO_CURRENT_USE_SVC_COUNT",
  "LIFESTYLE_TOBACCO_FORMER_SMOKER_SVC_COUNT",
  "LIFESTYLE_TOBACCO_ENVIRONMENTAL_EXPOSURE_SVC_COUNT",
  "LIFESTYLE_ALCOHOL_DISORDER_SVC_COUNT",
  "LIFESTYLE_OPIOID_DISORDER_SVC_COUNT",
  "LIFESTYLE_CANNABIS_DISORDER_SVC_COUNT",
  "LIFESTYLE_SEDATIVE_DISORDER_SVC_COUNT",
  "LIFESTYLE_COCAINE_DISORDER_SVC_COUNT",
  "LIFESTYLE_STIMULANT_DISORDER_SVC_COUNT",
  "LIFESTYLE_HALLUCINOGEN_DISORDER_SVC_COUNT",
  "LIFESTYLE_INHALANT_DISORDER_SVC_COUNT",
  "LIFESTYLE_POLYSUBSTANCE_DISORDER_SVC_COUNT",
  "LIFESTYLE_OPIOID_POISONING_SVC_COUNT",
  "LIFESTYLE_OBESITY_SVC_COUNT",
  "LIFESTYLE_BMI_CLASS_I_30_34_SVC_COUNT",
  "LIFESTYLE_BMI_CLASS_II_35_39_SVC_COUNT",
  "LIFESTYLE_BMI_CLASS_III_40_PLUS_SVC_COUNT",
  "LIFESTYLE_BMI_OVERWEIGHT_25_29_SVC_COUNT",
  "LIFESTYLE_BMI_UNDERWEIGHT_SVC_COUNT",
  "LIFESTYLE_BMI_OTHER_SVC_COUNT",
  "LIFESTYLE_PHYSICAL_INACTIVITY_SVC_COUNT",
  "LIFESTYLE_POOR_DIET_SVC_COUNT",
  "LIFESTYLE_GAMBLING_DISORDER_SVC_COUNT",
  "LIFESTYLE_MED_NONADHERENCE_INTENTIONAL_SVC_COUNT",
  "LIFESTYLE_MED_NONADHERENCE_UNINTENTIONAL_SVC_COUNT",
  "LIFESTYLE_MEDICAL_NONCOMPLIANCE_SVC_COUNT",
  "LIFESTYLE_NONCOMPLIANCE_OTHER_SVC_COUNT",
  "LIFESTYLE_VACCINE_REFUSAL_UNDERIMMUNIZATION_SVC_COUNT",
  "LIFESTYLE_SDOH_EDUCATION_LITERACY_SVC_COUNT",
  "LIFESTYLE_SDOH_EMPLOYMENT_SVC_COUNT",
  "LIFESTYLE_SDOH_OCCUPATIONAL_EXPOSURE_SVC_COUNT",
  "LIFESTYLE_SDOH_HOUSING_ECONOMIC_SVC_COUNT",
  "LIFESTYLE_SDOH_SOCIAL_ENVIRONMENT_SVC_COUNT",
  "LIFESTYLE_SDOH_UPBRINGING_SVC_COUNT",
  "LIFESTYLE_SDOH_FAMILY_CIRCUMSTANCES_SVC_COUNT",
  "LIFESTYLE_SDOH_PSYCHOSOCIAL_SVC_COUNT",
  "LIFESTYLE_SDOH_LEGAL_CIRCUMSTANCES_SVC_COUNT",
  "LIFESTYLE_SLEEP_APNEA_SVC_COUNT",
  "LIFESTYLE_SLEEP_OTHER_SVC_COUNT",
  "LIFESTYLE_SLEEP_BEHAVIORAL_SVC_COUNT",
  "LIFESTYLE_EATING_ANOREXIA_SVC_COUNT",
  "LIFESTYLE_EATING_BULIMIA_SVC_COUNT",
  "LIFESTYLE_EATING_BINGE_SVC_COUNT",
  "LIFESTYLE_EATING_OTHER_SVC_COUNT",
  "LIFESTYLE_TOBACCO_DEPENDENCE_MONTHS_SINCE_LAST",
  "LIFESTYLE_TOBACCO_CURRENT_USE_MONTHS_SINCE_LAST",
  "LIFESTYLE_TOBACCO_FORMER_SMOKER_MONTHS_SINCE_LAST",
  "LIFESTYLE_TOBACCO_ENVIRONMENTAL_EXPOSURE_MONTHS_SINCE_LAST",
  "LIFESTYLE_ALCOHOL_DISORDER_MONTHS_SINCE_LAST",
  "LIFESTYLE_OPIOID_DISORDER_MONTHS_SINCE_LAST",
  "LIFESTYLE_CANNABIS_DISORDER_MONTHS_SINCE_LAST",
  "LIFESTYLE_SEDATIVE_DISORDER_MONTHS_SINCE_LAST",
  "LIFESTYLE_COCAINE_DISORDER_MONTHS_SINCE_LAST",
  "LIFESTYLE_STIMULANT_DISORDER_MONTHS_SINCE_LAST",
  "LIFESTYLE_HALLUCINOGEN_DISORDER_MONTHS_SINCE_LAST",
  "LIFESTYLE_INHALANT_DISORDER_MONTHS_SINCE_LAST",
  "LIFESTYLE_POLYSUBSTANCE_DISORDER_MONTHS_SINCE_LAST",
  "LIFESTYLE_OPIOID_POISONING_MONTHS_SINCE_LAST",
  "LIFESTYLE_OBESITY_MONTHS_SINCE_LAST",
  "LIFESTYLE_BMI_CLASS_I_30_34_MONTHS_SINCE_LAST",
  "LIFESTYLE_BMI_CLASS_II_35_39_MONTHS_SINCE_LAST",
  "LIFESTYLE_BMI_CLASS_III_40_PLUS_MONTHS_SINCE_LAST",
  "LIFESTYLE_BMI_OVERWEIGHT_25_29_MONTHS_SINCE_LAST",
  "LIFESTYLE_BMI_UNDERWEIGHT_MONTHS_SINCE_LAST",
  "LIFESTYLE_BMI_OTHER_MONTHS_SINCE_LAST",
  "LIFESTYLE_PHYSICAL_INACTIVITY_MONTHS_SINCE_LAST",
  "LIFESTYLE_POOR_DIET_MONTHS_SINCE_LAST",
  "LIFESTYLE_GAMBLING_DISORDER_MONTHS_SINCE_LAST",
  "LIFESTYLE_MED_NONADHERENCE_INTENTIONAL_MONTHS_SINCE_LAST",
  "LIFESTYLE_MED_NONADHERENCE_UNINTENTIONAL_MONTHS_SINCE_LAST",
  "LIFESTYLE_MEDICAL_NONCOMPLIANCE_MONTHS_SINCE_LAST",
  "LIFESTYLE_NONCOMPLIANCE_OTHER_MONTHS_SINCE_LAST",
  "LIFESTYLE_VACCINE_REFUSAL_UNDERIMMUNIZATION_MONTHS_SINCE_LAST",
  "LIFESTYLE_SDOH_EDUCATION_LITERACY_MONTHS_SINCE_LAST",
  "LIFESTYLE_SDOH_EMPLOYMENT_MONTHS_SINCE_LAST",
  "LIFESTYLE_SDOH_OCCUPATIONAL_EXPOSURE_MONTHS_SINCE_LAST",
  "LIFESTYLE_SDOH_HOUSING_ECONOMIC_MONTHS_SINCE_LAST",
  "LIFESTYLE_SDOH_SOCIAL_ENVIRONMENT_MONTHS_SINCE_LAST",
  "LIFESTYLE_SDOH_UPBRINGING_MONTHS_SINCE_LAST",
  "LIFESTYLE_SDOH_FAMILY_CIRCUMSTANCES_MONTHS_SINCE_LAST",
  "LIFESTYLE_SDOH_PSYCHOSOCIAL_MONTHS_SINCE_LAST",
  "LIFESTYLE_SDOH_LEGAL_CIRCUMSTANCES_MONTHS_SINCE_LAST",
  "LIFESTYLE_SLEEP_APNEA_MONTHS_SINCE_LAST",
  "LIFESTYLE_SLEEP_OTHER_MONTHS_SINCE_LAST",
  "LIFESTYLE_SLEEP_BEHAVIORAL_MONTHS_SINCE_LAST",
  "LIFESTYLE_EATING_ANOREXIA_MONTHS_SINCE_LAST",
  "LIFESTYLE_EATING_BULIMIA_MONTHS_SINCE_LAST",
  "LIFESTYLE_EATING_BINGE_MONTHS_SINCE_LAST",
  "LIFESTYLE_EATING_OTHER_MONTHS_SINCE_LAST",
  "TOTAL_CHRONIC_DX_COUNT_24M",
  "TOTAL_ACUTE_DX_COUNT_24M",
  "TOTAL_UNIQUE_DX_24M",
  "CHRONIC_DX_RATE_PER_12M",
  "ACUTE_DX_RATE_PER_12M",
  "DX_DENSITY_PER_12M",
  "CHRONIC_TO_ACUTE_RATIO",
  "MAX_CHRONIC_DX_RECURRENCE",
  "UNIQUE_BODY_SYSTEMS_24M",
  "MULTI_SYSTEM_FLAG",
  "NEW_CHRONIC_DX_LAST_6M",
  "CHRONIC_DX_RATE_LAST_6M",
  "ACUTE_DX_RATE_LAST_6M",
  "DX_DENSITY_LAST_6M",
  "TOTAL_UNIQUE_NDCS_24M",
  "SPECIALTY_RX_MONTHS",
  "SPECIALTY_RX_RATE_24M",
  "SPECIALTY_RX_RATE_LAST_6M",
  "POLYPHARMACY_MAX_24M",
  "NEW_NDCS_LAST_6M",
  "IP_ADMISSIONS_24M",
  "ER_VISITS_24M",
  "IP_ADMITS_PER_12M",
  "ER_VISITS_PER_12M",
  "IP_ADMITS_LAST_6M",
  "ER_VISITS_LAST_6M",
  "IP_RATE_LAST_6M",
  "ER_RATE_LAST_6M",
  "TOTAL_SPEND_LAST_6M",
  "SPEND_RATE_LAST_6M",
  "UNIQUE_PROVIDERS_24M",
  "UNIQUE_SPECIALTIES_24M",
  "UNIQUE_TINS_24M",
  "PROVIDER_RATE_LAST_6M",
  "MONTHS_SINCE_LAST_IP",
  "MONTHS_SINCE_LAST_ER",
  "MONTHS_SINCE_LAST_SPECIALIST",
  "MONTHS_SINCE_LAST_RX_FILL",
  "MONTHS_SINCE_LAST_SPECIALTY_RX",
  "MONTHS_SINCE_LAST_CHRONIC_DX",
  "MONTHS_SINCE_LAST_ACUTE_DX",
  "MONTHS_SINCE_LAST_PROCEDURE",
  "MONTHS_SINCE_LAST_HIGH_COST_EVENT",
  "MONTHS_SINCE_LAST_PRIOR_AUTH",
  "CRF_FRAILTY_FLAG",
  "CRF_FRAILTY_CONCEPT_COUNT",
  "CRF_CANCER_ACTIVE_FLAG",
  "CRF_CANCER_HISTORY_FLAG",
  "CRF_DIALYSIS_FLAG",
  "CRF_PREGNANT_FLAG",
  "CRF_PAST_DELIVERY_FLAG",
  "CRF_DIABETES_FLAG",
  "CRF_CHF_FLAG",
  "CRF_COPD_FLAG",
  "CRF_ASTHMA_FLAG",
  "CRF_DEPRESSION_FLAG",
  "CRF_BIPOLAR_FLAG",
  "CRF_SCHIZOPHRENIA_FLAG",
  "CRF_SUD_FLAG",
  "CRF_HIV_FLAG",
  "CRF_TRANSPLANT_FLAG",
  "CRF_HYPERTENSION_FLAG",
  "CRF_ISCHEMIC_HEART_FLAG",
  "CRF_OSTEOPOROSIS_FLAG",
  "CRF_RHEUMATOID_ARTHRITIS_FLAG",
  "CRF_SEIZURE_FLAG",
  "CRF_PARKINSONS_FLAG",
  "CRF_DEMENTIA_FLAG",
  "CRF_OBESITY_FLAG",
  "CRF_MECHANICAL_VENTILATION_FLAG",
  "CRF_ON_INSULIN_FLAG",
  "CRF_ON_ANTINEOPLASTIC_FLAG",
  "CRF_ON_ANTICOAGULANT_FLAG",
  "CRF_ON_OPIOID_CHRONIC_FLAG",
  "CRF_ON_ANTIPSYCHOTIC_FLAG",
  "CRF_ON_ANTIDEPRESSANT_FLAG",
  "CRF_ON_IMMUNOSUPPRESSANT_FLAG",
  "CRF_CHRONIC_CONDITION_COUNT",
  "CRF_POLYPHARMACY_10PLUS_FLAG",
  "CRF_HIGH_RISK_COMPOSITE",
  "CRF_MULTI_CHRONIC_4PLUS_FLAG",
  "CRF_PREGNANCY_RISK_FLAG",
  "CRF_MAJOR_PROCEDURE_COUNT"
]
```

### `seq_columns` (324 entries)

```json
[
  "TOTAL_SPEND",
  "TOTAL_SPEND_IP",
  "TOTAL_SPEND_OP",
  "TOTAL_SPEND_PROF",
  "TOTAL_SPEND_RX",
  "TOTAL_SPEND_ER",
  "TOTAL_SPEND_MENTAL_HEALTH",
  "TOTAL_SPEND_SURGERY",
  "TOTAL_SPEND_RADIOLOGY",
  "TOTAL_SPEND_THERAPY",
  "TOTAL_SPEND_HOME_HEALTH",
  "TOTAL_SPEND_DME",
  "TOTAL_SPEND_CARDIOVASCULAR",
  "TOTAL_SPEND_MATERNITY",
  "TOTAL_SPEND_PREVENTIVE",
  "TOTAL_SPEND_SPECIALTY_RX",
  "TOTAL_SPEND_INJECTION_JQ",
  "TOTAL_SPEND_DIALYSIS",
  "TOTAL_SPEND_CHEMO_ADMIN",
  "TOTAL_UTIL",
  "TOTAL_UTIL_IP",
  "TOTAL_UTIL_OP",
  "TOTAL_UTIL_PROF",
  "TOTAL_UTIL_RX",
  "TOTAL_UTIL_ER",
  "TOTAL_UTIL_MENTAL_HEALTH",
  "TOTAL_UTIL_SURGERY",
  "TOTAL_UTIL_RADIOLOGY",
  "TOTAL_UTIL_HOME_HEALTH",
  "TOTAL_UTIL_INJECTION_JQ",
  "TOTAL_UTIL_PREVENTIVE",
  "TOTAL_RX_FILLS",
  "TOTAL_RX_FILLS_SPECIALTY",
  "CM_ANY_CASE_OPEN",
  "CM_TOTAL_CASES_OPEN",
  "CM_ANY_CASE_CLOSED",
  "CM_ENGAGED_ANY",
  "CM_HIGH_TIER_OPEN",
  "CM_MEDIUM_TIER_OPEN",
  "PRIOR_AUTH_COUNT",
  "PRIOR_AUTH_APPROVED_COUNT",
  "PRIOR_AUTH_DENIED_COUNT",
  "IP_NOTIF_COUNT",
  "IP_NOTIF_POST_ACUTE_COUNT",
  "WEEKS_SINCE_LAST_DX",
  "WEEKS_SINCE_LAST_IP",
  "WEEKS_SINCE_LAST_ER",
  "WEEKS_SINCE_LAST_PROCEDURE",
  "WEEKS_SINCE_LAST_RX_FILL",
  "WEEKS_SINCE_LAST_PRIOR_AUTH",
  "WEEKS_SINCE_LAST_HIGH_COST_EVENT",
  "TIME_SINCE_FIRST_OBSERVED_EVENT",
  "OBSERVED_THIS_PERIOD",
  "NUMBER_OF_OBSERVATIONS",
  "DAYS_WITH_ANY_ACTIVITY",
  "PHARMACY_OBSERVED_T",
  "MEDICAL_CLAIM_OBSERVED_T",
  "CARE_MANAGEMENT_OBSERVED_T",
  "CUM_SPEND_TOTAL",
  "CUM_SPEND_IP",
  "CUM_SPEND_RX",
  "CUM_SPEND_ER",
  "CUM_UTIL_TOTAL",
  "CUM_UTIL_IP",
  "CUM_UTIL_ER",
  "CUM_RX_FILLS",
  "SPEND_DELTA_4W",
  "SPEND_DELTA_12W",
  "UTIL_DELTA_4W",
  "UTIL_DELTA_12W",
  "SPEND_MA_4W",
  "SPEND_MA_12W",
  "UTIL_MA_4W",
  "UTIL_MA_12W",
  "NEW_DRUG_START",
  "POLYPHARMACY_COUNT",
  "SPECIALTY_DRUG_COUNT",
  "HIGH_COST_DRUG_COUNT",
  "INJECTABLE_DRUG_COUNT",
  "DAYS_SUPPLY_AVG",
  "CUM_UNIQUE_NDCS",
  "NEW_RX_THIS_PERIOD",
  "TOTAL_ACTIVE_NDCS",
  "RX_FLAG_T",
  "SPECIALTY_RX_FLAG_T",
  "NEW_SPECIALTY_RX_COUNT",
  "NDC_ACCUMULATION_RATE",
  "WEEKS_SINCE_LAST_NEW_NDC",
  "WEEKS_SINCE_LAST_SPECIALTY_RX",
  "NEW_PRIOR_AUTH",
  "PRIOR_AUTH_APPROVED_THIS_PERIOD",
  "PRIOR_AUTH_DENIED_THIS_PERIOD",
  "WEEKS_SINCE_PRIOR_AUTH_EVENT",
  "PRIOR_AUTH_FREQUENCY_4W",
  "PRIOR_AUTH_FREQUENCY_12W",
  "NUMBER_OF_ACTIVE_AUTHS",
  "COMPLEX_CARE_ENROLLMENT_THIS_PERIOD",
  "PA_FLAG_T",
  "CUM_PRIOR_AUTHS",
  "CUM_PA_DENIALS",
  "PA_ACCUMULATION_RATE",
  "WEEKS_SINCE_LAST_PA_DENIAL",
  "ELIGIBILITY_THIS_PERIOD",
  "MONTHS_SINCE_ENROLLMENT",
  "CONTINUOUS_ENROLLMENT_STREAK",
  "GAP_LENGTH",
  "NEW_MEMBER_FLAG_T",
  "RETURNING_MEMBER_FLAG_T",
  "RECENT_IP_ADMISSION",
  "RECENT_ER_VISIT",
  "IP_DAYS_LAST_4W",
  "IP_DAYS_LAST_12W",
  "ER_VISITS_LAST_4W",
  "ER_VISITS_LAST_12W",
  "READMISSION_WITHIN_30D",
  "DISCHARGE_TO_SNF",
  "DISCHARGE_TO_HOME",
  "ICU_DAYS_THIS_PERIOD",
  "OBSERVATION_DAYS_THIS_PERIOD",
  "OBSERVATION_TO_IP_THIS_PERIOD",
  "ER_TO_IP_THIS_PERIOD",
  "DISCHARGE_TO_LTACH",
  "DISCHARGE_TO_REHAB",
  "DISCHARGE_TO_PSYCH",
  "DISCHARGE_TO_HOSPICE",
  "DISCHARGE_TO_OTHER_FACILITY",
  "DISCHARGE_AMA",
  "EXPIRED_THIS_PERIOD",
  "IP_ADMISSION_COUNT_T",
  "ER_VISIT_COUNT_T",
  "CUM_IP_ADMISSIONS",
  "CUM_ER_VISITS",
  "IP_ACCUMULATION_RATE",
  "ER_ACCUMULATION_RATE",
  "WEEKS_SINCE_LAST_IP_ADMISSION",
  "WEEKS_SINCE_LAST_ER_VISIT",
  "WEEKS_SINCE_LAST_READMISSION",
  "WEEKS_SINCE_LAST_ICU",
  "UNIQUE_PROVIDERS_4W",
  "UNIQUE_PROVIDERS_12W",
  "UNIQUE_SPECIALTIES",
  "FACILITY_COUNT",
  "CUM_UNIQUE_PROVIDERS",
  "REFERRAL_COUNT",
  "NEW_PROVIDER_SPECIALTY_THIS_PERIOD",
  "REFERRAL_CHAIN_DEPTH",
  "CUM_REFERRALS",
  "REFERRAL_RATE_4W",
  "NEW_REFERRING_PROVIDER",
  "DISTINCT_REFERRING_SPECIALTIES",
  "OUT_OF_NETWORK_CLAIMS",
  "OUT_OF_NETWORK_FRACTION",
  "VALUE_BASED_NETWORK_CLAIMS",
  "VALUE_BASED_NETWORK_FRACTION",
  "UNIQUE_TINS",
  "UNIQUE_SUPERTINS",
  "TIN_CONCENTRATION_HHI",
  "CUM_UNIQUE_TINS",
  "NEW_PROVIDER_FLAG_T",
  "REFERRAL_FLAG_T",
  "PROVIDER_ACCUMULATION_RATE",
  "TIN_ACCUMULATION_RATE",
  "WEEKS_SINCE_LAST_NEW_PROVIDER",
  "WEEKS_SINCE_LAST_REFERRAL",
  "NEW_DX_THIS_PERIOD",
  "CUM_UNIQUE_DX",
  "DX_ACCUMULATION_RATE",
  "CHRONIC_DX_COUNT_T",
  "ACUTE_DX_COUNT_T",
  "NEW_CHRONIC_DX",
  "NEW_ACUTE_DX",
  "NEW_CHRONIC_DX_COUNT",
  "NEW_ACUTE_DX_COUNT",
  "CHRONIC_DX_FLAG_T",
  "ACUTE_DX_FLAG_T",
  "CUM_UNIQUE_CHRONIC_DX",
  "CUM_UNIQUE_ACUTE_DX",
  "CHRONIC_DX_ACCUMULATION_RATE",
  "ACUTE_DX_ACCUMULATION_RATE",
  "WEEKS_SINCE_LAST_CHRONIC_DX",
  "WEEKS_SINCE_LAST_ACUTE_DX",
  "WEEKS_SINCE_LAST_NEW_DX",
  "WEEKS_SINCE_LAST_NEW_CHRONIC_DX",
  "WEEKS_SINCE_LAST_NEW_ACUTE_DX",
  "LIFESTYLE_NEW_CATEGORY_THIS_PERIOD",
  "LIFESTYLE_CUM_UNIQUE_CATEGORIES",
  "LIFESTYLE_CUM_SVC_COUNT",
  "LIFESTYLE_TREND_4W",
  "LIFESTYLE_FLAG_T",
  "LIFESTYLE_NEW_CATEGORY_COUNT",
  "LIFESTYLE_SVC_COUNT_T",
  "LIFESTYLE_ACCUMULATION_RATE",
  "WEEKS_SINCE_LAST_LIFESTYLE_EVENT",
  "WEEKS_SINCE_LAST_NEW_LIFESTYLE_CATEGORY",
  "OBSERVATION_MATURITY",
  "PARTIAL_PERIOD_FLAG",
  "NEW_PROC_CD_THIS_PERIOD",
  "CUM_UNIQUE_PROC_CD",
  "PROC_CD_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_PROC_CD",
  "NEW_NDC_SUBSTANCE_NAME_THIS_PERIOD",
  "CUM_UNIQUE_NDC_SUBSTANCE_NAME",
  "NDC_SUBSTANCE_NAME_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_NDC_SUBSTANCE_NAME",
  "NEW_NDC_GPI_6_DESCR_THIS_PERIOD",
  "CUM_UNIQUE_NDC_GPI_6_DESCR",
  "NDC_GPI_6_DESCR_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_NDC_GPI_6_DESCR",
  "NEW_PAID_DRG_CD_THIS_PERIOD",
  "CUM_UNIQUE_PAID_DRG_CD",
  "PAID_DRG_CD_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_PAID_DRG_CD",
  "NEW_MILLIMAN_HCG_DRG_THIS_PERIOD",
  "CUM_UNIQUE_MILLIMAN_HCG_DRG",
  "MILLIMAN_HCG_DRG_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_MILLIMAN_HCG_DRG",
  "NEW_MILLIMAN_HCG_FAMILY1_THIS_PERIOD",
  "CUM_UNIQUE_MILLIMAN_HCG_FAMILY1",
  "MILLIMAN_HCG_FAMILY1_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_MILLIMAN_HCG_FAMILY1",
  "NEW_MILLIMAN_HCG_FAMILY2_THIS_PERIOD",
  "CUM_UNIQUE_MILLIMAN_HCG_FAMILY2",
  "MILLIMAN_HCG_FAMILY2_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_MILLIMAN_HCG_FAMILY2",
  "NEW_MILLIMAN_HCG_MR_LINE_CASE_THIS_PERIOD",
  "CUM_UNIQUE_MILLIMAN_HCG_MR_LINE_CASE",
  "MILLIMAN_HCG_MR_LINE_CASE_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_MILLIMAN_HCG_MR_LINE_CASE",
  "NEW_PROVIDER_SERVICING_TAXONOMY_THIS_PERIOD",
  "CUM_UNIQUE_PROVIDER_SERVICING_TAXONOMY",
  "PROVIDER_SERVICING_TAXONOMY_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_PROVIDER_SERVICING_TAXONOMY",
  "NEW_CPT_CCS_GROUP_THIS_PERIOD",
  "CUM_UNIQUE_CPT_CCS_GROUP",
  "CPT_CCS_GROUP_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_CPT_CCS_GROUP",
  "NEW_NDC_ROUTE_NAME_THIS_PERIOD",
  "CUM_UNIQUE_NDC_ROUTE_NAME",
  "NDC_ROUTE_NAME_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_NDC_ROUTE_NAME",
  "NEW_PRIOR_AUTH_DIAG_CCSR_OP_THIS_PERIOD",
  "CUM_UNIQUE_PRIOR_AUTH_DIAG_CCSR_OP",
  "PRIOR_AUTH_DIAG_CCSR_OP_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_PRIOR_AUTH_DIAG_CCSR_OP",
  "NEW_PRIOR_AUTH_DIAG_CCSR_IP_THIS_PERIOD",
  "CUM_UNIQUE_PRIOR_AUTH_DIAG_CCSR_IP",
  "PRIOR_AUTH_DIAG_CCSR_IP_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_PRIOR_AUTH_DIAG_CCSR_IP",
  "NEW_DIAGS_IP_CCSR_THIS_PERIOD",
  "CUM_UNIQUE_DIAGS_IP_CCSR",
  "DIAGS_IP_CCSR_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_DIAGS_IP_CCSR",
  "NEW_DIAGS_OP_CCSR_THIS_PERIOD",
  "CUM_UNIQUE_DIAGS_OP_CCSR",
  "DIAGS_OP_CCSR_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_DIAGS_OP_CCSR",
  "NEW_DIAGS_IP_CCSR_ACUTE_THIS_PERIOD",
  "CUM_UNIQUE_DIAGS_IP_CCSR_ACUTE",
  "DIAGS_IP_CCSR_ACUTE_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_DIAGS_IP_CCSR_ACUTE",
  "NEW_DIAGS_OP_CCSR_ACUTE_THIS_PERIOD",
  "CUM_UNIQUE_DIAGS_OP_CCSR_ACUTE",
  "DIAGS_OP_CCSR_ACUTE_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_DIAGS_OP_CCSR_ACUTE",
  "NEW_DIAGS_IP_CCSR_CHRONIC_THIS_PERIOD",
  "CUM_UNIQUE_DIAGS_IP_CCSR_CHRONIC",
  "DIAGS_IP_CCSR_CHRONIC_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_DIAGS_IP_CCSR_CHRONIC",
  "NEW_DIAGS_OP_CCSR_CHRONIC_THIS_PERIOD",
  "CUM_UNIQUE_DIAGS_OP_CCSR_CHRONIC",
  "DIAGS_OP_CCSR_CHRONIC_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_DIAGS_OP_CCSR_CHRONIC",
  "NEW_REVENUE_CD_THIS_PERIOD",
  "CUM_UNIQUE_REVENUE_CD",
  "REVENUE_CD_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_REVENUE_CD",
  "NEW_NDC_GPI_4_DESCR_THIS_PERIOD",
  "CUM_UNIQUE_NDC_GPI_4_DESCR",
  "NDC_GPI_4_DESCR_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_NDC_GPI_4_DESCR",
  "NEW_NDC_GPI_2_DESCR_THIS_PERIOD",
  "CUM_UNIQUE_NDC_GPI_2_DESCR",
  "NDC_GPI_2_DESCR_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_NDC_GPI_2_DESCR",
  "NEW_NDC_AHFS_CD_THIS_PERIOD",
  "CUM_UNIQUE_NDC_AHFS_CD",
  "NDC_AHFS_CD_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_NDC_AHFS_CD",
  "NEW_NDC_DOSAGE_FORM_NAME_THIS_PERIOD",
  "CUM_UNIQUE_NDC_DOSAGE_FORM_NAME",
  "NDC_DOSAGE_FORM_NAME_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_NDC_DOSAGE_FORM_NAME",
  "NEW_PROVIDER_SERVICING_TYPE_CD_THIS_PERIOD",
  "CUM_UNIQUE_PROVIDER_SERVICING_TYPE_CD",
  "PROVIDER_SERVICING_TYPE_CD_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_PROVIDER_SERVICING_TYPE_CD",
  "NEW_PAID_DRG_MDC_CD_THIS_PERIOD",
  "CUM_UNIQUE_PAID_DRG_MDC_CD",
  "PAID_DRG_MDC_CD_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_PAID_DRG_MDC_CD",
  "NEW_TREATMENT_TYPE_NAME_THIS_PERIOD",
  "CUM_UNIQUE_TREATMENT_TYPE_NAME",
  "TREATMENT_TYPE_NAME_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_TREATMENT_TYPE_NAME",
  "NEW_PROC_DESCRIPTION_THIS_PERIOD",
  "CUM_UNIQUE_PROC_DESCRIPTION",
  "PROC_DESCRIPTION_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_PROC_DESCRIPTION",
  "NEW_IP_NOTIF_ADMISSION_REASON_THIS_PERIOD",
  "CUM_UNIQUE_IP_NOTIF_ADMISSION_REASON",
  "IP_NOTIF_ADMISSION_REASON_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_IP_NOTIF_ADMISSION_REASON",
  "NEW_IP_NOTIF_DIAGS_CCSR_IP_THIS_PERIOD",
  "CUM_UNIQUE_IP_NOTIF_DIAGS_CCSR_IP",
  "IP_NOTIF_DIAGS_CCSR_IP_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_IP_NOTIF_DIAGS_CCSR_IP",
  "NEW_IP_NOTIF_TREATMENT_TYPE_THIS_PERIOD",
  "CUM_UNIQUE_IP_NOTIF_TREATMENT_TYPE",
  "IP_NOTIF_TREATMENT_TYPE_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_IP_NOTIF_TREATMENT_TYPE",
  "NEW_LIFESTYLE_DOMAIN_THIS_PERIOD",
  "CUM_UNIQUE_LIFESTYLE_DOMAIN",
  "LIFESTYLE_DOMAIN_ACCUM_RATE",
  "WEEKS_SINCE_LAST_NEW_LIFESTYLE_DOMAIN"
]
```

### `plan_cat_columns` (18 entries)

```json
[
  "MEMBER_GENDER",
  "PRODUCT_MKT_BUSN_SEG_DESCR",
  "PRODUCT_FULLY_SELF_INSURE_INDC",
  "PRODUCT_NETWORK_ROLLUP",
  "PRODUCT_NETWORK",
  "PLAN_METAL_LEVEL",
  "POLICY_COVERAGE_TIER",
  "RATING_AREA",
  "VALUE_BASED_NETWORK",
  "CARE_COORDINATION_RISK",
  "PRODUCT_ENTITY",
  "MEMBER_RELATIONSHIP_STND_DESCR",
  "ASSIGNED_PROV_TYPE_CD",
  "ASSIGNED_SPP_CARE_SYSTEM",
  "MEMBER_RES_STND_COUNTY",
  "MEMBER_RES_STND_STATE_CD",
  "RATE_CELL_MAJ_PROG",
  "RATE_CELL_ELIG_TYPE"
]
```

### `cluster_future_columns` (56 entries)

```json
[
  "IP_PCT",
  "IP_CHANGE_PCT",
  "ER_PCT",
  "ER_CHANGE_PCT",
  "COMPLEX_PREG_PCT",
  "COMPLEX_PREG_CHANGE_PCT",
  "ONC_TREATMENT_PCT",
  "ONC_TREATMENT_CHANGE_PCT",
  "DIALYSIS_PCT",
  "DIALYSIS_CHANGE_PCT",
  "PREDIABETES_PCT",
  "PREDIABETES_CHANGE_PCT",
  "T2D_PCT",
  "T2D_CHANGE_PCT",
  "HYPERTENSION_PCT",
  "HYPERTENSION_CHANGE_PCT",
  "CKD_ANY_PCT",
  "CKD_ANY_CHANGE_PCT",
  "CKD_EARLY_PCT",
  "CKD_EARLY_CHANGE_PCT",
  "CKD_ADVANCED_PCT",
  "CKD_ADVANCED_CHANGE_PCT",
  "HEART_FAILURE_PCT",
  "HEART_FAILURE_CHANGE_PCT",
  "FRAILTY_PCT",
  "FRAILTY_CHANGE_PCT",
  "OSTEOPOROSIS_PCT",
  "OSTEOPOROSIS_CHANGE_PCT",
  "FALL_EVENT_PCT",
  "FALL_EVENT_CHANGE_PCT",
  "FRACTURE_PCT",
  "FRACTURE_CHANGE_PCT",
  "AFIB_PCT",
  "AFIB_CHANGE_PCT",
  "COPD_PCT",
  "COPD_CHANGE_PCT",
  "SPECIALTY_RX_PCT",
  "SPECIALTY_RX_CHANGE_PCT",
  "AVG_SPEND_PER_MEMBER",
  "PCT_MEMBERS_ENROLLED",
  "PROFILE_MEMBER_COUNT",
  "PROFILE_EXPOSURE_MEMBER_MONTHS",
  "TRAIN_OBS_EPMPM",
  "TRAIN_PRE_MED_PMPM",
  "TRAIN_POST_MED_PMPM",
  "TRAIN_MED_PMPM_CHANGE",
  "TRAIN_MED_PMPM_RATIO",
  "TRAIN_PRE_RX_PMPM",
  "TRAIN_POST_RX_PMPM",
  "TRAIN_RX_PMPM_CHANGE",
  "TRAIN_RX_PMPM_RATIO",
  "TRAIN_PRE_TOTAL_PMPM",
  "TRAIN_POST_TOTAL_PMPM",
  "TRAIN_TOTAL_PMPM_CHANGE",
  "TRAIN_TOTAL_PMPM_RATIO",
  "PROFILE_AVAILABLE"
]
```

### `member_future_columns` (38 entries)

```json
[
  "IP_PCT",
  "ER_PCT",
  "COMPLEX_PREG_PCT",
  "ONC_TREATMENT_PCT",
  "DIALYSIS_PCT",
  "PREDIABETES_PCT",
  "T2D_PCT",
  "HYPERTENSION_PCT",
  "CKD_ANY_PCT",
  "CKD_EARLY_PCT",
  "CKD_ADVANCED_PCT",
  "HEART_FAILURE_PCT",
  "FRAILTY_PCT",
  "OSTEOPOROSIS_PCT",
  "FALL_EVENT_PCT",
  "FRACTURE_PCT",
  "AFIB_PCT",
  "COPD_PCT",
  "SPECIALTY_RX_PCT",
  "IP_ESCALATION_PCT",
  "ER_ESCALATION_PCT",
  "COMPLEX_PREG_ESCALATION_PCT",
  "ONC_TREATMENT_ESCALATION_PCT",
  "DIALYSIS_ESCALATION_PCT",
  "PREDIABETES_ESCALATION_PCT",
  "T2D_ESCALATION_PCT",
  "HYPERTENSION_ESCALATION_PCT",
  "CKD_ANY_ESCALATION_PCT",
  "CKD_EARLY_ESCALATION_PCT",
  "CKD_ADVANCED_ESCALATION_PCT",
  "HEART_FAILURE_ESCALATION_PCT",
  "FRAILTY_ESCALATION_PCT",
  "OSTEOPOROSIS_ESCALATION_PCT",
  "FALL_EVENT_ESCALATION_PCT",
  "FRACTURE_ESCALATION_PCT",
  "AFIB_ESCALATION_PCT",
  "COPD_ESCALATION_PCT",
  "SPECIALTY_RX_ESCALATION_PCT"
]
```

### `crude_future_columns` (19 entries)

```json
[
  "IP_PCT",
  "ER_PCT",
  "COMPLEX_PREG_PCT",
  "ONC_TREATMENT_PCT",
  "DIALYSIS_PCT",
  "PREDIABETES_PCT",
  "T2D_PCT",
  "HYPERTENSION_PCT",
  "CKD_ANY_PCT",
  "CKD_EARLY_PCT",
  "CKD_ADVANCED_PCT",
  "HEART_FAILURE_PCT",
  "FRAILTY_PCT",
  "OSTEOPOROSIS_PCT",
  "FALL_EVENT_PCT",
  "FRACTURE_PCT",
  "AFIB_PCT",
  "COPD_PCT",
  "SPECIALTY_RX_PCT"
]
```

### `prospective_static_columns` (56 entries)

```json
[
  "EST_POST_RATE_IP",
  "EST_POST_RATE_ER",
  "EST_POST_RATE_COMPLEX_PREG",
  "EST_POST_RATE_ONC_TREATMENT",
  "EST_POST_RATE_DIALYSIS",
  "EST_POST_RATE_PREDIABETES",
  "EST_POST_RATE_T2D",
  "EST_POST_RATE_HYPERTENSION",
  "EST_POST_RATE_CKD_ANY",
  "EST_POST_RATE_CKD_EARLY",
  "EST_POST_RATE_CKD_ADVANCED",
  "EST_POST_RATE_HEART_FAILURE",
  "EST_POST_RATE_FRAILTY",
  "EST_POST_RATE_OSTEOPOROSIS",
  "EST_POST_RATE_FALL_EVENT",
  "EST_POST_RATE_FRACTURE",
  "EST_POST_RATE_AFIB",
  "EST_POST_RATE_COPD",
  "EST_POST_RATE_SPECIALTY_RX",
  "P_ESC_IP_ESCALATION",
  "P_ESC_ER_ESCALATION",
  "P_ESC_COMPLEX_PREG_ESCALATION",
  "P_ESC_ONC_TREATMENT_ESCALATION",
  "P_ESC_DIALYSIS_ESCALATION",
  "P_ESC_PREDIABETES_ESCALATION",
  "P_ESC_T2D_ESCALATION",
  "P_ESC_HYPERTENSION_ESCALATION",
  "P_ESC_CKD_ANY_ESCALATION",
  "P_ESC_CKD_EARLY_ESCALATION",
  "P_ESC_CKD_ADVANCED_ESCALATION",
  "P_ESC_HEART_FAILURE_ESCALATION",
  "P_ESC_FRAILTY_ESCALATION",
  "P_ESC_OSTEOPOROSIS_ESCALATION",
  "P_ESC_FALL_EVENT_ESCALATION",
  "P_ESC_FRACTURE_ESCALATION",
  "P_ESC_AFIB_ESCALATION",
  "P_ESC_COPD_ESCALATION",
  "P_ESC_SPECIALTY_RX_ESCALATION",
  "PRE_RATE_IP",
  "PRE_RATE_ER",
  "PRE_RATE_COMPLEX_PREG",
  "PRE_RATE_ONC_TREATMENT",
  "PRE_RATE_PREDIABETES",
  "PRE_RATE_T2D",
  "PRE_RATE_HYPERTENSION",
  "PRE_RATE_CKD_ANY",
  "PRE_RATE_CKD_EARLY",
  "PRE_RATE_CKD_ADVANCED",
  "PRE_RATE_HEART_FAILURE",
  "PRE_RATE_FRAILTY",
  "PRE_RATE_OSTEOPOROSIS",
  "PRE_RATE_FALL_EVENT",
  "PRE_RATE_FRACTURE",
  "PRE_RATE_AFIB",
  "PRE_RATE_COPD",
  "PRE_RATE_SPECIALTY_RX"
]
```

### `train__CMD` categorical vocabularies and outcome months

```json
{
  "plan_cat_vocab_sizes": {
    "MEMBER_GENDER": 4,
    "PRODUCT_MKT_BUSN_SEG_DESCR": 16,
    "PRODUCT_FULLY_SELF_INSURE_INDC": 3,
    "PRODUCT_NETWORK_ROLLUP": 25,
    "PRODUCT_NETWORK": 68,
    "PLAN_METAL_LEVEL": 6,
    "POLICY_COVERAGE_TIER": 15,
    "RATING_AREA": 15,
    "VALUE_BASED_NETWORK": 34,
    "CARE_COORDINATION_RISK": 6,
    "PRODUCT_ENTITY": 8,
    "MEMBER_RELATIONSHIP_STND_DESCR": 6,
    "ASSIGNED_PROV_TYPE_CD": 4,
    "ASSIGNED_SPP_CARE_SYSTEM": 14,
    "MEMBER_RES_STND_COUNTY": 2279,
    "MEMBER_RES_STND_STATE_CD": 54,
    "RATE_CELL_MAJ_PROG": 11,
    "RATE_CELL_ELIG_TYPE": 18
  },
  "outcome_months": [
    "2023-01-01",
    "2023-02-01",
    "2023-03-01",
    "2023-04-01",
    "2023-05-01",
    "2023-06-01",
    "2023-07-01",
    "2023-08-01",
    "2023-09-01",
    "2023-10-01",
    "2023-11-01",
    "2023-12-01"
  ]
}
```

### `one_year_forward_test__CMD` categorical vocabularies and outcome months

```json
{
  "plan_cat_vocab_sizes": {
    "MEMBER_GENDER": 4,
    "PRODUCT_MKT_BUSN_SEG_DESCR": 16,
    "PRODUCT_FULLY_SELF_INSURE_INDC": 3,
    "PRODUCT_NETWORK_ROLLUP": 25,
    "PRODUCT_NETWORK": 68,
    "PLAN_METAL_LEVEL": 6,
    "POLICY_COVERAGE_TIER": 15,
    "RATING_AREA": 15,
    "VALUE_BASED_NETWORK": 34,
    "CARE_COORDINATION_RISK": 6,
    "PRODUCT_ENTITY": 8,
    "MEMBER_RELATIONSHIP_STND_DESCR": 6,
    "ASSIGNED_PROV_TYPE_CD": 4,
    "ASSIGNED_SPP_CARE_SYSTEM": 12,
    "MEMBER_RES_STND_COUNTY": 2278,
    "MEMBER_RES_STND_STATE_CD": 54,
    "RATE_CELL_MAJ_PROG": 8,
    "RATE_CELL_ELIG_TYPE": 18
  },
  "outcome_months": [
    "2024-01-01",
    "2024-02-01",
    "2024-03-01",
    "2024-04-01",
    "2024-05-01",
    "2024-06-01",
    "2024-07-01",
    "2024-08-01",
    "2024-09-01",
    "2024-10-01",
    "2024-11-01",
    "2024-12-01"
  ]
}
```
