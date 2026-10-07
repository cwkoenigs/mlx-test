"""Full-scale contract for the synthetic CMD arrays (transcribed from
docs/synthetic_cmd_generator_prompt.md, Appendices A-C). Pure data, no numpy needed.

Every list here is checked at import time against the counts printed in the source document
(255 static, 324 seq, 18 plan_cat, 56 cluster, 38 member, 19 crude, 56 prospective, 17 root
arrays, 50 code families).
"""

from __future__ import annotations

SEQ_LEN = 105
OUT_LEN = 12

# ------------------------------------------------------------------------------------------
# Splits
# ------------------------------------------------------------------------------------------
SPLITS = {
    "train__CMD": {
        "index": 0,
        "n_members": 863_154,
        "id_prefix": "SYN_TRAIN_",
        "partial_exposure_members": 119_410,
        "strat_new": 96_081,
        "strat_non_new": 767_073,
        "outcome_months": [f"2023-{m:02d}-01" for m in range(1, 13)],
        "plan_cat_vocab_sizes": {
            "MEMBER_GENDER": 4, "PRODUCT_MKT_BUSN_SEG_DESCR": 16, "PRODUCT_FULLY_SELF_INSURE_INDC": 3,
            "PRODUCT_NETWORK_ROLLUP": 25, "PRODUCT_NETWORK": 68, "PLAN_METAL_LEVEL": 6,
            "POLICY_COVERAGE_TIER": 15, "RATING_AREA": 15, "VALUE_BASED_NETWORK": 34,
            "CARE_COORDINATION_RISK": 6, "PRODUCT_ENTITY": 8, "MEMBER_RELATIONSHIP_STND_DESCR": 6,
            "ASSIGNED_PROV_TYPE_CD": 4, "ASSIGNED_SPP_CARE_SYSTEM": 14, "MEMBER_RES_STND_COUNTY": 2279,
            "MEMBER_RES_STND_STATE_CD": 54, "RATE_CELL_MAJ_PROG": 11, "RATE_CELL_ELIG_TYPE": 18,
        },
    },
    "one_year_forward_test__CMD": {
        "index": 1,
        "n_members": 847_568,
        "id_prefix": "SYN_TEST_",
        "partial_exposure_members": 117_136,
        "strat_new": 65_237,
        "strat_non_new": 782_331,
        # The source photo shows 2024-01-01 .. 2024-09-01 before being cut off; 12 months assumed.
        "outcome_months": [f"2024-{m:02d}-01" for m in range(1, 13)],
        "plan_cat_vocab_sizes": {
            "MEMBER_GENDER": 4, "PRODUCT_MKT_BUSN_SEG_DESCR": 16, "PRODUCT_FULLY_SELF_INSURE_INDC": 3,
            "PRODUCT_NETWORK_ROLLUP": 25, "PRODUCT_NETWORK": 68, "PLAN_METAL_LEVEL": 6,
            "POLICY_COVERAGE_TIER": 15, "RATING_AREA": 15, "VALUE_BASED_NETWORK": 34,
            "CARE_COORDINATION_RISK": 6, "PRODUCT_ENTITY": 8, "MEMBER_RELATIONSHIP_STND_DESCR": 6,
            "ASSIGNED_PROV_TYPE_CD": 4, "ASSIGNED_SPP_CARE_SYSTEM": 12, "MEMBER_RES_STND_COUNTY": 2278,
            "MEMBER_RES_STND_STATE_CD": 54, "RATE_CELL_MAJ_PROG": 8, "RATE_CELL_ELIG_TYPE": 18,
        },
    },
}
assert SPLITS["train__CMD"]["strat_new"] + SPLITS["train__CMD"]["strat_non_new"] == SPLITS["train__CMD"]["n_members"]
assert (SPLITS["one_year_forward_test__CMD"]["strat_new"] + SPLITS["one_year_forward_test__CMD"]["strat_non_new"]
        == SPLITS["one_year_forward_test__CMD"]["n_members"])

# ------------------------------------------------------------------------------------------
# Appendix A: root arrays (name, trailing shape after the member dimension, dtype)
# ------------------------------------------------------------------------------------------
ROOT_ARRAYS = [
    ("X_cluster_future", (OUT_LEN, 56), "float32"),
    ("X_crude_future", (OUT_LEN, 19), "float32"),
    ("X_member_future", (OUT_LEN, 38), "float32"),
    ("X_plan_cat", (18,), "int64"),
    ("X_prospective_static", (56,), "float32"),
    ("X_seq", (SEQ_LEN, 324), "float32"),
    ("X_static", (255,), "float32"),
    ("cluster_profile_mask", (OUT_LEN, 56), "float32"),
    ("crude_future_mask", (OUT_LEN, 19), "float32"),
    ("enrollment_mask", (SEQ_LEN,), "float32"),
    ("event_y", (OUT_LEN, 5), "float32"),
    ("member_future_mask", (OUT_LEN, 38), "float32"),
    ("outcome_enrollment_mask", (OUT_LEN,), "float32"),
    ("prospective_static_mask", (), "float32"),
    ("spend_archetype", (), "int8"),
    ("spend_archetype_multilabel", (6,), "float32"),
    ("y", (OUT_LEN,), "float32"),
]
assert len(ROOT_ARRAYS) == 17

# Source document GiB per file (train, forward test) for the size cross-check.
ROOT_ARRAY_GIB = {
    "X_cluster_future": (2.161, 2.122), "X_crude_future": (0.733, 0.720), "X_member_future": (1.466, 1.440),
    "X_plan_cat": (0.116, 0.114), "X_prospective_static": (0.180, 0.177), "X_seq": (109.391, 107.416),
    "X_static": (0.820, 0.805), "cluster_profile_mask": (2.161, 2.122), "crude_future_mask": (0.733, 0.720),
    "enrollment_mask": (0.338, 0.332), "event_y": (0.193, 0.189), "member_future_mask": (1.466, 1.440),
    "outcome_enrollment_mask": (0.039, 0.038), "prospective_static_mask": (0.003, 0.003),
    "spend_archetype": (0.001, 0.001), "spend_archetype_multilabel": (0.019, 0.019), "y": (0.039, 0.038),
}
TOTAL_GIB = {"train__CMD": 154.07, "one_year_forward_test__CMD": 151.34}
CODE_DIR_GIB = {"train__CMD": 34.205, "one_year_forward_test__CMD": 33.632}

# ------------------------------------------------------------------------------------------
# Appendix B: ragged code families (name, n_ids train, n_ids forward test, vocab train, vocab test)
# ------------------------------------------------------------------------------------------
CODE_FAMILIES = [
    ("admit_type_cd", 90_997_959, 89_377_525, 9, 9),
    ("auth_type_name", 90_631_779, 88_995_395, 9, 9),
    ("billtype", 90_726_872, 89_093_125, 37, 37),
    ("cpt_ccs_group", 98_432_142, 97_347_981, 91, 91),
    ("diags_ip_ccsr", 100_192_272, 99_450_888, 277, 277),
    ("diags_ip_ccsr_acute", 95_090_779, 93_835_051, 193, 193),
    ("diags_ip_ccsr_chronic", 93_924_391, 92_628_896, 167, 167),
    ("diags_op_ccsr", 103_496_160, 102_786_422, 278, 278),
    ("diags_op_ccsr_acute", 97_860_501, 96_636_345, 192, 192),
    ("diags_op_ccsr_chronic", 94_108_429, 92_820_189, 170, 170),
    ("discharge_status", 90_650_572, 89_015_497, 14, 14),
    ("drug_class_cd", 90_926_954, 89_392_888, 8, 8),
    ("drug_compound_indc", 90_641_081, 89_007_659, 5, 5),
    ("drug_formulary_indc", 90_660_970, 89_024_975, 5, 5),
    ("drug_generic_cd", 91_123_547, 89_586_241, 5, 5),
    ("esi_contracted_specialty_indc", 90_631_170, 88_994_640, 4, 4),
    ("esi_drug_type", 90_929_842, 89_396_296, 7, 7),
    ("ip_notif_admission_reason", 90_631_397, 88_994_869, 23, 23),
    ("ip_notif_auth_type", 90_631_268, 88_994_731, 7, 7),
    ("ip_notif_diags_ccsr_ip", 90_631_760, 88_995_174, 22, 22),
    ("ip_notif_treatment_type", 90_631_376, 88_994_848, 21, 21),
    ("lifestyle_domain", 90_927_432, 89_345_023, 38, 38),
    ("mail_order_retail", 90_650_422, 89_016_998, 5, 5),
    ("medicare_part_code", 90_631_170, 88_994_640, 5, 5),
    ("milliman_hcg_drg", 90_632_828, 88_996_253, 191, 191),
    ("milliman_hcg_family1", 100_333_706, 99_060_412, 133, 133),
    ("milliman_hcg_family2", 91_090_520, 89_519_059, 97, 97),
    ("milliman_hcg_mr_line_case", 103_175_307, 102_869_324, 109, 109),
    ("milliman_hcg_mr_line_case_svc_type_catgry", 91_824_872, 90_274_062, 6, 6),
    ("milliman_hcg_spec_type", 92_737_984, 91_107_164, 11, 11),
    ("ndc_ahfs_cd", 90_762_166, 89_128_310, 77, 77),
    ("ndc_dosage_form_name", 92_560_242, 91_580_378, 69, 69),
    ("ndc_gpi_6_descr", 94_049_642, 93_245_786, 217, 217),
    ("ndc_route_name", 91_340_848, 89_901_922, 36, 36),
    ("ndc_substance_name", 93_836_673, 93_014_818, 309, 309),
    ("new_refill_cd", 91_344_278, 89_893_627, 5, 5),
    ("paid_drg_cd", 90_632_265, 88_995_731, 215, 215),
    ("paid_drg_mdc_cd", 90_631_553, 88_995_022, 25, 25),
    ("pharmacy_class_code", 90_702_132, 89_095_791, 13, 13),
    ("place_of_service_cd", 92_789_784, 91_057_245, 25, 25),
    ("pos_type", 91_815_450, 90_264_528, 6, 6),
    ("prior_auth_diag_ccsr_ip", 90_631_240, 88_994_957, 98, 98),
    ("prior_auth_diag_ccsr_op", 90_631_253, 88_995_031, 105, 105),
    ("proc_cd", 108_195_489, 107_292_039, 642, 642),
    ("proc_description", 90_632_173, 88_995_844, 86, 86),
    ("provider_servicing_taxonomy", 94_678_550, 93_137_992, 101, 100),
    ("provider_servicing_type_cd", 94_605_854, 93_057_254, 79, 79),
    ("revenue_cd", 93_668_165, 92_211_955, 137, 137),
    ("treatment_type_name", 90_631_902, 88_995_415, 94, 94),
    ("ubh_indc", 90_631_170, 88_994_640, 4, 4),
]
assert len(CODE_FAMILIES) == 50
for _f in CODE_FAMILIES:  # every bag has at least one entry (zero = padding)
    assert _f[1] >= SPLITS["train__CMD"]["n_members"] * SEQ_LEN, _f
    assert _f[2] >= SPLITS["one_year_forward_test__CMD"]["n_members"] * SEQ_LEN, _f

# ------------------------------------------------------------------------------------------
# Appendix C: ordered metadata lists
# ------------------------------------------------------------------------------------------
_LIFESTYLE = [
    "TOBACCO_DEPENDENCE", "TOBACCO_CURRENT_USE", "TOBACCO_FORMER_SMOKER", "TOBACCO_ENVIRONMENTAL_EXPOSURE",
    "ALCOHOL_DISORDER", "OPIOID_DISORDER", "CANNABIS_DISORDER", "SEDATIVE_DISORDER", "COCAINE_DISORDER",
    "STIMULANT_DISORDER", "HALLUCINOGEN_DISORDER", "INHALANT_DISORDER", "POLYSUBSTANCE_DISORDER",
    "OPIOID_POISONING", "OBESITY", "BMI_CLASS_I_30_34", "BMI_CLASS_II_35_39", "BMI_CLASS_III_40_PLUS",
    "BMI_OVERWEIGHT_25_29", "BMI_UNDERWEIGHT", "BMI_OTHER", "PHYSICAL_INACTIVITY", "POOR_DIET",
    "GAMBLING_DISORDER", "MED_NONADHERENCE_INTENTIONAL", "MED_NONADHERENCE_UNINTENTIONAL",
    "MEDICAL_NONCOMPLIANCE", "NONCOMPLIANCE_OTHER", "VACCINE_REFUSAL_UNDERIMMUNIZATION",
    "SDOH_EDUCATION_LITERACY", "SDOH_EMPLOYMENT", "SDOH_OCCUPATIONAL_EXPOSURE", "SDOH_HOUSING_ECONOMIC",
    "SDOH_SOCIAL_ENVIRONMENT", "SDOH_UPBRINGING", "SDOH_FAMILY_CIRCUMSTANCES", "SDOH_PSYCHOSOCIAL",
    "SDOH_LEGAL_CIRCUMSTANCES", "SLEEP_APNEA", "SLEEP_OTHER", "SLEEP_BEHAVIORAL", "EATING_ANOREXIA",
    "EATING_BULIMIA", "EATING_BINGE", "EATING_OTHER",
]
assert len(_LIFESTYLE) == 45

STATIC_COLUMNS = [
    "AGE_AS_OF_FIRST_OF_MNTH", "CONTINUOUS_ENROLLED_MONTHS", "COVERAGE_GAP_MONTHS_PRIOR_12M",
    "ENROLLED_MONTHS_PRIOR_12M", "ENROLLED_MONTHS_PRIOR_3M", "ENROLLED_MONTHS_PRIOR_6M", "ENROLLMENT_FRACTION",
    "LOB_TRANSITION_FLAG", "MONTHS_SINCE_FIRST_ENROLLMENT", "MONTHS_TO_PLAN_EXPIRY", "NETWORK_TRANSITION_FLAG",
    "NEW_MEMBER_FLAG", "PLAN_EXPIRING_WITHIN_3M_FLAG", "RETURNING_MEMBER_FLAG", "SMOKER_INDC",
    "STATE_TRANSITION_FLAG",
    "ZIP5_AREA_SUMMARY_AGE_30S", "ZIP5_AREA_SUMMARY_AGE_60S", "ZIP5_AREA_SUMMARY_AGE_70S",
    "ZIP5_AREA_SUMMARY_AGE_MEDIAN", "ZIP5_AREA_SUMMARY_AGE_OVER_80", "ZIP5_AREA_SUMMARY_AGE_UNDER_10",
    "ZIP5_AREA_SUMMARY_FAMILY_SIZE", "ZIP5_AREA_SUMMARY_FARMER", "ZIP5_AREA_SUMMARY_INCOME_HOUSEHOLD_20_TO_25",
    "ZIP5_AREA_SUMMARY_INCOME_HOUSEHOLD_25_TO_35", "ZIP5_AREA_SUMMARY_LABOR_FORCE_PARTICIPATION",
    "ZIP5_AREA_SUMMARY_RACE_MULTIPLE", "ZIP5_AREA_SUMMARY_SELF_EMPLOYED", "ZIP5_AREA_SUMMARY_UNEMPLOYMENT_RATE",
    "ZIP5_AREA_SUMMARY_VETERAN", "ZIP5_AREA_SUMMARY_WIDOWED",
    "TOTAL_SPEND_PM_LAST_12M_PMPM", "TOTAL_UTIL_PM_LAST_12M_PMPM", "TOTAL_SPEND_ER_PM_LAST_12M_PMPM",
    "TOTAL_UTIL_ER_PM_LAST_12M_PMPM", "TOTAL_SPEND_IP_PM_LAST_12M_PMPM", "TOTAL_UTIL_IP_PM_LAST_12M_PMPM",
    "TOTAL_SPEND_OP_PM_LAST_12M_PMPM", "TOTAL_SPEND_RX_PM_LAST_12M_PMPM", "TOTAL_UTIL_RX_PM_LAST_12M_PMPM",
    "TOTAL_SPEND_MENTAL_HEALTH_PM_LAST_12M_PMPM", "TOTAL_UTIL_MENTAL_HEALTH_PM_LAST_12M_PMPM",
    "TOTAL_SPEND_ONCOLOGY_CHEMO_PM_LAST_12M_PMPM", "TOTAL_SPEND_SNF_PM_LAST_12M_PMPM",
    "TOTAL_SPEND_SUD_PM_LAST_12M_PMPM", "TOTAL_SPEND_DIALYSIS_PM_LAST_12M_PMPM",
    "TOTAL_SPEND_TREND_6M_12M", "TOTAL_UTIL_TREND_6M_12M", "TOTAL_SPEND_ER_TREND_6M_12M",
    "TOTAL_SPEND_IP_TREND_6M_12M", "TOTAL_SPEND_OP_TREND_6M_12M", "TOTAL_SPEND_RX_TREND_6M_12M",
    "TOTAL_SPEND_MENTAL_HEALTH_TREND_6M_12M", "TOTAL_SPEND_ONCOLOGY_CHEMO_TREND_6M_12M",
    "TOTAL_SPEND_SNF_TREND_6M_12M", "TOTAL_SPEND_SUD_TREND_6M_12M", "TOTAL_SPEND_DIALYSIS_TREND_6M_12M",
    "TOTAL_UTIL_ER_TREND_6M_12M", "TOTAL_UTIL_IP_TREND_6M_12M", "TOTAL_UTIL_RX_TREND_6M_12M",
    "TOTAL_UTIL_MENTAL_HEALTH_TREND_6M_12M",
    "RX_SPEND_SPECIALTY_PM_LAST_12M", "RX_SPEND_GLP1_INCRETIN_PM_LAST_12M", "RX_SPEND_ANTINEOPLASTIC_PM_LAST_12M",
    "RX_SPEND_OPIOID_PM_LAST_12M", "RX_TOTAL_FILLS_PM_LAST_12M",
    "CM_ANY_PROGRAM_OPEN_AT_INDEX", "CM_TOTAL_PROGRAMS_EVER", "CM_MONTHS_IN_ANY_PROGRAM", "CM_HIGH_TIER_EVER",
    "CM_MEDIUM_TIER_EVER", "CM_HIGH_TIER_MONTHS", "CM_MEDIUM_TIER_MONTHS",
    "PRIOR_AUTH_TOTAL_12M", "PRIOR_AUTH_DENIED_RATE_12M", "IP_NOTIF_TOTAL_12M", "IP_NOTIF_DISTINCT_CCSR_12M",
    "PREMIUM_RATE", "BILLED_PREMIUM_AMT", "RX_CARVE_OUT_INDC", "TERM_MEMBER_INDC",
    *[f"LIFESTYLE_{c}_SVC_COUNT" for c in _LIFESTYLE],
    *[f"LIFESTYLE_{c}_MONTHS_SINCE_LAST" for c in _LIFESTYLE],
    "TOTAL_CHRONIC_DX_COUNT_24M", "TOTAL_ACUTE_DX_COUNT_24M", "TOTAL_UNIQUE_DX_24M", "CHRONIC_DX_RATE_PER_12M",
    "ACUTE_DX_RATE_PER_12M", "DX_DENSITY_PER_12M", "CHRONIC_TO_ACUTE_RATIO", "MAX_CHRONIC_DX_RECURRENCE",
    "UNIQUE_BODY_SYSTEMS_24M", "MULTI_SYSTEM_FLAG", "NEW_CHRONIC_DX_LAST_6M", "CHRONIC_DX_RATE_LAST_6M",
    "ACUTE_DX_RATE_LAST_6M", "DX_DENSITY_LAST_6M", "TOTAL_UNIQUE_NDCS_24M", "SPECIALTY_RX_MONTHS",
    "SPECIALTY_RX_RATE_24M", "SPECIALTY_RX_RATE_LAST_6M", "POLYPHARMACY_MAX_24M", "NEW_NDCS_LAST_6M",
    "IP_ADMISSIONS_24M", "ER_VISITS_24M", "IP_ADMITS_PER_12M", "ER_VISITS_PER_12M", "IP_ADMITS_LAST_6M",
    "ER_VISITS_LAST_6M", "IP_RATE_LAST_6M", "ER_RATE_LAST_6M", "TOTAL_SPEND_LAST_6M", "SPEND_RATE_LAST_6M",
    "UNIQUE_PROVIDERS_24M", "UNIQUE_SPECIALTIES_24M", "UNIQUE_TINS_24M", "PROVIDER_RATE_LAST_6M",
    "MONTHS_SINCE_LAST_IP", "MONTHS_SINCE_LAST_ER", "MONTHS_SINCE_LAST_SPECIALIST", "MONTHS_SINCE_LAST_RX_FILL",
    "MONTHS_SINCE_LAST_SPECIALTY_RX", "MONTHS_SINCE_LAST_CHRONIC_DX", "MONTHS_SINCE_LAST_ACUTE_DX",
    "MONTHS_SINCE_LAST_PROCEDURE", "MONTHS_SINCE_LAST_HIGH_COST_EVENT", "MONTHS_SINCE_LAST_PRIOR_AUTH",
    "CRF_FRAILTY_FLAG", "CRF_FRAILTY_CONCEPT_COUNT", "CRF_CANCER_ACTIVE_FLAG", "CRF_CANCER_HISTORY_FLAG",
    "CRF_DIALYSIS_FLAG", "CRF_PREGNANT_FLAG", "CRF_PAST_DELIVERY_FLAG", "CRF_DIABETES_FLAG", "CRF_CHF_FLAG",
    "CRF_COPD_FLAG", "CRF_ASTHMA_FLAG", "CRF_DEPRESSION_FLAG", "CRF_BIPOLAR_FLAG", "CRF_SCHIZOPHRENIA_FLAG",
    "CRF_SUD_FLAG", "CRF_HIV_FLAG", "CRF_TRANSPLANT_FLAG", "CRF_HYPERTENSION_FLAG", "CRF_ISCHEMIC_HEART_FLAG",
    "CRF_OSTEOPOROSIS_FLAG", "CRF_RHEUMATOID_ARTHRITIS_FLAG", "CRF_SEIZURE_FLAG", "CRF_PARKINSONS_FLAG",
    "CRF_DEMENTIA_FLAG", "CRF_OBESITY_FLAG", "CRF_MECHANICAL_VENTILATION_FLAG", "CRF_ON_INSULIN_FLAG",
    "CRF_ON_ANTINEOPLASTIC_FLAG", "CRF_ON_ANTICOAGULANT_FLAG", "CRF_ON_OPIOID_CHRONIC_FLAG",
    "CRF_ON_ANTIPSYCHOTIC_FLAG", "CRF_ON_ANTIDEPRESSANT_FLAG", "CRF_ON_IMMUNOSUPPRESSANT_FLAG",
    "CRF_CHRONIC_CONDITION_COUNT", "CRF_POLYPHARMACY_10PLUS_FLAG", "CRF_HIGH_RISK_COMPOSITE",
    "CRF_MULTI_CHRONIC_4PLUS_FLAG", "CRF_PREGNANCY_RISK_FLAG", "CRF_MAJOR_PROCEDURE_COUNT",
]
assert len(STATIC_COLUMNS) == 255, len(STATIC_COLUMNS)

# The last 32 x 4 seq columns follow a fixed per-code-family pattern; the order below is the
# order they appear in the source document.
_SEQ_CODE_GROUPS = [
    "PROC_CD", "NDC_SUBSTANCE_NAME", "NDC_GPI_6_DESCR", "PAID_DRG_CD", "MILLIMAN_HCG_DRG",
    "MILLIMAN_HCG_FAMILY1", "MILLIMAN_HCG_FAMILY2", "MILLIMAN_HCG_MR_LINE_CASE", "PROVIDER_SERVICING_TAXONOMY",
    "CPT_CCS_GROUP", "NDC_ROUTE_NAME", "PRIOR_AUTH_DIAG_CCSR_OP", "PRIOR_AUTH_DIAG_CCSR_IP", "DIAGS_IP_CCSR",
    "DIAGS_OP_CCSR", "DIAGS_IP_CCSR_ACUTE", "DIAGS_OP_CCSR_ACUTE", "DIAGS_IP_CCSR_CHRONIC",
    "DIAGS_OP_CCSR_CHRONIC", "REVENUE_CD", "NDC_GPI_4_DESCR", "NDC_GPI_2_DESCR", "NDC_AHFS_CD",
    "NDC_DOSAGE_FORM_NAME", "PROVIDER_SERVICING_TYPE_CD", "PAID_DRG_MDC_CD", "TREATMENT_TYPE_NAME",
    "PROC_DESCRIPTION", "IP_NOTIF_ADMISSION_REASON", "IP_NOTIF_DIAGS_CCSR_IP", "IP_NOTIF_TREATMENT_TYPE",
    "LIFESTYLE_DOMAIN",
]
assert len(_SEQ_CODE_GROUPS) == 32

SEQ_COLUMNS = [
    "TOTAL_SPEND", "TOTAL_SPEND_IP", "TOTAL_SPEND_OP", "TOTAL_SPEND_PROF", "TOTAL_SPEND_RX", "TOTAL_SPEND_ER",
    "TOTAL_SPEND_MENTAL_HEALTH", "TOTAL_SPEND_SURGERY", "TOTAL_SPEND_RADIOLOGY", "TOTAL_SPEND_THERAPY",
    "TOTAL_SPEND_HOME_HEALTH", "TOTAL_SPEND_DME", "TOTAL_SPEND_CARDIOVASCULAR", "TOTAL_SPEND_MATERNITY",
    "TOTAL_SPEND_PREVENTIVE", "TOTAL_SPEND_SPECIALTY_RX", "TOTAL_SPEND_INJECTION_JQ", "TOTAL_SPEND_DIALYSIS",
    "TOTAL_SPEND_CHEMO_ADMIN",
    "TOTAL_UTIL", "TOTAL_UTIL_IP", "TOTAL_UTIL_OP", "TOTAL_UTIL_PROF", "TOTAL_UTIL_RX", "TOTAL_UTIL_ER",
    "TOTAL_UTIL_MENTAL_HEALTH", "TOTAL_UTIL_SURGERY", "TOTAL_UTIL_RADIOLOGY", "TOTAL_UTIL_HOME_HEALTH",
    "TOTAL_UTIL_INJECTION_JQ", "TOTAL_UTIL_PREVENTIVE", "TOTAL_RX_FILLS", "TOTAL_RX_FILLS_SPECIALTY",
    "CM_ANY_CASE_OPEN", "CM_TOTAL_CASES_OPEN", "CM_ANY_CASE_CLOSED", "CM_ENGAGED_ANY", "CM_HIGH_TIER_OPEN",
    "CM_MEDIUM_TIER_OPEN", "PRIOR_AUTH_COUNT", "PRIOR_AUTH_APPROVED_COUNT", "PRIOR_AUTH_DENIED_COUNT",
    "IP_NOTIF_COUNT", "IP_NOTIF_POST_ACUTE_COUNT",
    "WEEKS_SINCE_LAST_DX", "WEEKS_SINCE_LAST_IP", "WEEKS_SINCE_LAST_ER", "WEEKS_SINCE_LAST_PROCEDURE",
    "WEEKS_SINCE_LAST_RX_FILL", "WEEKS_SINCE_LAST_PRIOR_AUTH", "WEEKS_SINCE_LAST_HIGH_COST_EVENT",
    "TIME_SINCE_FIRST_OBSERVED_EVENT", "OBSERVED_THIS_PERIOD", "NUMBER_OF_OBSERVATIONS", "DAYS_WITH_ANY_ACTIVITY",
    "PHARMACY_OBSERVED_T", "MEDICAL_CLAIM_OBSERVED_T", "CARE_MANAGEMENT_OBSERVED_T",
    "CUM_SPEND_TOTAL", "CUM_SPEND_IP", "CUM_SPEND_RX", "CUM_SPEND_ER", "CUM_UTIL_TOTAL", "CUM_UTIL_IP",
    "CUM_UTIL_ER", "CUM_RX_FILLS", "SPEND_DELTA_4W", "SPEND_DELTA_12W", "UTIL_DELTA_4W", "UTIL_DELTA_12W",
    "SPEND_MA_4W", "SPEND_MA_12W", "UTIL_MA_4W", "UTIL_MA_12W",
    "NEW_DRUG_START", "POLYPHARMACY_COUNT", "SPECIALTY_DRUG_COUNT", "HIGH_COST_DRUG_COUNT",
    "INJECTABLE_DRUG_COUNT", "DAYS_SUPPLY_AVG", "CUM_UNIQUE_NDCS", "NEW_RX_THIS_PERIOD", "TOTAL_ACTIVE_NDCS",
    "RX_FLAG_T", "SPECIALTY_RX_FLAG_T", "NEW_SPECIALTY_RX_COUNT", "NDC_ACCUMULATION_RATE",
    "WEEKS_SINCE_LAST_NEW_NDC", "WEEKS_SINCE_LAST_SPECIALTY_RX",
    "NEW_PRIOR_AUTH", "PRIOR_AUTH_APPROVED_THIS_PERIOD", "PRIOR_AUTH_DENIED_THIS_PERIOD",
    "WEEKS_SINCE_PRIOR_AUTH_EVENT", "PRIOR_AUTH_FREQUENCY_4W", "PRIOR_AUTH_FREQUENCY_12W",
    "NUMBER_OF_ACTIVE_AUTHS", "COMPLEX_CARE_ENROLLMENT_THIS_PERIOD", "PA_FLAG_T", "CUM_PRIOR_AUTHS",
    "CUM_PA_DENIALS", "PA_ACCUMULATION_RATE", "WEEKS_SINCE_LAST_PA_DENIAL",
    "ELIGIBILITY_THIS_PERIOD", "MONTHS_SINCE_ENROLLMENT", "CONTINUOUS_ENROLLMENT_STREAK", "GAP_LENGTH",
    "NEW_MEMBER_FLAG_T", "RETURNING_MEMBER_FLAG_T",
    "RECENT_IP_ADMISSION", "RECENT_ER_VISIT", "IP_DAYS_LAST_4W", "IP_DAYS_LAST_12W", "ER_VISITS_LAST_4W",
    "ER_VISITS_LAST_12W", "READMISSION_WITHIN_30D", "DISCHARGE_TO_SNF", "DISCHARGE_TO_HOME",
    "ICU_DAYS_THIS_PERIOD", "OBSERVATION_DAYS_THIS_PERIOD", "OBSERVATION_TO_IP_THIS_PERIOD",
    "ER_TO_IP_THIS_PERIOD", "DISCHARGE_TO_LTACH", "DISCHARGE_TO_REHAB", "DISCHARGE_TO_PSYCH",
    "DISCHARGE_TO_HOSPICE", "DISCHARGE_TO_OTHER_FACILITY", "DISCHARGE_AMA", "EXPIRED_THIS_PERIOD",
    "IP_ADMISSION_COUNT_T", "ER_VISIT_COUNT_T", "CUM_IP_ADMISSIONS", "CUM_ER_VISITS", "IP_ACCUMULATION_RATE",
    "ER_ACCUMULATION_RATE", "WEEKS_SINCE_LAST_IP_ADMISSION", "WEEKS_SINCE_LAST_ER_VISIT",
    "WEEKS_SINCE_LAST_READMISSION", "WEEKS_SINCE_LAST_ICU",
    "UNIQUE_PROVIDERS_4W", "UNIQUE_PROVIDERS_12W", "UNIQUE_SPECIALTIES", "FACILITY_COUNT",
    "CUM_UNIQUE_PROVIDERS", "REFERRAL_COUNT", "NEW_PROVIDER_SPECIALTY_THIS_PERIOD", "REFERRAL_CHAIN_DEPTH",
    "CUM_REFERRALS", "REFERRAL_RATE_4W", "NEW_REFERRING_PROVIDER", "DISTINCT_REFERRING_SPECIALTIES",
    "OUT_OF_NETWORK_CLAIMS", "OUT_OF_NETWORK_FRACTION", "VALUE_BASED_NETWORK_CLAIMS",
    "VALUE_BASED_NETWORK_FRACTION", "UNIQUE_TINS", "UNIQUE_SUPERTINS", "TIN_CONCENTRATION_HHI",
    "CUM_UNIQUE_TINS", "NEW_PROVIDER_FLAG_T", "REFERRAL_FLAG_T", "PROVIDER_ACCUMULATION_RATE",
    "TIN_ACCUMULATION_RATE", "WEEKS_SINCE_LAST_NEW_PROVIDER", "WEEKS_SINCE_LAST_REFERRAL",
    "NEW_DX_THIS_PERIOD", "CUM_UNIQUE_DX", "DX_ACCUMULATION_RATE", "CHRONIC_DX_COUNT_T", "ACUTE_DX_COUNT_T",
    "NEW_CHRONIC_DX", "NEW_ACUTE_DX", "NEW_CHRONIC_DX_COUNT", "NEW_ACUTE_DX_COUNT", "CHRONIC_DX_FLAG_T",
    "ACUTE_DX_FLAG_T", "CUM_UNIQUE_CHRONIC_DX", "CUM_UNIQUE_ACUTE_DX", "CHRONIC_DX_ACCUMULATION_RATE",
    "ACUTE_DX_ACCUMULATION_RATE", "WEEKS_SINCE_LAST_CHRONIC_DX", "WEEKS_SINCE_LAST_ACUTE_DX",
    "WEEKS_SINCE_LAST_NEW_DX", "WEEKS_SINCE_LAST_NEW_CHRONIC_DX", "WEEKS_SINCE_LAST_NEW_ACUTE_DX",
    "LIFESTYLE_NEW_CATEGORY_THIS_PERIOD", "LIFESTYLE_CUM_UNIQUE_CATEGORIES", "LIFESTYLE_CUM_SVC_COUNT",
    "LIFESTYLE_TREND_4W", "LIFESTYLE_FLAG_T", "LIFESTYLE_NEW_CATEGORY_COUNT", "LIFESTYLE_SVC_COUNT_T",
    "LIFESTYLE_ACCUMULATION_RATE", "WEEKS_SINCE_LAST_LIFESTYLE_EVENT", "WEEKS_SINCE_LAST_NEW_LIFESTYLE_CATEGORY",
    "OBSERVATION_MATURITY", "PARTIAL_PERIOD_FLAG",
]
for _g in _SEQ_CODE_GROUPS:
    SEQ_COLUMNS += [f"NEW_{_g}_THIS_PERIOD", f"CUM_UNIQUE_{_g}", f"{_g}_ACCUM_RATE", f"WEEKS_SINCE_LAST_NEW_{_g}"]
assert len(SEQ_COLUMNS) == 324, len(SEQ_COLUMNS)
assert len(set(SEQ_COLUMNS)) == 324 and len(set(STATIC_COLUMNS)) == 255

PLAN_CAT_COLUMNS = [
    "MEMBER_GENDER", "PRODUCT_MKT_BUSN_SEG_DESCR", "PRODUCT_FULLY_SELF_INSURE_INDC", "PRODUCT_NETWORK_ROLLUP",
    "PRODUCT_NETWORK", "PLAN_METAL_LEVEL", "POLICY_COVERAGE_TIER", "RATING_AREA", "VALUE_BASED_NETWORK",
    "CARE_COORDINATION_RISK", "PRODUCT_ENTITY", "MEMBER_RELATIONSHIP_STND_DESCR", "ASSIGNED_PROV_TYPE_CD",
    "ASSIGNED_SPP_CARE_SYSTEM", "MEMBER_RES_STND_COUNTY", "MEMBER_RES_STND_STATE_CD", "RATE_CELL_MAJ_PROG",
    "RATE_CELL_ELIG_TYPE",
]
assert len(PLAN_CAT_COLUMNS) == 18
for _s in SPLITS.values():
    assert list(_s["plan_cat_vocab_sizes"]) == PLAN_CAT_COLUMNS

_CONDITIONS = [
    "IP", "ER", "COMPLEX_PREG", "ONC_TREATMENT", "DIALYSIS", "PREDIABETES", "T2D", "HYPERTENSION", "CKD_ANY",
    "CKD_EARLY", "CKD_ADVANCED", "HEART_FAILURE", "FRAILTY", "OSTEOPOROSIS", "FALL_EVENT", "FRACTURE", "AFIB",
    "COPD", "SPECIALTY_RX",
]
assert len(_CONDITIONS) == 19

CLUSTER_FUTURE_COLUMNS = [x for c in _CONDITIONS for x in (f"{c}_PCT", f"{c}_CHANGE_PCT")] + [
    "AVG_SPEND_PER_MEMBER", "PCT_MEMBERS_ENROLLED", "PROFILE_MEMBER_COUNT", "PROFILE_EXPOSURE_MEMBER_MONTHS",
    "TRAIN_OBS_EPMPM", "TRAIN_PRE_MED_PMPM", "TRAIN_POST_MED_PMPM", "TRAIN_MED_PMPM_CHANGE", "TRAIN_MED_PMPM_RATIO",
    "TRAIN_PRE_RX_PMPM", "TRAIN_POST_RX_PMPM", "TRAIN_RX_PMPM_CHANGE", "TRAIN_RX_PMPM_RATIO",
    "TRAIN_PRE_TOTAL_PMPM", "TRAIN_POST_TOTAL_PMPM", "TRAIN_TOTAL_PMPM_CHANGE", "TRAIN_TOTAL_PMPM_RATIO",
    "PROFILE_AVAILABLE",
]
MEMBER_FUTURE_COLUMNS = [f"{c}_PCT" for c in _CONDITIONS] + [f"{c}_ESCALATION_PCT" for c in _CONDITIONS]
CRUDE_FUTURE_COLUMNS = [f"{c}_PCT" for c in _CONDITIONS]
# PRE_RATE_* has no DIALYSIS entry in the source document (18 names), giving 19 + 19 + 18 = 56.
PROSPECTIVE_STATIC_COLUMNS = ([f"EST_POST_RATE_{c}" for c in _CONDITIONS]
                              + [f"P_ESC_{c}_ESCALATION" for c in _CONDITIONS]
                              + [f"PRE_RATE_{c}" for c in _CONDITIONS if c != "DIALYSIS"])
assert len(CLUSTER_FUTURE_COLUMNS) == 56 and len(MEMBER_FUTURE_COLUMNS) == 38
assert len(CRUDE_FUTURE_COLUMNS) == 19 and len(PROSPECTIVE_STATIC_COLUMNS) == 56

FUTURE_COLUMN_LISTS = {
    "X_cluster_future": ("cluster_future_columns", CLUSTER_FUTURE_COLUMNS),
    "X_member_future": ("member_future_columns", MEMBER_FUTURE_COLUMNS),
    "X_crude_future": ("crude_future_columns", CRUDE_FUTURE_COLUMNS),
    "X_prospective_static": ("prospective_static_columns", PROSPECTIVE_STATIC_COLUMNS),
}
# feature array -> its mask array (same shape for the three future arrays; [N] for prospective)
MASK_FOR = {
    "X_cluster_future": "cluster_profile_mask",
    "X_member_future": "member_future_mask",
    "X_crude_future": "crude_future_mask",
    "X_prospective_static": "prospective_static_mask",
}

GIB = 2**30


def n_members_at(split: str, scale: float) -> int:
    n = SPLITS[split]["n_members"]
    return n if scale >= 1.0 else max(16, int(round(n * scale)))


def scaled_count(full_count: int, split: str, n_scaled: int) -> int:
    """Proportional version of a full-size count (partial-exposure members, New members, ...)."""
    n_full = SPLITS[split]["n_members"]
    return full_count if n_scaled == n_full else int(round(full_count * n_scaled / n_full))


def expected_n_ids(family: tuple, split: str, n_scaled: int) -> int:
    """Flat-ID count for a family: exact at full size, proportional extras above one-per-bag otherwise."""
    name, n_train, n_test, _, _ = family
    n_full_ids = n_train if split == "train__CMD" else n_test
    n_full = SPLITS[split]["n_members"]
    if n_scaled == n_full:
        return n_full_ids
    extra_full = n_full_ids - n_full * SEQ_LEN
    return n_scaled * SEQ_LEN + int(round(extra_full * n_scaled / n_full))


def vocab_size(family: tuple, split: str) -> int:
    return family[3] if split == "train__CMD" else family[4]


def root_array_shape(name: str, n: int) -> tuple:
    for nm, trailing, _ in ROOT_ARRAYS:
        if nm == name:
            return (n, *trailing)
    raise KeyError(name)


def root_array_dtype(name: str) -> str:
    for nm, _, dt in ROOT_ARRAYS:
        if nm == name:
            return dt
    raise KeyError(name)


def metadata_lists() -> dict:
    return {
        "static_columns": STATIC_COLUMNS, "seq_columns": SEQ_COLUMNS, "plan_cat_columns": PLAN_CAT_COLUMNS,
        "cluster_future_columns": CLUSTER_FUTURE_COLUMNS, "member_future_columns": MEMBER_FUTURE_COLUMNS,
        "crude_future_columns": CRUDE_FUTURE_COLUMNS, "prospective_static_columns": PROSPECTIVE_STATIC_COLUMNS,
    }


if __name__ == "__main__":
    import json
    print(json.dumps({"splits": SPLITS, "root_arrays": [(n, list(s), d) for n, s, d in ROOT_ARRAYS],
                      "code_families": CODE_FAMILIES, **metadata_lists()}, indent=2))
