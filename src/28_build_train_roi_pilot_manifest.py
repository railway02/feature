#!/usr/bin/env python3
"""Build a 30-lesion Train ROI Pilot review-first preselection manifest.

Only the Train blinded registry, blinded review tables, and candidate timing
audit are read.  Entries remain pending until a human-reviewed series decision
is present.  This script does not calculate ROI features or read private labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path("/root/autodl-tmp/aneurysm")
PILOT_SIZE = 30
SELECTION_SEED = "api_fullseq_v3_train_roi_pilot_review_first_v1"
HUMAN_REVIEWED_STATUSES = {"exact_reviewed", "probable_reviewed"}

FORBIDDEN_OUTPUT_TOKENS = (
    "rroc",
    "adverse",
    "followup_time",
    "follow-up",
    "follow up",
    "随访",
    "不良转归",
    "术后即刻",
)

COLUMNS = [
    "roi_pilot_entry_id",
    "selection_order",
    "selection_hash",
    "split",
    "lesion_uid",
    "patient_id",
    "source_excel_row_id",
    "side_raw",
    "side_normalized",
    "location_raw",
    "location_normalized",
    "lesion_index_normalized",
    "multiple_aneurysm_normalized",
    "registry_status",
    "review_scope",
    "lesion_review_page",
    "review_item_id",
    "manual_decision_status",
    "candidate_source_type",
    "candidate_source_root",
    "candidate_discovery_rank",
    "candidate_series_id",
    "candidate_series_path",
    "candidate_valid",
    "candidate_selected_in_v2",
    "candidate_selection_status_in_v2",
    "candidate_clarity_basis",
    "pre_api_dir",
    "pre_selected_internal_series",
    "pre_n_frames",
    "pre_contiguous_pair_count",
    "pre_frame_span_frames",
    "pre_missing_frame_count",
    "pre_dimensions",
    "pre_fps",
    "pre_frame_time_ms",
    "pre_duration_seconds",
    "pre_timing_source",
    "pre_timing_reliability",
    "global_pre_peak_left_censored_qc",
    "global_pre_peak_right_censored_qc",
    "post_api_dir",
    "post_selected_internal_series",
    "post_n_frames",
    "post_contiguous_pair_count",
    "post_frame_span_frames",
    "post_missing_frame_count",
    "post_dimensions",
    "post_fps",
    "post_frame_time_ms",
    "post_duration_seconds",
    "post_timing_source",
    "post_timing_reliability",
    "global_post_peak_left_censored_qc",
    "global_post_peak_right_censored_qc",
    "coverage_frame_count_bin",
    "coverage_dimension_class",
    "coverage_new_categories_at_selection",
    "preselection_reason",
    "formal_roi_pilot_included",
    "roi_pilot_status",
    "formal_entry_gate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(lesion_uid: str) -> str:
    return sha256_text(f"{SELECTION_SEED}|{lesion_uid}")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("NaN", ""), errors="coerce")


def frame_bin(total: float) -> str:
    if total <= 30:
        return "frames_low"
    if total <= 70:
        return "frames_medium"
    return "frames_high"


def dimension_class(pre: str, post: str) -> str:
    if not pre and not post:
        return "dimensions_unknown"
    if pre == post:
        return f"dimensions_same_{pre}"
    return f"dimensions_mixed_{pre or 'none'}_{post or 'none'}"


def rarity_weight(value: str, counts: Counter[str]) -> float:
    return 1.0 / max(1, counts[value])


def greedy_diverse_select(frame: pd.DataFrame, size: int) -> list[tuple[int, list[str]]]:
    location_counts = Counter(frame["location_normalized"])
    source_counts = Counter(frame["candidate_source_type"])
    frame_counts = Counter(frame["coverage_frame_count_bin"])
    dimension_counts = Counter(frame["coverage_dimension_class"])
    status_counts = Counter(frame["current_registration_status"])

    selected: list[tuple[int, list[str]]] = []
    used_patients: set[str] = set()
    covered: dict[str, set[str]] = {
        "location": set(),
        "source": set(),
        "frame_bin": set(),
        "dimension": set(),
        "status": set(),
    }
    remaining = set(frame.index)

    while len(selected) < size:
        scored: list[tuple[float, str, int, list[str]]] = []
        for index in remaining:
            row = frame.loc[index]
            if row["patient_id"] in used_patients:
                continue
            new_categories: list[str] = []
            if row["location_normalized"] not in covered["location"]:
                new_categories.append(f"location={row['location_normalized']}")
            if row["candidate_source_type"] not in covered["source"]:
                new_categories.append(f"source={row['candidate_source_type']}")
            if row["coverage_frame_count_bin"] not in covered["frame_bin"]:
                new_categories.append(f"frame_bin={row['coverage_frame_count_bin']}")
            if row["coverage_dimension_class"] not in covered["dimension"]:
                new_categories.append(f"dimension={row['coverage_dimension_class']}")
            if row["current_registration_status"] not in covered["status"]:
                new_categories.append(f"status={row['current_registration_status']}")

            score = (
                30.0 * (row["location_normalized"] not in covered["location"])
                + 25.0 * (row["candidate_source_type"] not in covered["source"])
                + 14.0 * (row["coverage_frame_count_bin"] not in covered["frame_bin"])
                + 8.0 * (row["coverage_dimension_class"] not in covered["dimension"])
                + 5.0 * (row["current_registration_status"] not in covered["status"])
                + 5.0 * rarity_weight(row["location_normalized"], location_counts)
                + 4.0 * rarity_weight(row["candidate_source_type"], source_counts)
                + 2.0 * rarity_weight(row["coverage_frame_count_bin"], frame_counts)
                + 2.0 * rarity_weight(row["coverage_dimension_class"], dimension_counts)
                + rarity_weight(row["current_registration_status"], status_counts)
            )
            scored.append((score, row["selection_hash"], index, new_categories))
        if not scored:
            raise AssertionError(f"Only selected {len(selected)} unique-patient pilot entries")
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        _, _, chosen_index, new_categories = scored[0]
        chosen = frame.loc[chosen_index]
        selected.append((chosen_index, new_categories))
        used_patients.add(chosen["patient_id"])
        covered["location"].add(chosen["location_normalized"])
        covered["source"].add(chosen["candidate_source_type"])
        covered["frame_bin"].add(chosen["coverage_frame_count_bin"])
        covered["dimension"].add(chosen["coverage_dimension_class"])
        covered["status"].add(chosen["current_registration_status"])
        remaining.remove(chosen_index)
    return selected


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    df.to_csv(
        temp,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
        na_rep="NaN",
    )
    os.replace(temp, path)


def assert_blinded_text(text: str, name: str) -> None:
    folded = text.casefold()
    found = [token for token in FORBIDDEN_OUTPUT_TOKENS if token.casefold() in folded]
    if found:
        raise AssertionError(f"Forbidden token(s) in {name}: {found}")


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    train = pd.read_csv(
        root / "metadata/api_fullseq_v3/lesion_registry_train_blinded.csv",
        dtype=str,
        keep_default_na=False,
    )
    review = pd.read_csv(
        root / "reports/api_fullseq_v3/series_candidate_review.csv",
        dtype=str,
        keep_default_na=False,
    )
    decisions = pd.read_csv(
        root / "reports/api_fullseq_v3/lesion_series_decision.csv",
        dtype=str,
        keep_default_na=False,
    )
    timing = pd.read_csv(
        root / "metadata/api_fullseq_v3/candidate_timing_registry.csv",
        dtype=str,
        keep_default_na=False,
    )

    if set(train["split"]) != {"Train"}:
        raise AssertionError("Train blinded registry contains a non-Train split")
    train_decisions = decisions.loc[decisions["split"] == "Train"].copy()
    if train_decisions.empty:
        raise AssertionError("No Train lesions have entered blinded review")

    selected_review = review.loc[
        (review["split"] == "Train")
        & (review["candidate_selected_in_v2"] == "True")
        & (review["candidate_valid"] == "True")
        & (review["candidate_exclusion_reason"] == "")
    ].copy()
    selected_counts = selected_review.groupby("lesion_uid").size()
    clear_lesions = set(selected_counts[selected_counts == 1].index)
    selected_review = selected_review.loc[selected_review["lesion_uid"].isin(clear_lesions)]

    candidate_keys = [
        "split",
        "patient_id",
        "candidate_discovery_rank",
        "candidate_series_id",
        "candidate_series_path",
    ]
    merged = selected_review.merge(
        timing,
        on=candidate_keys,
        how="inner",
        validate="many_to_one",
        suffixes=("_review", "_timing"),
    )
    merged = merged.merge(
        train_decisions[
            [
                "lesion_uid",
                "review_scope",
                "lesion_review_page",
                "manual_decision_status",
            ]
        ],
        on=["lesion_uid", "review_scope", "lesion_review_page"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_decision"),
    )
    merged = merged.merge(
        train[
            [
                "lesion_uid",
                "registration_status",
                "valid_candidate_series_count",
            ]
        ],
        on="lesion_uid",
        how="inner",
        validate="one_to_one",
    )

    for column in (
        "pre_selected_n_frames_timing",
        "post_selected_n_frames_timing",
        "pre_selected_contiguous_pair_count",
        "post_selected_contiguous_pair_count",
        "pre_selected_frame_span_frames",
        "post_selected_frame_span_frames",
        "pre_selected_missing_frame_count",
        "post_selected_missing_frame_count",
    ):
        merged[column] = numeric(merged[column])

    eligible = merged.loc[
        (merged["pre_selected_n_frames_timing"] >= 2)
        & (merged["post_selected_n_frames_timing"] >= 2)
        & (merged["pre_selected_contiguous_pair_count"] >= 1)
        & (merged["post_selected_contiguous_pair_count"] >= 1)
        & (merged["pre_selected_dimensions"] != "")
        & (merged["post_selected_dimensions"] != "")
    ].copy()
    if len(eligible) < PILOT_SIZE:
        raise AssertionError(f"Only {len(eligible)} eligible review-first Train lesions")

    eligible["current_registration_status"] = eligible["registration_status"]
    eligible["candidate_source_type"] = eligible["candidate_source_type_timing"]
    eligible["selection_hash"] = eligible["lesion_uid"].map(stable_hash)
    eligible["coverage_frame_count_bin"] = (
        eligible["pre_selected_n_frames_timing"] + eligible["post_selected_n_frames_timing"]
    ).map(frame_bin)
    eligible["coverage_dimension_class"] = eligible.apply(
        lambda row: dimension_class(
            row["pre_selected_dimensions"], row["post_selected_dimensions"]
        ),
        axis=1,
    )

    chosen = greedy_diverse_select(eligible, PILOT_SIZE)
    output_rows: list[dict[str, Any]] = []
    for order, (index, new_categories) in enumerate(chosen, start=1):
        row = eligible.loc[index]
        manual_status = row["manual_decision_status"]
        formal = manual_status in HUMAN_REVIEWED_STATUSES
        output_rows.append(
            {
                "roi_pilot_entry_id": f"train_roi_pilot_{order:02d}",
                "selection_order": order,
                "selection_hash": row["selection_hash"],
                "split": "Train",
                "lesion_uid": row["lesion_uid"],
                "patient_id": row["patient_id"],
                "source_excel_row_id": row["source_excel_row_id"],
                "side_raw": row["side_raw"],
                "side_normalized": row["side_normalized"],
                "location_raw": row["location_raw"],
                "location_normalized": row["location_normalized"],
                "lesion_index_normalized": row["lesion_index_normalized"],
                "multiple_aneurysm_normalized": row["multiple_aneurysm_normalized"],
                "registry_status": row["registration_status"],
                "review_scope": row["review_scope"],
                "lesion_review_page": row["lesion_review_page"],
                "review_item_id": row["review_item_id"],
                "manual_decision_status": manual_status,
                "candidate_source_type": row["candidate_source_type_timing"],
                "candidate_source_root": row["candidate_source_root_timing"],
                "candidate_discovery_rank": row["candidate_discovery_rank"],
                "candidate_series_id": row["candidate_series_id"],
                "candidate_series_path": row["candidate_series_path"],
                "candidate_valid": row["candidate_valid_review"],
                "candidate_selected_in_v2": row["candidate_selected_in_v2_review"],
                "candidate_selection_status_in_v2": row[
                    "candidate_selection_status_in_v2_review"
                ],
                "candidate_clarity_basis": "one_valid_v2_selected_candidate_in_review_table",
                "pre_api_dir": row["pre_api_dir"],
                "pre_selected_internal_series": row["pre_selected_internal_series"],
                "pre_n_frames": int(row["pre_selected_n_frames_timing"]),
                "pre_contiguous_pair_count": int(row["pre_selected_contiguous_pair_count"]),
                "pre_frame_span_frames": int(row["pre_selected_frame_span_frames"]),
                "pre_missing_frame_count": int(row["pre_selected_missing_frame_count"]),
                "pre_dimensions": row["pre_selected_dimensions"],
                "pre_fps": row["pre_fps"],
                "pre_frame_time_ms": row["pre_frame_time_ms"],
                "pre_duration_seconds": row["pre_duration_seconds"],
                "pre_timing_source": row["pre_timing_source"],
                "pre_timing_reliability": row["pre_timing_reliability"],
                "global_pre_peak_left_censored_qc": row[
                    "global_pre_peak_left_censored_qc"
                ],
                "global_pre_peak_right_censored_qc": row[
                    "global_pre_peak_right_censored_qc"
                ],
                "post_api_dir": row["post_api_dir"],
                "post_selected_internal_series": row["post_selected_internal_series"],
                "post_n_frames": int(row["post_selected_n_frames_timing"]),
                "post_contiguous_pair_count": int(row["post_selected_contiguous_pair_count"]),
                "post_frame_span_frames": int(row["post_selected_frame_span_frames"]),
                "post_missing_frame_count": int(row["post_selected_missing_frame_count"]),
                "post_dimensions": row["post_selected_dimensions"],
                "post_fps": row["post_fps"],
                "post_frame_time_ms": row["post_frame_time_ms"],
                "post_duration_seconds": row["post_duration_seconds"],
                "post_timing_source": row["post_timing_source"],
                "post_timing_reliability": row["post_timing_reliability"],
                "global_post_peak_left_censored_qc": row[
                    "global_post_peak_left_censored_qc"
                ],
                "global_post_peak_right_censored_qc": row[
                    "global_post_peak_right_censored_qc"
                ],
                "coverage_frame_count_bin": row["coverage_frame_count_bin"],
                "coverage_dimension_class": row["coverage_dimension_class"],
                "coverage_new_categories_at_selection": "|".join(new_categories),
                "preselection_reason": (
                    "train_only_blinded_review_entry_with_valid_selected_prepost_candidate_"
                    "and_diversity_coverage"
                ),
                "formal_roi_pilot_included": formal,
                "roi_pilot_status": (
                    "ready_for_roi_annotation" if formal else "pending_manual_series_review"
                ),
                "formal_entry_gate": "human_reviewed_series_decision_required",
            }
        )

    output = pd.DataFrame(output_rows, columns=COLUMNS)
    if len(output) != PILOT_SIZE or not output["lesion_uid"].is_unique:
        raise AssertionError("ROI Pilot manifest must contain 30 unique lesions")
    if output["patient_id"].nunique() != PILOT_SIZE:
        raise AssertionError("ROI Pilot preselection must contain 30 unique patients")
    if set(output["split"]) != {"Train"}:
        raise AssertionError("ROI Pilot contains non-Train rows")
    if not set(output["lesion_uid"]).issubset(set(train_decisions["lesion_uid"])):
        raise AssertionError("Every ROI Pilot lesion must first enter blinded review")
    if output["formal_roi_pilot_included"].any():
        bad = output.loc[
            output["formal_roi_pilot_included"], "manual_decision_status"
        ]
        if not set(bad).issubset(HUMAN_REVIEWED_STATUSES):
            raise AssertionError("Formal ROI entry without a human-reviewed series status")
    if output["manual_decision_status"].eq("").all() and output[
        "formal_roi_pilot_included"
    ].any():
        raise AssertionError("Blank decisions cannot formally enter ROI Pilot")

    output_path = root / "manifests/api_fullseq_v3_roi_pilot_train.csv"
    csv_text = output.to_csv(index=False, lineterminator="\n", na_rep="NaN")
    assert_blinded_text(csv_text, output_path.name)
    atomic_write_csv(output, output_path)

    print(
        json.dumps(
            {
                "roi_pilot_preselection_rows": len(output),
                "unique_patients": output["patient_id"].nunique(),
                "locations": output["location_normalized"].value_counts().to_dict(),
                "sources": output["candidate_source_type"].value_counts().to_dict(),
                "frame_bins": output["coverage_frame_count_bin"].value_counts().to_dict(),
                "dimension_classes": output["coverage_dimension_class"].nunique(),
                "formal_roi_pilot_included": int(
                    output["formal_roi_pilot_included"].sum()
                ),
                "pending_manual_series_review": int(
                    (output["roi_pilot_status"] == "pending_manual_series_review").sum()
                ),
                "restricted_private_artifact_accessed": False,
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
