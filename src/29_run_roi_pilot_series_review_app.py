#!/usr/bin/env python3
"""Run the save-capable 30-case Train ROI Pilot series review application.

The application loads only lesion UIDs listed in the frozen Train ROI Pilot
manifest, then filters the blinded candidate review table to those UIDs.  It
never opens private labels, runs SEA-RAFT, trains a model, or calculates ROI
features.  Human review state is written to two new independent CSV files.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import html
import json
import mimetypes
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import quote

import pandas as pd
import tornado.ioloop
import tornado.web


PROJECT = Path("/root/autodl-tmp/aneurysm")
EXPECTED_PILOT_LESIONS = 30

FINAL_STATUSES = (
    "exact_reviewed",
    "probable_reviewed",
    "ambiguous",
    "unmatched",
)
EXACT_OR_PROBABLE = {"exact_reviewed", "probable_reviewed"}
BINARY_VALUES = {"", "0", "1"}

CANDIDATE_BINARY_FIELDS = [
    "lesion_visible_pre",
    "lesion_visible_post",
    "side_concordant",
    "location_concordant",
    "sac_opacified_pre",
    "sac_opacified_post",
    "sac_locatable_pre",
    "sac_locatable_post",
    "neck_visible_pre",
    "neck_visible_post",
    "parent_visible_pre",
    "parent_visible_post",
    "branch_visible_pre",
    "branch_visible_post",
    "roi_feasible_pre",
    "roi_feasible_post",
]
SELECTION_FIELDS = ["selected_for_pre", "selected_for_post"]
DECISION_BINARY_FIELDS = ["pre_post_same_lesion", "pre_post_view_comparable"]

CANDIDATE_SOURCE_COLUMNS = [
    "review_item_id",
    "lesion_uid",
    "split",
    "patient_id",
    "source_excel_row_id",
    "side_raw",
    "side_normalized",
    "location_raw",
    "location_normalized",
    "lesion_index_normalized",
    "multiple_aneurysm_normalized",
    "current_registration_status",
    "candidate_index_for_lesion",
    "candidate_count_for_lesion",
    "candidate_source_type",
    "candidate_source_root",
    "candidate_discovery_rank",
    "candidate_series_id",
    "candidate_series_path",
    "candidate_is_fixed_target",
    "candidate_valid",
    "candidate_selected_in_v2",
    "candidate_selection_status_in_v2",
    "candidate_exclusion_reason",
    "pre_api_dir",
    "pre_internal_series",
    "pre_selected_internal_series_in_v2",
    "pre_selected_n_frames",
    "pre_selected_n_contiguous_pairs",
    "post_api_dir",
    "post_internal_series",
    "post_selected_internal_series_in_v2",
    "post_selected_n_frames",
    "post_selected_n_contiguous_pairs",
    "pre_contact_sheet",
    "post_contact_sheet",
]

CANDIDATE_OUTPUT_COLUMNS = (
    CANDIDATE_SOURCE_COLUMNS
    + CANDIDATE_BINARY_FIELDS
    + SELECTION_FIELDS
    + ["updated_at_utc", "save_revision", "transaction_id"]
)

DECISION_OUTPUT_COLUMNS = [
    "lesion_uid",
    "split",
    "patient_id",
    "source_excel_row_id",
    "side_raw",
    "side_normalized",
    "location_raw",
    "location_normalized",
    "lesion_index_normalized",
    "multiple_aneurysm_normalized",
    "current_registration_status",
    "pre_post_same_lesion",
    "pre_post_view_comparable",
    "final_review_status",
    "final_review_reason",
    "reviewer_id",
    "selected_pre_review_item_id",
    "selected_post_review_item_id",
    "validation_status",
    "updated_at_utc",
    "save_revision",
    "transaction_id",
]

DISPLAY_LABELS = {
    "lesion_visible_pre": "Lesion visible",
    "lesion_visible_post": "Lesion visible",
    "side_concordant": "Side concordant",
    "location_concordant": "Location concordant",
    "sac_opacified_pre": "Sac opacified",
    "sac_opacified_post": "Sac opacified",
    "sac_locatable_pre": "Sac locatable",
    "sac_locatable_post": "Sac locatable",
    "neck_visible_pre": "Neck visible",
    "neck_visible_post": "Neck visible",
    "parent_visible_pre": "Parent visible",
    "parent_visible_post": "Parent visible",
    "branch_visible_pre": "Branch visible",
    "branch_visible_post": "Branch visible",
    "roi_feasible_pre": "ROI feasible",
    "roi_feasible_post": "ROI feasible",
}


class ReviewValidationError(Exception):
    def __init__(
        self,
        errors: list[str],
        candidate_draft: pd.DataFrame | None = None,
        decision_draft: dict[str, str] | None = None,
    ) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors
        self.candidate_draft = candidate_draft
        self.decision_draft = decision_draft


@dataclass(frozen=True)
class AppPaths:
    project_root: Path
    pilot_manifest: Path
    candidate_source: Path
    candidate_output: Path
    decision_output: Path
    backup_dir: Path
    review_report_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--pilot-manifest", type=Path)
    parser.add_argument("--candidate-source", type=Path)
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument("--decision-output", type=Path)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate the 30-case app state without starting a server.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Exercise validation, atomic save, backup, and resume in a temporary directory.",
    )
    return parser.parse_args()


def build_paths(args: argparse.Namespace) -> AppPaths:
    root = args.project_root.resolve()
    reports = root / "reports/api_fullseq_v3"
    return AppPaths(
        project_root=root,
        pilot_manifest=(
            args.pilot_manifest or root / "manifests/api_fullseq_v3_roi_pilot_train.csv"
        ).resolve(),
        candidate_source=(
            args.candidate_source or reports / "series_candidate_review.csv"
        ).resolve(),
        candidate_output=(
            args.candidate_output or reports / "roi_pilot_series_candidate_review.csv"
        ).resolve(),
        decision_output=(
            args.decision_output or reports / "roi_pilot_series_decision.csv"
        ).resolve(),
        backup_dir=(
            args.backup_dir or reports / "roi_pilot_series_review_backups"
        ).resolve(),
        review_report_root=reports.resolve(),
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as handle:
            frame.to_csv(
                handle,
                index=False,
                lineterminator="\n",
                quoting=csv.QUOTE_MINIMAL,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def first_form_value(form: Mapping[str, list[str]], name: str, default: str = "") -> str:
    values = form.get(name, [])
    return values[0] if values else default


def stream_filter_candidate_source(path: Path, lesion_uids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header in {path}")
        missing = sorted(set(CANDIDATE_SOURCE_COLUMNS) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        for row in reader:
            if row.get("lesion_uid", "") in lesion_uids:
                rows.append({column: row.get(column, "") for column in CANDIDATE_SOURCE_COLUMNS})
    return pd.DataFrame(rows, columns=CANDIDATE_SOURCE_COLUMNS)


class ReviewStore:
    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self._mutex = threading.RLock()
        self._lock_handle: Any = None
        self._acquire_process_lock()
        try:
            self.pilot = self._load_pilot()
            self.pilot_order = self.pilot["lesion_uid"].tolist()
            self.pilot_set = set(self.pilot_order)
            self.source_candidates = self._load_source_candidates()
            self.asset_paths, self.asset_tokens = self._build_asset_allowlist()
            self.candidate_state = self._blank_candidate_state()
            self.decision_state = self._blank_decision_state()
            self.revision = 0
            self._load_saved_state_with_recovery()
        except Exception:
            self.close()
            raise

    def _acquire_process_lock(self) -> None:
        lock_path = self.paths.candidate_output.with_suffix(
            self.paths.candidate_output.suffix + ".lock"
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_handle = lock_path.open("a+")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_handle.close()
            self._lock_handle = None
            raise RuntimeError(
                f"Another review app process holds the state lock: {lock_path}"
            ) from exc

    def close(self) -> None:
        if self._lock_handle is not None:
            try:
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock_handle.close()
                self._lock_handle = None

    def _load_pilot(self) -> pd.DataFrame:
        pilot = pd.read_csv(self.paths.pilot_manifest, dtype=str, keep_default_na=False)
        require_columns(
            pilot,
            [
                "selection_order",
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
            ],
            self.paths.pilot_manifest,
        )
        pilot = pilot.sort_values(
            "selection_order", key=lambda values: pd.to_numeric(values), kind="stable"
        ).reset_index(drop=True)
        if len(pilot) != EXPECTED_PILOT_LESIONS:
            raise AssertionError(
                f"Expected {EXPECTED_PILOT_LESIONS} Pilot lesions, got {len(pilot)}"
            )
        if not pilot["lesion_uid"].is_unique or not pilot["patient_id"].is_unique:
            raise AssertionError("Pilot lesion_uid and patient_id must both be unique")
        if set(pilot["split"]) != {"Train"}:
            raise AssertionError("Review app accepts Train Pilot rows only")
        return pilot

    def _load_source_candidates(self) -> pd.DataFrame:
        frame = stream_filter_candidate_source(
            self.paths.candidate_source, set(self.pilot["lesion_uid"])
        )
        if frame.empty:
            raise AssertionError("No candidate rows matched the 30 Pilot lesion UIDs")
        if not frame["review_item_id"].is_unique:
            raise AssertionError("Pilot candidate review_item_id values must be unique")
        if set(frame["lesion_uid"]) != set(self.pilot["lesion_uid"]):
            missing = sorted(set(self.pilot["lesion_uid"]) - set(frame["lesion_uid"]))
            raise AssertionError(f"Pilot lesions missing candidate rows: {missing}")
        frame["_pilot_order"] = frame["lesion_uid"].map(
            {uid: index for index, uid in enumerate(self.pilot["lesion_uid"])}
        )
        frame["_candidate_order"] = pd.to_numeric(
            frame["candidate_index_for_lesion"], errors="coerce"
        ).fillna(10**9)
        frame = frame.sort_values(
            ["_pilot_order", "_candidate_order", "review_item_id"], kind="stable"
        ).drop(columns=["_pilot_order", "_candidate_order"])
        return frame.reset_index(drop=True)

    def _resolve_asset(self, relative_value: str) -> Path | None:
        if not relative_value:
            return None
        candidate = (self.paths.review_report_root / relative_value).resolve()
        allowed_root = (self.paths.review_report_root / "review_app").resolve()
        try:
            candidate.relative_to(allowed_root)
        except ValueError as exc:
            raise AssertionError(f"Contact sheet outside review_app: {candidate}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    def _build_asset_allowlist(self) -> tuple[dict[str, Path], dict[str, str]]:
        token_to_path: dict[str, Path] = {}
        relative_to_token: dict[str, str] = {}
        for column in ("pre_contact_sheet", "post_contact_sheet"):
            for relative in sorted(set(self.source_candidates[column]) - {""}):
                path = self._resolve_asset(relative)
                assert path is not None
                token = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:24]
                existing = token_to_path.get(token)
                if existing is not None and existing != path:
                    raise AssertionError("Asset token collision")
                token_to_path[token] = path
                relative_to_token[relative] = token
        return token_to_path, relative_to_token

    def _blank_candidate_state(self) -> pd.DataFrame:
        frame = self.source_candidates[CANDIDATE_SOURCE_COLUMNS].copy()
        for column in CANDIDATE_BINARY_FIELDS + SELECTION_FIELDS:
            frame[column] = ""
        frame["updated_at_utc"] = ""
        frame["save_revision"] = "0"
        frame["transaction_id"] = ""
        return frame[CANDIDATE_OUTPUT_COLUMNS]

    def _blank_decision_state(self) -> pd.DataFrame:
        pilot_lookup = self.pilot.set_index("lesion_uid")
        source_first = self.source_candidates.groupby("lesion_uid", sort=False).first()
        rows: list[dict[str, str]] = []
        for lesion_uid in self.pilot_order:
            pilot = pilot_lookup.loc[lesion_uid]
            source = source_first.loc[lesion_uid]
            rows.append(
                {
                    "lesion_uid": lesion_uid,
                    "split": "Train",
                    "patient_id": str(pilot["patient_id"]),
                    "source_excel_row_id": str(pilot["source_excel_row_id"]),
                    "side_raw": str(pilot["side_raw"]),
                    "side_normalized": str(pilot["side_normalized"]),
                    "location_raw": str(pilot["location_raw"]),
                    "location_normalized": str(pilot["location_normalized"]),
                    "lesion_index_normalized": str(pilot["lesion_index_normalized"]),
                    "multiple_aneurysm_normalized": str(
                        pilot["multiple_aneurysm_normalized"]
                    ),
                    "current_registration_status": str(source["current_registration_status"]),
                    "pre_post_same_lesion": "",
                    "pre_post_view_comparable": "",
                    "final_review_status": "",
                    "final_review_reason": "",
                    "reviewer_id": "",
                    "selected_pre_review_item_id": "",
                    "selected_post_review_item_id": "",
                    "validation_status": "draft",
                    "updated_at_utc": "",
                    "save_revision": "0",
                    "transaction_id": "",
                }
            )
        return pd.DataFrame(rows, columns=DECISION_OUTPUT_COLUMNS)

    def _overlay_saved_candidate_state(self, saved: pd.DataFrame) -> None:
        require_columns(saved, CANDIDATE_OUTPUT_COLUMNS, self.paths.candidate_output)
        if saved["review_item_id"].duplicated().any():
            raise AssertionError("Saved candidate state has duplicate review_item_id")
        if not set(saved["review_item_id"]).issubset(
            set(self.candidate_state["review_item_id"])
        ):
            raise AssertionError("Saved candidate state contains unknown review items")
        saved_lookup = saved.set_index("review_item_id")
        overlay_columns = (
            CANDIDATE_BINARY_FIELDS
            + SELECTION_FIELDS
            + ["updated_at_utc", "save_revision", "transaction_id"]
        )
        for column in overlay_columns:
            self.candidate_state[column] = [
                str(saved_lookup.at[item_id, column])
                if item_id in saved_lookup.index
                else current
                for item_id, current in zip(
                    self.candidate_state["review_item_id"], self.candidate_state[column]
                )
            ]

    def _overlay_saved_decision_state(self, saved: pd.DataFrame) -> None:
        require_columns(saved, DECISION_OUTPUT_COLUMNS, self.paths.decision_output)
        if saved["lesion_uid"].duplicated().any():
            raise AssertionError("Saved decision state has duplicate lesion_uid")
        if not set(saved["lesion_uid"]).issubset(set(self.pilot_order)):
            raise AssertionError("Saved decision state contains non-Pilot lesion UIDs")
        saved_lookup = saved.set_index("lesion_uid")
        overlay_columns = DECISION_BINARY_FIELDS + [
            "final_review_status",
            "final_review_reason",
            "reviewer_id",
            "selected_pre_review_item_id",
            "selected_post_review_item_id",
            "validation_status",
            "updated_at_utc",
            "save_revision",
            "transaction_id",
        ]
        for column in overlay_columns:
            self.decision_state[column] = [
                str(saved_lookup.at[lesion_uid, column])
                if lesion_uid in saved_lookup.index
                else current
                for lesion_uid, current in zip(
                    self.decision_state["lesion_uid"], self.decision_state[column]
                )
            ]

    def _transaction_state_consistent(self) -> bool:
        decisions = self.decision_state.set_index("lesion_uid")
        for lesion_uid in self.pilot_order:
            candidate_txs = set(
                self.candidate_state.loc[
                    self.candidate_state["lesion_uid"] == lesion_uid, "transaction_id"
                ]
            )
            if len(candidate_txs) != 1:
                return False
            if next(iter(candidate_txs)) != decisions.at[lesion_uid, "transaction_id"]:
                return False
        return True

    def _latest_valid_backup(self) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        if not self.paths.backup_dir.is_dir():
            return None
        manifests = sorted(
            self.paths.backup_dir.glob("backup_*_manifest.json"), reverse=True
        )
        for manifest_path in manifests:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                candidate_path = self.paths.backup_dir / manifest["candidate_file"]
                decision_path = self.paths.backup_dir / manifest["decision_file"]
                if (
                    sha256_file(candidate_path) != manifest["candidate_sha256"]
                    or sha256_file(decision_path) != manifest["decision_sha256"]
                ):
                    continue
                candidate = pd.read_csv(candidate_path, dtype=str, keep_default_na=False)
                decision = pd.read_csv(decision_path, dtype=str, keep_default_na=False)
                return candidate, decision
            except Exception:
                continue
        return None

    def _restore_latest_backup(self) -> bool:
        backup = self._latest_valid_backup()
        if backup is None:
            return False
        candidate, decision = backup
        atomic_write_csv(candidate, self.paths.candidate_output)
        atomic_write_csv(decision, self.paths.decision_output)
        return True

    def _load_saved_state_once(self) -> None:
        candidate_exists = self.paths.candidate_output.is_file()
        decision_exists = self.paths.decision_output.is_file()
        if candidate_exists != decision_exists:
            raise AssertionError("Only one of the paired review state files exists")
        if not candidate_exists:
            return
        candidate = pd.read_csv(
            self.paths.candidate_output, dtype=str, keep_default_na=False
        )
        decision = pd.read_csv(
            self.paths.decision_output, dtype=str, keep_default_na=False
        )
        self._overlay_saved_candidate_state(candidate)
        self._overlay_saved_decision_state(decision)
        if not self._transaction_state_consistent():
            raise AssertionError("Paired review state transaction IDs are inconsistent")
        revisions = [
            safe_int(value)
            for value in list(self.candidate_state["save_revision"])
            + list(self.decision_state["save_revision"])
        ]
        self.revision = max(revisions, default=0)

    def _load_saved_state_with_recovery(self) -> None:
        try:
            self._load_saved_state_once()
        except Exception:
            if not self._restore_latest_backup():
                raise
            self.candidate_state = self._blank_candidate_state()
            self.decision_state = self._blank_decision_state()
            self.revision = 0
            self._load_saved_state_once()

    def _backup_current_state(self) -> None:
        self.paths.backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{time.time_ns()}_r{self.revision:06d}"
        candidate_name = f"backup_{stamp}_candidate.csv"
        decision_name = f"backup_{stamp}_decision.csv"
        candidate_path = self.paths.backup_dir / candidate_name
        decision_path = self.paths.backup_dir / decision_name
        manifest_path = self.paths.backup_dir / f"backup_{stamp}_manifest.json"
        atomic_write_csv(self.candidate_state, candidate_path)
        atomic_write_csv(self.decision_state, decision_path)
        manifest = {
            "created_at_utc": utc_now(),
            "revision": self.revision,
            "candidate_file": candidate_name,
            "candidate_sha256": sha256_file(candidate_path),
            "decision_file": decision_name,
            "decision_sha256": sha256_file(decision_path),
        }
        atomic_write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), manifest_path
        )

    def candidate_rows(self, lesion_uid: str) -> pd.DataFrame:
        if lesion_uid not in self.pilot_set:
            raise KeyError(lesion_uid)
        return self.candidate_state.loc[
            self.candidate_state["lesion_uid"] == lesion_uid
        ].copy()

    def decision_row(self, lesion_uid: str) -> dict[str, str]:
        if lesion_uid not in self.pilot_set:
            raise KeyError(lesion_uid)
        row = self.decision_state.loc[
            self.decision_state["lesion_uid"] == lesion_uid
        ].iloc[0]
        return {column: str(row[column]) for column in DECISION_OUTPUT_COLUMNS}

    def is_complete(self, lesion_uid: str) -> bool:
        decision = self.decision_row(lesion_uid)
        return (
            decision["final_review_status"] in FINAL_STATUSES
            and decision["validation_status"] == "valid_final"
        )

    def filtered_uids(self, incomplete_only: bool) -> list[str]:
        if not incomplete_only:
            return list(self.pilot_order)
        return [uid for uid in self.pilot_order if not self.is_complete(uid)]

    def completion_counts(self) -> tuple[int, int]:
        complete = sum(self.is_complete(uid) for uid in self.pilot_order)
        return complete, len(self.pilot_order) - complete

    def _prepare_form_state(
        self, lesion_uid: str, form: Mapping[str, list[str]]
    ) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
        candidates = self.candidate_rows(lesion_uid)
        decision = self.decision_row(lesion_uid)
        known_items = set(candidates["review_item_id"])
        errors: list[str] = []

        for index, row in candidates.iterrows():
            item_id = row["review_item_id"]
            for field in CANDIDATE_BINARY_FIELDS:
                name = f"{item_id}__{field}"
                value = first_form_value(form, name, str(row[field]))
                if value not in BINARY_VALUES:
                    errors.append(f"{item_id} {field} must be blank, 0, or 1")
                candidates.at[index, field] = value

        selected_pre_values = form.get("selected_for_pre", [])
        selected_post_values = form.get("selected_for_post", [])
        selected_pre = [value for value in selected_pre_values if value]
        selected_post = [value for value in selected_post_values if value]
        if len(set(selected_pre)) > 1:
            errors.append("Each lesion may select at most one Pre candidate")
        if len(set(selected_post)) > 1:
            errors.append("Each lesion may select at most one Post candidate")
        if any(value not in known_items for value in selected_pre + selected_post):
            errors.append("Selected candidate is not part of this Pilot lesion")
        pre_item = selected_pre[0] if len(set(selected_pre)) == 1 else ""
        post_item = selected_post[0] if len(set(selected_post)) == 1 else ""
        for index, row in candidates.iterrows():
            candidates.at[index, "selected_for_pre"] = (
                "1" if row["review_item_id"] == pre_item else "0" if pre_item else ""
            )
            candidates.at[index, "selected_for_post"] = (
                "1" if row["review_item_id"] == post_item else "0" if post_item else ""
            )

        for field in DECISION_BINARY_FIELDS:
            value = first_form_value(form, field, decision[field])
            if value not in BINARY_VALUES:
                errors.append(f"{field} must be blank, 0, or 1")
            decision[field] = value
        decision["final_review_status"] = first_form_value(
            form, "final_review_status", decision["final_review_status"]
        )
        decision["final_review_reason"] = first_form_value(
            form, "final_review_reason", decision["final_review_reason"]
        ).strip()
        decision["reviewer_id"] = first_form_value(
            form, "reviewer_id", decision["reviewer_id"]
        ).strip()
        decision["selected_pre_review_item_id"] = pre_item
        decision["selected_post_review_item_id"] = post_item

        status = decision["final_review_status"]
        if status not in {"", *FINAL_STATUSES}:
            errors.append("final_review_status is not allowed")
        if status in EXACT_OR_PROBABLE:
            if not pre_item or not post_item:
                errors.append("exact/probable review requires one selected Pre and one selected Post")
            if decision["pre_post_same_lesion"] != "1":
                errors.append("exact/probable review requires pre_post_same_lesion=1")
            candidate_lookup = candidates.set_index("review_item_id")
            if pre_item and pre_item in candidate_lookup.index:
                for field in ("sac_locatable_pre", "parent_visible_pre", "roi_feasible_pre"):
                    if candidate_lookup.at[pre_item, field] != "1":
                        errors.append(f"Selected Pre candidate requires {field}=1")
            if post_item and post_item in candidate_lookup.index:
                for field in (
                    "sac_locatable_post",
                    "parent_visible_post",
                    "roi_feasible_post",
                ):
                    if candidate_lookup.at[post_item, field] != "1":
                        errors.append(f"Selected Post candidate requires {field}=1")

        decision["validation_status"] = "valid_final" if status else "draft"
        return candidates, decision, errors

    def validate_form(
        self, lesion_uid: str, form: Mapping[str, list[str]]
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        candidates, decision, errors = self._prepare_form_state(lesion_uid, form)
        if errors:
            raise ReviewValidationError(errors, candidates, decision)
        return candidates, decision

    def save_form(
        self, lesion_uid: str, form: Mapping[str, list[str]]
    ) -> dict[str, Any]:
        with self._mutex:
            candidates, decision = self.validate_form(lesion_uid, form)
            self._backup_current_state()
            new_candidate_state = self.candidate_state.copy()
            new_decision_state = self.decision_state.copy()
            revision = self.revision + 1
            transaction_id = uuid.uuid4().hex
            timestamp = utc_now()
            candidate_indices = new_candidate_state.index[
                new_candidate_state["lesion_uid"] == lesion_uid
            ]
            candidates = candidates.copy()
            candidates["updated_at_utc"] = timestamp
            candidates["save_revision"] = str(revision)
            candidates["transaction_id"] = transaction_id
            new_candidate_state.loc[candidate_indices, CANDIDATE_OUTPUT_COLUMNS] = (
                candidates[CANDIDATE_OUTPUT_COLUMNS].to_numpy()
            )
            decision["updated_at_utc"] = timestamp
            decision["save_revision"] = str(revision)
            decision["transaction_id"] = transaction_id
            decision_index = new_decision_state.index[
                new_decision_state["lesion_uid"] == lesion_uid
            ][0]
            for column in DECISION_OUTPUT_COLUMNS:
                new_decision_state.at[decision_index, column] = decision[column]

            atomic_write_csv(new_candidate_state, self.paths.candidate_output)
            atomic_write_csv(new_decision_state, self.paths.decision_output)
            self.candidate_state = new_candidate_state
            self.decision_state = new_decision_state
            self.revision = revision
            return {
                "lesion_uid": lesion_uid,
                "revision": revision,
                "transaction_id": transaction_id,
                "complete": self.is_complete(lesion_uid),
            }

    def asset_token(self, relative_path: str) -> str:
        return self.asset_tokens.get(relative_path, "")

    def summary(self) -> dict[str, Any]:
        complete, incomplete = self.completion_counts()
        return {
            "pilot_lesions": len(self.pilot_order),
            "pilot_candidates": len(self.source_candidates),
            "contact_sheet_assets": len(self.asset_paths),
            "complete": complete,
            "incomplete": incomplete,
            "revision": self.revision,
            "candidate_output_exists": self.paths.candidate_output.is_file(),
            "decision_output_exists": self.paths.decision_output.is_file(),
            "restricted_private_artifact_accessed": False,
        }


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def binary_select(name: str, value: str) -> str:
    options = [("", "Not reviewed"), ("1", "Yes"), ("0", "No")]
    rendered = "".join(
        f'<option value="{option}"{" selected" if option == value else ""}>{label}</option>'
        for option, label in options
    )
    return f'<select name="{escape(name)}">{rendered}</select>'


def status_select(value: str) -> str:
    options = [("", "Not completed")] + [(status, status) for status in FINAL_STATUSES]
    return "<select name=\"final_review_status\">" + "".join(
        f'<option value="{escape(option)}"{" selected" if option == value else ""}>{escape(label)}</option>'
        for option, label in options
    ) + "</select>"


def case_url(lesion_uid: str, incomplete_only: bool = False, saved: bool = False) -> str:
    query: list[str] = []
    if incomplete_only:
        query.append("incomplete=1")
    if saved:
        query.append("saved=1")
    suffix = f"?{'&'.join(query)}" if query else ""
    return f"/case/{quote(lesion_uid, safe='')}{suffix}"


def candidate_phase_fields(item_id: str, row: Mapping[str, Any], phase: str) -> str:
    fields = [
        f"lesion_visible_{phase}",
        f"sac_opacified_{phase}",
        f"sac_locatable_{phase}",
        f"neck_visible_{phase}",
        f"parent_visible_{phase}",
        f"branch_visible_{phase}",
        f"roi_feasible_{phase}",
    ]
    return "".join(
        f'<label>{escape(DISPLAY_LABELS[field])}{binary_select(f"{item_id}__{field}", str(row[field]))}</label>'
        for field in fields
    )


def render_case_page(
    store: ReviewStore,
    lesion_uid: str,
    xsrf_html: str = "",
    incomplete_only: bool = False,
    errors: list[str] | None = None,
    saved: bool = False,
    candidate_override: pd.DataFrame | None = None,
    decision_override: dict[str, str] | None = None,
) -> str:
    candidates = candidate_override if candidate_override is not None else store.candidate_rows(lesion_uid)
    decision = decision_override if decision_override is not None else store.decision_row(lesion_uid)
    pilot_position = store.pilot_order.index(lesion_uid)
    display_uids = store.filtered_uids(incomplete_only)
    if lesion_uid in display_uids:
        display_position = display_uids.index(lesion_uid)
        previous_uid = display_uids[display_position - 1] if display_position > 0 else ""
        next_uid = (
            display_uids[display_position + 1]
            if display_position + 1 < len(display_uids)
            else ""
        )
    else:
        previous_uid = ""
        next_uid = display_uids[0] if display_uids else ""
    complete, incomplete = store.completion_counts()
    selected_pre = decision["selected_pre_review_item_id"]
    selected_post = decision["selected_post_review_item_id"]

    error_html = ""
    if errors:
        error_html = '<section class="alert error"><strong>Not saved:</strong><ul>' + "".join(
            f"<li>{escape(error)}</li>" for error in errors
        ) + "</ul></section>"
    saved_html = (
        '<section class="alert success">Saved atomically with a paired backup.</section>'
        if saved
        else ""
    )

    cards: list[str] = []
    for _, row_series in candidates.iterrows():
        row = row_series.to_dict()
        item_id = str(row["review_item_id"])
        pre_token = store.asset_token(str(row["pre_contact_sheet"]))
        post_token = store.asset_token(str(row["post_contact_sheet"]))
        pre_image = (
            f'<img src="/asset/{pre_token}" alt="Pre contact sheet">'
            if pre_token
            else '<div class="missing">No Pre contact sheet</div>'
        )
        post_image = (
            f'<img src="/asset/{post_token}" alt="Post contact sheet">'
            if post_token
            else '<div class="missing">No Post contact sheet</div>'
        )
        cards.append(
            f"""
            <article class="candidate">
              <h2>Candidate {escape(row['candidate_index_for_lesion'])}/{escape(row['candidate_count_for_lesion'])}: {escape(row['candidate_series_id'])}</h2>
              <div class="candidate-meta">
                <div><b>Review item</b><span>{escape(item_id)}</span></div>
                <div><b>Source/rank</b><span>{escape(row['candidate_source_type'])} / {escape(row['candidate_discovery_rank'])}</span></div>
                <div><b>Valid / v2 selected</b><span>{escape(row['candidate_valid'])} / {escape(row['candidate_selected_in_v2'])}</span></div>
                <div class="wide"><b>Path</b><span>{escape(row['candidate_series_path'])}</span></div>
                <div><b>Pre internal</b><span>{escape(row['pre_internal_series'])}</span></div>
                <div><b>Pre selected / frames / pairs</b><span>{escape(row['pre_selected_internal_series_in_v2'])} / {escape(row['pre_selected_n_frames'])} / {escape(row['pre_selected_n_contiguous_pairs'])}</span></div>
                <div><b>Post internal</b><span>{escape(row['post_internal_series'])}</span></div>
                <div><b>Post selected / frames / pairs</b><span>{escape(row['post_selected_internal_series_in_v2'])} / {escape(row['post_selected_n_frames'])} / {escape(row['post_selected_n_contiguous_pairs'])}</span></div>
              </div>
              <div class="concordance">
                <label>{escape(DISPLAY_LABELS['side_concordant'])}{binary_select(f"{item_id}__side_concordant", str(row['side_concordant']))}</label>
                <label>{escape(DISPLAY_LABELS['location_concordant'])}{binary_select(f"{item_id}__location_concordant", str(row['location_concordant']))}</label>
              </div>
              <div class="phase-grid">
                <section class="phase"><h3>Pre</h3>{pre_image}<div class="fields">{candidate_phase_fields(item_id, row, 'pre')}</div>
                  <label class="selection"><input type="radio" name="selected_for_pre" value="{escape(item_id)}"{' checked' if selected_pre == item_id else ''}> Select this candidate for Pre</label>
                </section>
                <section class="phase"><h3>Post</h3>{post_image}<div class="fields">{candidate_phase_fields(item_id, row, 'post')}</div>
                  <label class="selection"><input type="radio" name="selected_for_post" value="{escape(item_id)}"{' checked' if selected_post == item_id else ''}> Select this candidate for Post</label>
                </section>
              </div>
            </article>
            """
        )

    previous_link = (
        f'<a class="button" href="{case_url(previous_uid, incomplete_only)}">Previous case</a>'
        if previous_uid
        else '<span class="button disabled">Previous case</span>'
    )
    next_link = (
        f'<a class="button" href="{case_url(next_uid, incomplete_only)}">Next case</a>'
        if next_uid
        else '<span class="button disabled">Next case</span>'
    )
    filter_link = (
        f'<a href="{case_url(lesion_uid, False)}">Show all 30</a>'
        if incomplete_only
        else f'<a href="{case_url(lesion_uid, True)}">Show incomplete only</a>'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ROI Pilot series review - {escape(lesion_uid)}</title>
  <style>
    :root {{ color-scheme: light; }} body {{ font-family: system-ui,sans-serif; margin:0; background:#eef1f4; color:#17202a; }}
    header {{ position:sticky; top:0; z-index:5; background:#172a3a; color:white; padding:12px 20px; display:flex; gap:18px; align-items:center; flex-wrap:wrap; }}
    header a {{ color:#bde1ff; }} main {{ max-width:1500px; margin:18px auto; padding:0 18px 80px; }}
    .summary,.candidate,.decision {{ background:white; border-radius:9px; padding:16px; margin:14px 0; box-shadow:0 1px 5px #bbc4cc; }}
    .summary-grid,.candidate-meta {{ display:grid; grid-template-columns:repeat(3,minmax(180px,1fr)); gap:8px 16px; }}
    .candidate-meta div {{ display:grid; gap:3px; }} .candidate-meta .wide {{ grid-column:1/-1; overflow-wrap:anywhere; }}
    .phase-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }} .phase img {{ width:100%; height:auto; border:1px solid #566; background:#111; }}
    .missing {{ padding:50px; background:#eee; color:#555; text-align:center; }} .fields {{ display:grid; grid-template-columns:repeat(2,minmax(180px,1fr)); gap:8px; margin-top:10px; }}
    label {{ display:grid; gap:4px; font-weight:600; }} select,input[type=text],textarea {{ font:inherit; padding:7px; }}
    .concordance {{ display:flex; gap:18px; margin:12px 0; }} .selection {{ display:block; padding:10px; background:#fff6ce; margin-top:10px; }}
    .decision-grid {{ display:grid; grid-template-columns:repeat(3,minmax(180px,1fr)); gap:12px; }} .decision-grid .wide {{ grid-column:1/-1; }} textarea {{ min-height:90px; }}
    .actions {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:16px; }} .button,button {{ border:0; border-radius:6px; padding:10px 14px; background:#1769aa; color:white; text-decoration:none; cursor:pointer; font:inherit; }}
    .button.secondary {{ background:#526575; }} .button.disabled {{ background:#a9b2b9; cursor:not-allowed; }} .alert {{ padding:12px; border-radius:7px; margin:12px 0; }} .error {{ background:#ffe1e1; }} .success {{ background:#dff6e5; }}
    @media (max-width:900px) {{ .phase-grid,.summary-grid,.candidate-meta,.decision-grid {{ grid-template-columns:1fr; }} .candidate-meta .wide,.decision-grid .wide {{ grid-column:auto; }} }}
  </style>
</head>
<body>
  <header>
    <strong>Train ROI Pilot series review</strong>
    <span>Pilot order {pilot_position + 1}/30</span><span>Complete {complete} · Incomplete {incomplete}</span>
    <a href="/cases{'?incomplete=1' if incomplete_only else ''}">Case list</a>{filter_link}
  </header>
  <main>
    {error_html}{saved_html}
    <section class="summary"><h1>{escape(lesion_uid)}</h1>
      <div class="summary-grid">
        <div><b>Patient</b><br>{escape(decision['patient_id'])}</div><div><b>Side</b><br>{escape(decision['side_raw'])} / {escape(decision['side_normalized'])}</div>
        <div><b>Location</b><br>{escape(decision['location_raw'])} / {escape(decision['location_normalized'])}</div>
        <div><b>Lesion index</b><br>{escape(decision['lesion_index_normalized'])}</div><div><b>Current status</b><br>{escape(decision['current_registration_status'])}</div>
        <div><b>Last revision</b><br>{escape(decision['save_revision'])}</div>
      </div>
    </section>
    <form method="post" action="/save">
      {xsrf_html}<input type="hidden" name="lesion_uid" value="{escape(lesion_uid)}"><input type="hidden" name="incomplete" value="{'1' if incomplete_only else '0'}">
      <div class="concordance">
        <label><input type="radio" name="selected_for_pre" value=""{' checked' if not selected_pre else ''}> No Pre selection yet</label>
        <label><input type="radio" name="selected_for_post" value=""{' checked' if not selected_post else ''}> No Post selection yet</label>
      </div>
      {''.join(cards)}
      <section class="decision"><h2>Lesion-level decision</h2><div class="decision-grid">
        <label>Pre/Post same lesion{binary_select('pre_post_same_lesion', decision['pre_post_same_lesion'])}</label>
        <label>Pre/Post view comparable{binary_select('pre_post_view_comparable', decision['pre_post_view_comparable'])}</label>
        <label>Final review status{status_select(decision['final_review_status'])}</label>
        <label>Reviewer ID<input type="text" name="reviewer_id" value="{escape(decision['reviewer_id'])}"></label>
        <label class="wide">Final review reason<textarea name="final_review_reason">{escape(decision['final_review_reason'])}</textarea></label>
      </div>
      <p><b>Automatic gate:</b> exact/probable requires selected Pre and Post, same-lesion=Yes, and sac, parent, and ROI-feasible=Yes for both selected phases.</p>
      <div class="actions">{previous_link}{next_link}<button type="submit" name="action" value="save">Save</button><button type="submit" name="action" value="save_next">Save and next case</button></div>
      </section>
    </form>
  </main>
</body>
</html>"""


def render_case_list(store: ReviewStore, incomplete_only: bool) -> str:
    uids = store.filtered_uids(incomplete_only)
    rows = []
    for uid in uids:
        decision = store.decision_row(uid)
        rows.append(
            f'<tr><td><a href="{case_url(uid, incomplete_only)}">{escape(uid)}</a></td>'
            f'<td>{escape(decision["patient_id"])}</td><td>{escape(decision["side_normalized"])}</td>'
            f'<td>{escape(decision["location_normalized"])}</td><td>{escape(decision["final_review_status"] or "incomplete")}</td>'
            f'<td>{escape(decision["reviewer_id"])}</td><td>{escape(decision["save_revision"])}</td></tr>'
        )
    complete, incomplete = store.completion_counts()
    toggle = (
        '<a href="/cases">Show all 30</a>'
        if incomplete_only
        else '<a href="/cases?incomplete=1">Show incomplete only</a>'
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ROI Pilot review cases</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:8px;text-align:left}}th{{background:#eef2f5}}a{{color:#1769aa}}</style></head>
<body><h1>Train ROI Pilot review cases</h1><p>Complete {complete} · Incomplete {incomplete} · Showing {len(uids)}</p><p>{toggle}</p>
<table><thead><tr><th>Lesion UID</th><th>Patient</th><th>Side</th><th>Location</th><th>Status</th><th>Reviewer</th><th>Revision</th></tr></thead><tbody>{''.join(rows)}</tbody></table></body></html>"""


class BaseHandler(tornado.web.RequestHandler):
    @property
    def store(self) -> ReviewStore:
        return self.application.settings["review_store"]


class RootHandler(BaseHandler):
    def get(self) -> None:
        self.redirect(case_url(self.store.pilot_order[0]))


class CaseListHandler(BaseHandler):
    def get(self) -> None:
        incomplete_only = self.get_query_argument("incomplete", "0") == "1"
        self.write(render_case_list(self.store, incomplete_only))


class CaseHandler(BaseHandler):
    def get(self, lesion_uid: str) -> None:
        if lesion_uid not in self.store.pilot_set:
            raise tornado.web.HTTPError(404)
        incomplete_only = self.get_query_argument("incomplete", "0") == "1"
        saved = self.get_query_argument("saved", "0") == "1"
        xsrf = self.xsrf_form_html()
        if isinstance(xsrf, bytes):
            xsrf = xsrf.decode("utf-8")
        self.write(
            render_case_page(
                self.store,
                lesion_uid,
                xsrf_html=xsrf,
                incomplete_only=incomplete_only,
                saved=saved,
            )
        )


class SaveHandler(BaseHandler):
    def post(self) -> None:
        form = {
            key: [value.decode("utf-8") for value in values]
            for key, values in self.request.body_arguments.items()
        }
        lesion_uid = first_form_value(form, "lesion_uid")
        if lesion_uid not in self.store.pilot_set:
            raise tornado.web.HTTPError(400, "Unknown Pilot lesion UID")
        incomplete_only = first_form_value(form, "incomplete") == "1"
        action = first_form_value(form, "action", "save")
        try:
            self.store.save_form(lesion_uid, form)
        except ReviewValidationError as exc:
            self.set_status(400)
            xsrf = self.xsrf_form_html()
            if isinstance(xsrf, bytes):
                xsrf = xsrf.decode("utf-8")
            self.write(
                render_case_page(
                    self.store,
                    lesion_uid,
                    xsrf_html=xsrf,
                    incomplete_only=incomplete_only,
                    errors=exc.errors,
                    candidate_override=exc.candidate_draft,
                    decision_override=exc.decision_draft,
                )
            )
            return

        target_uid = lesion_uid
        if action == "save_next":
            available = self.store.filtered_uids(incomplete_only)
            if incomplete_only and lesion_uid not in available:
                later = [
                    uid
                    for uid in available
                    if self.store.pilot_order.index(uid)
                    > self.store.pilot_order.index(lesion_uid)
                ]
                target_uid = later[0] if later else available[0] if available else lesion_uid
            elif lesion_uid in available:
                position = available.index(lesion_uid)
                if position + 1 < len(available):
                    target_uid = available[position + 1]
        self.redirect(case_url(target_uid, incomplete_only, saved=True))


class AssetHandler(BaseHandler):
    def get(self, token: str) -> None:
        path = self.store.asset_paths.get(token)
        if path is None:
            raise tornado.web.HTTPError(404)
        content_type, _ = mimetypes.guess_type(path.name)
        self.set_header("Content-Type", content_type or "application/octet-stream")
        self.set_header("Cache-Control", "private, max-age=3600")
        self.write(path.read_bytes())


def make_application(store: ReviewStore) -> tornado.web.Application:
    return tornado.web.Application(
        [
            (r"/", RootHandler),
            (r"/cases", CaseListHandler),
            (r"/case/(.+)", CaseHandler),
            (r"/save", SaveHandler),
            (r"/asset/([0-9a-f]{24})", AssetHandler),
        ],
        review_store=store,
        xsrf_cookies=True,
        cookie_secret=uuid.uuid4().hex + uuid.uuid4().hex,
        debug=False,
        autoreload=False,
    )


def self_test(paths: AppPaths) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="roi_pilot_review_selftest_") as temp_dir:
        temp = Path(temp_dir)
        test_paths = AppPaths(
            project_root=paths.project_root,
            pilot_manifest=paths.pilot_manifest,
            candidate_source=paths.candidate_source,
            candidate_output=temp / "candidate.csv",
            decision_output=temp / "decision.csv",
            backup_dir=temp / "backups",
            review_report_root=paths.review_report_root,
        )
        store = ReviewStore(test_paths)
        lesion_uid = store.pilot_order[0]
        candidates = store.candidate_rows(lesion_uid)
        first_item = str(candidates.iloc[0]["review_item_id"])
        invalid_form: dict[str, list[str]] = {
            "final_review_status": ["exact_reviewed"],
            "pre_post_same_lesion": ["0"],
            "selected_for_pre": [""],
            "selected_for_post": [""],
        }
        invalid_rejected = False
        try:
            store.save_form(lesion_uid, invalid_form)
        except ReviewValidationError:
            invalid_rejected = True
        if not invalid_rejected:
            raise AssertionError("Self-test invalid exact review was not rejected")

        valid_form: dict[str, list[str]] = {
            "final_review_status": ["exact_reviewed"],
            "pre_post_same_lesion": ["1"],
            "pre_post_view_comparable": ["1"],
            "reviewer_id": ["self_test_reviewer"],
            "final_review_reason": ["self test only"],
            "selected_for_pre": [first_item],
            "selected_for_post": [first_item],
            f"{first_item}__sac_opacified_pre": ["1"],
            f"{first_item}__sac_locatable_pre": ["1"],
            f"{first_item}__parent_visible_pre": ["1"],
            f"{first_item}__roi_feasible_pre": ["1"],
            f"{first_item}__sac_opacified_post": ["0"],
            f"{first_item}__sac_locatable_post": ["1"],
            f"{first_item}__parent_visible_post": ["1"],
            f"{first_item}__roi_feasible_post": ["1"],
        }
        result = store.save_form(lesion_uid, valid_form)
        page = render_case_page(store, lesion_uid)
        if not result["complete"] or "self_test_reviewer" not in page:
            raise AssertionError("Self-test valid save/render failure")
        if not test_paths.candidate_output.is_file() or not test_paths.decision_output.is_file():
            raise AssertionError("Self-test outputs were not created")
        backup_manifests = list(test_paths.backup_dir.glob("backup_*_manifest.json"))
        if not backup_manifests:
            raise AssertionError("Self-test backup was not created")
        store.close()

        resumed = ReviewStore(test_paths)
        resumed_decision = resumed.decision_row(lesion_uid)
        if resumed_decision["final_review_status"] != "exact_reviewed":
            raise AssertionError("Self-test resume did not restore saved review")
        summary = resumed.summary()
        resumed.close()
        return {
            "invalid_exact_rejected": True,
            "valid_exact_saved": True,
            "atomic_outputs_created": True,
            "backup_created": True,
            "resume_verified": True,
            "pilot_lesions": summary["pilot_lesions"],
            "pilot_candidates": summary["pilot_candidates"],
            "real_outputs_modified": False,
        }


def main() -> int:
    args = parse_args()
    if not (1 <= args.port <= 65535):
        raise ValueError("--port must be between 1 and 65535")
    paths = build_paths(args)
    if args.self_test:
        print(json.dumps(self_test(paths), ensure_ascii=False, indent=2))
        return 0

    store = ReviewStore(paths)
    if args.validate_only:
        try:
            print(json.dumps(store.summary(), ensure_ascii=False, indent=2))
        finally:
            store.close()
        return 0

    application = make_application(store)
    application.listen(args.port, address=args.host)
    print(
        json.dumps(
            {
                **store.summary(),
                "host": args.host,
                "port": args.port,
                "url": f"http://{args.host}:{args.port}/",
                "candidate_output": str(paths.candidate_output),
                "decision_output": str(paths.decision_output),
                "backup_dir": str(paths.backup_dir),
                "model_training": False,
                "sea_raft_run": False,
                "roi_generated": False,
                "features_calculated": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    try:
        tornado.ioloop.IOLoop.current().start()
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
