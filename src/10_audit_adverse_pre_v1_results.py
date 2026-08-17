#!/root/autodl-tmp/envs/aneurysm-ml/bin/python
"""Independent, read-only verification of adverse_pre_v1 persisted results.

The script never imports or executes the training module, never loads a fitted
model, and never reads the original Train/Valid input CSV files. It reads only
the locked training source and the persisted adverse_pre_v1 output/report tree,
then exclusively creates reports/adverse_pre_v1/independent_verification.md.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SOURCE = PROJECT_ROOT / "code/09_train_adverse_pre_v1.py"
OUTPUT_DIR = PROJECT_ROOT / "outputs/baselines/adverse_pre_v1"
REPORT_DIR = PROJECT_ROOT / "reports/adverse_pre_v1"
AUDIT_REPORT = REPORT_DIR / "independent_verification.md"

MODEL_ORDER = ["Dummy", "Logistic", "CatBoost"]
MODEL_SLUG = {"Dummy": "dummy", "Logistic": "logistic", "CatBoost": "catboost"}
EXPECTED_ROWS = {"Train OOF": 794, "Valid": 209}
FLOAT_METRICS = [
    "auroc",
    "auprc",
    "balanced_accuracy",
    "f1",
    "precision",
    "sensitivity",
    "specificity",
    "brier_score",
]
COUNT_METRICS = ["tp", "tn", "fp", "fn"]
BOOTSTRAP_METRICS = FLOAT_METRICS + COUNT_METRICS
EXPECTED_BOOTSTRAP_REPEATS = 2000
EXPECTED_RANDOM_SEED = 42
ABS_TOL = 1e-12
REL_TOL = 1e-10

EXPECTED_OUTPUT_FILES = [
    "train_oof_predictions.csv",
    "valid_predictions.csv",
    "model_comparison.csv",
    "train_oof_metrics.csv",
    "valid_metrics.csv",
    "bootstrap_confidence_intervals.csv",
    "selected_vs_dummy_bootstrap.csv",
    "cv_results.csv",
    "outer_fold_metrics.csv",
    "threshold_search.csv",
    "final_hyperparameters.json",
    "thresholds.json",
    "feature_names.json",
    "configuration.json",
    "frozen_selection.json",
    "environment.txt",
    "gpu_verification.txt",
    "run.log",
    "exit_status.txt",
    "models/dummy.joblib",
    "models/logistic.joblib",
    "models/catboost_pipeline.joblib",
    "models/catboost.cbm",
    "logistic_coefficients.csv",
    "catboost_feature_importance.csv",
    "plots/train_oof_roc.png",
    "plots/train_oof_pr.png",
    "plots/valid_roc.png",
    "plots/valid_pr.png",
    "plots/valid_calibration.png",
    "plots/valid_confusion_dummy.png",
    "plots/valid_confusion_logistic.png",
    "plots/valid_confusion_catboost.png",
    "plots/logistic_coefficients.png",
    "plots/catboost_feature_importance.png",
]
EXPECTED_REPORT_FILES = ["report.md", "data_validation.md", "execution_summary.md"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_state(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, sha256_file(path)


def input_snapshot() -> dict[str, tuple[int, int, str]]:
    paths = [TRAINING_SOURCE]
    paths.extend(path for path in OUTPUT_DIR.rglob("*") if path.is_file())
    paths.extend(
        path
        for path in REPORT_DIR.rglob("*")
        if path.is_file() and path.resolve() != AUDIT_REPORT.resolve()
    )
    return {
        str(path.relative_to(PROJECT_ROOT)): file_state(path)
        for path in sorted(paths)
    }


def parse_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        if "=" not in raw_line or raw_line.startswith("---"):
            continue
        key, value = raw_line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def utc_mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def close_float(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=REL_TOL, abs_tol=ABS_TOL)
    except (TypeError, ValueError):
        return False


def bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def format_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.15g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def markdown_cell(value: Any) -> str:
    return format_value(value).replace("|", "\\|").replace("\n", "<br>")


def extract_required_files_from_source(tree: ast.AST) -> list[str] | None:
    for node in getattr(tree, "body", []):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "REQUIRED_RELATIVE_FILES"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            return [str(item) for item in value]
    return None


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else math.nan
    specificity = float(tn / (tn + fp)) if tn + fp else math.nan
    return {
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


@dataclass
class RequirementResult:
    number: int
    title: str
    passed: bool
    evidence: str
    issues: list[str] = field(default_factory=list)


class Audit:
    def __init__(self) -> None:
        self.started_at = datetime.now(timezone.utc)
        self.requirements: list[RequirementResult] = []
        self.metric_rows: list[dict[str, Any]] = []
        self.artifact_rows: list[dict[str, Any]] = []
        self.differences: list[str] = []
        self.warnings: list[str] = [
            "按验收禁令，CatBoost GPU 结论仅核对既有配置、日志和 gpu_verification.txt；未重新执行 GPU 训练测试，也未加载模型重新预测。",
            "冻结先后关系依据持久化 JSON 时间、文件 mtime_ns、run.log 行序和训练源码控制流交叉验证；不存在外部不可篡改时间戳服务。",
        ]
        self.source_text = ""
        self.source_tree: ast.AST | None = None
        self.train_predictions = pd.DataFrame()
        self.valid_predictions = pd.DataFrame()
        self.train_recorded = pd.DataFrame()
        self.valid_recorded = pd.DataFrame()
        self.thresholds: dict[str, float] = {}
        self.frozen: dict[str, Any] = {}
        self.configuration: dict[str, Any] = {}
        self.recalculated: dict[str, dict[str, dict[str, Any]]] = {}
        self.snapshot_before: dict[str, tuple[int, int, str]] = {}

    def add_requirement(
        self,
        number: int,
        title: str,
        issues: list[str],
        evidence: str,
    ) -> None:
        passed = not issues
        self.requirements.append(RequirementResult(number, title, passed, evidence, issues))
        if issues:
            self.differences.extend(f"要求 {number}（{title}）：{issue}" for issue in issues)

    def load_inputs(self) -> None:
        self.snapshot_before = input_snapshot()
        self.source_text = TRAINING_SOURCE.read_text(encoding="utf-8")
        self.source_tree = ast.parse(self.source_text, filename=str(TRAINING_SOURCE))
        self.train_predictions = pd.read_csv(
            OUTPUT_DIR / "train_oof_predictions.csv", encoding="utf-8-sig"
        )
        self.valid_predictions = pd.read_csv(
            OUTPUT_DIR / "valid_predictions.csv", encoding="utf-8-sig"
        )
        self.train_recorded = pd.read_csv(
            OUTPUT_DIR / "train_oof_metrics.csv", encoding="utf-8-sig"
        )
        self.valid_recorded = pd.read_csv(
            OUTPUT_DIR / "valid_metrics.csv", encoding="utf-8-sig"
        )
        self.thresholds = {
            str(key): float(value)
            for key, value in json.loads(
                (OUTPUT_DIR / "thresholds.json").read_text(encoding="utf-8")
            ).items()
        }
        self.frozen = json.loads(
            (OUTPUT_DIR / "frozen_selection.json").read_text(encoding="utf-8")
        )
        self.configuration = json.loads(
            (OUTPUT_DIR / "configuration.json").read_text(encoding="utf-8")
        )

    def check_status(self) -> None:
        issues: list[str] = []
        success_path = OUTPUT_DIR / ".SUCCESS"
        running_path = OUTPUT_DIR / ".RUNNING"
        exit_path = OUTPUT_DIR / "exit_status.txt"
        success_values: dict[str, str] = {}
        if not success_path.is_file() or success_path.stat().st_size <= 0:
            issues.append(".SUCCESS 不存在或为空")
        else:
            success_values = parse_key_values(success_path.read_text(encoding="utf-8"))
            if success_values.get("status") != "SUCCESS":
                issues.append(f".SUCCESS status={success_values.get('status')!r}，期望 SUCCESS")
            if success_values.get("exit_status") != "0":
                issues.append(
                    f".SUCCESS exit_status={success_values.get('exit_status')!r}，期望 0"
                )
        if not exit_path.is_file() or exit_path.stat().st_size <= 0:
            issues.append("exit_status.txt 不存在或为空")
            exit_text = "<missing>"
        else:
            exit_text = exit_path.read_text(encoding="utf-8").strip()
            if exit_text != "0":
                issues.append(f"exit_status.txt={exit_text!r}，期望 '0'")
        if running_path.exists():
            issues.append(".RUNNING 仍然存在")
        evidence = (
            f".SUCCESS status={success_values.get('status', '<missing>')}, "
            f"exit_status={success_values.get('exit_status', '<missing>')}; "
            f"exit_status.txt={exit_text}; .RUNNING={'存在' if running_path.exists() else '不存在'}"
        )
        self.add_requirement(1, "完成标记与退出状态", issues, evidence)

    def check_rows(self) -> None:
        issues: list[str] = []
        train_n = len(self.train_predictions)
        valid_n = len(self.valid_predictions)
        if train_n != EXPECTED_ROWS["Train OOF"]:
            issues.append(f"Train OOF 行数={train_n}，期望 794")
        if valid_n != EXPECTED_ROWS["Valid"]:
            issues.append(f"Valid 行数={valid_n}，期望 209")
        if set(self.train_predictions.get("split", pd.Series(dtype=str)).astype(str)) != {"train"}:
            issues.append("Train OOF split 列不严格等于 train")
        if set(self.valid_predictions.get("split", pd.Series(dtype=str)).astype(str)) != {"valid"}:
            issues.append("Valid split 列不严格等于 valid")
        self.add_requirement(
            2,
            "预测行数",
            issues,
            f"Train OOF={train_n} 行；Valid={valid_n} 行",
        )

    def check_patients(self) -> None:
        issues: list[str] = []
        for label, frame in [("Train OOF", self.train_predictions), ("Valid", self.valid_predictions)]:
            if "patient_id" not in frame.columns:
                issues.append(f"{label} 缺少 patient_id")
                continue
            if frame["patient_id"].isna().any():
                issues.append(f"{label} patient_id 存在缺失")
            if not frame["patient_id"].is_unique:
                duplicates = int(frame["patient_id"].duplicated(keep=False).sum())
                issues.append(f"{label} patient_id 非唯一，涉及 {duplicates} 行")
        overlap = set(self.train_predictions["patient_id"]).intersection(
            set(self.valid_predictions["patient_id"])
        )
        if overlap:
            issues.append(f"Train/Valid patient_id 交集={len(overlap)}")
        self.add_requirement(
            3,
            "患者唯一性与集合隔离",
            issues,
            f"Train 唯一={self.train_predictions['patient_id'].is_unique}; "
            f"Valid 唯一={self.valid_predictions['patient_id'].is_unique}; 交集={len(overlap)}",
        )

    def check_probabilities(self) -> None:
        issues: list[str] = []
        evidence_parts: list[str] = []
        for dataset_name, frame in [
            ("Train OOF", self.train_predictions),
            ("Valid", self.valid_predictions),
        ]:
            for model in MODEL_ORDER:
                column = f"{MODEL_SLUG[model]}_probability"
                if column not in frame.columns:
                    issues.append(f"{dataset_name} 缺少 {column}")
                    continue
                numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
                missing = int(np.isnan(numeric).sum())
                infinite = int(np.isinf(numeric).sum())
                out_of_range = int(((numeric < 0.0) | (numeric > 1.0)).sum())
                if missing or infinite or out_of_range:
                    issues.append(
                        f"{dataset_name}/{model}: missing={missing}, infinite={infinite}, "
                        f"out_of_[0,1]={out_of_range}"
                    )
                finite = numeric[np.isfinite(numeric)]
                if finite.size:
                    evidence_parts.append(
                        f"{dataset_name}/{model}=[{finite.min():.6g}, {finite.max():.6g}]"
                    )
        self.add_requirement(
            4,
            "三模型概率完整性与范围",
            issues,
            "; ".join(evidence_parts),
        )

    def recompute_and_compare_metrics(self) -> None:
        issues_recompute: list[str] = []
        issues_compare: list[str] = []
        self.recalculated = {}
        dataset_inputs = [
            ("Train OOF", self.train_predictions, self.train_recorded),
            ("Valid", self.valid_predictions, self.valid_recorded),
        ]
        max_abs_delta = 0.0
        nonzero_deltas: list[str] = []
        for dataset_name, predictions, recorded_frame in dataset_inputs:
            self.recalculated[dataset_name] = {}
            if set(recorded_frame.get("model", pd.Series(dtype=str)).astype(str)) != set(MODEL_ORDER):
                issues_compare.append(
                    f"{dataset_name} 指标表模型集合={sorted(recorded_frame.get('model', []))}"
                )
            labels_numeric = pd.to_numeric(predictions["adverse"], errors="coerce")
            if labels_numeric.isna().any() or set(labels_numeric.astype(int)) != {0, 1}:
                issues_recompute.append(f"{dataset_name} adverse 标签不是完整二分类 0/1")
            y_true = labels_numeric.astype(int).to_numpy()
            for model in MODEL_ORDER:
                slug = MODEL_SLUG[model]
                probability_column = f"{slug}_probability"
                prediction_column = f"{slug}_prediction"
                threshold_column = f"{slug}_threshold"
                threshold = self.thresholds.get(model)
                if threshold is None:
                    issues_recompute.append(f"thresholds.json 缺少 {model}")
                    continue
                probabilities = predictions[probability_column].to_numpy(dtype=float)
                recalculated = calculate_metrics(y_true, probabilities, threshold)
                self.recalculated[dataset_name][model] = recalculated

                stored_thresholds = predictions[threshold_column].to_numpy(dtype=float)
                if not np.isfinite(stored_thresholds).all() or not np.allclose(
                    stored_thresholds, threshold, rtol=0.0, atol=ABS_TOL
                ):
                    issues_recompute.append(
                        f"{dataset_name}/{model} 逐行 threshold 与 thresholds.json 不一致"
                    )
                expected_predictions = (probabilities >= threshold).astype(int)
                stored_predictions = pd.to_numeric(
                    predictions[prediction_column], errors="coerce"
                ).to_numpy(dtype=float)
                if not np.array_equal(stored_predictions, expected_predictions.astype(float)):
                    mismatch = int(np.sum(stored_predictions != expected_predictions))
                    issues_recompute.append(
                        f"{dataset_name}/{model} prediction 列与 probability>=冻结阈值不一致 {mismatch} 行"
                    )

                matching_rows = recorded_frame[recorded_frame["model"].astype(str) == model]
                if len(matching_rows) != 1:
                    issues_compare.append(
                        f"{dataset_name}/{model} 指标表匹配行数={len(matching_rows)}，期望 1"
                    )
                    continue
                recorded = matching_rows.iloc[0]
                threshold_ok = close_float(recorded["threshold"], threshold)
                self.metric_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model,
                        "metric": "threshold",
                        "recalculated": threshold,
                        "recorded": recorded["threshold"],
                        "delta": float(recorded["threshold"]) - threshold,
                        "passed": threshold_ok,
                    }
                )
                if not threshold_ok:
                    issues_compare.append(
                        f"{dataset_name}/{model}/threshold: frozen={threshold}, recorded={recorded['threshold']}"
                    )

                for metric in FLOAT_METRICS:
                    left = float(recalculated[metric])
                    right = float(recorded[metric])
                    delta = right - left
                    passed = close_float(left, right)
                    max_abs_delta = max(max_abs_delta, abs(delta))
                    if delta != 0.0:
                        nonzero_deltas.append(
                            f"{dataset_name}/{model}/{metric}: recorded-recomputed={delta:+.3e}"
                        )
                    self.metric_rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model,
                            "metric": metric,
                            "recalculated": left,
                            "recorded": right,
                            "delta": delta,
                            "passed": passed,
                        }
                    )
                    if not passed:
                        issues_compare.append(
                            f"{dataset_name}/{model}/{metric}: recomputed={left:.15g}, "
                            f"recorded={right:.15g}, delta={delta:+.3e}"
                        )

                for metric in COUNT_METRICS:
                    left = int(recalculated[metric])
                    right = int(recorded[metric])
                    passed = left == right
                    self.metric_rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model,
                            "metric": metric,
                            "recalculated": left,
                            "recorded": right,
                            "delta": right - left,
                            "passed": passed,
                        }
                    )
                    if not passed:
                        issues_compare.append(
                            f"{dataset_name}/{model}/{metric}: recomputed={left}, recorded={right}"
                        )

                try:
                    recorded_matrix = json.loads(str(recorded["confusion_matrix"]))
                except json.JSONDecodeError:
                    recorded_matrix = None
                matrix_ok = recorded_matrix == recalculated["confusion_matrix"]
                self.metric_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model,
                        "metric": "confusion_matrix",
                        "recalculated": recalculated["confusion_matrix"],
                        "recorded": recorded_matrix,
                        "delta": "—",
                        "passed": matrix_ok,
                    }
                )
                if not matrix_ok:
                    issues_compare.append(
                        f"{dataset_name}/{model}/confusion_matrix: "
                        f"recomputed={recalculated['confusion_matrix']}, recorded={recorded_matrix}"
                    )

        self.add_requirement(
            5,
            "从预测 CSV 独立重算全部指标",
            issues_recompute,
            f"完成 2 个数据集 × 3 个模型；使用 thresholds.json 冻结阈值，未搜索新阈值",
        )
        self.add_requirement(
            6,
            "与正式指标 CSV 逐项对照",
            issues_compare,
            f"逐项比较 {len(self.metric_rows)} 项；最大浮点绝对差={max_abs_delta:.3e}",
        )
        if nonzero_deltas:
            self.differences.extend(nonzero_deltas)

    def check_freeze_order(self) -> None:
        issues: list[str] = []
        frozen_path = OUTPUT_DIR / "frozen_selection.json"
        valid_path = OUTPUT_DIR / "valid_predictions.csv"
        frozen_mtime = utc_mtime(frozen_path)
        valid_mtime = utc_mtime(valid_path)
        try:
            frozen_at = parse_utc(str(self.frozen["frozen_at_utc"]))
        except (KeyError, ValueError) as exc:
            issues.append(f"frozen_at_utc 无法解析：{exc}")
            frozen_at = frozen_mtime
        if frozen_path.stat().st_mtime_ns >= valid_path.stat().st_mtime_ns:
            issues.append("frozen_selection.json 文件 mtime_ns 不早于 valid_predictions.csv")
        if frozen_at >= valid_mtime:
            issues.append("frozen_at_utc 不早于 valid_predictions.csv 文件写入时间")

        write_marker = 'write_json_sync(OUTPUT_DIR / "frozen_selection.json", frozen_selection)'
        valid_predict_marker = "final_models[model_name].predict_proba(x_valid)"
        write_position = self.source_text.find(write_marker)
        predict_position = self.source_text.find(valid_predict_marker)
        if write_position < 0 or predict_position < 0 or write_position >= predict_position:
            issues.append("训练源码未证明 frozen_selection 持久化发生在 Valid predict_proba 之前")

        run_log = (OUTPUT_DIR / "run.log").read_text(encoding="utf-8")
        log_freeze = run_log.find("frozen_selection.json durably written before any Valid prediction")
        log_valid = run_log.find("official Valid prediction complete")
        if log_freeze < 0 or log_valid < 0 or log_freeze >= log_valid:
            issues.append("run.log 中冻结记录未严格位于首条正式 Valid 预测完成记录之前")

        margin_ms = (valid_mtime - frozen_mtime).total_seconds() * 1000.0
        evidence = (
            f"frozen mtime={frozen_mtime.isoformat()}; frozen_at_utc={frozen_at.isoformat()}; "
            f"valid_predictions mtime={valid_mtime.isoformat()}; mtime 领先 {margin_ms:.3f} ms；"
            f"源码位置 {write_position}<{predict_position}；日志位置 {log_freeze}<{log_valid}"
        )
        self.add_requirement(7, "冻结文件早于正式 Valid 预测", issues, evidence)

    def check_selection_basis(self) -> None:
        issues: list[str] = []
        train_metrics = self.recalculated.get("Train OOF", {})
        ranking = sorted(
            MODEL_ORDER,
            key=lambda model: (
                -float(train_metrics[model]["auprc"]),
                -float(train_metrics[model]["auroc"]),
                float(train_metrics[model]["brier_score"]),
                MODEL_ORDER.index(model),
            ),
        )
        frozen_records = sorted(
            self.frozen.get("train_oof_model_ranking", []), key=lambda item: item.get("rank", 999)
        )
        frozen_ranking = [str(item.get("model")) for item in frozen_records]
        if frozen_ranking != ranking:
            issues.append(f"冻结排名={frozen_ranking}，Train OOF 独立重算排名={ranking}")
        if self.frozen.get("selected_model") != ranking[0]:
            issues.append(
                f"selected_model={self.frozen.get('selected_model')!r}，Train OOF 第一名={ranking[0]!r}"
            )
        if self.frozen.get("valid_used_before_freeze") is not False:
            issues.append("frozen_selection.valid_used_before_freeze 不是 false")
        expected_rule = ["AUPRC descending", "AUROC descending", "Brier Score ascending"]
        if self.frozen.get("selection_rule") != expected_rule:
            issues.append(f"冻结 selection_rule={self.frozen.get('selection_rule')!r}")
        expected_config_rule = [
            "Train pooled OOF AUPRC descending",
            "Train pooled OOF AUROC descending",
            "Train pooled OOF Brier Score ascending",
        ]
        if self.configuration.get("model_selection") != expected_config_rule:
            issues.append("configuration.model_selection 不是预定 Train pooled OOF 三级排序")
        cat_config = self.configuration.get("catboost", {})
        if cat_config.get("valid_eval_set") is not False or cat_config.get("early_stopping") is not False:
            issues.append("configuration 显示 Valid eval_set 或 early stopping 未禁用")

        rank_marker = "ranking = model_ranking(train_metrics)"
        selected_marker = "selected_model = ranking[0]"
        freeze_marker = 'write_json_sync(OUTPUT_DIR / "frozen_selection.json", frozen_selection)'
        valid_marker = "final_models[model_name].predict_proba(x_valid)"
        positions = [self.source_text.find(marker) for marker in [rank_marker, selected_marker, freeze_marker, valid_marker]]
        if any(position < 0 for position in positions) or positions != sorted(positions):
            issues.append(f"训练源码模型选择/冻结/Valid 顺序异常：positions={positions}")

        for item in frozen_records:
            model = str(item.get("model"))
            if model not in train_metrics:
                issues.append(f"冻结排名含未知模型 {model}")
                continue
            comparisons = {
                "train_oof_auprc": train_metrics[model]["auprc"],
                "train_oof_auroc": train_metrics[model]["auroc"],
                "train_oof_brier_score": train_metrics[model]["brier_score"],
            }
            for key, recalculated in comparisons.items():
                if not close_float(item.get(key), recalculated):
                    issues.append(
                        f"冻结排名 {model}/{key}={item.get(key)}，重算={recalculated}"
                    )

        comparison = pd.read_csv(OUTPUT_DIR / "model_comparison.csv", encoding="utf-8-sig")
        if set(comparison["model"].astype(str)) != set(MODEL_ORDER):
            issues.append("model_comparison.csv 模型集合不完整")
        selected_rows = comparison[comparison["selected"].map(bool_value)]
        if len(selected_rows) != 1 or str(selected_rows.iloc[0]["model"]) != ranking[0]:
            issues.append("model_comparison.csv selected 标记与 Train OOF 第一名不一致")
        for expected_rank, model in enumerate(ranking, start=1):
            row = comparison[comparison["model"].astype(str) == model]
            if len(row) != 1 or int(row.iloc[0]["train_oof_rank"]) != expected_rank:
                issues.append(f"model_comparison.csv {model} 排名不一致")

        valid_auprc_best = max(
            MODEL_ORDER,
            key=lambda model: float(self.recalculated["Valid"][model]["auprc"]),
        )
        evidence = (
            f"Train OOF 独立排名={' > '.join(ranking)}；冻结入选={self.frozen.get('selected_model')}；"
            f"Valid AUPRC 最高={valid_auprc_best}，但 selected 仍保持 {ranking[0]}；源码顺序={positions}"
        )
        self.add_requirement(8, "模型选择仅依据 Train pooled OOF", issues, evidence)

    def check_bootstrap(self) -> None:
        issues: list[str] = []
        bootstrap = pd.read_csv(
            OUTPUT_DIR / "bootstrap_confidence_intervals.csv", encoding="utf-8-sig"
        )
        paired = pd.read_csv(
            OUTPUT_DIR / "selected_vs_dummy_bootstrap.csv", encoding="utf-8-sig"
        )
        expected_pairs = {(model, metric) for model in MODEL_ORDER for metric in BOOTSTRAP_METRICS}
        actual_pairs = set(zip(bootstrap["model"].astype(str), bootstrap["metric"].astype(str)))
        if actual_pairs != expected_pairs or len(bootstrap) != len(expected_pairs):
            issues.append(
                f"bootstrap 主表组合数={len(actual_pairs)}/行数={len(bootstrap)}，期望 36 个唯一组合"
            )
        if not (pd.to_numeric(bootstrap["valid_repeats"]) == EXPECTED_BOOTSTRAP_REPEATS).all():
            issues.append("bootstrap_confidence_intervals.csv 并非所有 valid_repeats=2000")
        if not (pd.to_numeric(paired["valid_repeats"]) == EXPECTED_BOOTSTRAP_REPEATS).all():
            issues.append("selected_vs_dummy_bootstrap.csv 并非所有 valid_repeats=2000")
        if set(paired["metric"].astype(str)) != {
            "auroc_improvement",
            "auprc_improvement",
            "brier_improvement",
        } or len(paired) != 3:
            issues.append("配对 Bootstrap 指标集合或行数不正确")
        for frame_name, frame in [("主表", bootstrap), ("配对表", paired)]:
            repeats = pd.to_numeric(frame["valid_repeats"]).astype(int)
            attempts = pd.to_numeric(frame["sampling_attempts"]).astype(int)
            if (attempts < repeats).any():
                issues.append(f"Bootstrap {frame_name} sampling_attempts 小于 valid_repeats")
        if "skipped_single_class" in bootstrap.columns:
            expected_skipped = (
                pd.to_numeric(bootstrap["sampling_attempts"]).astype(int)
                - pd.to_numeric(bootstrap["valid_repeats"]).astype(int)
            )
            if not np.array_equal(
                pd.to_numeric(bootstrap["skipped_single_class"]).astype(int), expected_skipped
            ):
                issues.append("skipped_single_class != sampling_attempts-valid_repeats")
        config_repeats = self.configuration.get("bootstrap", {}).get("valid_patient_repeats")
        if config_repeats != EXPECTED_BOOTSTRAP_REPEATS:
            issues.append(f"configuration bootstrap repeats={config_repeats}")
        run_log = (OUTPUT_DIR / "run.log").read_text(encoding="utf-8")
        if "Valid bootstrap complete | effective=2000" not in run_log:
            issues.append("run.log 未记录 effective=2000")
        attempts_values = sorted(set(pd.to_numeric(bootstrap["sampling_attempts"]).astype(int)))
        skipped_values = sorted(set(pd.to_numeric(bootstrap["skipped_single_class"]).astype(int)))
        self.add_requirement(
            9,
            "Bootstrap 有效重复数",
            issues,
            f"主表={len(bootstrap)} 行，配对表={len(paired)} 行；valid_repeats=2000；"
            f"attempts={attempts_values}；skipped_single_class={skipped_values}",
        )

    def check_artifacts(self) -> None:
        issues: list[str] = []
        assert self.source_tree is not None
        source_required = extract_required_files_from_source(self.source_tree)
        if source_required is None:
            issues.append("无法从训练源码解析 REQUIRED_RELATIVE_FILES")
        elif source_required != EXPECTED_OUTPUT_FILES:
            missing_in_source = sorted(set(EXPECTED_OUTPUT_FILES) - set(source_required))
            extra_in_source = sorted(set(source_required) - set(EXPECTED_OUTPUT_FILES))
            issues.append(
                f"训练源码必需产物清单与锁定清单不同：missing={missing_in_source}, extra={extra_in_source}"
            )

        for relative in EXPECTED_OUTPUT_FILES:
            path = OUTPUT_DIR / relative
            exists = path.is_file()
            size = path.stat().st_size if exists else 0
            passed = exists and size > 0
            category = (
                "模型"
                if path.suffix in {".joblib", ".cbm"}
                else "CSV"
                if path.suffix == ".csv"
                else "JSON"
                if path.suffix == ".json"
                else "图片"
                if path.suffix == ".png"
                else "证据/日志"
            )
            self.artifact_rows.append(
                {
                    "category": category,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "size": size,
                    "passed": passed,
                }
            )
            if not passed:
                issues.append(f"缺失或为空：{path.relative_to(PROJECT_ROOT)}")
        for relative in EXPECTED_REPORT_FILES:
            path = REPORT_DIR / relative
            exists = path.is_file()
            size = path.stat().st_size if exists else 0
            passed = exists and size > 0
            self.artifact_rows.append(
                {
                    "category": "报告",
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "size": size,
                    "passed": passed,
                }
            )
            if not passed:
                issues.append(f"缺失或为空：{path.relative_to(PROJECT_ROOT)}")
        if not TRAINING_SOURCE.is_file() or TRAINING_SOURCE.stat().st_size <= 0:
            issues.append("训练源码不存在或为空")
        counts: dict[str, int] = {}
        for row in self.artifact_rows:
            counts[row["category"]] = counts.get(row["category"], 0) + 1
        self.add_requirement(
            10,
            "必需模型、CSV、JSON、图片与报告完整性",
            issues,
            "；".join(f"{category}={count}" for category, count in sorted(counts.items())),
        )

    def check_gpu_evidence(self) -> None:
        issues: list[str] = []
        gpu_text = (OUTPUT_DIR / "gpu_verification.txt").read_text(encoding="utf-8")
        gpu = parse_key_values(gpu_text)
        environment = parse_key_values(
            (OUTPUT_DIR / "environment.txt").read_text(encoding="utf-8")
        )
        expected_pairs = {
            "task_type": "GPU",
            "devices": "0",
            "allow_writing_files": "False",
            "cpu_fallback": "False",
            "nvidia_smi_returncode": "0",
            "minimal_catboost_gpu_training_test": "PASS",
        }
        for key, expected in expected_pairs.items():
            if gpu.get(key) != expected:
                issues.append(f"gpu_verification {key}={gpu.get(key)!r}，期望 {expected!r}")
        try:
            gpu_count = int(gpu.get("get_gpu_device_count()", "0"))
        except ValueError:
            gpu_count = 0
        if gpu_count < 1:
            issues.append(f"get_gpu_device_count()={gpu.get('get_gpu_device_count()')!r}")
        if "NVIDIA GeForce RTX 4090" not in gpu_text:
            issues.append("nvidia-smi 证据中未找到 NVIDIA GeForce RTX 4090")
        if environment.get("catboost_task_type") != "GPU" or environment.get("catboost_devices") != "0":
            issues.append("environment.txt 的 CatBoost GPU 配置不一致")
        fixed = self.configuration.get("catboost", {}).get("fixed", {})
        if fixed.get("task_type") != "GPU" or str(fixed.get("devices")) != "0":
            issues.append("configuration.json 的 CatBoost task_type/devices 不是 GPU/0")
        if self.configuration.get("catboost", {}).get("cpu_fallback") is not False:
            issues.append("configuration.json cpu_fallback 不是 false")
        frozen_model = (
            self.frozen.get("final_hyperparameters", {})
            .get("CatBoost", {})
            .get("pipeline", {})
            .get("model", {})
        )
        if frozen_model.get("task_type") != "GPU" or str(frozen_model.get("devices")) != "0":
            issues.append("frozen_selection.json 的最终 CatBoost 配置不是 GPU/0")
        run_log = (OUTPUT_DIR / "run.log").read_text(encoding="utf-8")
        gpu_log = run_log.find("GPU verification passed")
        train_log = run_log.find("Starting true nested OOF procedure")
        if gpu_log < 0 or train_log < 0 or gpu_log >= train_log:
            issues.append("run.log 未证明 GPU 验证在正式模型训练前完成")
        evidence = (
            f"GPU count={gpu_count}; nvidia-smi rc={gpu.get('nvidia_smi_returncode')}; "
            f"device=RTX 4090; minimal test={gpu.get('minimal_catboost_gpu_training_test')}; "
            f"task_type={gpu.get('task_type')}; devices={gpu.get('devices')}; cpu_fallback={gpu.get('cpu_fallback')}"
        )
        self.add_requirement(11, "CatBoost GPU 证据", issues, evidence)

    def check_read_only_and_conclusion(self) -> None:
        snapshot_after = input_snapshot()
        issues: list[str] = []
        before_keys = set(self.snapshot_before)
        after_keys = set(snapshot_after)
        if before_keys != after_keys:
            issues.append(
                f"只读输入文件集合变化：removed={sorted(before_keys-after_keys)}, "
                f"added={sorted(after_keys-before_keys)}"
            )
        changed = sorted(
            key
            for key in before_keys.intersection(after_keys)
            if self.snapshot_before[key] != snapshot_after[key]
        )
        if changed:
            issues.append(f"只读输入内容/大小/mtime 发生变化：{changed}")
        prior_failures = sum(not item.passed for item in self.requirements)
        if prior_failures:
            issues.append(f"前 11 项中有 {prior_failures} 项未通过")
        self.add_requirement(
            12,
            "最终结论与只读性复核",
            issues,
            f"复核 {len(snapshot_after)} 个只读输入文件的 size/mtime_ns/SHA256；变化文件=0；"
            f"前置失败项={prior_failures}",
        )

    def run(self) -> None:
        self.load_inputs()
        self.check_status()
        self.check_rows()
        self.check_patients()
        self.check_probabilities()
        self.recompute_and_compare_metrics()
        self.check_freeze_order()
        self.check_selection_basis()
        self.check_bootstrap()
        self.check_artifacts()
        self.check_gpu_evidence()
        self.check_read_only_and_conclusion()

    @property
    def passed(self) -> bool:
        return len(self.requirements) == 12 and all(item.passed for item in self.requirements)

    def render(self) -> str:
        ended_at = datetime.now(timezone.utc)
        overall = "PASS" if self.passed else "FAIL"
        lines = [
            "# adverse_pre_v1 独立只读验收",
            "",
            "## 最终结论",
            "",
            f"**{overall}**",
            "",
            f"- 审计时间（UTC）：{self.started_at.isoformat()} 至 {ended_at.isoformat()}",
            f"- 审计脚本：`{Path(__file__).relative_to(PROJECT_ROOT)}`",
            "- 只读输入：`code/09_train_adverse_pre_v1.py`、`outputs/baselines/adverse_pre_v1/`、`reports/adverse_pre_v1/`（本报告除外）",
            "- 唯一写入：`reports/adverse_pre_v1/independent_verification.md`",
            "- 未执行：训练脚本、模型加载/预测、调参、阈值搜索、基于 Valid 的模型更换、输入 CSV 读取/修改、依赖安装",
            f"- 浮点比较容差：abs={ABS_TOL:g}，rel={REL_TOL:g}",
            "",
            "## 12 项验收总览",
            "",
            "| # | 验收项 | 结论 | 证据摘要 |",
            "| ---: | --- | --- | --- |",
        ]
        for item in sorted(self.requirements, key=lambda value: value.number):
            lines.append(
                f"| {item.number} | {markdown_cell(item.title)} | "
                f"{'PASS' if item.passed else 'FAIL'} | {markdown_cell(item.evidence)} |"
            )

        lines.extend(
            [
                "",
                "## 指标逐项独立复算与对照",
                "",
                "以下指标全部由预测 CSV 中的 `adverse`、三模型概率和 `thresholds.json` 冻结阈值重算；未重新选择阈值。`delta` 为 `记录值 - 重算值`。",
            ]
        )
        for dataset_name in ["Train OOF", "Valid"]:
            lines.extend(
                [
                    "",
                    f"### {dataset_name}",
                    "",
                    "| Model | Item | Recomputed | Recorded | Delta | Result |",
                    "| --- | --- | ---: | ---: | ---: | --- |",
                ]
            )
            for row in self.metric_rows:
                if row["dataset"] != dataset_name:
                    continue
                lines.append(
                    f"| {row['model']} | {row['metric']} | "
                    f"{markdown_cell(row['recalculated'])} | {markdown_cell(row['recorded'])} | "
                    f"{markdown_cell(row['delta'])} | {'PASS' if row['passed'] else 'FAIL'} |"
                )

        lines.extend(
            [
                "",
                "## 必需产物完整性",
                "",
                "| Category | Path | Bytes | Result |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for row in self.artifact_rows:
            lines.append(
                f"| {row['category']} | `{row['path']}` | {row['size']} | "
                f"{'PASS' if row['passed'] else 'FAIL'} |"
            )

        lines.extend(["", "## 差异", ""])
        if self.differences:
            lines.extend(f"- {difference}" for difference in self.differences)
        else:
            lines.append("- 未发现指标、阈值、混淆矩阵、排名、状态或产物完整性差异。")

        lines.extend(["", "## 警告与验收边界", ""])
        lines.extend(f"- {warning}" for warning in self.warnings)

        failed_details = [item for item in self.requirements if not item.passed]
        lines.extend(["", "## 失败明细", ""])
        if not failed_details:
            lines.append("- 无。")
        else:
            for item in failed_details:
                for issue in item.issues:
                    lines.append(f"- 要求 {item.number}（{item.title}）：{issue}")

        lines.extend(
            [
                "",
                "## 只读性声明",
                "",
                "审计前后对训练源码、既有输出和既有报告逐文件比较了文件集合、大小、mtime_ns 与 SHA256。除本审计报告的创建外，脚本不包含任何既有结果写入路径。",
                "",
            ]
        )
        return "\n".join(lines)


def main() -> int:
    if AUDIT_REPORT.exists():
        print(f"Refusing to overwrite existing audit report: {AUDIT_REPORT}", file=sys.stderr)
        return 2
    audit = Audit()
    try:
        audit.run()
    except Exception as exc:  # Ensure an auditable FAIL report for unexpected read/parse errors.
        audit.differences.append(f"审计执行异常：{type(exc).__name__}: {exc}")
        present_numbers = {item.number for item in audit.requirements}
        for number in range(1, 13):
            if number not in present_numbers:
                audit.requirements.append(
                    RequirementResult(
                        number=number,
                        title="未完成的验收项",
                        passed=False,
                        evidence="审计执行异常后未完成",
                        issues=[f"{type(exc).__name__}: {exc}"],
                    )
                )
    report = audit.render()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with AUDIT_REPORT.open("x", encoding="utf-8") as handle:
        handle.write(report)
        handle.flush()
        os.fsync(handle.fileno())
    if AUDIT_REPORT.stat().st_size <= 0:
        print("Audit report was created but is empty", file=sys.stderr)
        return 2
    print(f"{('PASS' if audit.passed else 'FAIL')}: {AUDIT_REPORT}")
    return 0 if audit.passed else 1


if __name__ == "__main__":
    sys.exit(main())
