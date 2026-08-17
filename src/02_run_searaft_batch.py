# 模型只加载一次；
# 自动按帧号排序；
# 分别处理Pre和Post；
# 自动填充不同尺寸图像；
# 计算光流和不确定性；
# 输出帧对统计特征；
# Pilot阶段可选保存原始数组和可视化；
# 支持断点续跑

# 加载SEA-RAFT一次
#         ↓
# 读取pilot_manifest.csv
#         ↓
# 依次处理每位患者的Pre-Image
#         ↓
# 依次处理每位患者的Post-Image
#         ↓
# 相邻帧计算光流
#         ↓
# 保存帧对统计和少量可视化

作用：
患者Pre/Post影像目录
→ 图像按帧号排序
→ 相邻帧两两配对
→ SEA-RAFT GPU推理
→ 光流场和不确定性图
→ 帧对级统计
→ 序列级统计

输入：
读取一个 Manifest

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


REPO_ROOT = Path(
    "/root/autodl-tmp/aneurysm/code/SEA-RAFT"
)

sys.path.insert(0, str(REPO_ROOT / "core"))
sys.path.insert(0, str(REPO_ROOT))

from raft import RAFT
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff",
}


PAIR_FEATURES = [
    "mag_mean",
    "mag_median",
    "mag_std",
    "mag_p90",
    "mag_p95",
    "mag_max",
    "mag_norm_mean",
    "mag_norm_p90",
    "u_mean",
    "u_std",
    "v_mean",
    "v_std",
    "direction_entropy",
    "uncertainty_mean",
    "uncertainty_std",
    "uncertainty_p90",
    "runtime_s",
]


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)

    return [
        int(part) if part.isdigit()
        else part.lower()
        for part in parts
    ]


def list_frames(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []

    return sorted(
        [
            path
            for path in directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTS
            )
        ],
        key=natural_key,
    )


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(f"图像读取失败：{path}")

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )


def image_to_tensor(
    image: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    return (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        .to(device)
    )


def pad_pair(
    image1: torch.Tensor,
    image2: torch.Tensor,
    multiple: int,
):
    if image1.shape != image2.shape:
        raise ValueError(
            "同一帧对图像尺寸不同："
            f"{tuple(image1.shape)} vs "
            f"{tuple(image2.shape)}"
        )

    height, width = image1.shape[-2:]

    pad_height = (-height) % multiple
    pad_width = (-width) % multiple

    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top

    padding = (left, right, top, bottom)

    image1 = F.pad(
        image1,
        padding,
        mode="replicate",
    )

    image2 = F.pad(
        image2,
        padding,
        mode="replicate",
    )

    return image1, image2, padding


def unpad(
    tensor: torch.Tensor,
    padding,
):
    left, right, top, bottom = padding

    height, width = tensor.shape[-2:]

    end_h = height - bottom if bottom else height
    end_w = width - right if right else width

    return tensor[
        ...,
        top:end_h,
        left:end_w,
    ]


def uncertainty_from_info(
    info: torch.Tensor,
    var_min: float,
    var_max: float,
):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)

    weights = info[:, :2].softmax(dim=1)

    log_b[:, 0] = torch.clamp(
        raw_b[:, 0],
        min=0,
        max=var_max,
    )

    log_b[:, 1] = torch.clamp(
        raw_b[:, 1],
        min=var_min,
        max=0,
    )

    return (
        log_b * weights
    ).sum(
        dim=1,
        keepdim=True,
    )


@torch.inference_mode()
def infer_pair(
    model,
    model_args,
    image1: np.ndarray,
    image2: np.ndarray,
    device: torch.device,
):
    tensor1 = image_to_tensor(image1, device)
    tensor2 = image_to_tensor(image2, device)

    # spring-M配置scale=-1：
    # 模型推理前会缩小到1/2。
    # 所以原图先填充为16的倍数，
    # 缩小后仍然是8的倍数。
    scale = int(model_args.scale)

    if scale < 0:
        multiple = 8 * (2 ** (-scale))
    else:
        multiple = 8

    tensor1, tensor2, padding = pad_pair(
        tensor1,
        tensor2,
        multiple,
    )

    padded_size = tensor1.shape[-2:]
    scale_factor = 2.0 ** scale

    if scale_factor != 1:
        scaled_height = int(
            round(padded_size[0] * scale_factor)
        )
        scaled_width = int(
            round(padded_size[1] * scale_factor)
        )

        tensor1_scaled = F.interpolate(
            tensor1,
            size=(scaled_height, scaled_width),
            mode="bilinear",
            align_corners=False,
        )

        tensor2_scaled = F.interpolate(
            tensor2,
            size=(scaled_height, scaled_width),
            mode="bilinear",
            align_corners=False,
        )
    else:
        tensor1_scaled = tensor1
        tensor2_scaled = tensor2

    output = model(
        tensor1_scaled,
        tensor2_scaled,
        iters=model_args.iters,
        test_mode=True,
    )

    flow = output["flow"][-1]
    info = output["info"][-1]

    if flow.shape[-2:] != padded_size:
        flow = F.interpolate(
            flow,
            size=padded_size,
            mode="bilinear",
            align_corners=False,
        )

        # 流量值也必须恢复到原始像素尺度。
        flow = flow / scale_factor

        info = F.interpolate(
            info,
            size=padded_size,
            mode="area",
        )

    flow = unpad(flow, padding)
    info = unpad(info, padding)

    uncertainty = uncertainty_from_info(
        info,
        var_min=float(model_args.var_min),
        var_max=float(model_args.var_max),
    )

    return (
        flow[0].permute(1, 2, 0)
        .detach().cpu().numpy(),
        uncertainty[0, 0]
        .detach().cpu().numpy(),
    )


def direction_entropy(
    flow: np.ndarray,
    magnitude: np.ndarray,
    bins: int = 18,
) -> float:
    if magnitude.size == 0:
        return float("nan")

    threshold = np.percentile(
        magnitude,
        75,
    )

    active = magnitude > max(
        threshold,
        1e-6,
    )

    if active.sum() < bins:
        return 0.0

    angle = np.arctan2(
        flow[..., 1],
        flow[..., 0],
    )

    histogram, _ = np.histogram(
        angle[active],
        bins=bins,
        range=(-np.pi, np.pi),
    )

    total = histogram.sum()

    if total == 0:
        return 0.0

    probability = histogram / total
    probability = probability[
        probability > 0
    ]

    entropy = -np.sum(
        probability * np.log(probability)
    )

    return float(
        entropy / np.log(bins)
    )


def calculate_pair_features(
    flow: np.ndarray,
    uncertainty: np.ndarray,
):
    u = flow[..., 0]
    v = flow[..., 1]

    magnitude = np.sqrt(
        u ** 2 + v ** 2
    )

    height, width = magnitude.shape
    diagonal = max(
        math.hypot(height, width),
        1.0,
    )

    magnitude_normalized = (
        magnitude / diagonal
    )

    return {
        "height": height,
        "width": width,

        "mag_mean": float(
            np.mean(magnitude)
        ),
        "mag_median": float(
            np.median(magnitude)
        ),
        "mag_std": float(
            np.std(magnitude)
        ),
        "mag_p90": float(
            np.percentile(magnitude, 90)
        ),
        "mag_p95": float(
            np.percentile(magnitude, 95)
        ),
        "mag_max": float(
            np.max(magnitude)
        ),

        "mag_norm_mean": float(
            np.mean(magnitude_normalized)
        ),
        "mag_norm_p90": float(
            np.percentile(
                magnitude_normalized,
                90,
            )
        ),

        "u_mean": float(np.mean(u)),
        "u_std": float(np.std(u)),
        "v_mean": float(np.mean(v)),
        "v_std": float(np.std(v)),

        "direction_entropy": (
            direction_entropy(
                flow,
                magnitude,
            )
        ),

        "uncertainty_mean": float(
            np.mean(uncertainty)
        ),
        "uncertainty_std": float(
            np.std(uncertainty)
        ),
        "uncertainty_p90": float(
            np.percentile(
                uncertainty,
                90,
            )
        ),
    }


def normalized_heatmap(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    finite = np.isfinite(values)

    if not finite.any():
        normalized = np.zeros_like(
            values,
            dtype=np.uint8,
        )
    else:
        valid = values[finite]

        low = np.percentile(valid, 1)
        high = np.percentile(valid, 99)

        if high <= low:
            normalized = np.zeros_like(
                values,
                dtype=np.uint8,
            )
        else:
            clipped = np.clip(
                values,
                low,
                high,
            )

            normalized = (
                255
                * (clipped - low)
                / (high - low)
            ).astype(np.uint8)

    return cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET,
    )


def save_visualizations(
    output_dir: Path,
    pair_name: str,
    flow: np.ndarray,
    uncertainty: np.ndarray,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    flow_image = flow_to_image(
        flow,
        convert_to_bgr=True,
    )

    magnitude = np.linalg.norm(
        flow,
        axis=-1,
    )

    cv2.imwrite(
        str(output_dir / f"{pair_name}_flow.jpg"),
        flow_image,
    )

    cv2.imwrite(
        str(output_dir / f"{pair_name}_magnitude.jpg"),
        normalized_heatmap(magnitude),
    )

    cv2.imwrite(
        str(output_dir / f"{pair_name}_uncertainty.jpg"),
        normalized_heatmap(uncertainty),
    )


def summarize_phase(
    pair_df: pd.DataFrame,
    patient_id: str,
    phase: str,
):
    row = {
        "patient_id": patient_id,
        "phase": phase,
        "n_pairs": len(pair_df),
    }

    for feature in PAIR_FEATURES:
        values = pd.to_numeric(
            pair_df[feature],
            errors="coerce",
        )

        row[f"{feature}_mean"] = (
            values.mean()
        )

        row[f"{feature}_std"] = (
            values.std(ddof=0)
        )

        row[f"{feature}_max"] = (
            values.max()
        )

    return row


def parse_bool_series(series):
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
        .fillna(False)
    )


def process_phase(
    patient_id: str,
    phase: str,
    phase_dir: Path,
    phase_output: Path,
    model,
    model_args,
    device,
    save_arrays: bool,
    vis_every: int,
    overwrite: bool,
):
    frames = list_frames(phase_dir)

    if len(frames) < 2:
        return None, {
            "patient_id": patient_id,
            "phase": phase,
            "error": (
                f"可用帧少于2：{len(frames)}"
            ),
        }

    pair_csv = (
        phase_output / "pair_features.csv"
    )

    if (
        pair_csv.exists()
        and not overwrite
    ):
        existing = pd.read_csv(pair_csv)
        print(
            f"[SKIP] {patient_id} {phase}: "
            f"已有结果 {len(existing)} 对"
        )
        return existing, None

    phase_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    visualization_dir = (
        phase_output / "visualizations"
    )

    array_dir = phase_output / "arrays"

    if save_arrays:
        array_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    rows = []

    for index in range(len(frames) - 1):
        frame1_path = frames[index]
        frame2_path = frames[index + 1]

        pair_name = (
            f"pair_{index:03d}_"
            f"{index + 1:03d}"
        )

        start = time.perf_counter()

        image1 = read_image(frame1_path)
        image2 = read_image(frame2_path)

        flow, uncertainty = infer_pair(
            model=model,
            model_args=model_args,
            image1=image1,
            image2=image2,
            device=device,
        )

        runtime = (
            time.perf_counter() - start
        )

        features = calculate_pair_features(
            flow,
            uncertainty,
        )

        features.update({
            "patient_id": patient_id,
            "phase": phase,
            "pair_index": index,
            "frame1": frame1_path.name,
            "frame2": frame2_path.name,
            "runtime_s": runtime,
        })

        rows.append(features)

        if save_arrays:
            np.save(
                array_dir / f"{pair_name}_flow.npy",
                flow.astype(np.float16),
            )

            np.save(
                array_dir
                / f"{pair_name}_uncertainty.npy",
                uncertainty.astype(np.float16),
            )

        should_visualize = (
            vis_every > 0
            and (
                index % vis_every == 0
                or index == len(frames) - 2
            )
        )

        if should_visualize:
            save_visualizations(
                visualization_dir,
                pair_name,
                flow,
                uncertainty,
            )

        print(
            f"[OK] {patient_id} {phase} "
            f"{index + 1}/{len(frames) - 1} "
            f"{runtime:.2f}s"
        )

    pair_df = pd.DataFrame(rows)

    pair_df.to_csv(
        pair_csv,
        index=False,
        encoding="utf-8-sig",
    )

    return pair_df, None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--cfg",
        default=str(
            REPO_ROOT
            / "config/eval/spring-M.json"
        ),
    )

    parser.add_argument(
        "--url",
        default=(
            "/root/autodl-tmp/aneurysm/"
            "models/"
            "Tartan-C-T-TSKH-spring540x960-M"
        ),
    )

    parser.add_argument(
        "--path",
        default=None,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--patient-id",
        default=None,
    )

    parser.add_argument(
        "--save-arrays",
        action="store_true",
    )

    parser.add_argument(
        "--vis-every",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--exclude-patch",
        action="store_true",
    )

    cli = parser.parse_args()

    if (
        cli.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA不可用，请检查GPU环境"
        )

    output_root = Path(cli.output)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(cli.cfg, "r") as file:
        config = json.load(file)

    config.update({
        "cfg": cli.cfg,
        "url": cli.url,
        "path": cli.path,
        "device": cli.device,
    })

    model_args = argparse.Namespace(
        **config
    )

    device = torch.device(cli.device)

    print("正在加载SEA-RAFT模型……")

    if cli.path:
        model = RAFT(model_args)
        load_ckpt(model, cli.path)
    else:
        model = RAFT.from_pretrained(
            cli.url,
            args=model_args,
        )

    model = model.to(device)
    model.eval()

    torch.backends.cudnn.benchmark = True

    print(
        f"模型加载完成："
        f"iters={model_args.iters}, "
        f"scale={model_args.scale}"
    )

    manifest = pd.read_csv(
        cli.manifest,
        dtype={"patient_id": str},
    )
    
    required_columns = {
        "patient_id",
        "pre_image_dir",
        "post_image_dir",
    }

    missing_columns = (
        required_columns - set(manifest.columns)
    )

    if missing_columns:
        raise KeyError(
            "Manifest缺少必要列："
            f"{sorted(missing_columns)}；"
            f"当前列为：{list(manifest.columns)}"
        )

    if cli.patient_id:
        manifest = manifest[
            manifest["patient_id"]
            == str(cli.patient_id)
        ].copy()

    if (
        cli.exclude_patch
        and "patch_case" in manifest.columns
    ):
        patch_case = parse_bool_series(
            manifest["patch_case"]
        )

        manifest = manifest[
            ~patch_case
        ].copy()

    if manifest.empty:
        raise RuntimeError(
            "筛选后Manifest为空"
        )

    all_pair_rows = []
    sequence_rows = []
    failures = []

    required_columns = {
        "patient_id",
        "pre_image_dir",
        "post_image_dir",
    }

    missing_columns = (
        required_columns - set(manifest.columns)
    )

    if missing_columns:
        raise KeyError(
            "Manifest缺少必要列："
            f"{sorted(missing_columns)}；"
            f"当前列为：{list(manifest.columns)}"
        )

    for progress_index, (_, row) in enumerate(
        manifest.iterrows(),
        start=1,
    ):
        patient_id = str(
            row["patient_id"]
        ).strip()

        print(
            "\n" + "=" * 70
        )
        print(
            f"患者 {patient_id} "
            f"({progress_index}/{len(manifest)})"
        )

        for phase in ("pre", "post"):
            run_column = f"run_{phase}"
            directory_column = (
                f"{phase}_image_dir"
            )

            # 根据Manifest判断该阶段能否运行。
            if run_column in manifest.columns:
                run_value = row[run_column]

                if pd.isna(run_value):
                    run_phase = False
                elif isinstance(
                    run_value,
                    (bool, np.bool_),
                ):
                    run_phase = bool(run_value)
                else:
                    run_phase = (
                        str(run_value)
                        .strip()
                        .lower()
                        in {
                            "true",
                            "1",
                            "yes",
                        }
                    )

                if not run_phase:
                    print(
                        f"[SKIP] {patient_id} "
                        f"{phase}: "
                        "该阶段可用帧少于2"
                    )
                    continue

            directory_value = row[
                directory_column
            ]

            if (
                pd.isna(directory_value)
                or not str(
                    directory_value
                ).strip()
            ):
                failure = {
                    "patient_id": patient_id,
                    "phase": phase,
                    "error": (
                        f"{directory_column}为空"
                    ),
                }

                failures.append(failure)

                print(
                    f"[ERROR] {patient_id} "
                    f"{phase}: "
                    f"{directory_column}为空"
                )
                continue

            phase_dir = Path(
                str(directory_value)
            )

            if not phase_dir.is_dir():
                failure = {
                    "patient_id": patient_id,
                    "phase": phase,
                    "error": (
                        "影像目录不存在："
                        f"{phase_dir}"
                    ),
                }

                failures.append(failure)

                print(
                    f"[ERROR] {patient_id} "
                    f"{phase}: 目录不存在 "
                    f"{phase_dir}"
                )
                continue

            phase_output = (
                output_root
                / patient_id
                / phase
            )

            try:
                pair_df, failure = process_phase(
                    patient_id=patient_id,
                    phase=phase,
                    phase_dir=phase_dir,
                    phase_output=phase_output,
                    model=model,
                    model_args=model_args,
                    device=device,
                    save_arrays=cli.save_arrays,
                    vis_every=cli.vis_every,
                    overwrite=cli.overwrite,
                )

                if failure:
                    failures.append(failure)

                if (
                    pair_df is not None
                    and not pair_df.empty
                ):
                    all_pair_rows.append(
                        pair_df
                    )

                    sequence_rows.append(
                        summarize_phase(
                            pair_df,
                            patient_id,
                            phase,
                        )
                    )

            except Exception as exc:
                print(
                    f"[ERROR] {patient_id} "
                    f"{phase}: {exc}"
                )

                failures.append({
                    "patient_id": patient_id,
                    "phase": phase,
                    "error": repr(exc),
                    "traceback": (
                        traceback.format_exc()
                    ),
                })

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    stem = Path(cli.manifest).stem

    if all_pair_rows:
        all_pairs = pd.concat(
            all_pair_rows,
            ignore_index=True,
        )

        all_pairs.to_csv(
            output_root
            / f"{stem}_pair_features.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if sequence_rows:
        sequence_df = pd.DataFrame(
            sequence_rows
        )

        sequence_df.to_csv(
            output_root
            / f"{stem}_sequence_features.csv",
            index=False,
            encoding="utf-8-sig",
        )

    failure_df = pd.DataFrame(
        failures
    )

    failure_df.to_csv(
        output_root
        / f"{stem}_failures.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== 运行完成 ==========")
    print("Manifest患者数:", len(manifest))
    print("成功阶段数:", len(sequence_rows))
    print("失败阶段数:", len(failures))
    print("输出目录:", output_root)


if __name__ == "__main__":
    main()