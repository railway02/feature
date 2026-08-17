#!/usr/bin/env bash
set -euo pipefail

ENV_PATH=${CAVE_ENV:-/root/autodl-tmp/envs/cave-dsa}
REPO=${CAVE_REPO:-/root/autodl-tmp/CAVE_DSA}
CAVE_COMMIT=c3b0c215db4029368c9499a12417178014d58d6f
CKPT_REL=checkpoints/sequence_av_sigmoid_image512_ConvGRU_logical-star-1097.pt
EXPECTED_SHA=c90b7e066e32039cf61352993a9c57784caac6aa1fdb042dc4801df6dc729651
EXPECTED_SIZE=332731061
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  source /root/miniconda3/etc/profile.d/conda.sh
elif [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  echo "[FAIL] conda.sh not found" >&2
  exit 2
fi

[[ -d "$ENV_PATH" ]] || conda create -p "$ENV_PATH" python=3.10 -y
conda activate "$ENV_PATH"
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 scikit-image==0.23.2 \
  pillow==10.4.0 pydicom==2.4.4 nibabel==5.2.1 natsort==8.4.0 \
  tqdm==4.66.4 matplotlib==3.9.1 opencv-contrib-python-headless==4.10.0.84 \
  pyarrow==17.0.0 openpyxl==3.1.5

command -v git-lfs >/dev/null 2>&1 || {
  apt-get update
  apt-get install -y git-lfs
}
git lfs install

if [[ ! -d "$REPO/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/RuishengSu/CAVE_DSA.git "$REPO"
fi

if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "[FAIL] Existing CAVE repository is dirty: $REPO" >&2
  git -C "$REPO" status --short >&2
  exit 2
fi

git -C "$REPO" fetch --all --tags --prune
git -C "$REPO" checkout --detach "$CAVE_COMMIT"
git -C "$REPO" lfs fetch --include="$CKPT_REL" origin "$CAVE_COMMIT"
git -C "$REPO" lfs checkout "$CKPT_REL"

ACTUAL_COMMIT=$(git -C "$REPO" rev-parse HEAD)
ACTUAL_SHA=$(sha256sum "$REPO/$CKPT_REL" | awk '{print $1}')
ACTUAL_SIZE=$(stat -c%s "$REPO/$CKPT_REL")
[[ "$ACTUAL_COMMIT" == "$CAVE_COMMIT" ]]
[[ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]]
[[ "$ACTUAL_SIZE" == "$EXPECTED_SIZE" ]]

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

python "$SCRIPT_DIR/test_synthetic.py"
python "$SCRIPT_DIR/smoke_test.py" \
  --cave-repo "$REPO" \
  --checkpoint "$REPO/$CKPT_REL"

python -m pip freeze > "$SCRIPT_DIR/environment_pip_freeze.txt"
echo "$ACTUAL_COMMIT" > "$SCRIPT_DIR/CAVE_COMMIT.txt"
echo "$ACTUAL_SHA  $REPO/$CKPT_REL" > "$SCRIPT_DIR/CAVE_CHECKPOINT_SHA256.txt"

echo "[PASS] CAVE environment, pinned repository, checkpoint, synthetic tests and GPU smoke"
