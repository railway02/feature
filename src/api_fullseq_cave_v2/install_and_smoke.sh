#!/usr/bin/env bash
set -euo pipefail
ENV=/root/autodl-tmp/envs/cave-dsa
REPO=/root/autodl-tmp/CAVE_DSA
CKPT_REL=checkpoints/sequence_av_sigmoid_image512_ConvGRU_logical-star-1097.pt
EXPECTED_SHA=c90b7e066e32039cf61352993a9c57784caac6aa1fdb042dc4801df6dc729651
EXPECTED_SIZE=332731061
source /root/miniconda3/etc/profile.d/conda.sh
[[ -d "$ENV" ]] || conda create -p "$ENV" python=3.10 -y
conda activate "$ENV"
python -m pip install -U pip setuptools wheel
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 scikit-image==0.23.2 pillow==10.4.0 pydicom==2.4.4 nibabel==5.2.1 natsort==8.4.0 tqdm==4.66.4 matplotlib==3.9.1 opencv-contrib-python-headless==4.10.0.84 pyarrow==17.0.0
command -v git-lfs >/dev/null 2>&1 || { apt-get update && apt-get install -y git-lfs; }
git lfs install
if [[ ! -d "$REPO/.git" ]]; then GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/RuishengSu/CAVE_DSA.git "$REPO"; fi
cd "$REPO"
git lfs fetch --include="$CKPT_REL"
git lfs checkout "$CKPT_REL"
[[ "$(sha256sum "$CKPT_REL" | awk '{print $1}')" == "$EXPECTED_SHA" ]]
[[ "$(stat -c%s "$CKPT_REL")" == "$EXPECTED_SIZE" ]]
git rev-parse HEAD
python - <<'PY'
import torch
assert torch.cuda.is_available()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
