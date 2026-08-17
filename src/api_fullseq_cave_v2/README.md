# api_fullseq_cave_v2_featurebank

这是对 `cave_featurebank_v1` 的生产级升级，仅负责二维 DSA 特征提取；Manifest、Excel 审计、任务映射和训练全部复用 `api_fullseq_v3`。

## 冻结方法

- CAVE sequence AV ConvGRU, 512×512, kernel=3, layers=2。
- 长序列固定双视图：`uniform_full20` 与官方式 `contrast_core20`。
- 小 gap（最多缺 2 帧）保持有序输入；更大 gap 分 block。
- 正式血管权重：`max(artery, vein)`；概率并集仅作辅助缓存。
- 主 embedding：10×512=5120 维；Top-K 主向量保存绝对响应强度，signed mean 另存。
- 每个 view/block 保存 f4/f5 last map、原生与 16 点时序轨迹、概率图和辅助 embedding。
- CAVE 概率图逆变换回原始分辨率，再与冻结 v3 的 FOV、baseline、polarity、activity、kinetic 和 filling 原语结合。
- 可解释标量固定 206/phase：形态 51 + 空间关系 10 + 区域 TDC 64 + 动静脉时序关系 10 + kinetic 57 + filling 14。
- 所有写入为临时目录完成后再替换；任何 phase 失败时进程返回非零。

## 放置

```bash
mkdir -p /root/autodl-tmp/aneurysm/code/api_fullseq_cave_v2
cp *.py /root/autodl-tmp/aneurysm/code/api_fullseq_cave_v2/
cp frozen_config.example.json /root/autodl-tmp/aneurysm/configs/api_fullseq_cave_v2_frozen.json
```

## Smoke

```bash
bash install_and_smoke.sh
/root/autodl-tmp/envs/cave-dsa/bin/python smoke_test.py \
  --cave-repo /root/autodl-tmp/CAVE_DSA \
  --checkpoint /root/autodl-tmp/CAVE_DSA/checkpoints/sequence_av_sigmoid_image512_ConvGRU_logical-star-1097.pt
```

再用正式提取命令加 `--max-series 5` 检查真实落盘；只修程序错误，不改 frozen config。

## 全量命令模板

```bash
PY=/root/autodl-tmp/envs/cave-dsa/bin/python
CODE=/root/autodl-tmp/aneurysm/code/api_fullseq_cave_v2
ROOT=/root/autodl-tmp/aneurysm
COMMON="\
 --cave-repo /root/autodl-tmp/CAVE_DSA \
 --checkpoint /root/autodl-tmp/CAVE_DSA/checkpoints/sequence_av_sigmoid_image512_ConvGRU_logical-star-1097.pt \
 --v3-extractor $ROOT/code/api_fullseq_v3/extract_pairdata.py \
 --v3-base-config <V3_BASE_CONFIG_PATH> \
 --v3-override-config $ROOT/configs/api_fullseq_v3_improved_overrides.json \
 --frozen-config $ROOT/configs/api_fullseq_cave_v2_frozen.json \
 --output-root $ROOT/outputs/api_fullseq_cave_v2_featurebank"

nohup $PY $CODE/extract_cave_featurebank.py \
  --manifest $ROOT/source_manifests/train_all_series_manifest.csv $COMMON \
  > $ROOT/logs/cave_v2_train.log 2>&1 < /dev/null &

nohup $PY $CODE/extract_cave_featurebank.py \
  --manifest $ROOT/source_manifests/valid_all_series_manifest.csv $COMMON \
  > $ROOT/logs/cave_v2_valid.log 2>&1 < /dev/null &
```

`<V3_BASE_CONFIG_PATH>` 使用服务器上 v3 已经冻结并实际运行的 base config，不新建一套参数。

## 硬验收

Train：1147 series、940 Pre、1147 Post、2087 phase。  
Valid：287 series、248 Pre、287 Post、535 phase。

```bash
$PY $CODE/audit_featurebank.py \
 --manifest $ROOT/source_manifests/train_all_series_manifest.csv \
 --feature-root $ROOT/outputs/api_fullseq_cave_v2_featurebank \
 --output $ROOT/reports/api_fullseq_cave_v2/train_audit.json \
 --expected-series 1147 --expected-pre 940 --expected-post 1147 \
 --expected-config-hash <SHA256_OF_CANONICAL_FROZEN_JSON>
```

构建表：

```bash
$PY $CODE/build_feature_tables.py \
 --manifest $ROOT/source_manifests/train_all_series_manifest.csv \
 --feature-root $ROOT/outputs/api_fullseq_cave_v2_featurebank \
 --output-dir $ROOT/outputs/api_fullseq_cave_v2_tables/train
```
