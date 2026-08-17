# api_fullseq_cave_v3_full_auto_models

基于冻结 `api_fullseq_v3` Manifest 的 CAVE 二维 DSA 全自动生产流程：

```text
Full Train CAVE 特征提取
→ Train 硬审计与 phase/series/patient 母表
→ 冻结 Train release
→ Full Valid 特征提取
→ Valid 硬审计与母表
→ 复用 v3 标签与 record-to-series mapping
→ Dummy / Logistic / MLP 训练
→ Train OOF + 独立 Valid 评估
```

## 1. 特征提取

固定使用：

- CAVE sequence AV ConvGRU；
- commit `c3b0c215db4029368c9499a12417178014d58d6f`；
- checkpoint `sequence_av_sigmoid_image512_ConvGRU_logical-star-1097.pt`；
- 完整 Pre/Post 灰度 DSA；
- strict frame-gap block；
- uniform/contrast-core 双时间视图；
- f4/f5、动脉/静脉/血管概率图；
- 5120 维/phase 深层特征；
- 206 维/phase CAVE-mask 形态、TDC、kinetic、filling 标量；
- phase/series/patient 三级母表及患者 label-blind median。

特征提取阶段禁止读取 RROC、随访 RROC 和不良转归标签。

## 2. 预测输入

每条任务样本读取完整母特征：

- Deep：Pre/Post `2×5120=10240`；
- Scalar：Pre/Post、允许的 Delta、10组 embedding distance；
- Missing：`missing_pre/missing_post`。

不把一万多列复制进 CSV，而是使用压缩 NPZ 与小型 metadata CSV。

## 3. 折内降维与模型

每个 outer development fold 内：

- Deep：NaN→0（另有 missing flag）→ sparse random projection 512 → StandardScaler → PCA64；
- Scalar：Train-fold 缺失/方差过滤 → median imputation → missing indicators → RobustScaler → PCA32；
- Missing：2维直接保留。

模型：

- Dummy；
- Logistic_deep；
- Logistic_scalar；
- Logistic_fusion；
- MLP_fusion（128→32）。

患者分组 5-fold OOF；C、早停和 Youden 阈值全部只使用 Train。官方 Valid 仅预测和评估。

## 4. 部署更新包

```bash
cd /root/autodl-tmp
unzip -q cave_featurebank_v3_full_auto_models.zip
cd cave_featurebank_v3_full_auto_models
bash deploy.sh
```

部署会更新：

```text
/root/autodl-tmp/aneurysm/code/api_fullseq_cave_v3
```

不会删除已完成的 5-series smoke 缓存，也不会覆盖冻结配置。

## 5. 一条命令跑完

```bash
cd /root/autodl-tmp/aneurysm/code/api_fullseq_cave_v3
mkdir -p /root/autodl-tmp/aneurysm/logs

nohup bash run_pipeline.sh full-auto \
  > /root/autodl-tmp/aneurysm/logs/cave_v3_full_auto_models.log \
  2>&1 < /dev/null &

echo "PID=$!"
```

查看：

```bash
tail -f /root/autodl-tmp/aneurysm/logs/cave_v3_full_auto_models.log
bash run_pipeline.sh status
```

## 6. 预测依赖

默认预测环境：

```text
/root/autodl-tmp/envs/aneurysm-ml/bin/python
```

要求已有 `numpy pandas scikit-learn joblib torch openpyxl pyarrow`。运行：

```bash
bash run_pipeline.sh install
```

会同时检查 CAVE 环境和预测环境，但不会重建已存在的 `aneurysm-ml`。

## 7. 输出

```text
outputs/api_fullseq_cave_v3_featurebank/
outputs/api_fullseq_cave_v3_tables/train/
outputs/api_fullseq_cave_v3_tables/valid/
outputs/api_fullseq_cave_v3_tasks/
outputs/api_fullseq_cave_v3_models/
reports/api_fullseq_cave_v3/full_auto_summary.json
reports/api_fullseq_cave_v3/selected_model_metrics.csv
reports/api_fullseq_cave_v3/.FULL_AUTO_WITH_MODELS_SUCCESS
```

存在旧 SEA-RAFT 指标时，还会输出：

```text
reports/api_fullseq_cave_v3/cave_vs_searaft_valid_metrics.csv
```

## 8. 正式数量

Train：1147 series / 1055 patients / 940 Pre / 1147 Post / 2087 phases。

Valid：287 series / 264 patients / 248 Pre / 287 Post / 535 phases。
