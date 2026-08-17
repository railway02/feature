# api_fullseq_fusion_v3

将冻结的 SEA-RAFT `api_fullseq_v3` 特征与 CAVE `api_fullseq_cave_v3` 特征按同一任务样本做早期融合预测。

训练前会逐行强制核对：

- `patient_id`，record 任务还核对 `record_uid/series_uid`；
- 行数、行顺序与二分类标签；
- Train/Valid 病人交集为 0。

模型输入：

- CAVE deep：10240 → random projection 512 → PCA64；
- CAVE scalar：折内过滤、填补、RobustScaler → PCA32；
- SEA-RAFT：212 个冻结特征，折内过滤、填补、RobustScaler → PCA64；
- 缺失 phase 标记：2 维；
- 每个最终 variant 再做 development-fitted StandardScaler。

对比模型使用完全相同的患者分组 outer folds：

- `Logistic_searaft`
- `Logistic_cave_deep`
- `Logistic_cave_scalar`
- `Logistic_cave_fusion`
- `Logistic_multimodal_fusion`
- `MLP_multimodal_fusion`

Logistic 的 C 在每个 outer-development 内用 grouped 3-fold OOF AUPRC 选择。所有预处理都在对应 inner/outer development 内拟合；官方 Valid 只预测和评估。

先做输入审计：

```bash
/root/autodl-tmp/envs/aneurysm-ml/bin/python \
  code/api_fullseq_fusion_v3/train_fusion_prediction_models.py \
  --searaft-task-root outputs/api_fullseq_v3_tasks \
  --cave-task-root outputs/api_fullseq_cave_v3_tasks \
  --output-dir reports/api_fullseq_fusion_v3_alignment \
  --audit-only
```

正式训练：

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
/root/autodl-tmp/envs/aneurysm-ml/bin/python \
  code/api_fullseq_fusion_v3/train_fusion_prediction_models.py \
  --searaft-task-root outputs/api_fullseq_v3_tasks \
  --cave-task-root outputs/api_fullseq_cave_v3_tasks \
  --output-dir outputs/api_fullseq_fusion_v3_models \
  --device cuda:0
```

核心结果：

```text
outputs/api_fullseq_fusion_v3_models/all_task_metrics.csv
outputs/api_fullseq_fusion_v3_models/fusion_gains.csv
outputs/api_fullseq_fusion_v3_models/<task>/train_oof_predictions.csv
outputs/api_fullseq_fusion_v3_models/<task>/valid_predictions.csv
outputs/api_fullseq_fusion_v3_models/<task>/logistic_convergence_audit.csv
```
