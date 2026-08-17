# api_fullseq_keyfusion_v3

患者级不良转归关键表征融合，主体输入为：

- CAVE 完整 Pre/Post `2×10×512` embedding；
- SEA-RAFT 已缓存 dense `pair_maps.npz` 的关键 flow/output maps；
- Train.xlsx/valid.xlsx 中经过泄漏筛查的患者、解剖和治疗信息。

不使用无监督 PCA。CAVE 全部 embedding 通道由共享监督投影头学习；SEA dense maps 保留早/中/晚、时间波动、峰值和 `16×16` 空间结构，再由监督 CNN 编码。

明确不进入模型的 Excel 字段：不良转归标签、即刻 RROC、随访 RROC、姓名、病案号、DSA/随访日期和随访间隔。

构建数据：

```bash
/root/autodl-tmp/envs/aneurysm-ml/bin/python \
  code/api_fullseq_keyfusion_v3/build_adverse_keyfusion_dataset.py \
  --output-dir outputs/api_fullseq_keyfusion_v3_dataset \
  --workers 8
```

训练：

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
/root/autodl-tmp/envs/aneurysm-ml/bin/python \
  code/api_fullseq_keyfusion_v3/train_adverse_keyfusion.py \
  --dataset-dir outputs/api_fullseq_keyfusion_v3_dataset \
  --output-dir outputs/api_fullseq_keyfusion_v3_models \
  --device cuda:0
```
