# v4 → v5 修正

1. 删除“只对已可靠匹配 Mask 的 matched 子集提取”的主逻辑。
2. 先将冻结 all-series manifest 展开为 2622 个必须完成的 source phase。
3. 每个 source phase 必须得到恰好一个已有 GT Mask；缺失/冲突时停止，不静默删样本。
4. 生成完整的 1147 行 Train 和 287 行 Valid Local-CAVE manifest，保持原顺序与 Pre/Post flag。
5. 第一阶段只做 Mask 映射、ROI、manifest 和 QA；不训练分割模型。
6. 第二阶段对完整 2087 Train phase 与 535 Valid phase 全量提取 Local-CAVE。
7. 增加 `.STAGE1_FULLMASK_SUCCESS.json` 和 `.FULL_FEATUREBANK_SUCCESS.json` 两个闭合锁。
8. 增加分片提取断点续跑：已在 consolidated featurebank 完成的 series 不再重复计算。
9. 默认不保存成千上万张局部图片；在内存逐帧裁切后直接送入 CAVE。
10. 将输入、输出、主键和 manifest 字段写入独立 I/O 规范。
