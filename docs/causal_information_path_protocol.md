# 信息通路因果消融协议

目标是定位损失发生在哪条信息通路；固定 seed=42、十个已有 split、500 epoch 上限、patience=50，按验证损失选 checkpoint，不做超参数搜索。主环境维持原 NumPy/PyTorch 版本。

## 优先组：只移除额外 A^kX

`run_no_akx.py` 在原 PEGFAN 中只将输入列表的 `AX, A²X, A³X` 三项置零。`X`、原 cached Haar 对 `AX` 的所有频带、FSGNN 参数量/输入槽位、通道 attention、bias、初始化、优化器及 dropout 均保留。零输入槽位仍可提供共享的 learned bias，但不再提供逐节点传播特征。这是信息移除对照，不是减少分类器宽度。

须注意原 framelet 频带仍包含 AX 的信息，因此不能说这组“移除了所有一阶传播”。它精确移除了用户指定的额外 propagation channels。重新训练的 PEGFAN control 保存 checkpoint，便于后续同权重恒等性诊断。

优先组覆盖 Film、Chameleon、Squirrel、Texas、Wisconsin；错误 Cornell 先排除。修复的 Cornell 使用单独目录和新数据哈希重新训练，绝不与旧 Cornell 汇总混合。

## Pool → immediate lift

在 PEGFAN classifier 的每一个原输入矩阵 H 上插入空间 pooling/lifting。测试两种 hierarchy：原 hierarchy + 原 equal-child transfers；已冻结的 Sheaf dyadic hierarchy + mass transfers。cut 均为该 hierarchy 第一个 ≤512 的现有层。

- global-only：逐层 pool 后直接 lift，相当于复合等距矩阵 P 下的 `P P^T H`。重新训练原 classifier，记为 `pool_global` / `pool_sheaf_global`。
- exact reconstruction：保留每层 `H-P(P^T H)`，逆序 lift 并加回。对同一个训练好的 PEGFAN checkpoint 比较输入误差、logit 误差、全节点预测及测试准确率；不另做一个随机训练后把差异误认为重构错误。

重构不会调用 Transformer、新 Haar 或 learned fusion。float32 下检查容差，不承诺实数恒等式在浮点下逐 bit 相同。原树和新树每 split 都单独检查。

## Feature dimensionality bottleneck

原 FSGNN **本来就对每个 channel 投影到 hidden=64**。完整模型更早共享一个 `F→64` 编码器，因此本轮测试的是额外的、位于原 FSGNN 之前的共享 feature bottleneck，而不是重复修改已有 hidden width。

保留图、hierarchy、Haar、FSGNN，向每个原输入 channel 插入同一个 `F→64→F`：

1. `svd64`：原 row-normalized X 的 exact rank-64 uncentered SVD，固定正交 projector；无 labels、无参数学习，记录保留的特征能量。
2. `learned64`：用同一 SVD 初始化的共享线性 encoder/decoder，额外 lr=0.001；没有 residual bypass、没有额外激活函数，确保 rank bottleneck 不被绕过。FSGNN 沿用原学习率。

由于线性 feature projection 与左侧图算子可交换，这等价于对输入特征投影后使用原传播/原 framelet。learner 组仍是一个带优化因素的对照，不能把它的下降完全解释为信息论必然损失。

## 逐项构造完整链条

使用完整模型相同的 64 维 encoder、local sheaf update、Transformer、Haar fusion、detail projection 和 classifier 初始化。所有阶段先固定同 split 已有的 Sheaf dyadic hierarchy，以免模块变化同时改变 partition。被冻结的 hierarchy 来源于前轮同 split 监督训练的 scorer，附原重放审计；这是一项两阶段诊断，不是新训练算法的最终 benchmark。

阶段顺序：

1. `chain_local`：仅 local sheaf signal。
2. `chain_pool_global`：pool → lift，只保留 global。
3. `chain_transformer_global`：pool → Transformer → lift。
4. Haar analysis 后**直接求和**应为恒等；在阶段 3 的同一个 checkpoint 上检查 signal/logits/预测，不重复训练等价函数。
5. `chain_haar_global`：pool → Transformer → post-Haar learned band fusion → lift，只保留 global。
6. `chain_detail_only`：仅保留局部 details 经原 detail projection 后 lift 的结果。
7. `chain_global_detail`：同阶段 5 加回局部 detail。
8. `chain_full_skip`：再加回完整模型额外的原节点 hidden skip。

为了得到真正的 global-only/detail-only，前七项共用的 classifier 第二个输入槽位均置零；阶段 8 才恢复 hidden skip。否则把 `concat(global, h)` 称作 global-only 会混入局部旁路。各组实例化相同模块和参数量；未启用模块不参与预测，实际有效容量自然随干预改变。

## 完整动态模型重新接回 A^kX

`full_no_akx` 与 `full_with_akx` 均使用完整动态 Sheaf hierarchy，每个 forward 重新评分/粗化，保持 Transformer → post-Haar → lift → detail fusion 和 hidden skip。

新增三个原图 propagation 输入，各经 `F→64`、L2 normalize、learned channel weighting 后拼接到最终分类器。第一层从 `2h→h` 扩为 `5h→h`，原列和 bias 原样复制，新增列零初始化；初始输出严格等于原完整模型。新 projection 学习率为 PEGFAN 原值 0.01，channel weights 为 0.02；原完整模型参数仍为 0.001。

两组具有相同初始化、模块和参数量，区别仅在新增的 local propagation features 是否置零。关闭组的无效列保持零，因此实现了匹配新增容量的 on/off 对照。它是本轮新训练的完整模型 control，不能把它的数值无条件等同于上一轮不同 RNG 消耗顺序的训练结果。

扩展因果链优先覆盖 Film、Chameleon、Squirrel；小图在 512 预算下不发生粗化，优先组仍覆盖有效小图。所有新增运行保存 checkpoint、逐 epoch 日志、源码/数据/split 哈希。

## Cornell 数据修复与旧统计

旧 Cornell 从两轮正式 CSV/Markdown/图表剔除，原始 JSONL 与原报告仅作审计归档。修复从 Geom-GCN 固定提交获取图、特征、labels、十个 splits，校验节点 ID 连续性、183 个节点、1703 个 features、5 类及 masks 互斥覆盖。

已发现旧图结构同样与上游不匹配，故必须重建 h=4/8 的原 Ward Haar。使用仓库原 `scikit-network==0.28.3`，在隔离路径搭配兼容的 NumPy，只影响缓存构建，不升级主要训练环境。修复审计完成前禁止把 Cornell 加回正式统计。
