# 单因素消融协议与解释边界

本轮不搜索超参数，也不把 cluster distortion gate 混入新的 hierarchy。三组实验共享仓库原数据、十个 split、seed=42、验证损失选 checkpoint、500 epoch 上限、50 epoch patience。原 PEGFAN 数值复用上一轮；新实验保存 checkpoint、逐 epoch 日志、数据划分与源码哈希。

## 1. 只替换 hierarchy

分类器仍是原 FSGNN；保留 `X, AX, A²X, A³X`，framelet 输入仍是 `AX`；学习率、正则、dropout、层归一化、每个通道的输入维度全部不变。

为了连 FSGNN 的通道数和参数量也保持不变，用原 hierarchy 每一层的 region count 查询新 filtration，取第一个 count 小于等于目标的 cut。同分边始终整体处理；若多个目标落到同一个 cut，保留重复 cut（相应 detail channel 为零），不拆分 ties。这里测的是新的分区结构在原 PEGFAN 多尺度接口上的作用，不是上一轮 dyadic level schedule 的完整复现。

硬分区没有可供 restriction maps 学习的梯度。因此先以同一个 split、相同 seed 和上一轮配置重放完整 Sheaf 模型，在上一轮**由验证集选定**的 epoch 冻结评分；随后从相同 seed 重新初始化 FSGNN，只传入分区，绝不把 scorer 的隐藏特征、local update 或 logits 输入分类器。scorer 增加了预训练成本，也使用了该 split 的训练监督，不能称为无监督分区实验。上一轮没有保存模型权重，本轮保存重放权重；各次重放与原验证损失的偏差见 `scorers/*_seed42.json`，不能声称精确恢复了上一轮 checkpoint。

仍使用原 PEGFAN **equal-child** Haar 算子。设一个 parent 有 k 个 child，其原成对 Haar 矩阵为 `A = [(e_i-e_j)^T] / sqrt(k)`，则 `A^T A = I - 11^T/k`。隐式计算只是避免大分支产生 O(k²) 个成对系数；并未更换归一化规则。六个原缓存的所有 band 都通过随机探针校验，实际 `AX` 频带也单独核对。不能把此计算优化与下一组的 mass normalization 混为一谈。

## 2. 原 hierarchy + 新 projector normalization

原 hierarchy 直接从已有 pairwise Haar 缓存的 support 恢复，未重新调用可能产生不同结果的 Ward 版本。唯一修改为 `p_i = sqrt(child 原节点质量 / parent 原节点质量)`，替换原 `p_i = 1/sqrt(child 数)`。保留所有层与通道、`A^kX`、FSGNN 和训练配置。

因此这组检查的是 **归一化改变**，而非 `Psi.T @ Psi` 与其代数等价的隐式求值方式。

## 3. 原 hierarchy + Transformer

在原 hierarchy 已有层中选第一个 region count ≤512 的 cut；原 equal-child pooling 给出 `Z`。增加

`Z' = Z + W_out Transformer(W_in Z)`。

Transformer 宽度 64、4 heads、2 layers、FFN 宽度 256、dropout=0.5，与完整模型中的 attention 设置一致。`W_out` 零初始化，初始函数严格等于原 PEGFAN。新增参数 lr=0.001，其余参数仍用 PEGFAN 原学习率。保留下层全部 detail，只对 coarse source 作残差更新，再使用原 framelet 通道和原 FSGNN；`A^kX` 通道不动。

这是**残差插入**的单因素实验，不能把结论直接推广到完整新模型的非残差替换、不同输入投影及 multiscale fusion。adapter 是为了让宽度 64 的 Transformer 接入原始特征维度，属于本组新增模块。原 tree 的 token 数未必接近 512，这是预算上界，不是固定目标。

## 4. 完整分区诊断

Film、Chameleon、Squirrel 每个 split 保存三套树：原 hierarchy、按原层数/目标 count 查询的新 filtration、上一轮模型使用的 dyadic hierarchy。每个层都保存：

- 最大 region 占总节点比例。
- singleton / region 数与 singleton / 总节点数，两个口径均提供。
- region size Gini、节点质量分布熵 `-sum (|C|/N) log(|C|/N)`，以及除以 `log K` 的归一化熵。
- 原 row-normalized features 的 `sum_C sum_v ||x_v-mean_C x||²/N` 及其占全图总中心化方差的比例；同时报告 scorer hidden 的相同统计。
- 内部无向边占比、内部边 Sheaf energy 的 edge-weighted mean（每条边等权）、有内部边 region 的 macro mean。没有内部边时使用 null，而不是把“没有观测”当作零能量。

这里的 feature variance 对应均值压缩的失真；它不是原 equal-child、逐层非均匀 scaling vector 的直接投影误差。分区有 giant region 不自动推出分类必然失败，尤其在完整保留局部 detail 与 `A^kX` 时。

全部训练完成之后，才执行 `label_diagnostics_posthoc`：分别报告所有节点 labels 与仅 test labels 下的 node-weighted cluster label entropy/purity，并附有效节点/region 数。labels 不进入分区或参数选择。只比较 test-label purity 时应注意 singleton 很多的分区有大量无 test labels 的 region。

## 5. Film 配对比较

上一轮完整新模型在十个 split 均胜于 PEGFAN，均值 +1.7961 pp；逐 split CSV 保存在 `film_paired_previous.csv`。本轮三组新增实验的逐 split 差值另存 `paired_splits.csv`。这是同 seed 的 paired split 证据；十个划分并非十个独立采样数据集，不能视为多 seed 稳定性证明。

## 对 cluster-level gate 的后续要求

当前 `q=-s`；给 s 乘严格正标量不改变排序和 exact ties，因此数学上的 CC filtration 不变。单纯调整这种 gamma 不能修复 chaining。若另加非均匀 affinity，则需要单独分析排序变化。

接受/拒绝必须作用于整组候选 parent，并以 permutation-invariant statistics 决策。固定原 CC merge tree、拒绝时保留 children 可以维持 laminar/nested partitions。但拒绝 merge 后不再自动保证能达到 token budget；必须报告“预算不可达”或明确定义放宽策略，不能悄悄拆 ties 或以 node ID 排序。

平均内部边 Sheaf energy **单独使用仍不足以保证区域可压缩**。例如 identity restrictions、链图节点 `h_i=i`：每条边 energy=1，整个 region 的 mean energy 永远为 1，但均值失真为 `(n²-1)/12`。因此建议下一轮直接检查 aggregate reconstruction distortion，Sheaf energy 作为补充约束，而不是认为其均值天然解决 chaining。使用 invariant gate 的等变性不等于预算、复杂度、精度都自动保留。

## 数据适用范围

当前仓库 Texas 与 Cornell 的 `out1_node_feature_label.txt` 逐字节相同（包括 labels），在原始 Git HEAD 中也相同。上游哈希比对确认本地 Texas 匹配 [Geom-GCN Texas](https://github.com/bingzhewei/geom-gcn/blob/master/new_data/texas/out1_node_feature_label.txt)，本地 Cornell 则不匹配 [Geom-GCN Cornell](https://github.com/bingzhewei/geom-gcn/blob/master/new_data/cornell/out1_node_feature_label.txt)。这不是本轮代码产生的问题；Cornell 只能作为当前仓库数据上的内部控制，不能直接声称标准 Cornell 结果。需另行更正 Cornell 文件并核对对应 splits，再独立复跑。哈希见 `results/targeted_ablations/webkb_data_audit.json`。

经典 Chameleon/Squirrel 存在 duplicate-node leakage 问题，见 [Platonov et al., ICLR 2023](https://arxiv.org/abs/2302.11640)。后续正式 benchmark 应加入清理版或[该论文公开的新数据集](https://github.com/yandex-research/heterophilous-graphs)。本轮沿用同一版本只为诊断已有相对性能下降。
