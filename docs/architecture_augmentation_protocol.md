# Exact PEGFAN + gated nonlocal branch：小规模架构诊断

本轮只做 48 次正式训练：Film / Chameleon / Squirrel / 已修复 Cornell，固定 split 0、1、2，seed=42。四个配置是原 PEGFAN、Transformer64 增量分支、Transformer + post-T Haar64 增量分支、后一分支宽度 128。保留此前 500 epoch / patience=50 / 最小 validation NLL 选 checkpoint 的规则，不根据本轮 test 结果增加组合或调参。三个 split 是架构筛查，不能替代十 split / 多 seed 的最终比较；结果只和同三个 split 的 PEGFAN 配对。

## 原 PEGFAN 保持完整

直接调用原 `FSGNN.forward`，原 cached hierarchy / Haar、X、AX/A²X/A³X、所有 framelet channels、逐 channel F→64、归一化、softmax channel attention、分类器和 dropout=.5 全部保留。输入的 F 维特征不经过新增的共享 bottleneck；原模型本身已有的逐 channel hidden=64 也不改。

原 backbone 从同一 seed 初始化并照常训练，不冻结、不从前轮最佳 test 结果选择权重。通过临时 forward hook 读取其最终 fc2 的 raw logits L，不修改 FSGNN 源码或计算。融合为：

`log_prob = log_softmax(L + alpha * G)`，`alpha` 为所有节点共享的可学习标量，初值为 0。

G 的输出头正常随机初始化，不能和 alpha 同时零初始化，否则门控和分支可能都没有启动梯度。alpha=0 时，原模型的 logits、log_prob 和训练梯度保持原样；第一步 alpha 有梯度，随后分支开始学习。门控是有符号标量，不能仅根据 alpha 的绝对值判断贡献，另记录 `||center_class(alpha G)|| / ||center_class(L)||`。

构造分支时保存并恢复原 RNG；分支训练 dropout 使用独立的 `100042 + training_forward_index` 序列，随后恢复 PEGFAN RNG。关闭门控时已测试连续 optimizer 更新、输出、参数和 RNG 与原 PEGFAN 一致。评估时 dropout 关闭。两种 64 维配置实例化相同分支模块和相同初始参数，区别只在是否执行 post-T Haar/fusion。

## 新分支及第三项 bottleneck 实验

`X → branch-only F→d_T + GELU → local sheaf update / edge residual → dynamic sheaf hierarchy → Pool → Transformer → optional post-T Haar + learned fusion → Lift → linear class logits G`。

- 原 local PEGFAN 路径从未被 F→d_T 压缩；因此“取消全局 64 维 bottleneck”已经落实在全部增量配置中。额外比较 d_T=64 与 128 只改新分支，不再复制一条损坏本地路径的模型。
- 每个 forward 用当前 restriction maps 的 residual 建立 tie-atomic threshold-CC filtration。选第一个 regions≤512 的 cut。预算是上界，不强行凑足 512。
- 同一 hierarchy 的 cut 以下做 pooling/lifting，cut 以上做 Transformer 后的 projector-form Haar。T-only 配置跳过后者及 learned scale fusion。
- 非局部分支只 lift coarse global signal，不另插入局部 detail 或 hidden skip；PEGFAN 主干已经保留原有局部、多跳、多尺度输入。
- Transformer：2 层，4 heads，FFN=4d_T，dropout=.5，无 node-ID positional encoding。
- 新分支 lr=.001、weight decay=.0005；alpha lr=.001、无 weight decay；原 PEGFAN FC lr=.01、channel attention lr=.02 和 weight decay=.0005 不变。
- 从随机初始化共同训练。这是 logit-level gated fusion 的具体实现，不是证明所有 gated fusion 或 post-T framelets 都有效。
- 新分支在 eval 下满足 permutation equivariance；整模型仍继承原 PEGFAN cached hierarchy 的性质，不能据此宣称重新运行原 Ward generator 也严格 equivariant。

## Cornell Transformer diagnostic

Cornell 已修复为上游固定版本，183 nodes≤512，所以实际 token 数必须为183，不发生空间 pooling。新架构每个 Cornell trial 记录初始化和 validation-selected checkpoint 两个时点。另只读加载前轮修复后 `full_no_akx` 的10个 checkpoint，复核其选模指标，不新增训练。

每个时点记录：

- Transformer 输入、每层 attention 的输出（尚未加 residual）、每个完整 encoder layer 输出、最终 Transformer 输出。
- feature variance：各维度节点总体方差的平均。
- mean pairwise cosine：所有不同节点对的平均余弦相似度，排除对角；零向量记零相似度并另计数。
- centered energy fraction：`||H-mean_nodes(H)||² / ||H||²`，辅助区分 LayerNorm 带来的整体缩放与节点间差异收缩。
- 两层各4个 head的 attention entropy，使用实际 Q/K 权重重算无 mask 的 softmax probabilities；在 eval 下没有 attention dropout。记录 entropy（nats）、entropy/log(N)、平均最大 attention probability。已经与原 PyTorch 返回的 head-averaged weights 做数值核对。

不能把高 attention entropy 单独当作 oversmoothing 的证据。attention 输出接近一致，也不代表 residual + FFN 后的完整表示仍然一致；需要同时观察最终 cosine 和相对离散程度，并区分训练前后的变化。GPU scatter 运算可有微小数值误差，诊断前后 forward 使用浮点容差并要求所有预测不变，同时记录无 hook 重复 forward 的误差。

## 保存、审计与解释边界

保存全部 checkpoint、逐 epoch loss / gate / gate gradient、数据/split/cache/source hashes、初始主干参数 hash、初始函数误差、实际 tokens、最大 region 比例。最佳 checkpoint 另测试关闭门控后的准确率；这里的主干已经参与联合训练，因此它不等同于独立训练的 PEGFAN 对照，不能混用。

结果位于 `results/architecture_augmentation/`。8次两 epoch的GPU smoke test位于独立目录，不计入48次正式训练。Cornell旧checkpoint的诊断复用不计为新增训练。部分只读诊断与训练同时运行，训练耗时不作严格速度比较。

复现：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_augmentation.py
# 若已有相同协议的运行，使用 --resume
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/diagnose_cornell_transformer.py
python scripts/analyze_augmentation.py
```
