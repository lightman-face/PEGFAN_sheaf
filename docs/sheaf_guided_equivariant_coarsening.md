# Sheaf-Guided Equivariant Coarsening：理论论证与实现规格

状态：已按本规格实现，2026-09-23。本文件给出方法的定义、证明、限制与实现契约；实现位于 sheaf_coarsening.py，统一实验入口为 compare_pegfan.py。所有复杂度均区分评分、filtration 构建、层次物化与神经网络计算。

## 1. 结论与必要条件

用户提出的“同分边整批加入 → 连通分量 filtration → attention budget cut → projector Haar”可以建立完整的置换等变性论证。需要补齐以下条件：

- merge score 越大越适合合并；若使用 sheaf 不一致性能量，应取其负值或严格递减变换。
- 一次建树使用固定的原图边分数；合并中途不重新计算区域分数。
- 同分批次处理完毕后才记录分区、检查预算；不以节点编号或簇大小上限拆分该批次。
- 断连图的纯边 filtration 最终停在原图连通分量，需要显式预算可达条件或对称的 virtual root。
- 对数深度属于抽样后的 token 尺度序列，完整 filtration 最坏仍有线性数量的不同分区。
- 不显式构造大簇内的所有 pairwise framelets，也不构造稠密的 `I - p p.T`。
- 严格等变性指理想精确算术、固定参数、关闭 dropout 的前向映射。独立随机 dropout 一般只具有分布等变性；若要求逐次输出对应，随机掩码也须同步置换。

## 2. 评分和 filtration 的定义

设无向图 G=(V,E)，n=|V|≥1，m=|E|，输入特征 X，attention budget M 为正整数。节点重排由 Π 表示；比较时 X'=ΠX，A'=ΠAΠᵀ，模型参数 θ 保持相同。

使用逐节点共享编码器或已证明等变的编码器得到 H。对于无向边 e={u,v}，共享端点函数生成：

\[
R_{u,e}=\Phi_\theta(h_u,h_v),\qquad
R_{v,e}=\Phi_\theta(h_v,h_u).
\]

轻量版本可以使用当前实现的对角 restriction maps。定义

\[
s_e=\|R_{u,e}h_u-R_{v,e}h_v\|_2^2,
\qquad q_e=-s_e.
\]

这样按 q 从大到小处理等价于按不一致性能量从小到大处理。交换边方向会交换两端映射并反转差向量，因此分数不变。可采用其他对称评分，但必须单独说明其定义与等变性；首版不需要加入额外边界头或学习 Haar 权重。

取所有不同的有限分数 τ₁>…>τ_T，定义

\[
E_j=\{e\in E:q_e\ge\tau_j\},\qquad
\mathcal P_j=\operatorname{CC}(V,E_j).
\]

另设初始分区 P₀={{v}:v∈V}。只保留造成分区变化的事件；加入环内边但不改变连通分量的事件不产生新层。此处的分区是无序块集合，簇编号与并查集代表元不属于数学输出。

一次构建完成后，整个 filtration 与 M 无关。更新模型参数后，可以在下一次构建重新计算 q。

## 3. 断连图与预算的补全

设 c=|CC(G)|。原图边全部加入以后仍有 c 个块，因此：

\[
\exists j:\ |\mathcal P_j|\le M
\quad\Longleftrightarrow\quad c\le M.
\]

两个明确的实现策略：

1. `disconnected_policy="forest"`：保留森林；M<c 时报告预算不可达。最终 lowpass 可以有 c 个坐标，不需要强行合并为一个根。
2. `disconnected_policy="virtual_root"`：在真实边 filtration 完成后，若 c>1，将全部 c 个连通分量同时接到一个抽象根。该事件不依赖 M、不依赖节点编号、不添加图边，并标注 `virtual=True`。此时任意 M≥1 都有合法 cut。

建议首版采用第二种策略以满足全图 token 上限，同时保留第一种作为可解释性对照。必须报告代价：当 c>M 时，唯一新增的预算可达层是单根，可能只有一个 token；不能宣称仍有接近 M 个连通 coarse regions。

不采用“按节点编号把断连区域分成 M 组”的回退策略。若增加基于特征的跨分量连接，那是另一个需要定义和分析的算法。

## 4. 预算切分与尺度抽样

记不同分区按由细到粗排列为 P₀,…,P_L，n_j=|P_j|。在预算可达的前提下定义

\[
j^\star=\min\{j:n_j\le M\},\qquad m_\star=n_{j^\star}.
\]

这是该 filtration 中满足预算的最细分区。若 n≤M，则 j*=0，不做特征粗化。若某个同分批次将块数从 100 降为 7，即使 M=64，也必须选择 7，不能从批次内部制造 64 个块。

用户提出的目标 M, floor(M/2), …, 1 可以直接使用。为减少重复目标，建议从实际 token 数出发：

\[
t_r=\left\lfloor\frac{m_\star}{2^r}\right\rfloor,
\quad r=0,\ldots,\lfloor\log_2m_\star\rfloor,
\qquad
j_r=\min\{j\ge j^\star:n_j\le t_r\}.
\]

在 virtual-root 策略下所有目标均可达；按顺序删除重复 j_r。遇到大规模同时合并时，多个目标可以命中同一分区，不插入人为二叉化层。forest 策略则把目标截到 c，并在原图连通分量层终止。

切分层以下若需要多尺度 local details，可类似地从 n,n/2,…抽取尚未越过 j* 的事件，最后加入 j*。因此整条用于特征计算的层次可以有 O(1+log n) 层；token cut 以上的 framelet 尺度数则是 O(1+log M)。两者不能混用。

## 5. 命题一：hierarchy equivariance

**命题。** 若 H 和评分器满足节点/边等变性，分数按完整同分批次处理，则对任意置换 π，

\[
\mathcal F_\theta(\Pi A\Pi^\top,\Pi X)
=\pi\bigl(\mathcal F_\theta(A,X)\bigr),
\]

其中等号指各层分区的块对应，而非数组中的簇编号相同。上述两种断连策略均保持这一性质。

**证明。** 共享端点函数使重排后的对应边满足 q'_{π(e)}=q_e，因此不同阈值集合相同，且 E'_j=π(E_j)。路径在节点重排下双射对应，故 CC(π(V),π(E_j))=π(CC(V,E_j))。删除不改变分区的事件只依赖连通分量是否减少；该条件在重排后保持。virtual root 的全部孩子是最终连通分量集合，重排只改变孩子的排列。证毕。

不必给并查集选择“等变的代表节点”：内部代表元和数组顺序可以不同，只要它们不影响分区、评分、后续截断及簇内权重。

## 6. 命题二：nestedness 与 budget

**命题。** 对固定边分数，删除重复分区后有严格的由细到粗嵌套关系。预算可达时，上述 j* 存在且满足 n_{j*}≤M；所有更细的事件层均有 n_j>M。预算选择和计数驱动的尺度选择也置换等变。

**证明。** 阈值递减只增加边，不删除边，故已连通的节点以后仍连通：每个旧块包含在唯一新块中。重复分区被删去后每次变化至少减少一个连通分量。预算可达性由第 3 节给出；最细性直接来自 j* 的最小定义。根据命题一，置换不改变阈值事件或各层块数，因此选择相同事件。证毕。

这一定理不保证 n_{j*}=M，也不给出 m* 相对 M 的正比例下界。同分的连通图可能从 n 个单点直接变成一个块。

## 7. 命题三：深度与复杂度

### 7.1 选取的 framelet 尺度数

设 K 是包含 token cut 在内、去重后的所选分区数量。使用第 4 节的实际 token 数目标时，

\[
K\le 1+\lfloor\log_2m_\star\rfloor
\le 1+\lfloor\log_2M\rfloor.
\]

证明：候选目标只有这么多个，去重不增加数量。使用 M 起始的目标直接得到后一个上界。若 K 计数的是高通层或转移次数，则对应上界减一。

完整 filtration 没有该对数界。例：n 节点路径，所有 n−1 条边分数不同，每次加边恰好减少一个连通分量，因而有 n 个不同分区。

### 7.2 紧凑 filtration 的构建

分数已经给定时，排序边、逐批执行带路径压缩和按秩合并的并查集，并记录多叉 merge forest，可达到

\[
T_{\mathcal F}
=O\bigl(n+m\log(1+m)+m\alpha(n)\bigr),
\qquad S_{\mathcal F}=O(n+m).
\]

排序贡献 m log(1+m)，并查集初始化贡献 n，处理边贡献 m α(n)。每个内部合并节点至少减少一个当前块，所以真实内部节点不超过 n−c；virtual root 至多再加一个。存储同分整批合并后的多叉节点，不存储顺序 union 时任意产生的二叉中间层。

在非平凡连通图、分数已给定的通常情形，可以把上述时间简写为 O(m log m)。对有大量孤立节点的图不能省略 n。sheaf 神经网络计算分数的代价 C_q 需要另加。

如果每个分数事件后都扫描所有 n 个节点并保存完整 assignment，时间和空间最坏会到 O(nL)=O(n²)，此时不能再使用上述紧凑构建复杂度。

### 7.3 下游物化与计算需要单独计费

- 从紧凑 merge forest 物化原节点的预算分区可用 O(n) 遍历。
- 在预算 cut 以上，只对 m* 个 token 查询选定层；一种直接实现需要 O(m*K) 生成各层 token assignment。这不是 O(n) 紧凑森林存储的一部分。
- 若同时保留全部 K 个 token 空间的 d 维 band signals，它们本身就占 O(m*K*d) 空间；逐 band 提升/融合也应单独计费。
- 对固定 Transformer 层数，标准 attention 的计算项是 O(m*² d+m*d²)，由 m*≤M 控制。它不控制 sheaf 的原图边计算量。
- 不在全部 filtration 事件上重新构造 coarse adjacency。标准 token Transformer 与 projector Haar 只需要分区和投影；如确需 coarse adjacency，仅在所需 cut 构造并另记费用。

## 8. 命题四：projector Haar、跳层一致性与重构

### 8.1 建议的归一化

任意单位范数 p 都能产生互补投影 ppᵀ 与 I−ppᵀ。不过若要让“抽取 filtration 的若干层”与完整层次的算子保持一致，建议使用原节点数量作为 cluster mass。

对分区 P_j，定义归一化指示矩阵 U_j∈R^{n×n_j}：

\[
(U_j)_{v,C}=\frac{\mathbf1\{v\in C\}}{\sqrt{|C|}}.
\]

显然 U_jᵀU_j=I。对任意两个嵌套的、可不相邻的分区 P_a≼P_b，定义

\[
R_{a\to b}=U_a^\top U_b,
\qquad
(R_{a\to b})_{C,D}
=\begin{cases}\sqrt{|C|/|D|},&C\subseteq D,\\0,&\text{otherwise.}\end{cases}
\]

于是

\[
R_{a\to b}^\top R_{a\to b}=I,
\qquad U_aR_{a\to b}=U_b,
\qquad R_{a\to b}R_{b\to c}=R_{a\to c}.
\]

证明：一个父簇的孩子互不相交，且孩子的质量和等于父簇质量；矩阵乘法中每个孩子只经过唯一父簇，平方根质量比相乘后消去中间质量。

因此 skipping levels 不改变端点之间的低通投影。局部 p_C=√(|C|/|D|) 在孩子等质量时退化为 1/√k。当前代码的“每层对孩子等权”仍可重构并等变，但在孩子原节点数量不等时，重新抽层后的算子通常不等于原来逐层算子的复合；不能把这两种归一化混为一谈。

PEGFAN 的 Theorem 1 允许一般单位范数 scaling 向量；这里的质量归一化与跳层公式是基于该构造作出的具体推导：[PEGFAN, Section III-B](https://arxiv.org/html/2306.04265v3#S3.SS2)。

### 8.2 单层重构与能量恒等式

记 R 为当前细层到粗层的转移，RᵀR=I。对任意信号 Y 定义

\[
Z=R^\top Y,\qquad D=(I-RR^\top)Y.
\]

则

\[
Y=RZ+D,\qquad R^\top D=0,
\qquad\|Y\|_F^2=\|Z\|_F^2+\|D\|_F^2.
\]

证明：第一式代入即得；第二式使用 RᵀR=I；因此 RZ 和 D 正交，且 ||RZ||=||Z||，得到第三式。对每层递归应用即可重构原始输入。

对一个含 k 个孩子的父簇，计算应写成 `low = p @ (p.T @ Y)`、`detail = Y - low`。代价 O(kd)，额外局部存储 O(k+d)（不计输入/输出），不枚举 O(k²) 个 pairwise 系数，也不存储 k×k 的投影矩阵。

### 8.3 Transformer 输出的多尺度分解

在 token cut 空间，令 V₀=I_{m*}，V_r 为从 token 层到第 r 个被选层的转移乘积，E_r=V_rV_rᵀ。嵌套性保证这些是嵌套子空间的正交投影。令末层序号为 K−1，则

\[
B_{\mathrm{low}}=E_{K-1},\qquad
B_r=E_r-E_{r+1}\quad(0\le r<K-1).
\]

这些 band projectors 两两正交，并满足

\[
B_{\mathrm{low}}+\sum_r B_r=I,
\qquad
Y=B_{\mathrm{low}}Y+\sum_r B_rY,
\qquad
\|Y\|_F^2=\|B_{\mathrm{low}}Y\|_F^2+\sum_r\|B_rY\|_F^2.
\]

证明：嵌套投影有 E_rE_s=E_sE_r=E_max(r,s)，故差投影幂等且互相正交；求和望远镜消去得到 I。重构与能量结论随之成立。forest 策略的最终低通子空间可以是多维，证明不变。

这里 Y 是 Transformer 增强后的 token signal；重构和能量恒等式针对 learned band fusion 之前的分析/合成，不声称任意学习后的融合仍保能量。

### 8.4 全链路等变性

设 Π_a、Π_b 分别表示两个尺度的簇重排。质量归一化给出

\[
R'_{a\to b}=\Pi_aR_{a\to b}\Pi_b^\top.
\]

所以 pooling 满足 R'ᵀΠ_aY=Π_bRᵀY，lifting 满足 R'Π_bZ=Π_aRZ，局部高通满足

\[
I-R'R'^\top=\Pi_a(I-RR^\top)\Pi_a^\top.
\]

无节点编号位置编码的确定性 self-attention 满足 T(Π_*Z)=Π_*T(Z)：QKᵀ 在 token 重排下共轭变换，逐行 softmax 保持该对应关系，乘 V 后输出按 Π_* 重排。共享逐 token FFN、LayerNorm 与残差连接也保持该性质。若未来引入 attention mask/bias，它们也必须同步重排。

各尺度 band projectors 同样按 Π_* 共轭变换。使用共享逐节点融合、只依赖层数/簇数的尺度描述，以及对应提升和逐节点分类器，得到固定 θ 下

\[
\boxed{f_\theta(\Pi X,\Pi A\Pi^\top)=\Pi f_\theta(X,A).}
\]

当前可微局部 sheaf 更新也符合端点共享映射和对 incident edges 求和的形式，可放在该链路中。局部细节融合须使用对应层次的投影，不能另建一个任意分区。

## 9. 不应由上述命题推出的结论

### 9.1 边界绝对保留

高 sheaf 能量边本身加入得晚，但其两个端点可能已经通过其他低能量边连通。

可实现的反例：三角图，标量特征 (0,1,2)，identity restrictions。三条边 (0,1)、(1,2)、(0,2) 的能量分别为 1、1、4。能量为 1 的两条边同时加入时，三个节点已经处于同一个块；能量为 4 的边两端也已经合并。

因此该算法具有单链连接效应，不能直接证明“所有高频边界始终保留”。更强的可证条件是：在某阈值下，预期区域内部连通，且所有跨区域边均未达到加入阈值，则该阈值分区恰为这些区域。若还希望 Transformer cut 不越过此分区，则其块数需不超过预算 M。

### 9.2 一般扰动稳定性

置换等变性不等于对特征扰动、评分扰动或训练更新的稳定性。三节点路径的两条边同分时可直接得到 [3,1]；将其中一条边的分数提高任意小量，就出现 [3,2,1]。M=2 时 token 数和选中的分区会改变。

若有效合并阈值之间存在正间隔，且扰动不足以改变相关严格排序、也不打破相关并列关系，则对应分区可保持不变；缺少这些条件时不能宣称统一的连续性或 Lipschitz 稳定性。

### 9.3 离散建树可微性

整批 CC 与排序仍不可微。当前模型 restriction maps 的任务梯度来自可微局部 sheaf 更新；新的 filtration 不会自动增加穿过离散簇成员的梯度路径。可微簇内权重属于后续独立扩展，不是这四个命题成立的前提。

### 9.4 创新定位

按阈值取连通分量对应图上的 single-linkage 结构。single linkage 的最小跨簇距离定义见 [SciPy 官方 linkage 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)。因此不应把 threshold-CC 本身、单位投影恒等式或简单目标计数界称为新发明。

可检验的贡献定位是：任务学习的 sheaf 评分如何决定层次；同分对称性与预算切分的统一处理；同一 filtration 上的 token 全局建模及隐式 Haar 分析；以及这些组件在异亲任务中的实证收益。正式新颖性判断仍需要更完整的相关工作比较。

## 10. 实现接口与数据结构规格

建议把“生成”与“选择”拆开，防止 M 隐式改变树。

```python
build_sheaf_filtration(
    num_nodes, edge_sources, edge_targets, merge_scores,
    disconnected_policy="virtual_root",
) -> MergeFiltration

select_hierarchy_levels(
    filtration, token_budget,
) -> HierarchySelection  # local/global schedules are dyadic

materialize_projections(
    filtration, selection,
) -> SheafHierarchy  # normalization is fixed to node mass
```

`MergeFiltration` 至少保存：初始节点数、叶子与多叉内部节点、父子关系、每个簇的原节点质量、合并事件阈值、每个有效事件后的块数、原图连通分量数、virtual-root 标志。边分数不要求为正，但必须有限。禁止 self-loop 参与合并；平行边若存在，应定义为任一边达到阈值即可连通，或者事先明确等价聚合规则。

`HierarchySelection` 保存预算 M、实际 m*、cut 事件、local/global 被选事件及目标、去重映射。空图拒绝；n=1、M=1、n≤M 均须明确定义。

`SheafHierarchy` 保存所选层的簇数、嵌套 assignment、原节点质量和稀疏转移 R。簇数不再依赖于存储每层 adjacency。token cut 以下保存 local details；cut 以上对 Transformer 输出作 Haar 分析；整条路径共享同一组转移。

构建器内部顺序：

1. 验证端点、分数和节点数；计算一次完整排序。
2. 将完全相同的分数视为一个 batch；收集该 batch 连接的旧 components。
3. 完成整个 batch 的 unions；将每个发生变化的最终 component 记录成一个多叉父节点。
4. 仅在完整 batch 后记录 component count；不检查“已经刚好达到 M”并提前中断。
5. 丢弃不引起 component 变化的 batch；最终按策略保留森林或添加统一 virtual root。
6. 对完成的 filtration 选择 budget cut 和 framelet levels；按需物化投影。

并查集的二叉实现细节不能泄露成 framelet 层次。生成器不再接收 `max_cluster_size` 或 `residual_quantile`；这两个旧参数与完整 threshold-CC 的定义不一致。Ward 也不出现在新路径中。

浮点实现默认按实际计算出的完全相同分数组批。若采用容差，应使用明确的、逐边置换一致的量化映射后再分组；不使用依赖遍历顺序、非传递的链式 `isclose` 分组。量化会改变方法，需记录在配置中。数学等变性与浮点近似误差分别报告。

固定 filtration 可以缓存结构。若周期性重建，应明确更新时机，并在 checkpoint 中保存影响预测的缓存结构/版本或给出能重建相同状态的规则。不得把陈旧缓存与新权重混合后仍声称与每次重建相同。

## 11. 后续实现必须覆盖的验证

| 场景 | 必须检查的结果 |
|---|---|
| 随机图 + 大量同分边 | 节点置换后 co-membership 矩阵共轭对应；不能只比较簇编号 |
| 边输入顺序打乱、端点交换 | 每个阈值分区不变 |
| 同分连通图 | 允许一次大分支合并；不生成依赖输入顺序的中间层 |
| 不同 M | 同一输入与参数的完整 filtration 完全一致；只有 selection 改变 |
| 三节点路径同分，M=2 | 可选分区是 [3,1]，实际 token 数为 1 |
| 五个孤立节点，M=2 | forest 策略报告不可达；virtual-root 策略得到 [5,1] |
| 不同边分数的路径 | 完整层数线性；所选 token 尺度满足对数上界 |
| 不等质量孩子 + 跳层 | RᵀR=I，且逐层复合等于直接跨层投影 |
| 大分支父节点 | 隐式结果等于小例子的显式投影；实现不申请 k×k 矩阵 |
| token Haar 与 local details | 融合前重构、能量分解及梯度均正确 |
| 全模型节点置换 | 关闭 dropout，从评分到重新建树、预测完整检查；使用容差而非逐位相同 |
| 训练与恢复 | restriction maps 有任务梯度，cached hierarchy 与 checkpoint 行为一致 |

复杂度验证需分别测建树、层次物化、Transformer、Haar 与峰值内存，不能仅以 token 数替代性能评估。

## 12. 本轮独立数值核对

以下是在临时小规模参考计算上得到的检查结果，记录理论阶段的独立核查，也不替代上述数学证明：

- 100 节点不同分数路径：完整 filtration 有 100 个分区；M=16 时选取 [16,8,4,2,1]。
- identity-sheaf 三角形反例：残差 [1,1,4]，块数 [3,1]，高残差边端点提前连通。
- 五个孤立节点：原图 filtration 停在 5；添加统一 virtual root 后为 [5,1]。
- 30 个带离散同分分数的随机图：重排节点后，每个有效事件的 co-membership 检查通过，包括断连情况。
- 不等簇质量的嵌套分区：跳层复合检查通过；double 精度重构最大误差约 2.22e-16，能量差为 0。

上述参考计算逐阈值重算 CC，仅用于核查定义和反例；不具有第 7 节紧凑生产实现的复杂度保证。
