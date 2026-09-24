# 36次固定矩阵：global branch 的独立预测价值

固定 Chameleon、Squirrel，split 0/1/2，seed=42；六个配置共36次正式训练：PEGFAN、+T128、+T128+Haar、+stop-grad T128、+stop-grad T128+Haar、frozen PEGFAN+T128+Haar。不追加配置或调参。此前12次两 epoch的GPU短跑存放在独立 smoke 目录，不计正式训练。原环境/原PEGFAN优化器不变，最多500 epochs，patience=50，validation NLL选模。

## 所有模型的共同结构

原 PEGFAN 的 X、AX/A²X/A³X、原 cached Haar 对AX的频带、FSGNN、逐通道特征维度和 dropout 均保留。新分支只在自身投影为128维：Sheaf local update →固定分区Pool→2层4-head Transformer→可选post-T Haar→Lift→class logits G。融合仍在原分类 logits L 上进行：

`log_prob = log_softmax(L + alpha * G)`，`alpha_0 = 0`。

这里实际融合空间为N×C的class logits。记录的H_PEG=L，不将这些指标冒充宽隐藏层的feature指标。新分支参数lr=.001、weight decay=.0005，alpha lr=.001且无weight decay；原PEGFAN FC lr=.01、channel weights lr=.02。新分支初始化和dropout继续与原主干RNG隔离。

所有Haar配置按本轮给出的`Fuse{Z_T, F_j(Z_T)}`保留直接的Transformer信号，具体实现：

`Z_TF = Z_T + existing_learned_fusion(projector_Haar_bands(Z_T))`。

这是本轮固定使用的identity-plus-band残差融合；前轮仅对bands做learned fusion。因此本轮结果仅在本轮矩阵内作直接因果比较，不把前轮Haar128数值混入本轮表。所有五种分支模型实例化同样的128维模块、同样的初始权重，T-only跳过Haar/fusion的执行。

## 固定hierarchy，而非继续优化coarsening

不修改既有threshold-CC/tie-atomic filtration规则，不引入新的cluster条件。每个dataset/split只读加载前轮validation-selected Haar128 checkpoint，复现其验证损失，提取已经学得的partition；本轮该split的五种分支配置全程共用同一份分区和mass-normalized transfers。生成来源、checkpoint与分区SHA、node counts见`hierarchies/manifest.json`。

冻结的是实际partition，不只是generator配置；训练中不重新评分/重建分区。新分支的sheaf local update仍可学习，但其scores不改变固定结构。Pool、post-T Haar和Lift使用同一tree。原PEGFAN主干的原hierarchy完全不变。

参考树来自前轮同split训练监督和validation选模，是已有预训练结构。它不是无监督新树，也不是本轮学习到的accuracy来源；复用成本和来源不能在最终论文中省略。树来自Haar128参考模型，结论条件于这份共同结构，不能替代不同hierarchy下的普遍验证。

## 为什么stop-grad不能只对输入或G调用detach

当前branch只使用原始X，与PEGFAN没有共享的可学习encoder。因此对X执行detach没有干预。即使G.detach，`CE(L+alpha*G,y)`对L的梯度仍然依赖G，依旧会改变主干训练。

本轮stop-grad使用：

`loss_local = CE(L,y)`

`loss_branch = CE(stopgrad(L) + alpha*G,y)`

优化目标为两者之和，不平均。主干只得到原local CE梯度，分支和gate只得到fused CE梯度。已测试非零gate下完整主干参数与RNG轨迹逐值等于独立PEGFAN，而不仅是某条梯度为零。

### 同时去除选模对主干的间接影响

只隔离梯度仍不充分：若联合validation loss选择了不同local epoch，测试差异仍会混入主干选模变化。因此先训练六个独立PEGFAN基线，记录每个epoch的主干参数hash及其validation-selected checkpoint。

- **joint T/Haar**：常规联合训练，联合validation loss早停。
- **stop-grad T/Haar**：从同一随机初始化开始独立训练local CE，同时训练branch；主干逐epoch hash必须与baseline轨迹一致。到baseline自己选定的最佳epoch，主干固定为该状态；只有达到该时点后才允许选择branch checkpoint。后续主干eval、参数不更新。
- **frozen T+Haar**：从一开始加载同一validation-selected baseline并设为eval/冻结，仅训练branch和gate。

stop-grad与frozen最终的主干参数hash和gate-off全节点logits都必须与baseline checkpoint完全相同。两者区别是branch在主干学习期间是否已有训练，而非最终主干强弱。stop-grad最多500步含独立local训练阶段；frozen的branch最多500步从完整baseline开始；预训练基线的成本另列，包含在36个run中的6个baseline。

两种隔离方式都把完整baseline、alpha=0作为明确的validation候选，称为`baseline_reference`；若所有合格训练checkpoint均未降低其validation NLL，则返回它。`best_epoch=0`专指这个候选，不声称stop-grad实际训练的第0步已经具有预训练主干。patience从合格选模阶段开始计；被拒绝的训练过程、alpha及gate指标仍保存在history中。测试labels只用于最终诊断，不参与候选选择。

## Gate指标

逐epoch保存全图和validation的alpha及贡献指标；选定checkpoint额外保存全图/train/validation/test四个scope：

`R_g = ||alpha G||_F / ||L||_F`

`cosine = <L, alpha G>_F / (||L||_F ||alpha G||_F)`。

这两个是用户要求的原始指标，同时计算每个节点class维去均值后的同名指标。softmax对每个节点加一个class公共偏置不敏感，因此解释“抵消分类信号”时优先参考centered版本；raw版本原样保留，不用centered偷换定义。另保存平均逐节点cosine，区别于展平矩阵的Frobenius cosine。零gate时R_g=0，cosine无定义，JSON记null，不伪造为0。

选定checkpoint记录gate-on/off accuracy与changed predictions。对于stop-grad/frozen，gate-off严格等于独立PEGFAN；对于joint，gate-off是联合训练后的主干，仍不能等同于独立baseline。负cosine与accuracy下降同时出现可以提示抵消，但cosine本身不使用labels，不能单独证明分支有害。验证选择了零gate表示本配置未验证出增量，不证明所有Transformer都无效。

## 运行与交付

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/prepare_isolation_hierarchy.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_gradient_isolation.py
# 已有相同协议时加 --resume
python scripts/analyze_gradient_isolation.py
```

结果位于`results/gradient_isolation/`：checkpoint、histories、gates、共同hierarchy来源、逐split差值及审计。36次小矩阵检验的是当前128维分支、固定tree与训练协议；3个重叠split、单seed不构成普遍性或统计显著性证明。若要讨论严格frozen情况下Haar比Transformer-alone更好，本轮只有frozen Haar，需明确没有frozen T-only这一对照；其证据主要由stop-grad T/Haar配对提供。
