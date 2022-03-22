# DL&RL_PlayGround

# 😄DL&RL游乐园

- The code repo contains multiple code reproduction processes of various SOTA deep learning algorithms.
- 这个代码仓库包含了经典、热门、以及SOTA的DL/RL算法的复现。
- 我会在业余时间，对感兴趣的DL/RL算法进行复现，并第一时间更新维护该Repo。

## 关键依赖项

- transformers
  - `pip install transformers`
- torch  1.10.1
- torchvision                   0.11.2
- torchtext                     0.11.1

## I. Seq2Seq(TO-DO)

## II. Transformer Basic

#### ① Step By Step之Transformer基础实现

- 项目跳转链接🔗: [Step By Step之Transformer基础实现](https://github.com/HzcIrving/DeepLearning_PlayGround/tree/main/TransformerBasic)
- [🚀️Colab](https://github.com/HzcIrving/DeepLearning_PlayGround/blob/main/TransformerBasic/Transformer%E5%9F%BA%E7%A1%80%E5%AE%9E%E7%8E%B0StepByStep.ipynb)
- Encoder
  - Positional Encoding
  - Attention Machanism
  - Trick ---- Padding Mask
  - Add & Norm Layer
- Decoder
  - Masked Self-Attention
  - Masked Encoder-Decoder Attention

## III. VIT

#### ① Transformers包中的VIT-Demo

- 该接口可以直接调用预训练的VIT模型对给定图片进行分类。
- [VIT.py脚本](https://github.com/HzcIrving/DeepLearning_PlayGround/blob/main/VIT/VITDemo/VIT.py)

#### ② Step By Step之Transformer基础实现

- 项目跳转链接🔗: [Step By Step之Transformer基础实现](https://github.com/HzcIrving/DeepLearning_PlayGround/tree/main/VIT/BasicVIT)
- VIT是Transformer在CV图片分类种的一种应用，VIT的实验结论是，在预训练Dataset足够大的前提下，所有数据集的表现是超过ResNet的。
- VIT的本质是一个Transformer的Encoder网络。
- 🚀️ [Colab ](https://colab.research.google.com/drive/1eCH380s0Yrt4DMERH1cQkbDZbK0Dufqt)

#### ③ Pre-trained VIT

- 项目跳转链接🔗: [Pre-trained VIT](https://github.com/HzcIrving/DeepLearning_PlayGround/tree/main/VIT)
- 基于 `ViT-B_16`预训练模型 + VIT Model

## IV. Swin Transformer

#### ① Step By Step 之 Swin Transformer实现

- 项目跳转链接🔗：[Swin Transformer](https://github.com/HzcIrving/DeepLearning_PlayGround/tree/main/Swin-Transformer)
- Swin Transformer 被视为CNN的理想替代方案，其在设计时也融合了很多CNN的思想。
- Swin Transformer 结合CNN思想，引入层次化构建方式构建层次化的Transformer，使得SwinT可以做层级式的特征提取（方便下游多尺度的检测、分割任务）。证明了Swin Transformer可以作为通用的视觉任务Backbone网络。
- 详情：[知乎: DLPlayGround之Swin-Transformer(v1)](https://zhuanlan.zhihu.com/p/467158838)

## V. Meta Learning Part(TO-DO)

## VI. Offline RL

#### ① Offline RL Introduction

- 基础知识点笔记跳转链接🔗: [Offline RL -- Introduction](https://github.com/HzcIrving/DeepLearning_PlayGround/blob/main/Offline%20RL/Introduction/OFFLINE_RL.pdf)

#### ② Decision Transformer

DT将RL看成一个序列建模问题（Sequence Modeling Problem ），不用传统RL方法，而使用网络直接输出动作进行决策。

- 项目跳转链接🔗: [DecisionTransformer_StepbyStep](https://github.com/HzcIrving/DecisionTransformer_StepbyStep)

#### ③ BCQ

Batch-Constrained deep Q- Learning(BCQ)

* 优化Value函数时候加入future uncertainty的衡量；
* 加入了距离限制，通过state-conditioned generative model完成；
* Q网络选择最高价值的动作；
* 在价值更新时候，利用Double Q的估计取soft minimum; $r+\gamma max_{a_i}[\lambda min_{j=1,2}Q_{\theta' *j}(s',a_i)+(1-\lambda)max* {j=1,2}Q_{\theta'_j}(s',a_i)$ 是Convex Combination 而不是 Hard Minimum ...

- 项目跳转链接🔗: [BCQ](https://github.com/HzcIrving/DLRL-PlayGround/tree/main/Offline%20RL/BCQ)

#### ④ AWAC 

关键点：

- Trains well offline
- Fine-tunes quickly online
- Does not need to estimate a behavior model.
- 项目跳转链接🔗: AWAC

## Distributional RL

#### ① C51

- 项目跳转链接🔗: [C51](https://github.com/HzcIrving/DLRL-PlayGround/tree/main/Distributional%20RL/C51)

#### ② D4PG

Distributed Distributional Determinisitic Policy Gradient (D4PG)

D4PG将经验收集的Actor和策略学习的Learner分开：

* 使用多个并行的Actor进行数据收集，即分布式的采样；
* 分享一个大的经验数据缓存区，发送给Learner进行学习，Learner从Buffer中采样，将更新后的权重在同步到各个Actor上（ApeX)；
* 使用TD(N-steps)的方式进行处理，减小Bias；
* 可以使用PER技术（优先经验回放）；
* Critic Net -- C51-based method.
* 项目跳转链接🔗: [D4PG](https://github.com/HzcIrving/DLRL-PlayGround/tree/main/Distributional%20RL/D4PG/)
