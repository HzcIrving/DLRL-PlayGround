# DeepLearning_PlayGround

# 😄DL游乐园

- The code repo contains multiple code reproduction processes of various SOTA deep learning algorithms.
- 这个代码仓库包含了经典、热门、以及SOTA的DL算法的复现。
- 我会在业余时间，对感兴趣的DL算法进行复现，并第一时间更新维护该Repo。

## 关键依赖项
- transformers 
  - `pip install transformers`
- torch  1.10.1 
- torchvision                   0.11.2 
- torchtext                     0.11.1

## I. Seq2Seq(TO-DO)

## II. Transformer Basic

### [Step By Step之Transformer基础实现](https://github.com/HzcIrving/DeepLearning_PlayGround/tree/main/TransformerBasic)

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

### 1. Transformers包中的VIT-Demo

- 该接口可以直接调用预训练的VIT模型对给定图片进行分类。
- [VIT.py脚本](https://github.com/HzcIrving/DeepLearning_PlayGround/blob/main/VIT/VITDemo/VIT.py)

### 2. [Step By Step之Transformer基础实现](https://github.com/HzcIrving/DeepLearning_PlayGround/tree/main/VIT/BasicVIT)

- VIT是Transformer在CV图片分类种的一种应用，VIT的实验结论是，在预训练Dataset足够大的前提下，所有数据集的表现是超过ResNet的。
- VIT的本质是一个Transformer的Encoder网络。
- 🚀️ [Colab ](https://colab.research.google.com/drive/1eCH380s0Yrt4DMERH1cQkbDZbK0Dufqt)

### 3. [Pre-trained VIT](https://github.com/HzcIrving/DeepLearning_PlayGround/tree/main/VIT)

- 基于 `ViT-B_16`预训练模型 + VIT Model

## IV. Swin Transformer(TO-DO)

## V. Meta Learning Part(TO-DO)

## VI. Reinforcement SOTA

### 1. Decision Transformer

DT将RL看成一个序列建模问题（Sequence Modeling Problem ），不用传统RL方法，而使用网络直接输出动作进行决策。

🚀️ [DecisionTransformer_StepbyStep](https://github.com/HzcIrving/DecisionTransformer_StepbyStep)
