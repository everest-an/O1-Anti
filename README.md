# NVIDIA Nemotron Model Reasoning Challenge - MT-LNN Edition

This repository is dedicated to the Kaggle [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).

## 理论与比赛规则的结合 (Theory & Competition Constraints)

由于该 Kaggle 比赛**严格要求**提交流准的 LoRA 适配器（最大 rank 32），并且**后台使用 vLLM 固定原生架构推理**，我们无法在推理时直接引入原项目中的 `mt_lnn_layer.py` 或是连续时间的物理微分方程结构。 

**我们的破局方案（The Solution）：将理论前置到“训练和蒸馏”阶段。**

核心逻辑：**利用 MT-LNN（微管液态神经网络）与量子耦合物理正则化（Quantum Coupling Regularization）来干预 LoRA 矩阵的学习轨迹。**

1. **MT-LNN 辅助数据构建 / 蒸馏**：利用你的 O1 完整结构的输出分布，作为高质量的软标签（Soft-labels）或筛选数据的依据，转移给 Nemotron 的 LoRA 进行学习。
2. **定制物理启发 Loss (Physics-inspired Loss)**：在标准的交叉熵损失之外，加入类似于原系统中的全局相干性（`global_coherence`）约束，迫使 LoRA 产生的特征空间扰动服从 MT-LNN 的几何流形。
3. **LoRA 矩阵正交化与谱初始化（Spectral Initialization）**：采用原项目中的 `phi_spectral.py` 逻辑来初始化 LoRA 的 $A$ 和 $B$ 矩阵，使其天然具备更好的动力学稳定性。

## 项目结构
- `train_lora.py`: 适配 Nemotron-3-Nano-30B 的核心训练脚本。
- `mt_physics_loss.py`: 将你的液态神经网络和量子耦合理论转化为训练时的正则化损失。
- `data_synthesis.py`: 结合你原先的架构进行复杂推理链的数据生成。

## 如何上传到你的 GitHub
由于我无法直接读取你的 GitHub 密码以创建云端仓库，你只需在 GitHub 页面点击 "New Repository"（命名为 `Nemotron-MT-Reasoning`），然后在终端执行：
```bash
git remote add origin https://github.com/everest-an/Nemotron-MT-Reasoning.git
git branch -M main
git add .
git commit -m "Initial commit for Kaggle Nemotron Challenge"
git push -u origin main
```
