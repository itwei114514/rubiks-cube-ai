# 权重目录说明（ckpt/）

## ✅ 主权重（系统默认使用）
- **`agent_1e10.pt`** — **1e10 训练模型**（单 agent，最佳 @step 80000）。
  - 训练验证：40/40 解出，平均 27.1 步（@beam 256）
  - 最终评测：**100% 解出，平均 23.3 步**（@beam 4096，40 个深打乱）
  - **用途：所有求解入口（server.py / evaluate.py / run_demo.py）默认加载它。**
  - 从这以后**统一使用该模型**。

## 🧪 训练 / 续训状态（非服务权重）
- **`agent0_train.pt`** — 训练断点（含 model + optimizer + LR 调度器 + scaler + RNG + step）。
  - 仅用于训练中断后 `--resume` 续训，**不是最终服务权重**。

## 🗂 非主要权重（历史基线，不建议作为主用）
`not_primary/` 下均为早期 **1e8 基线**，仅供对比实验，**不是主模型**：
- `not_primary/agent_1e8_model0.pt` — 1e8 基线模型 #0（@beam4096 解出率约 70%，平均约 31.7 步）
- `not_primary/agent_1e8_model1.pt` — 1e8 基线模型 #1
- `not_primary/agent_1e8_model1_dup.pt` — agent1 的重复副本
- `not_primary/1e8_backup/` — 旧的 1e8 备份目录（含重复文件）

> 说明：本机策略禁止删除文件，因此未直接删除旧权重；已全部归并到 `not_primary/` 并以本 README 注明“非主要”。
> 若确认不需要，可自行删除 `not_primary/` 目录。