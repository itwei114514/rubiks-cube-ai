# 三阶魔方复原 Agent —— 扩散距离 ResMLP + 束搜索 + 3D 网页

> 项目：`Rubik's Cube_agent`
> GitHub：https://github.com/itwei114514/rubiks-cube-ai
> 硬件：NVIDIA RTX 5060 Ti（8 GB，实际可用约 6.3 GB）
> 环境：Python 3.12.10 · torch 2.11.0+cu128 · CUDA 12.8 · sm_120 (Blackwell)

一个**零人工知识**的三阶魔方求解 Agent：
训练数据完全由“从复原态随机打乱”自动生成（无监督/自监督），
网络学习“状态 → 到复原态的扩散距离”，求解时用**束搜索**寻找复原路径，
并配套一个**实时 3D 网页**（可打乱、可 AI 复原演示、可自定义盘面）。

---

## 1. 技术方案

| 组件 | 实现 |
|---|---|
| 状态表示 | 54 贴纸 × 6 颜色 **one-hot → 324 维** |
| 动作 | **QTM**，12 种（U,D,L,R,F,B × 顺/逆 90°） |
| 网络 | **ResMLP**：输入 324 → Linear(1024)+BN+ReLU → **4 个残差块**(1024↔768) → Linear(1) |
| 输出 | 1 维标量：预测扩散距离（到复原态的步数） |
| 损失 | SmoothL1 / Huber（预测距离 vs 随机游走深度），也可 MSE |
| 数据 | 在线生成随机游走，`K_max = 60`，记录所有前缀 `(state, depth)` |
| 训练 | FP16 AMP + **余弦 LR 调度**，Adam，batch=61440（num_walks=1024），支持断点续训 |
| 求解 | GPU 向量化束搜索（**全局去重**），`beam` 可选 1024/2048/4096/5120，`max_depth` 默认 100 |
| 模型 | **单 agent（1e10 训练）**，主权重 `ckpt/agent_1e10.pt` |

**核心思想**：扩散距离 = 随机游走到达该状态的步数 `k`，不需要人工标注、也不需要最优求解器；
它与真实最短距离强相关。网络学到这个“粗糙但有用”的距离打分，束搜索据此把每个状态逼近复原态。

---

## 2. 项目结构

```
Rubik's Cube_agent/
  README.md  config.py  requirements.txt  .gitignore  .gitattributes
  cube/        state.py(魔方状态/动作/校验)  scramble.py(在线随机游走/打乱)
  model/       resmlp.py(ResMLP)  train.py(FP16 训练+余弦LR+续训)
  solve/       beam.py(GPU 向量化束搜索+去重)  multiagent.py  agent.py(加载/求解)
  server.py    # 网页后端(FastAPI + /api/solve /api/solve_facelet /api/shutdown)
  train_main.py  evaluate.py  run_demo.py
  eval_1000.py  eval_stats.py  run_serial_beams.py  compare_models.py   # 评测脚本
  start_ui.bat # 一键开启网站
  ui/          index.html  app.js  style.css   # 实时 3D 网页
  ckpt/        agent_1e10.pt(主模型)  README.md  agent0_train.pt(续训)  not_primary/(旧权重)
```

---

## 3. 用法

### 训练
```powershell
# 用满 5h 预算(自适应步数, 吃远超 1e8 的样本)
python train_main.py --max_hours 5 --num_walks 1024 --num_agents 1

# 或按“总数据量”指定(例如 1e10 总量)
python train_main.py --total_data 10000000000 --num_walks 1024 --num_agents 1

# 断点续训(从上次保存的 agent0_train.pt 继续)
python train_main.py --total_data 10000000000 --num_walks 1024 --resume
```
主模型保存为 `ckpt/agent_1e10.pt`。

### 评测 / 演示
```powershell
python evaluate.py --num 50 --beam 4096
python run_demo.py --scramble "R U R' U'"
```

### 网页 UI(推荐)
```powershell
python server.py          # 或双击 start_ui.bat
# 浏览器打开 http://127.0.0.1:8000/
```
功能：
- **实时 3D 魔方**，拖动旋转视角；
- **🎲 随机打乱** / **手动转动**(12 个面转按钮，自己参与打乱)；
- **✨ 复原**：AI 用所选 beam 计算并自动 3D 演示；
- **打乱/复原速度分开**(0.1–5 s/步，100 ms 一档)；
- **两种独立模式**：🧊 普通模式 / 🎨 自定义模式；
  - 自定义模式：涂色对照手中魔方 → 3D 预览 → 简单合法性检查 → 求解(**3D 演示复原** + 文字步骤)；
  - 无法找到解/配色非法会提示**无解/不合法**。
- **beam** 可选 1024 / 2048 / 4096 / 5120；
- **出入口**：`start_ui.bat` 开启、网页内 **🛑 关闭网站** 按钮关闭。

---

## 4. 权重说明
| 文件 | 说明 |
|---|---|
| `ckpt/agent_1e10.pt` | **主模型(1e10 训练，单 agent)**，默认加载 |
| `ckpt/agent0_train.pt` | 训练/续训断点(含优化器/LR调度器/scaler/RNG)，非服务权重 |
| `ckpt/not_primary/` | 早期 1e8 基线权重，**非主模型** |
`ckpt/README.md` 有详细说明。

---

## 5. 实测结果(1e10 模型)

| 配置 | 解出率 | 平均步长 | 单次耗时 |
|---|---|---|---|
| @beam 256 · 40 深打乱 | 100% | 27.8 | 0.2s |
| @beam 4096 · 40 深打乱 | 100% | 23.3 | 2.34s |
| **@beam 5120 · 1000 深打乱** | **100%** | **23.03** | 2.86s |
| @beam 2048 · 100 深打乱 | 100% | 23.73(std 1.89) | 1.42s |

100 题 @beam2048 统计：中位 24.0，范围 **15–28** 步，P95=26，**最差 5% 平均 27.2 步**。
> QTM God's number = 26，任意合法魔方 ≤26 步可解；模型给出接近最优的解(平均 23 步)，远超“≤100 步”目标。

---

## 6. 环境
```
python 3.12.10
torch 2.11.0+cu128
cuda 12.8
device_capability (12, 0)  ->  sm_120 (Blackwell)
GPU NVIDIA GeForce RTX 5060 Ti
```
校验依赖：`kociemba`(仅用于魔方正确性验证，不影响训练/求解)。
`requirements.txt`：torch、numpy；验证额外需要 kociemba；网页需要 fastapi、uvicorn。

---

## 7. GitHub / 后续
- 仓库：https://github.com/itwei114514/rubiks-cube-ai
- 权重/数据/日志不入库(见 `.gitignore`)。
- 可扩展方向：更高阶魔方(4×4×4/5×5×5，同 pipeline 可复用)、更高维超立方体(难度/显存急剧上升)。