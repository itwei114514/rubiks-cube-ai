# 三阶魔方复原 Agent — 扩散距离 ResMLP + 束搜索 + 多智能体

> 项目目录：`F:\VScode_program\.vscode\Rubik's Cube_agent`
> 硬件：NVIDIA RTX 5060 Ti（8 GB，桌面/壁纸已占约 1.7 GB，实际可用约 6.3 GB）
> Python：`.venv` = `F:\VScode_program\Agent\.venv`（Python 3.12.10，`torch 2.11.0+cu128`，CUDA 12.8，`device_capability=(12,0)` sm_120）

这是一个**零人工知识**的三阶魔方求解 Agent：
训练数据完全由“从复原态随机打乱”自动生成（无监督/自监督），
网络学习“状态 → 到复原态的扩散距离”，求解时用束搜索 + 多智能体寻找复原路径。

---

## 1. 技术方案

| 组件 | 实现 |
|---|---|
| 状态表示 | 54 贴纸 × 6 颜色 **one-hot → 324 维** |
| 动作 | **QTM，12 种**（U,D,L,R,F,B × 顺/逆 90°） |
| 网络 | **ResMLP**：输入 324 → Linear(1024)+BN+ReLU → **4 个残差块**(每块 1024↔768) → Linear(1) |
| 输出 | **1 维标量**：预测扩散距离（到复原态的步数） |
| 损失 | SmoothL1/Huber（预测距离 vs 随机游走深度），也可切 MSE |
| 数据 | **在线生成**随机游走，`K_max = 60`，每条游走记录所有前缀 `(state, depth)` |
| 训练 | **FP16 AMP**，Adam（lr=1e-3），批 = 256 条游走 × 60 = 15360 对，默认总样本 **1e8** |
| 求解 | **GPU 向量化束搜索**，`beam = 2^12 = 4096`，`max_depth = 100` |
| 多智能体 | **agent = 2**，各自独立种子/数据训练，取最短解 |

**核心思想（为什么无监督 + 只管用）**：扩散距离 = 随机游走到达该状态的步数 `k`。
`k` 不需要任何人给，也不需要最优求解器；它和“真实最短距离”强相关。网络学会这个“粗糙但有用”的打分，束搜索用它把每个状态逼近复原态。

---

## 2. 项目结构

```
Rubik's Cube_agent/
  README.md                # 本文件
  config.py                # 所有参数（N1,N2,Nr,K_max,batch,beam,agents...）
  requirements.txt
  cube/
    state.py               # 魔方状态/动作/324维编码/校验（已与 kociemba 交叉验证）
    scramble.py            # 在线随机游走 / 打乱生成（K_max=60）
  model/
    resmlp.py              # ResMLP 结构
    train.py               # FP16 训练循环 + 验证 + 保存
  solve/
    beam.py                # GPU 向量化束搜索
    multiagent.py          # 多智能体取最短解
    agent.py               # 高层接口 CubeAgent / load_agent
  train_main.py            # 训练入口
  evaluate.py              # 评测入口
  run_demo.py              # 演示入口
  ckpt/                    # 训练好的 agent0.pt / agent1.pt
  tests/                   # 校验脚本
```

---

## 3. 用法

### 3.1 训练
```powershell
# 默认：每个 agent 训练 1e8 个样本（= 总样本 1 亿个，每个 agent 自己 1 亿）
python train_main.py

# 用满 5 小时预算（自适应：先测每步耗时，再按 5h 推算步数，会吃远超 1e8 的样本）
python train_main.py --max_hours 5

# 改参数示例
python train_main.py --num_agents 2 --num_walks 256 --total_samples 100000000 --k_max 60

# 快速自检（极小模型，几十秒）
python train_main.py --smoke
```
保存：`ckpt/agent0.pt`、`ckpt/agent1.pt`。

### 3.2 评测
```powershell
python evaluate.py --num 50 --beam 4096 --max_depth 100
```
输出：解出率、平均解长、平均求解时间。

### 3.3 演示（输入一串打乱 / 随机打乱）
```powershell
python run_demo.py --scramble "R U R' U'"
python run_demo.py --beam 4096
```

---

## 4. 当前实测结果（在 5060 Ti 上）

用默认参数训练了 **2 个 agent × 1e8 样本**（约 9–15 分钟完成），然后在 **深度 20–60 的“任意打乱”** 上以 `beam=4096`、`max_depth=100` 评测：

| 指标 | 结果 |
|---|---|
| 解出率 | **7/10（70%）** |
| 平均解长 | **27.7 步**（目标 ≤100，达标） |
| 平均求解时间 | 约 17 秒/个（beam=4096） |

**结论**：方法端到端可用，且“平均 ≤100 步”在解出的盘面上轻松达标。
但 **1e8 样本只是参考论文里的“小数据集”，会欠训练**，导致“任意深打乱”解出率不高（70%）。

### 关键提示：1 亿样本 vs 训练 5 小时
- 在你机器上，1e8 样本训练**仅约 10 分钟**（每步约 0.08 s）。
- 要真正达到“训练 5 小时、任意打乱高解出率”，需要**远超 1e8 的样本/步数**。
- 因此请用：
  ```powershell
  python train_main.py --max_hours 5
  ```
  它会按 5 小时预算自适应训练（预计每个 agent 约 20–40 万步，即约 3×10^9–6×10^9 样本），
  通常能显著提高解出率（参考 CayleyPy：1.28e8 样本 + 更大的 beam/agent 达 98% 最优）。

**把 `total_samples`（默认 1e8）当作“每阶段至少跑这么多”，用 `--max_hours` 控制真正时长。**

---

## 5. 常用调参方向

- **解不出 / 解出率低** → 加大 `beam`、`--num_agents`（增益最明显），或 `--max_hours` 拉长训练。
- **求解太慢** → 适当降 `beam`（如 2^10）。
- **想更优解** → 提高 `beam` 与 `agent`。
- **显存不足**（6 GB 可用）→ 用 FP16（默认开）、调小 `num_walks` 或 `beam`。

---

## 6. 关键实现说明

### 魔方正确性（已交叉验证）
`cube/state.py` 用“几何旋转”定义 12 种 QTM 动作（把贴纸当格点+法线，转动某一层的格点），
并用 `kociemba` 作为 oracle 通过：
- 任意打乱 → 每色恰好 9 个；
- 打乱 + 逆序逆操作 → 回到复原态；
- 把状态喂给 `kociemba.solve`，其返回的解在我的魔方上也能复原（一致性）。

### 在线无监督数据
`cube/scramble.py::generate_random_walks` 每条游走从复原态随机走 `K_max` 步（同面不连续，避免 R R′ 自抵消），
记录**所有前缀** `(state, depth)`；打乱/验证集用 `generate_scrambles`（随机长度 20–60）。

### 束搜索（向量化，支持大 beam）
`solve/beam.py` 用“节点表”（状态 + 父指针 + 生成动作）实现，每层把所有束内状态各扩展 12 步，
用网络打分后保留预测距离最小的 `beam` 个；命中复原态时沿父指针回溯出动作串。
避免了逐候选 Python 循环，beam=2^12 下每个深度都是纯批量张量运算。

### 多智能体
每个 agent 用不同种子（独立数据流与初始化）训练，得到“各有偏差”的距离估计；
同一打乱各跑一遍，取**最短**解。

---

## 7. 环境已确认

```
python 3.12.10
torch 2.11.0+cu128
cuda 12.8
device_capability (12, 0)  ->  sm_120 (Blackwell)
GPU NVIDIA GeForce RTX 5060 Ti
```

校验依赖：`kociemba`（仅用于魔方正确性验证，不影响训练/求解）。
`requirements.txt` 已列出训练所需：`torch`、`numpy`，验证额外需要 `kociemba`。

---

## 8. 最终使用说明(1e10 模型为主)

### 统一主模型
从这以后**统一使用 `ckpt/agent_1e10.pt`**(1e10 训练,单 agent)。
- 所有求解入口(server.py / evaluate.py / run_demo.py)默认加载它。
- 早期 1e8 权重已归并到 `ckpt/not_primary/`,**不是主模型**(详见 `ckpt/README.md`)。
- `ckpt/agent0_train.pt` 仅为训练续训断点,非服务权重。

### 网页版 UI(推荐)
```powershell
python server.py
# 浏览器打开 http://127.0.0.1:8000/
```
功能:实时 3D 魔方、拖动旋转视角、一键打乱、AI 复原自动演示、打乱/复原速度分开可调(0.1–5s)、
beam 可选 1024 / 2048 / 4096 / 5120、两种独立模式(普通/自定义)、自定义可涂色并 3D 演示复原。

### 命令行
```powershell
python run_demo.py --scramble "R U R' U'"
python evaluate.py --num 50 --beam 4096
```

### 当前实测效果(1e10 模型)
| 配置 | 解出率 | 平均步长 | 单次耗时 |
|---|---|---|---|
| @beam 256, 40 深打乱 | 100% | 27.8 | 0.2s |
| @beam 4096, 40 深打乱 | **100%** | **23.3** | **2.34s** |
| @beam 5120, 1000 深打乱 | 进行中(100%,约22.8步) | — | ~2.7s |

