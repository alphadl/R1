# R1 - Lightweight GRPO Training Framework

A lightweight Group-based Preference Optimization (GRPO) training framework built on Hugging Face transformers, optimized for inference model training.

[English](#english) | [中文](#中文)

---

## English

### Quick Start

```bash
# Setup environment
conda create -n r1 python=3.10
conda activate r1
pip install -e .

# Run training demo
python train_enhanced_demo.py
```

### Features

- **Enhanced GRPO Trainer**: Extended from `trl.GRPOTrainer` with real-time evaluation
- **Modular Reward System**: Multiple reward functions (accuracy, format, reasoning, etc.)
- **YAML Configuration**: Flexible training parameter management
- **Built-in Utilities**: Logging, formatting, and data processing tools

### Configuration

Basic example:
```yaml
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
dataset_name: open-r1/Mixture-of-Thoughts
learning_rate: 3.0e-6
num_train_epochs: 2
reward_funcs: [accuracy, format, tag_count]
reward_weights: [1.0, 1.0, 1.0]
```

### Project Structure

```
src/r1/
├── enhanced_grpo_trainer.py    # Main trainer implementation
├── rewards.py                  # Reward function library
├── configs.py                  # Configuration classes
├── grpo.py                     # Training script
└── utils/                      # Utility modules
    ├── logging.py
    ├── formatting.py
    └── data.py
```

### Reward Functions

| Function | Description |
|----------|-------------|
| `accuracy` | Answer correctness using math-verify |
| `format` | Tag structure validation |
| `reasoning_steps` | Logical reasoning quality |
| `tag_count` | Required tag presence |
| `length` | Response length control |
| `mathematical_rigor` | Math reasoning evaluation |

### Training Scripts

- `train_enhanced_demo.py` - Demo with enhanced features
- `src/r1/grpo.py` - Standard GRPO training
- `src/r1/sft.py` - Supervised fine-tuning

---

## 中文

### 快速开始

```bash
# 环境配置
conda create -n r1 python=3.10
conda activate r1
pip install -e .

# 运行训练演示
python train_enhanced_demo.py
```

### 主要功能

- **增强GRPO训练器**: 基于`trl.GRPOTrainer`扩展，支持实时评估
- **模块化奖励系统**: 多种奖励函数（准确性、格式、推理等）
- **YAML配置管理**: 灵活的训练参数配置
- **内置工具库**: 日志记录、格式化、数据处理工具

### 配置示例

基础配置:
```yaml
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
dataset_name: open-r1/Mixture-of-Thoughts
learning_rate: 3.0e-6
num_train_epochs: 2
reward_funcs: [accuracy, format, tag_count]
reward_weights: [1.0, 1.0, 1.0]
```

### 项目结构

```
src/r1/
├── enhanced_grpo_trainer.py    # 主要训练器实现
├── rewards.py                  # 奖励函数库
├── configs.py                  # 配置类
├── grpo.py                     # 训练脚本
└── utils/                      # 工具模块
    ├── logging.py
    ├── formatting.py
    └── data.py
```

### 奖励函数

| 函数 | 说明 |
|------|------|
| `accuracy` | 使用math-verify验证答案正确性 |
| `format` | 标签结构验证 |
| `reasoning_steps` | 逻辑推理质量评估 |
| `tag_count` | 必需标签存在性检查 |
| `length` | 响应长度控制 |
| `mathematical_rigor` | 数学推理评估 |

### 训练脚本

- `train_enhanced_demo.py` - 增强功能演示
- `src/r1/grpo.py` - 标准GRPO训练
- `src/r1/sft.py` - 监督微调

## License

Apache-2.0
