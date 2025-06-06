# R1 - Lightweight GRPO Training Framework

A lightweight Group-based Preference Optimization (GRPO) training framework built on Hugging Face transformers, optimized for reasoning model training.

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

### Enhanced Features

**🚀 What makes this "Enhanced"?**

Compared to standard GRPO implementations, this framework provides:

| Feature | Standard GRPO | Enhanced GRPO |
|---------|---------------|---------------|
| **Reward Functions** | 1-2 basic rewards | **6 specialized rewards** with customizable weights |
| **Training Monitoring** | Basic logging | **Real-time evaluation** every N steps |
| **Progress Tracking** | Step counter only | **Detailed statistics** + ETA estimation |
| **Evaluation** | End-of-training only | **Pre-training baseline** + continuous monitoring |
| **Reward Quality** | Simple scoring | **Multi-dimensional scoring** with bonuses |
| **Configurability** | Fixed parameters | **Flexible YAML configuration** |

**Key Enhancements:**

1. **Advanced Reward System**: 6 reward functions with intelligent bonuses
   - `accuracy`: Math verification + confidence detection
   - `format`: Structure validation + completeness scoring  
   - `reasoning_steps`: Logic quality assessment
   - `tag_count`: Smart tag validation with penalty system
   - `length`: Optimal response length control
   - `mathematical_rigor`: Mathematical reasoning evaluation

2. **Real-time Training Insights**:
   - Live evaluation during training
   - Progress estimation with ETA
   - Best metric tracking
   - Comprehensive training summary

3. **Modular Architecture**: Easy to extend with new reward functions and evaluation metrics

### Configuration

Basic example:
```yaml
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
dataset_name: open-r1/Mixture-of-Thoughts
learning_rate: 3.0e-6
num_train_epochs: 2
# All available reward functions
reward_funcs: [accuracy, format, reasoning_steps, tag_count, length, mathematical_rigor]
reward_weights: [4.0, 1.0, 2.0, 1.0, 0.8, 1.5]  # Customizable weights for each function
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

### 增强功能

**🚀 "增强"体现在哪里？**

相比标准GRPO实现，本框架提供：

| 功能 | 标准GRPO | 增强GRPO |
|------|---------|---------|
| **奖励函数** | 1-2个基础奖励 | **6个专门奖励**，可自定义权重 |
| **训练监控** | 基础日志 | **实时评估**，每N步一次 |
| **进度追踪** | 仅步数统计 | **详细统计** + ETA预估 |
| **评估方式** | 仅训练结束时 | **训练前基线** + 持续监控 |
| **奖励质量** | 简单打分 | **多维度打分**，带奖励机制 |
| **可配置性** | 固定参数 | **灵活YAML配置** |

**核心增强功能：**

1. **高级奖励系统**: 6个奖励函数，含智能奖励机制
   - `accuracy`: 数学验证 + 置信度检测
   - `format`: 结构验证 + 完整性评分
   - `reasoning_steps`: 逻辑质量评估
   - `tag_count`: 智能标签验证，含惩罚机制
   - `length`: 最优响应长度控制
   - `mathematical_rigor`: 数学推理评估

2. **实时训练洞察**:
   - 训练中实时评估
   - 进度预估和ETA
   - 最佳指标追踪
   - 全面训练总结

3. **模块化架构**: 易于扩展新奖励函数和评估指标

### 配置示例

基础配置:
```yaml
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
dataset_name: open-r1/Mixture-of-Thoughts
learning_rate: 3.0e-6
num_train_epochs: 2
# 所有可用的奖励函数
reward_funcs: [accuracy, format, reasoning_steps, tag_count, length, mathematical_rigor]
reward_weights: [4.0, 1.0, 2.0, 1.0, 0.8, 1.5]  # 每个函数可自定义权重
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
