<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>R1 - Lightweight GRPO Inference Model Training Framework</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; }
        .lang-toggle { position: fixed; top: 20px; right: 20px; z-index: 1000; }
        .lang-btn { 
            background: #0969da; 
            color: white; 
            border: none; 
            padding: 8px 16px; 
            border-radius: 6px; 
            cursor: pointer; 
            margin: 0 2px;
            font-size: 14px;
        }
        .lang-btn:hover { background: #0860ca; }
        .lang-btn.active { background: #2da44e; }
        .lang-content { display: none; }
        .lang-content.active { display: block; }
        .emoji { font-size: 1.2em; }
    </style>
</head>
<body>

<div class="lang-toggle">
    <button class="lang-btn active" onclick="switchLang('en')">English</button>
    <button class="lang-btn" onclick="switchLang('zh')">中文</button>
</div>

<!-- English Version -->
<div id="lang-en" class="lang-content active">

# R1 - Lightweight GRPO Inference Model Training Framework

🚀 **GRPO (Group-based Preference Optimization) Training Framework Optimized for Inference Tasks**

Built on the Hugging Face ecosystem, focused on training high-quality inference models that excel at mathematical reasoning, logical reasoning, and other complex cognitive tasks.

## ✨ Core Features

### 🧠 Enhanced GRPO Training
- **Complete GRPO Implementation**: Enhanced trainer based on `trl.GRPOTrainer`
- **Real-time Evaluation System**: Automatic model performance evaluation during training
- **Reject Sampling Mechanism**: Intelligent resampling to improve generation quality
- **Modular Design**: Pluggable reward functions and configuration system

### 🎯 Multi-dimensional Reward Functions
- **Answer Accuracy** (`accuracy`): Precise verification using `math-verify`
- **Format Compliance** (`format`): Validates `<think>` and `<answer>` tag structure
- **Reasoning Steps** (`reasoning_steps`): Evaluates logical reasoning process quality
- **Tag Completeness** (`tag_count`): Ensures correct tag usage
- **Text Length** (`length`): Controls reasonable generation length
- **Mathematical Rigor** (`mathematical_rigor`): Specialized evaluation for mathematical reasoning

### ⚙️ Flexible Configuration System
- **YAML-Driven**: All training parameters managed through configuration files
- **Adjustable Weights**: Independent weight settings for each reward function
- **Dynamic Loading**: Smart separation of training and model parameters
- **Environment Adaptation**: Supports multiple accelerators and distributed training

### 🔧 Developer-Friendly
- **Complete Toolchain**: Logging, formatting, data processing utilities
- **Testing Framework**: Comprehensive functional testing and validation
- **Debug Support**: Detailed logging and debug mode

## 📦 Quick Start

### Environment Setup
```bash
# Requires Python 3.10.9+
conda create -n r1 python=3.10
conda activate r1

# Clone the project
git clone <repository-url>
cd r1

# Install dependencies (PyTorch ecosystem based)
pip install -e .
```

### Basic Training
```bash
# Quick start with preset configuration
python train_enhanced_demo.py

# Custom configuration file
python train_enhanced_demo.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_enhanced_demo.yaml

# Debug mode (no actual training)
python train_enhanced_demo.py --debug --dry-run
```

### Advanced Training
```bash
# Using native TRL interface
python src/r1/grpo.py --config <config_file>

# SFT pre-training
python src/r1/sft.py --config <sft_config_file>
```

## 📋 Configuration Details

### Basic Configuration Example
```yaml
# Model settings
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
torch_dtype: bfloat16

# Dataset configuration
dataset_name: open-r1/Mixture-of-Thoughts
dataset_prompt_column: prompt

# Training parameters
learning_rate: 3.0e-6
num_train_epochs: 2
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
max_completion_length: 2048

# Basic reward functions
reward_funcs:
  - accuracy
  - format
  - tag_count
reward_weights: [1.0, 1.0, 1.0]
```

### Enhanced Features Configuration
```yaml
# Real-time evaluation 🔍
eval_steps: 50                    # Evaluate every 50 steps
eval_on_start: true              # Evaluate before training
max_eval_samples: 100            # Evaluation sample limit
eval_dataset_name: gsm8k         # Independent evaluation dataset

# Reject sampling 🎲
use_reject_sampling: true        # Enable reject sampling
max_resample_attempts: 3         # Maximum retry attempts
reject_sampling_threshold: 0.6   # Quality threshold

# Multi-dimensional rewards 🏆
reward_funcs:
  - accuracy           # Answer accuracy
  - format            # Format compliance
  - reasoning_steps   # Reasoning quality
  - mathematical_rigor # Mathematical rigor
reward_weights: [4.0, 1.0, 2.0, 1.5]  # Weight configuration

# Performance optimization ⚡
torch_empty_cache_steps: 5       # GPU cache cleanup frequency
gradient_checkpointing: true     # Gradient checkpointing
use_liger_kernel: true          # Liger optimization kernel

# Monitoring & logging 📊
wandb_project: R1-Training
log_completion_details: true
save_completions: false
debug_mode: true
```

## 🏗️ Project Architecture

```
r1/
├── 📁 src/r1/                    # Core modules
│   ├── 📄 configs.py            # Configuration classes (16KB, 448 lines)
│   ├── 📄 enhanced_grpo_trainer.py  # Enhanced GRPO trainer (10KB, 284 lines)
│   ├── 📄 trainer.py            # Trainer interface
│   ├── 📄 rewards.py            # Reward function library (13KB, 354 lines)
│   ├── 📄 callbacks.py          # Training callbacks
│   ├── 📄 grpo.py               # GRPO training script (6.5KB, 183 lines)
│   ├── 📄 sft.py                # SFT training script (6KB, 172 lines)
│   └── 📁 utils/                # Utility library
│       ├── 📄 logging.py        # Logging system (4.6KB, 152 lines)
│       ├── 📄 formatting.py     # Text formatting (3.6KB, 152 lines)
│       ├── 📄 data.py           # Data processing (2.6KB, 66 lines)
│       ├── 📄 model_utils.py    # Model utilities (1.6KB, 43 lines)
│       ├── 📄 import_utils.py   # Dependency checking (1.4KB, 55 lines)
│       ├── 📄 callbacks.py      # Callback utilities (938B, 30 lines)
│       └── 📄 wandb_logging.py  # W&B integration (480B, 14 lines)
├── 📁 recipes/                  # Configuration templates
│   └── 📁 Qwen2.5-1.5B-Instruct/grpo/
│       ├── 📄 config_demo.yaml          # Basic configuration
│       └── 📄 config_enhanced_demo.yaml # Enhanced configuration
├── 📄 train_enhanced_demo.py     # Training demo script (3KB, 92 lines)
├── 📄 test_enhanced_features.py  # Feature testing script (4.8KB, 164 lines)
└── 📄 setup.py                  # Package configuration (3.7KB, 125 lines)
```

## 📊 Reward Functions Explained

### 🎯 accuracy - Answer Accuracy
- Uses `math-verify` and `latex2sympy2_extended` for precise verification
- Supports multiple mathematical expression formats
- Includes confidence-based reward mechanism

### 📝 format - Format Compliance
- Validates completeness of `<think>` and `<answer>` tags
- Checks structural reasonableness of thinking process and answers
- Rewards clear logical segmentation

### 🧠 reasoning_steps - Reasoning Steps
- Detects step indicators (Step 1, First, Then, etc.)
- Evaluates use of logical transition words
- Supports both English and Chinese reasoning patterns

### 🏷️ tag_count - Tag Completeness
- Precisely counts tag quantities
- Penalizes excessive or missing tags
- Ensures format consistency

### 📏 length - Text Length
- Gaussian distribution-based length rewards
- Configurable target length ranges
- Avoids overly short or long generations

### 🔬 mathematical_rigor - Mathematical Rigor
- Detects mathematical terms and symbols
- Evaluates logical structure completeness
- Optimized specifically for mathematical reasoning tasks

## 🚀 Advanced Features

### Real-time Evaluation System
```python
# Automatically evaluate model during training
trainer = EnhancedGRPOTrainer(
    eval_dataset=eval_data,
    eval_steps=50,
    eval_on_start=True,
    max_eval_samples=100
)
```

### Reject Sampling Mechanism
```python
# Intelligent resampling to improve generation quality
completions = trainer._generate_completions_with_reject_sampling(
    prompts=prompts,
    max_resample_attempts=3,
    reject_sampling_threshold=0.6
)
```

### Modular Reward Functions
```python
# Create custom reward function combinations
from src.r1.rewards import get_reward_calculator

calculator = get_reward_calculator(
    reward_funcs=['accuracy', 'format', 'mathematical_rigor'],
    reward_weights=[4.0, 1.0, 2.0]
)
```

## 📈 Monitoring & Debugging

### Weights & Biases Integration
```yaml
# Complete experiment tracking
wandb_project: R1-Experiments
wandb_entity: your-team
run_name: qwen-grpo-v1
report_to: [wandb]
```

### Detailed Logging System
```python
# Multi-level logging
from src.r1.utils.logging import setup_logger, TrainingLogger

logger = setup_logger("r1_training", level=logging.DEBUG)
training_logger = TrainingLogger(log_file="training.log")
```

### Feature Testing
```bash
# Run comprehensive feature tests
python test_enhanced_features.py

# Example output:
# 🎯 Overall Result: 🎉 All tests passed!
```

## 🛠️ Technology Stack

### Core Dependencies
- **PyTorch 2.6.0** - Deep learning framework
- **Transformers 4.52.3** - Model library
- **TRL 0.18.0** - Reinforcement learning training
- **Accelerate 1.4.0** - Distributed training
- **DeepSpeed 0.16.8** - Memory optimization

### Specialized Tools
- **math-verify 0.5.2** - Mathematical verification
- **latex2sympy2_extended** - LaTeX parsing
- **liger-kernel** - GPU optimization
- **bitsandbytes** - Quantized training

## 📝 Development Guide

### Adding New Reward Functions
```python
def custom_reward(completions, **kwargs) -> list[float]:
    """Custom reward function"""
    # Implement evaluation logic
    return scores

# Register to system
REWARD_FUNCS_REGISTRY["custom"] = custom_reward
```

### Custom Configuration
```python
from src.r1.configs import GRPOScriptArguments

@dataclass
class CustomConfig(GRPOScriptArguments):
    custom_param: float = field(default=1.0)
```

## 🤝 Contributing

1. **Fork the project** and create a feature branch
2. **Run tests** to ensure functionality: `python test_enhanced_features.py`
3. **Follow code style** using black and isort formatting
4. **Submit PR** with description of changes

## 📄 License

Apache License 2.0 - See LICENSE file for details

---

> 💡 **Lightweight GRPO framework focused on inference tasks - making training high-quality reasoning models simple and efficient!**

</div>

<!-- Chinese Version -->
<div id="lang-zh" class="lang-content">

# R1 - 轻量级GRPO推理模型训练框架

🚀 **专为推理任务优化的GRPO (Group-based Preference Optimization) 训练框架**

基于Hugging Face生态系统构建，专注于训练高质量的推理模型，支持数学推理、逻辑推理等复杂任务。

## ✨ 核心特性

### 🧠 增强版GRPO训练
- **完整GRPO实现**：基于`trl.GRPOTrainer`的增强版训练器
- **实时评估系统**：训练过程中自动评估模型性能
- **拒绝采样机制**：智能重采样提升生成质量
- **模块化设计**：可插拔的奖励函数和配置系统

### 🎯 多维度奖励函数
- **答案准确性** (`accuracy`)：基于`math-verify`的精确验证
- **格式规范性** (`format`)：检查`<think>`和`<answer>`标签结构
- **推理步骤** (`reasoning_steps`)：评估逻辑推理过程质量
- **标签完整性** (`tag_count`)：确保正确的标签使用
- **文本长度** (`length`)：控制生成长度的合理性
- **数学严谨性** (`mathematical_rigor`)：专门针对数学推理的评估

### ⚙️ 灵活配置系统
- **YAML驱动**：所有训练参数通过配置文件管理
- **权重可调**：每个奖励函数支持独立权重设置
- **动态加载**：智能分离训练参数和模型参数
- **环境适配**：支持多种加速器和分布式训练

### 🔧 开发者友好
- **完整工具链**：日志、格式化、数据处理等实用工具
- **测试框架**：全面的功能测试和验证
- **调试支持**：详细的日志记录和调试模式

## 📦 快速开始

### 环境准备
```bash
# 要求Python 3.10.9+
conda create -n r1 python=3.10
conda activate r1

# 克隆项目
git clone <repository-url>
cd r1

# 安装依赖（基于PyTorch生态系统）
pip install -e .
```

### 基础训练
```bash
# 使用预设配置快速开始
python train_enhanced_demo.py

# 自定义配置文件
python train_enhanced_demo.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_enhanced_demo.yaml

# 调试模式（不执行实际训练）
python train_enhanced_demo.py --debug --dry-run
```

### 高级训练
```bash
# 使用原生TRL接口
python src/r1/grpo.py --config <config_file>

# SFT预训练
python src/r1/sft.py --config <sft_config_file>
```

## 📋 配置详解

### 基础配置示例
```yaml
# 模型设置
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
torch_dtype: bfloat16

# 数据集配置
dataset_name: open-r1/Mixture-of-Thoughts
dataset_prompt_column: prompt

# 训练参数
learning_rate: 3.0e-6
num_train_epochs: 2
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
max_completion_length: 2048

# 基础奖励函数
reward_funcs:
  - accuracy
  - format
  - tag_count
reward_weights: [1.0, 1.0, 1.0]
```

### 增强功能配置
```yaml
# 实时评估 🔍
eval_steps: 50                    # 每50步评估一次
eval_on_start: true              # 训练前评估
max_eval_samples: 100            # 评估样本数限制
eval_dataset_name: gsm8k         # 独立评估数据集

# 拒绝采样 🎲
use_reject_sampling: true        # 启用拒绝采样
max_resample_attempts: 3         # 最大重试次数
reject_sampling_threshold: 0.6   # 质量阈值

# 多维度奖励 🏆
reward_funcs:
  - accuracy           # 答案准确性
  - format            # 格式规范
  - reasoning_steps   # 推理质量
  - mathematical_rigor # 数学严谨性
reward_weights: [4.0, 1.0, 2.0, 1.5]  # 权重配置

# 性能优化 ⚡
torch_empty_cache_steps: 5       # GPU缓存清理频率
gradient_checkpointing: true     # 梯度检查点
use_liger_kernel: true          # Liger优化内核

# 监控日志 📊
wandb_project: R1-Training
log_completion_details: true
save_completions: false
debug_mode: true
```

## 🏗️ 项目架构

```
r1/
├── 📁 src/r1/                    # 核心模块
│   ├── 📄 configs.py            # 配置类定义（16KB, 448行）
│   ├── 📄 enhanced_grpo_trainer.py  # 增强版GRPO训练器（10KB, 284行）
│   ├── 📄 trainer.py            # 训练器接口
│   ├── 📄 rewards.py            # 奖励函数库（13KB, 354行）
│   ├── 📄 callbacks.py          # 训练回调
│   ├── 📄 grpo.py               # GRPO训练脚本（6.5KB, 183行）
│   ├── 📄 sft.py                # SFT训练脚本（6KB, 172行）
│   └── 📁 utils/                # 工具库
│       ├── 📄 logging.py        # 日志系统（4.6KB, 152行）
│       ├── 📄 formatting.py     # 文本格式化（3.6KB, 152行）
│       ├── 📄 data.py           # 数据处理（2.6KB, 66行）
│       ├── 📄 model_utils.py    # 模型工具（1.6KB, 43行）
│       ├── 📄 import_utils.py   # 依赖检查（1.4KB, 55行）
│       ├── 📄 callbacks.py      # 回调工具（938B, 30行）
│       └── 📄 wandb_logging.py  # W&B集成（480B, 14行）
├── 📁 recipes/                  # 配置模板
│   └── 📁 Qwen2.5-1.5B-Instruct/grpo/
│       ├── 📄 config_demo.yaml          # 基础配置
│       └── 📄 config_enhanced_demo.yaml # 增强配置
├── 📄 train_enhanced_demo.py     # 训练演示脚本（3KB, 92行）
├── 📄 test_enhanced_features.py  # 功能测试脚本（4.8KB, 164行）
└── 📄 setup.py                  # 包配置（3.7KB, 125行）
```

## 📊 奖励函数详解

### 🎯 accuracy - 答案准确性
- 使用`math-verify`和`latex2sympy2_extended`进行精确验证
- 支持多种数学表达式格式
- 包含置信度奖励机制

### 📝 format - 格式规范性
- 验证`<think>`和`<answer>`标签完整性
- 检查思考过程和答案的结构合理性
- 奖励清晰的逻辑分段

### 🧠 reasoning_steps - 推理步骤
- 检测步骤指示词（Step 1, First, Then等）
- 评估逻辑过渡词的使用
- 支持中英文推理模式

### 🏷️ tag_count - 标签完整性
- 精确计算标签数量
- 惩罚多余或缺失的标签
- 确保格式一致性

### 📏 length - 文本长度
- 基于高斯分布的长度奖励
- 可配置目标长度范围
- 避免过短或过长的生成

### 🔬 mathematical_rigor - 数学严谨性
- 检测数学术语和符号
- 评估逻辑结构完整性
- 专为数学推理任务优化

## 🚀 高级功能

### 实时评估系统
```python
# 自动在训练过程中评估模型
trainer = EnhancedGRPOTrainer(
    eval_dataset=eval_data,
    eval_steps=50,
    eval_on_start=True,
    max_eval_samples=100
)
```

### 拒绝采样机制
```python
# 智能重采样提升生成质量
completions = trainer._generate_completions_with_reject_sampling(
    prompts=prompts,
    max_resample_attempts=3,
    reject_sampling_threshold=0.6
)
```

### 模块化奖励函数
```python
# 创建自定义奖励函数组合
from src.r1.rewards import get_reward_calculator

calculator = get_reward_calculator(
    reward_funcs=['accuracy', 'format', 'mathematical_rigor'],
    reward_weights=[4.0, 1.0, 2.0]
)
```

## 📈 监控与调试

### Weights & Biases集成
```yaml
# 完整的实验跟踪
wandb_project: R1-Experiments
wandb_entity: your-team
run_name: qwen-grpo-v1
report_to: [wandb]
```

### 详细日志系统
```python
# 多级别日志记录
from src.r1.utils.logging import setup_logger, TrainingLogger

logger = setup_logger("r1_training", level=logging.DEBUG)
training_logger = TrainingLogger(log_file="training.log")
```

### 功能测试
```bash
# 运行全面的功能测试
python test_enhanced_features.py

# 输出示例：
# 🎯 总体结果: 🎉 所有测试通过!
```

## 🛠️ 技术栈

### 核心依赖
- **PyTorch 2.6.0** - 深度学习框架
- **Transformers 4.52.3** - 模型库
- **TRL 0.18.0** - 强化学习训练
- **Accelerate 1.4.0** - 分布式训练
- **DeepSpeed 0.16.8** - 内存优化

### 专业工具
- **math-verify 0.5.2** - 数学验证
- **latex2sympy2_extended** - LaTeX解析
- **liger-kernel** - GPU优化
- **bitsandbytes** - 量化训练

## 📝 开发指南

### 添加新奖励函数
```python
def custom_reward(completions, **kwargs) -> list[float]:
    """自定义奖励函数"""
    # 实现评估逻辑
    return scores

# 注册到系统
REWARD_FUNCS_REGISTRY["custom"] = custom_reward
```

### 自定义配置
```python
from src.r1.configs import GRPOScriptArguments

@dataclass
class CustomConfig(GRPOScriptArguments):
    custom_param: float = field(default=1.0)
```

## 🤝 贡献指南

1. **Fork项目** 并创建特性分支
2. **运行测试** 确保功能正常：`python test_enhanced_features.py`
3. **遵循代码风格** 使用black和isort格式化
4. **提交PR** 并描述变更内容

## 📄 许可证

Apache License 2.0 - 详见LICENSE文件

---

> 💡 **专注于推理任务的轻量级GRPO框架，让训练高质量推理模型变得简单高效！**

</div>

<script>
function switchLang(lang) {
    // Hide all language content
    document.querySelectorAll('.lang-content').forEach(el => {
        el.classList.remove('active');
    });
    
    // Show selected language content
    document.getElementById('lang-' + lang).classList.add('active');
    
    // Update button states
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Update page title and lang attribute
    if (lang === 'zh') {
        document.title = 'R1 - 轻量级GRPO推理模型训练框架';
        document.documentElement.lang = 'zh-CN';
    } else {
        document.title = 'R1 - Lightweight GRPO Inference Model Training Framework';
        document.documentElement.lang = 'en';
    }
}
</script>

</body>
</html>
