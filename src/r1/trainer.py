"""
GRPO训练器模块
提供增强版GRPO训练器的简化接口
"""

from .enhanced_grpo_trainer import EnhancedGRPOTrainer, ProgressCallback

# 为了向后兼容，提供简化的别名
GRPOTrainer = EnhancedGRPOTrainer

__all__ = ['GRPOTrainer', 'EnhancedGRPOTrainer', 'ProgressCallback'] 