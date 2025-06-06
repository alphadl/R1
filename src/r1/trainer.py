"""
GRPO training module
Provides a simplified interface for the enhanced GRPO trainer
"""

from .enhanced_grpo_trainer import EnhancedGRPOTrainer, ProgressCallback

# For backward compatibility, provide simplified aliases
from .enhanced_grpo_trainer import EnhancedGRPOTrainer as GRPOTrainer

__all__ = ['GRPOTrainer', 'EnhancedGRPOTrainer', 'ProgressCallback'] 