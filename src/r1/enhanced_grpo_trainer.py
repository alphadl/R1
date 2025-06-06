# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Enhanced GRPO trainer with real-time evaluation and detailed logging."""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import torch
from transformers import TrainerCallback, TrainerState, TrainerControl
from trl import GRPOTrainer
from .rewards import get_reward_calculator


logger = logging.getLogger(__name__)


@dataclass 
class TrainingStats:
    """Training statistics tracking"""
    
    # Training statistics
    start_time: float = field(default_factory=time.time)
    total_steps: int = 0
    completed_steps: int = 0
    total_samples_processed: int = 0
    step_times: List[float] = field(default_factory=list)
    
    # Performance metrics
    best_reward_score: float = 0.0
    best_eval_loss: float = float('inf')
    recent_reward_scores: List[float] = field(default_factory=list)
    recent_eval_losses: List[float] = field(default_factory=list)


class EnhancedGRPOTrainer(GRPOTrainer):
    """Enhanced GRPO trainer with real-time evaluation and detailed logging."""
    
    def __init__(self, *args, **kwargs):
        # Extract custom arguments
        self.eval_steps = kwargs.pop("eval_steps", 100)
        self.log_completion_details = kwargs.pop("log_completion_details", True)
        self.reward_config = kwargs.pop("reward_config", {})
        
        super().__init__(*args, **kwargs)
        
        # Initialize tracking
        self.training_stats = TrainingStats()
        self.reward_calculator = get_reward_calculator(self.reward_config)
        
        # Pre-training evaluation
        if self.eval_dataset is not None:
            logger.info("Running pre-training evaluation...")
            self._evaluate_before_training()

    def train(self, *args, **kwargs):
        """Override training method to add real-time evaluation functionality"""
        self.training_stats.total_steps = self.args.max_steps or (
            len(self.get_train_dataloader()) * self.args.num_train_epochs
        )
        self.training_stats.start_time = time.time()
        
        logger.info(f"Starting enhanced GRPO training...")
        logger.info(f"  Total steps: {self.training_stats.total_steps}")
        logger.info(f"  Evaluation interval: {self.eval_steps} steps")
        logger.info(f"  Reward configuration: {self.reward_config}")
        
        result = super().train(*args, **kwargs)
        
        # Final summary
        self._log_training_summary()
        
        return result
    
    def _evaluate_before_training(self):
        """Evaluate model before training starts"""
        eval_results = self.evaluate()
        logger.info(f"Pre-training evaluation - Loss: {eval_results.get('eval_loss', 'N/A')}")
        
    def compute_rewards(self, completions, **kwargs):
        """Compute evaluation reward scores"""
        if not completions:
            return []
            
        try:
            # Use the reward calculator to compute scores
            rewards = self.reward_calculator.calculate_reward(completions, **kwargs)
            
            # Format reward
            format_scores = []
            for completion in completions:
                # Reasoning step reward
                step_count = completion.count('<thinking>') + completion.count('<reasoning>')
                reasoning_score = min(step_count / 4.0, 1.0)  # Normalize to 0-1
                format_scores.append(reasoning_score)
            
            # Combine rewards
            combined_rewards = []
            for i, completion in enumerate(completions):
                base_reward = rewards[i] if i < len(rewards) else 0.0
                format_reward = format_scores[i] if i < len(format_scores) else 0.0
                combined_reward = (base_reward + format_reward) / 2.0
                combined_rewards.append(combined_reward)
                
            return combined_rewards
            
        except Exception as e:
            logger.warning(f"Error computing rewards: {e}")
            return [0.0] * len(completions)
    
    def _log_training_progress(self, logs: Dict[str, Any]):
        """Log training progress and statistics"""
        current_step = logs.get('step', 0)
        self.training_stats.completed_steps = current_step
        
        # Calculate estimated remaining time
        elapsed_time = time.time() - self.training_stats.start_time
        if current_step > 0:
            avg_time_per_step = elapsed_time / current_step
            remaining_steps = self.training_stats.total_steps - current_step
            estimated_remaining = avg_time_per_step * remaining_steps
            
            logger.info(f"Step {current_step}/{self.training_stats.total_steps} "
                       f"({current_step/self.training_stats.total_steps*100:.1f}%) - "
                       f"Est. remaining: {self._format_time(estimated_remaining)}")
        
        # Regular evaluation
        if current_step % self.eval_steps == 0 and current_step > 0:
            self._perform_real_time_evaluation()
            
            # Log generation statistics
            if hasattr(self, 'generation_stats'):
                logger.info(f"Generation stats: {self.generation_stats}")
    
    def _perform_real_time_evaluation(self):
        """Perform real-time evaluation during training"""
        if self.eval_dataset is None:
            return
            
        try:
            eval_results = self.evaluate()
            eval_loss = eval_results.get('eval_loss', float('inf'))
            
            # Track best results
            if eval_loss < self.training_stats.best_eval_loss:
                self.training_stats.best_eval_loss = eval_loss
                logger.info(f"🎉 New best evaluation loss: {eval_loss:.4f}")
            
            # Store recent results
            self.training_stats.recent_eval_losses.append(eval_loss)
            if len(self.training_stats.recent_eval_losses) > 10:
                self.training_stats.recent_eval_losses.pop(0)
                
        except Exception as e:
            logger.warning(f"Real-time evaluation failed: {e}")

    def _log_step_time(self, step_time: float):
        """Record step time"""
        self.training_stats.step_times.append(step_time)
        if len(self.training_stats.step_times) > 100:  # Keep last 100 steps
            self.training_stats.step_times.pop(0)
        
        # Let the trainer log progress
        if len(self.training_stats.step_times) % self.eval_steps == 0:
            avg_step_time = sum(self.training_stats.step_times) / len(self.training_stats.step_times)
            logger.info(f"Average step time: {avg_step_time:.2f}s")
    
    def _log_training_summary(self):
        """Log training summary at the end"""
        total_time = time.time() - self.training_stats.start_time
        avg_step_time = sum(self.training_stats.step_times) / len(self.training_stats.step_times) if self.training_stats.step_times else 0
        
        logger.info("="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total training time: {self._format_time(total_time)}")
        logger.info(f"Steps completed: {self.training_stats.completed_steps}/{self.training_stats.total_steps}")
        logger.info(f"Average step time: {avg_step_time:.2f}s")
        logger.info(f"Best evaluation loss: {self.training_stats.best_eval_loss:.4f}")
        
        if self.training_stats.recent_reward_scores:
            avg_reward = sum(self.training_stats.recent_reward_scores) / len(self.training_stats.recent_reward_scores)
            logger.info(f"Average reward score: {avg_reward:.4f}")
            
        logger.info("="*60)

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary statistics"""
        total_time = time.time() - self.training_stats.start_time
        avg_step_time = sum(self.training_stats.step_times) / len(self.training_stats.step_times) if self.training_stats.step_times else 0
        
        return {
            "total_training_time": total_time,
            "completed_steps": self.training_stats.completed_steps,
            "total_steps": self.training_stats.total_steps,
            "average_step_time": avg_step_time,
            "best_eval_loss": self.training_stats.best_eval_loss,
            "best_reward_score": self.training_stats.best_reward_score,
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time display"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


class ProgressCallback(TrainerCallback):
    """Progress callback for detailed training logging"""
    
    def __init__(self, trainer: EnhancedGRPOTrainer):
        self.trainer = trainer
        self.step_start_time = None
    
    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.step_start_time = time.time()
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            self.trainer._log_step_time(step_time)
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        logs = kwargs.get('logs', {})
        self.trainer._log_training_progress(logs) 