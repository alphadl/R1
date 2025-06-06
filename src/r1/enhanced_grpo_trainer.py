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

"""Enhanced GRPO Trainer with real-time evaluation and reject sampling."""

import time
from typing import Any, Dict, List, Optional, Union
import torch
import numpy as np
from tqdm import tqdm
from transformers import TrainerCallback
from trl import GRPOTrainer
import wandb


class EnhancedGRPOTrainer(GRPOTrainer):
    """
    Enhanced GRPO Trainer with additional features:
    - Real-time evaluation during training
    - Reject sampling for better generation quality
    - Detailed logging and time estimation
    - Modularized generation and scoring
    """
    
    def __init__(
        self,
        *args,
        eval_dataset=None,
        eval_steps: int = 100,
        eval_on_start: bool = True,
        max_resample_attempts: int = 3,
        reject_sampling_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.eval_dataset = eval_dataset
        self.eval_steps = eval_steps
        self.eval_on_start = eval_on_start
        self.max_resample_attempts = max_resample_attempts
        self.reject_sampling_threshold = reject_sampling_threshold
        
        # 训练统计信息
        self.training_stats = {
            'start_time': None,
            'step_times': [],
            'generation_lengths': [],
            'reward_scores': [],
            'eval_scores': []
        }
    
    def train(self, *args, **kwargs):
        """重写训练方法，添加实时评估功能"""
        self.training_stats['start_time'] = time.time()
        
        # 开始前评估
        if self.eval_on_start and self.eval_dataset is not None:
            self._evaluate_model()
        
        return super().train(*args, **kwargs)
    
    def _generate_completions_with_reject_sampling(
        self, 
        prompts: List[str],
        **generation_kwargs
    ) -> List[str]:
        """Generate high-quality responses using reject sampling"""
        completions = []
        
        for prompt in tqdm(prompts, desc="Generating responses"):
            best_completion = None
            best_score = -float('inf')
            
            for attempt in range(self.max_resample_attempts):
                # Generate candidate response
                candidate = self._generate_single_completion(prompt, **generation_kwargs)
                
                # Quick evaluation of candidate response quality
                score = self._quick_score_completion(candidate)
                
                if score > best_score:
                    best_score = score
                    best_completion = candidate
                
                # Accept response if threshold is reached
                if score >= self.reject_sampling_threshold:
                    break
            
            completions.append(best_completion)
            
        return completions
    
    def _generate_single_completion(self, prompt: str, **kwargs) -> str:
        """Generate single response"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids.to(self.model.device),
                **kwargs
            )
        
        completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return completion[len(prompt):]  # Only return generated part
    
    def _quick_score_completion(self, completion: str) -> float:
        """Quick evaluation of response quality (simplified reward function)"""
        score = 0.0
        
        # Check format
        if "<think>" in completion and "</think>" in completion:
            score += 0.3
        if "<answer>" in completion and "</answer>" in completion:
            score += 0.3
        
        # Check reasonable length
        if 50 < len(completion) < 2000:
            score += 0.2
        
        # Check for reasoning steps
        if any(keyword in completion.lower() for keyword in ["first", "then", "next", "finally", "step"]):
            score += 0.2
        
        return score
    
    def _evaluate_model(self) -> Dict[str, float]:
        """Evaluate model on evaluation dataset"""
        if self.eval_dataset is None:
            return {}
        
        self.model.eval()
        eval_results = {}
        
        eval_prompts = self.eval_dataset[:min(100, len(self.eval_dataset))]  # Limit evaluation sample count
        
        with torch.no_grad():
            # Generate responses
            completions = []
            for prompt in tqdm(eval_prompts, desc="Evaluation generation"):
                completion = self._generate_single_completion(
                    prompt,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True
                )
                completions.append(completion)
            
            # Calculate reward scores
            rewards = self._compute_eval_rewards(completions, eval_prompts)
            
            # Aggregate results
            eval_results = {
                'eval_mean_reward': np.mean(rewards),
                'eval_std_reward': np.std(rewards),
                'eval_completion_length': np.mean([len(c) for c in completions])
            }
        
        self.model.train()
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log(eval_results)
        
        self.training_stats['eval_scores'].append(eval_results['eval_mean_reward'])
        
        return eval_results
    
    def _compute_eval_rewards(self, completions: List[str], prompts: List[str]) -> List[float]:
        """计算评估奖励分数"""
        rewards = []
        
        for completion in completions:
            reward = 0.0
            
            # 格式奖励
            if "<think>" in completion and "</think>" in completion:
                reward += 0.25
            if "<answer>" in completion and "</answer>" in completion:
                reward += 0.25
            
            # 推理步骤奖励
            reasoning_indicators = len([word for word in ["first", "then", "next", "finally", "step"] 
                                     if word in completion.lower()])
            reward += min(0.5, reasoning_indicators * 0.1)
            
            rewards.append(reward)
        
        return rewards
    
    def log_training_progress(self, step: int, logs: Dict[str, Any]):
        """记录训练进度和统计信息"""
        current_time = time.time()
        if self.training_stats['start_time'] is not None:
            elapsed_time = current_time - self.training_stats['start_time']
            
            # 计算预估剩余时间
            if step > 0:
                avg_step_time = elapsed_time / step
                remaining_steps = self.args.max_steps - step if self.args.max_steps > 0 else 0
                estimated_remaining = avg_step_time * remaining_steps
                
                logs.update({
                    'elapsed_time': elapsed_time,
                    'estimated_remaining': estimated_remaining,
                    'avg_step_time': avg_step_time
                })
        
        # 定期评估
        if step % self.eval_steps == 0 and step > 0:
            eval_results = self._evaluate_model()
            logs.update(eval_results)
        
        # 记录生成统计
        if 'generation_length' in logs:
            self.training_stats['generation_lengths'].append(logs['generation_length'])
        if 'reward_score' in logs:
            self.training_stats['reward_scores'].append(logs['reward_score'])
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练总结统计"""
        if not self.training_stats['start_time']:
            return {}
        
        total_time = time.time() - self.training_stats['start_time']
        
        summary = {
            'total_training_time': total_time,
            'total_training_time_formatted': self._format_time(total_time),
            'avg_generation_length': np.mean(self.training_stats['generation_lengths']) if self.training_stats['generation_lengths'] else 0,
            'avg_reward_score': np.mean(self.training_stats['reward_scores']) if self.training_stats['reward_scores'] else 0,
            'final_eval_score': self.training_stats['eval_scores'][-1] if self.training_stats['eval_scores'] else 0
        }
        
        return summary
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours}h {minutes}m {seconds}s"


class ProgressCallback(TrainerCallback):
    """进度回调，用于详细的训练日志记录"""
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.step_start_time = None
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
    
    def on_step_end(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            logs = {}
        
        # 记录步骤时间
        if self.step_start_time:
            step_time = time.time() - self.step_start_time
            logs['step_time'] = step_time
        
        # 让训练器记录进度
        if hasattr(self.trainer, 'log_training_progress'):
            self.trainer.log_training_progress(state.global_step, logs)
    
    def on_train_end(self, args, state, control, **kwargs):
        # 训练结束时打印总结
        if hasattr(self.trainer, 'get_training_summary'):
            summary = self.trainer.get_training_summary()
            print("\n=== 训练总结 ===")
            for key, value in summary.items():
                print(f"{key}: {value}") 