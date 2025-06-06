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

"""Enhanced reward functions for GRPO training."""

import re
import numpy as np
from typing import Callable, Optional, List, Dict, Any
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


class RewardCalculator:
    """模块化奖励计算器"""
    
    def __init__(self, reward_funcs: List[str], reward_weights: List[float] = None):
        self.reward_funcs = reward_funcs
        self.reward_weights = reward_weights or [1.0] * len(reward_funcs)
        
        if len(self.reward_funcs) != len(self.reward_weights):
            raise ValueError("奖励函数数量和权重数量必须一致")
    
    def compute_rewards(self, completions: list[list[dict[str, str]]], **kwargs) -> Dict[str, List[float]]:
        """计算所有奖励函数的分数"""
        all_rewards = {}
        
        for func_name, weight in zip(self.reward_funcs, self.reward_weights):
            if func_name in REWARD_FUNCS_REGISTRY:
                rewards = REWARD_FUNCS_REGISTRY[func_name](completions, **kwargs)
                all_rewards[func_name] = rewards
                all_rewards[f"{func_name}_weighted"] = [r * weight for r in rewards]
        
        # 计算加权总分
        total_rewards = []
        for i in range(len(completions)):
            total = sum(all_rewards[f"{func}_weighted"][i] for func in self.reward_funcs)
            total_rewards.append(total)
        
        all_rewards["total"] = total_rewards
        return all_rewards


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Enhanced accuracy reward function"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        try:
            gold_parsed = parse(sol, extraction_mode="first_match")
            
            if len(gold_parsed) != 0:
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                
                # 计算准确性奖励
                try:
                    base_reward = float(verify(gold_parsed, answer_parsed))
                    
                    # 添加置信度奖励
                    confidence_bonus = _calculate_confidence_bonus(content)
                    reward = base_reward + confidence_bonus
                    
                except Exception as e:
                    print(f"验证失败: {e}, 答案: {answer_parsed}, 标准答案: {gold_parsed}")
                    reward = None
            else:
                reward = None
                print("无法解析标准答案: ", sol)
                
        except Exception as e:
            print(f"解析错误: {e}")
            reward = None
            
        rewards.append(reward)
    
    return rewards


def _calculate_confidence_bonus(content: str) -> float:
    """Calculate confidence reward"""
    confidence_indicators = [
        "therefore", "thus", "hence", "clearly", "obviously",
        "确实", "显然", "因此", "所以", "综上"
    ]
    
    bonus = 0.0
    for indicator in confidence_indicators:
        if indicator in content.lower():
            bonus += 0.05
    
    return min(bonus, 0.2)  # Maximum 0.2 confidence bonus


def format_reward(completions, **kwargs):
    """Enhanced format reward function"""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in completion_contents:
        base_reward = 1.0 if re.match(pattern, content, re.DOTALL | re.MULTILINE) else 0.0
        
        # 添加结构完整性奖励
        structure_bonus = _calculate_structure_bonus(content)
        total_reward = base_reward + structure_bonus
        
        rewards.append(total_reward)
    
    return rewards


def _calculate_structure_bonus(content: str) -> float:
    """Calculate structural integrity reward"""
    bonus = 0.0
    
    # Check for clear thinking process
    if "<think>" in content and "</think>" in content:
        think_content = re.search(r"<think>\n(.*?)\n</think>", content, re.DOTALL)
        if think_content:
            think_text = think_content.group(1)
            
            # Reasonable thinking content length
            if 50 < len(think_text) < 1500:
                bonus += 0.1
            
            # Contains mathematical reasoning
            if any(symbol in think_text for symbol in ["=", "+", "-", "*", "/", "\\frac", "\\sqrt"]):
                bonus += 0.1
    
    # Check answer section
    if "<answer>" in content and "</answer>" in content:
        answer_content = re.search(r"<answer>\n(.*?)\n</answer>", content, re.DOTALL)
        if answer_content:
            answer_text = answer_content.group(1).strip()
            
            # Answer is not empty and not too long
            if 1 < len(answer_text) < 200:
                bonus += 0.1
    
    return bonus


def tag_count_reward(completions, **kwargs) -> list[float]:
    """增强的标签计数奖励函数"""
    def count_tags_enhanced(text: str) -> float:
        score = 0.0
        
        # 基础标签检查
        if text.count("<think>\n") == 1:
            score += 0.25
        if text.count("\n</think>\n") == 1:
            score += 0.25
        if text.count("\n<answer>\n") == 1:
            score += 0.25
        if text.count("\n</answer>") == 1:
            score += 0.25
        
        # 惩罚多余的标签
        excess_penalty = 0.0
        if text.count("<think>") > 1:
            excess_penalty += 0.1 * (text.count("<think>") - 1)
        if text.count("</think>") > 1:
            excess_penalty += 0.1 * (text.count("</think>") - 1)
        if text.count("<answer>") > 1:
            excess_penalty += 0.1 * (text.count("<answer>") - 1)
        if text.count("</answer>") > 1:
            excess_penalty += 0.1 * (text.count("</answer>") - 1)
        
        return max(0.0, score - excess_penalty)

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags_enhanced(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    """Enhanced reasoning steps reward function"""
    patterns = {
        'step_indicators': r"(Step \d+:|^\d+\.|\n-|\n\*)",
        'transition_words': r"(First,|Second,|Next,|Then,|Finally,|Therefore,)",
        'mathematical_reasoning': r"(Since|Because|Given that|Let|Assume|Suppose)",
        'chinese_reasoning': r"(首先|其次|然后|接下来|最后|因为|由于|设|假设)"
    }
    
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in completion_contents:
        score = 0.0
        
        # Calculate various reasoning indicators
        for pattern_type, pattern in patterns.items():
            matches = len(re.findall(pattern, content, re.MULTILINE))
            
            if pattern_type == 'step_indicators':
                score += min(0.4, matches * 0.1)  # Maximum 0.4 points
            elif pattern_type == 'transition_words':
                score += min(0.3, matches * 0.05)  # Maximum 0.3 points
            elif pattern_type in ['mathematical_reasoning', 'chinese_reasoning']:
                score += min(0.15, matches * 0.03)  # Maximum 0.15 points each
        
        # Check reasoning coherence
        coherence_bonus = _calculate_coherence_bonus(content)
        total_score = min(1.0, score + coherence_bonus)
        
        rewards.append(total_score)
    
    return rewards


def _calculate_coherence_bonus(content: str) -> float:
    """Calculate reasoning coherence reward"""
    bonus = 0.0
    
    # Check for logical connectors
    logical_connectors = ["however", "moreover", "furthermore", "in addition", "on the other hand"]
    chinese_connectors = ["但是", "而且", "此外", "另外", "另一方面"]
    
    all_connectors = logical_connectors + chinese_connectors
    connector_count = sum(1 for connector in all_connectors if connector in content.lower())
    
    bonus += min(0.1, connector_count * 0.02)
    
    return bonus


def length_reward(completions, target_length: int = 500, **kwargs) -> list[float]:
    """Length reward function, encourages appropriate response length"""
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in completion_contents:
        length = len(content)
        
        # Use Gaussian distribution to calculate length reward
        optimal_range = (target_length * 0.8, target_length * 1.2)
        
        if optimal_range[0] <= length <= optimal_range[1]:
            reward = 1.0
        else:
            # The farther from optimal range, the lower the reward
            distance = min(abs(length - optimal_range[0]), abs(length - optimal_range[1]))
            reward = max(0.0, 1.0 - distance / target_length)
        
        rewards.append(reward)
    
    return rewards


def mathematical_rigor_reward(completions, **kwargs) -> list[float]:
    """Mathematical rigor reward function"""
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    mathematical_terms = [
        # English mathematical terms
        "equation", "function", "derivative", "integral", "theorem", "proof", "lemma",
        "hypothesis", "conclusion", "contradiction", "sufficient", "necessary",
        # Chinese mathematical terms
        "方程", "函数", "导数", "积分", "定理", "证明", "引理", "假设", "结论", "矛盾", "充分", "必要"
    ]
    
    for content in completion_contents:
        score = 0.0
        
        # Check mathematical terminology
        term_count = sum(1 for term in mathematical_terms if term in content.lower())
        score += min(0.3, term_count * 0.05)
        
        # Check mathematical symbols and formulas
        math_symbols = ["\\frac", "\\sqrt", "\\sum", "\\int", "\\lim", "\\infty", "\\theta", "\\pi"]
        symbol_count = sum(1 for symbol in math_symbols if symbol in content)
        score += min(0.4, symbol_count * 0.1)
        
        # Check logical structure
        logical_structure = ["if", "then", "therefore", "since", "because", "如果", "那么", "因此", "由于"]
        logic_count = sum(1 for logic in logical_structure if logic in content.lower())
        score += min(0.3, logic_count * 0.1)
        
        rewards.append(min(1.0, score))
    
    return rewards


# 奖励函数注册表
REWARD_FUNCS_REGISTRY = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "tag_count": tag_count_reward,
    "reasoning_steps": reasoning_steps_reward,
    "length": length_reward,
    "mathematical_rigor": mathematical_rigor_reward,
}


def create_reward_functions(reward_names: List[str]) -> List[Callable]:
    """
    根据名称列表创建奖励函数列表
    
    Args:
        reward_names: 奖励函数名称列表
        
    Returns:
        List[Callable]: 奖励函数列表
    """
    reward_funcs = []
    
    for name in reward_names:
        if name in REWARD_FUNCS_REGISTRY:
            reward_funcs.append(REWARD_FUNCS_REGISTRY[name])
        else:
            available_funcs = list(REWARD_FUNCS_REGISTRY.keys())
            raise ValueError(f"未知的奖励函数: {name}. 可用的函数: {available_funcs}")
    
    return reward_funcs


def get_available_reward_functions() -> List[str]:
    """获取所有可用的奖励函数名称"""
    return list(REWARD_FUNCS_REGISTRY.keys())


def get_reward_calculator(reward_funcs: List[str], reward_weights: List[float] = None) -> RewardCalculator:
    """获取奖励计算器实例"""
    return RewardCalculator(reward_funcs, reward_weights)
