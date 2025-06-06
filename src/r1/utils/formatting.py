"""
Text formatting utility module
Provides various text processing and formatting functions
"""

import re
from typing import List, Dict, Any, Optional


def format_chat_template(messages: List[Dict[str, str]], template: str = None) -> str:
    """
    Format chat template
    
    Args:
        messages: Message list, each message contains role and content
        template: Optional custom template
        
    Returns:
        str: Formatted text
    """
    if template is None:
        # Default Qwen template
        template = """{% for message in messages %}
{% if message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}
{% endfor %}
<|im_start|>assistant"""
    
    # Simple template rendering
    result = ""
    for message in messages:
        if message['role'] == 'user':
            result += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
        elif message['role'] == 'assistant':
            result += f"<|im_start|>assistant\n{message['content']}<|im_end|>\n"
    
    result += "<|im_start|>assistant"
    return result


def extract_thinking_and_answer(text: str) -> Dict[str, Optional[str]]:
    """
    Extract thinking process and answer from text
    
    Args:
        text: Input text
        
    Returns:
        Dict: Dictionary containing thinking and answer
    """
    result = {"thinking": None, "answer": None}
    
    # Extract thinking process
    think_match = re.search(r"<think>\n(.*?)\n</think>", text, re.DOTALL)
    if think_match:
        result["thinking"] = think_match.group(1).strip()
    
    # Extract answer
    answer_match = re.search(r"<answer>\n(.*?)\n</answer>", text, re.DOTALL)
    if answer_match:
        result["answer"] = answer_match.group(1).strip()
    
    return result


def clean_completion(text: str) -> str:
    """
    Clean generated text
    
    Args:
        text: Original text
        
    Returns:
        str: Cleaned text
    """
    # Remove excess whitespace characters
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove leading and trailing whitespace
    text = text.strip()
    
    return text


def format_completion_for_display(completion: str, max_length: int = 200) -> str:
    """
    Format completion for display purposes
    
    Args:
        completion: Raw completion text
        max_length: Maximum display length
        
    Returns:
        str: Formatted completion for display
    """
    # Clean the completion
    cleaned = clean_completion(completion)
    
    # Truncate if too long
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."
    
    return cleaned


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Truncation suffix
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_mathematical_expressions(text: str) -> List[str]:
    """
    Extract mathematical expressions from text
    
    Args:
        text: Input text
        
    Returns:
        List[str]: List of mathematical expressions
    """
    # Pattern for LaTeX math expressions
    latex_pattern = r'\\[a-zA-Z]+\{[^}]*\}|\\[a-zA-Z]+'
    
    # Pattern for simple math expressions
    math_pattern = r'[0-9]+\s*[\+\-\*\/\=]\s*[0-9]+'
    
    expressions = []
    expressions.extend(re.findall(latex_pattern, text))
    expressions.extend(re.findall(math_pattern, text))
    
    return expressions


def format_reward_info(rewards: Dict[str, List[float]]) -> str:
    """
    Format reward information for display
    
    Args:
        rewards: Reward dictionary
        
    Returns:
        str: Formatted reward information
    """
    lines = []
    for reward_name, scores in rewards.items():
        if scores:
            mean_score = sum(scores) / len(scores)
            lines.append(f"{reward_name}: {mean_score:.3f}")
    
    return " | ".join(lines)


def format_training_progress(
    step: int, 
    total_steps: int, 
    loss: float, 
    rewards: Dict[str, float] = None
) -> str:
    """
    Format training progress information
    
    Args:
        step: Current step
        total_steps: Total steps
        loss: Loss value
        rewards: Reward information
        
    Returns:
        str: Formatted progress information
    """
    progress = f"Step {step}/{total_steps} | Loss: {loss:.4f}"
    
    if rewards:
        reward_str = format_reward_info(rewards)
        progress += f" | {reward_str}"
    
    return progress 