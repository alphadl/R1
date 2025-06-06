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

from dataclasses import dataclass, field
from typing import Any, Optional, List, Union

import trl
import yaml
from pathlib import Path


@dataclass
class DatasetConfig:
    """Configuration for a dataset in a mixture."""

    id: str
    config: Optional[str] = None
    split: str = "train"
    columns: Optional[list[str]] = None
    weight: Optional[float] = None


@dataclass
class DatasetMixtureConfig:
    """Configuration for a mixture of datasets."""

    datasets: list[DatasetConfig]
    seed: int = 0
    test_split_size: Optional[float] = None


@dataclass
class ScriptArguments(trl.ScriptArguments):
    """
    Extended version of ScriptArguments with support for dataset mixtures.

    Args:
        dataset_mixture (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Configuration for creating dataset mixtures with advanced options.
            Format:
              dataset_mixture:
                datasets:
                  - id: dataset_id1
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                  - id: dataset_id2
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                seed: 42
                test_split_size: 0.1
    """

    # Override the dataset_name to make it optional
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Dataset name. Can be omitted if using dataset_mixture."}
    )
    dataset_mixture: Optional[dict[str, Any]] = field(
        default=None,
        metadata={"help": "Configuration for creating dataset mixtures with advanced options like shuffling."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.dataset_mixture is None:
            raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")

        if self.dataset_mixture is not None:
            if not isinstance(self.dataset_mixture, dict) or "datasets" not in self.dataset_mixture:
                raise ValueError(
                    "dataset_mixture must be a dictionary with a 'datasets' key. "
                    "Expected format: {'datasets': [...], 'seed': int}"
                )

            datasets_list = []
            datasets_data = self.dataset_mixture.get("datasets", [])

            if isinstance(datasets_data, list):
                for dataset_config in datasets_data:
                    datasets_list.append(
                        DatasetConfig(
                            id=dataset_config.get("id"),
                            config=dataset_config.get("config"),
                            split=dataset_config.get("split", "train"),
                            columns=dataset_config.get("columns"),
                            weight=dataset_config.get("weight", 1.0),
                        )
                    )
            else:
                raise ValueError("'datasets' must be a list of dataset configurations")

            self.dataset_mixture = DatasetMixtureConfig(
                datasets=datasets_list,
                seed=self.dataset_mixture.get("seed", 0),
                test_split_size=self.dataset_mixture.get("test_split_size", None),
            )

            # Check that column names are consistent across all dataset configs
            columns_sets = [set(dataset.columns) for dataset in datasets_list if dataset.columns is not None]
            if columns_sets:
                first_columns = columns_sets[0]
                if not all(columns == first_columns for columns in columns_sets):
                    raise ValueError(
                        "Column names must be consistent across all dataset configurations in a mixture. "
                        f"Found different column sets: {[list(cols) for cols in columns_sets]}"
                    )


@dataclass
class GRPOConfig(trl.GRPOConfig):
    """Enhanced configuration for GRPO training."""

    # 添加模型路径字段
    model_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    num_completions_to_print: int = field(default=0, metadata={"help": "Number of completions to print."})
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )
    
    # Enhanced evaluation settings
    eval_steps: int = field(default=100, metadata={"help": "Steps between evaluations."})
    eval_on_start: bool = field(default=True, metadata={"help": "Whether to evaluate before training starts."})
    max_eval_samples: int = field(default=100, metadata={"help": "Maximum number of samples for evaluation."})
    eval_temperature: float = field(default=0.7, metadata={"help": "Temperature for evaluation generation."})
    eval_top_p: float = field(default=0.95, metadata={"help": "Top-p for evaluation generation."})
    eval_max_new_tokens: int = field(default=1024, metadata={"help": "Max new tokens for evaluation."})
    
    # Reject sampling settings
    max_resample_attempts: int = field(default=3, metadata={"help": "Maximum attempts for reject sampling."})
    reject_sampling_threshold: float = field(default=0.5, metadata={"help": "Threshold for reject sampling."})
    use_reject_sampling: bool = field(default=False, metadata={"help": "Whether to use reject sampling."})
    
    # Advanced logging
    log_completion_details: bool = field(default=True, metadata={"help": "Whether to log detailed completion info."})
    save_completions: bool = field(default=False, metadata={"help": "Whether to save completions to file."})
    
    # Performance optimization
    torch_empty_cache_steps: int = field(default=10, metadata={"help": "Steps between cache clearing."})
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None, 
        metadata={"help": "Additional kwargs for gradient checkpointing."}
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """Enhanced configuration for SFT training."""

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Enhanced script arguments for the GRPO training script.

    Args:
        model_name_or_path (`str`):
            Path to pretrained model or model identifier from huggingface.co/models.
        reward_funcs (`list[str]`):
            List of reward functions. Available: 'accuracy', 'format', 'reasoning_steps', 
            'tag_count', 'length', 'mathematical_rigor'.
        reward_weights (`list[float]`):
            Weights for each reward function. Must match length of reward_funcs.
        dataset_prompt_column (`str`):
            Column to use as prompts for training.
        eval_dataset_name (`str`):
            Dataset name for evaluation.
        eval_dataset_config (`str`):
            Configuration for evaluation dataset.
    """

    # 添加模型路径字段
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."},
    )

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Available: 'accuracy', 'format', 'reasoning_steps', 'tag_count', 'length', 'mathematical_rigor'"
        },
    )
    
    reward_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "Weights for each reward function. Must match length of reward_funcs."},
    )

    dataset_prompt_column: str = field(
        default="prompt",
        metadata={"help": "Column to use as prompts for training."},
    )
    
    # Evaluation dataset settings
    eval_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset name for evaluation."},
    )
    
    eval_dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Configuration for evaluation dataset."},
    )
    
    eval_dataset_split: str = field(
        default="test",
        metadata={"help": "Split to use for evaluation dataset."},
    )
    
    # Advanced generation settings
    generation_temperature: float = field(
        default=0.7,
        metadata={"help": "Temperature for generation during training."},
    )
    
    generation_top_p: float = field(
        default=0.95,
        metadata={"help": "Top-p for generation during training."},
    )
    
    generation_top_k: int = field(
        default=50,
        metadata={"help": "Top-k for generation during training."},
    )
    
    # Performance and debugging
    max_num_train_samples: int = field(
        default=-1,
        metadata={"help": "Maximum number of training samples. -1 for all."},
    )
    
    max_num_eval_samples: int = field(
        default=100,
        metadata={"help": "Maximum number of evaluation samples."},
    )
    
    debug_mode: bool = field(
        default=False,
        metadata={"help": "Enable debug mode with additional logging."},
    )
    
    save_strategy_steps: int = field(
        default=500,
        metadata={"help": "Steps between model saves."},
    )

    def __post_init__(self):
        super().__post_init__()
        
        # Validate reward weights
        if self.reward_weights is not None:
            if len(self.reward_weights) != len(self.reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(self.reward_weights)}) must match "
                    f"number of reward functions ({len(self.reward_funcs)})"
                )
        else:
            # Set default weights (all equal)
            self.reward_weights = [1.0] * len(self.reward_funcs)


@dataclass
class SFTScriptArguments(ScriptArguments):
    """Enhanced script arguments for the SFT training script."""
    
    # Enhanced SFT settings
    max_seq_length_override: Optional[int] = field(
        default=None,
        metadata={"help": "Override max sequence length for specific datasets."},
    )
    
    use_chat_format: bool = field(
        default=True,
        metadata={"help": "Whether to use chat format for training."},
    )
    
    filter_long_sequences: bool = field(
        default=True,
        metadata={"help": "Whether to filter out sequences longer than max_seq_length."},
    )
    
    debug_mode: bool = field(
        default=False,
        metadata={"help": "Enable debug mode with additional logging."},
    )


def load_config(config_path: Union[str, Path]) -> GRPOScriptArguments:
    """
    Load GRPO configuration from YAML file
    
    Args:
        config_path: Configuration file path
        
    Returns:
        GRPOScriptArguments: Loaded configuration object
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML file
    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
    
    if yaml_data is None:
        raise ValueError(f"Configuration file is empty: {config_path}")
    
    # Get GRPOScriptArguments fields
    script_args_fields = set(GRPOScriptArguments.__dataclass_fields__.keys())
    
    # Separate fields belonging to ScriptArguments and other fields
    script_args_data = {}
    other_data = {}
    
    for key, value in yaml_data.items():
        if value is not None:
            if key in script_args_fields:
                script_args_data[key] = value
            else:
                other_data[key] = value
    
    try:
        # Create configuration object
        config = GRPOScriptArguments(**script_args_data)
        
        # Store other configurations in config object for later use
        for key, value in other_data.items():
            setattr(config, key, value)
        
        return config
    except Exception as e:
        raise ValueError(f"Configuration parameter error: {e}")
        
        
def create_grpo_config_from_script_args(script_args: GRPOScriptArguments) -> GRPOConfig:
    """
    从ScriptArguments创建GRPOConfig
    
    Args:
        script_args: 脚本参数
        
    Returns:
        GRPOConfig: GRPO配置对象
    """
    # 获取GRPOConfig的字段
    grpo_config_fields = set(GRPOConfig.__dataclass_fields__.keys())
    
    # 从script_args中提取GRPOConfig相关的字段
    grpo_config_data = {}
    
    for field_name in grpo_config_fields:
        if hasattr(script_args, field_name):
            value = getattr(script_args, field_name)
            if value is not None:
                grpo_config_data[field_name] = value
    
    return GRPOConfig(**grpo_config_data)


def save_config(config: GRPOScriptArguments, config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration object
        config_path: Save path
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert configuration object to dictionary
    config_dict = {}
    for field_info in config.__dataclass_fields__.values():
        field_name = field_info.name
        field_value = getattr(config, field_name)
        
        # Skip None values and default values
        if field_value is not None:
            config_dict[field_name] = field_value
    
    # Save to YAML file
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
