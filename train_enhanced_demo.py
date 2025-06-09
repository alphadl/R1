#!/usr/bin/env python3
"""
Enhanced GRPO Training Demo Script
Demonstrates usage of new features
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import R1 modules
from src.r1.configs import load_config, GRPOScriptArguments, GRPOConfig
from src.r1.utils.logging import setup_logger
from src.r1.rewards import get_reward_funcs
from src.r1.utils import get_dataset, get_model, get_tokenizer
from trl import ModelConfig, get_peft_config, GRPOTrainer


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Enhanced GRPO Training Demo")
    parser.add_argument(
        "--config", 
        default="recipes/Qwen2.5-1.5B-Instruct/grpo/config_enhanced_demo.yaml",
        help="Configuration file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--dry-run", action="store_true", help="Only validate config without training")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logger("enhanced_demo", level=20 if args.debug else 30)
    
    if args.debug:
        logger.setLevel(10)

    logger.info("🚀 Starting Enhanced GRPO Training Demo")
    logger.info(f"Config file: {args.config}")
    
    try:
        # Load configuration from YAML
        config = load_config(args.config)
        logger.info("✓ Configuration loaded successfully")
        
        # Use the loaded config directly as it's already a GRPOScriptArguments object
        script_args = config
        
        if args.dry_run:
            logger.info("🧪 Dry run mode - configuration validation completed")
            return
        
        # Load dataset, model, and tokenizer
        logger.info("📊 Loading dataset...")
        dataset = get_dataset(script_args)
        
        # Load evaluation dataset if specified
        eval_dataset = None
        eval_strategy = "no"  # 默认禁用评估
        
        if hasattr(script_args, 'eval_dataset_name') and getattr(script_args, 'eval_dataset_name', None):
            logger.info(f"📊 Loading evaluation dataset: {script_args.eval_dataset_name}")
            try:
                import datasets
                eval_ds = datasets.load_dataset(
                    script_args.eval_dataset_name, 
                    getattr(script_args, 'eval_dataset_config', None),
                    split=getattr(script_args, 'eval_dataset_split', 'test')
                )
                # Limit eval samples if specified
                max_eval_samples = getattr(script_args, 'max_eval_samples', None)
                if max_eval_samples and len(eval_ds) > max_eval_samples:
                    eval_dataset = eval_ds.select(range(max_eval_samples))
                else:
                    eval_dataset = eval_ds
                eval_strategy = "steps"  # 如果有评估数据集，启用评估
                logger.info(f"✓ Loaded {len(eval_dataset)} evaluation samples")
            except Exception as e:
                logger.warning(f"⚠️ Could not load evaluation dataset: {e}")
                logger.info("🔄 Continuing without evaluation dataset")
                eval_dataset = None
                eval_strategy = "no"
        
        # Create training args from loaded config (after determining eval strategy)
        training_args = GRPOConfig(
            output_dir=getattr(config, 'output_dir', './data/enhanced-demo'),
            per_device_train_batch_size=getattr(config, 'per_device_train_batch_size', 2),
            gradient_accumulation_steps=getattr(config, 'gradient_accumulation_steps', 4),
            learning_rate=getattr(config, 'learning_rate', 3e-6),
            num_train_epochs=getattr(config, 'num_train_epochs', 2),
            logging_steps=getattr(config, 'logging_steps', 10),
            save_steps=getattr(config, 'save_steps', 100),
            warmup_steps=100,
            bf16=True,
            remove_unused_columns=False,
            
            # GRPO specific parameters
            max_prompt_length=getattr(config, 'max_prompt_length', 512),
            num_generations=getattr(config, 'num_generations', 8),  # Number of generations per prompt
            max_completion_length=getattr(config, 'max_completion_length', 256),  # Maximum length of generated completion
            beta=getattr(config, 'beta', 0.0),  # KL coefficient (0.0 to disable KL divergence)
            epsilon=getattr(config, 'epsilon', 0.2),  # Epsilon value for clipping
            loss_type=getattr(config, 'loss_type', "bnpo"),  # Loss type: "grpo", "bnpo", or "dr_grpo"
            scale_rewards=getattr(config, 'scale_rewards', True),  # Whether to scale rewards by standard deviation
            
            # Generation parameters
            temperature=getattr(config, 'temperature', 1.0),
            top_p=getattr(config, 'top_p', 1.0),
            
            # Evaluation
            eval_strategy=eval_strategy,  # 使用动态确定的评估策略
            eval_steps=getattr(config, 'eval_steps', 50),
            eval_on_start=getattr(config, 'eval_on_start', False),  # 添加eval_on_start参数
            
            # Logging
            report_to=["tensorboard"],
            logging_dir=f"{getattr(config, 'output_dir', './data/enhanced-demo')}/logs",
        )
        
        model_args = ModelConfig(
            model_name_or_path=getattr(config, 'model_name_or_path', None),
        )
        
        # Display new feature configurations
        new_features = {
            'Real-time Evaluation': training_args.eval_steps if eval_strategy != "no" else "Disabled",
            'Reject Sampling': getattr(config, 'use_reject_sampling', False),
            'Enhanced Rewards': getattr(script_args, 'reward_funcs', []),
            'Performance Optimization': getattr(config, 'use_liger_kernel', False),
        }
        
        logger.info("🎯 New Features Configuration:")
        for feature, value in new_features.items():
            logger.info(f"  {feature}: {value}")
        
        logger.info("🤖 Loading model and tokenizer...")
        tokenizer = get_tokenizer(model_args, training_args)
        model = get_model(model_args, training_args)
        
        # Get reward functions
        logger.info("🎯 Setting up reward functions...")
        reward_funcs = get_reward_funcs(script_args)
        
        # Create and setup enhanced trainer
        logger.info("🔧 Creating GRPO Trainer...")
        
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset[getattr(script_args, 'dataset_train_split', 'train')],
            eval_dataset=eval_dataset,  # 提供评估数据集
            processing_class=tokenizer,  # 使用processing_class而不是tokenizer
            reward_funcs=reward_funcs,  # 添加奖励函数
            peft_config=get_peft_config(model_args),
        )
        logger.info("✅ GRPO Trainer created successfully!")
        
        # Start training
        logger.info("🚀 Starting enhanced training...")
        results = trainer.train()
        
        logger.info("✓ Training completed successfully!")
        logger.info(f"Final results: {results}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 