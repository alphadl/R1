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
from src.r1.configs import load_config
from src.r1.enhanced_grpo_trainer import EnhancedGRPOTrainer
from src.r1.utils.logging import setup_logger


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
        # Load configuration
        config = load_config(args.config)
        logger.info("✓ Configuration loaded successfully")
        
        # Validate configuration
        if not hasattr(config, 'model_name_or_path') or not config.model_name_or_path:
            raise ValueError("model_name_or_path must be specified")
        
        # Display new feature configurations
        new_features = {
            'Real-time Evaluation': getattr(config, 'eval_steps', None),
            'Reject Sampling': getattr(config, 'use_reject_sampling', False),
            'Enhanced Rewards': getattr(config, 'reward_funcs', []),
            'Performance Optimization': getattr(config, 'use_liger_kernel', False),
        }
        
        logger.info("🎯 New Features Configuration:")
        for feature, value in new_features.items():
            logger.info(f"  {feature}: {value}")
        
        if args.dry_run:
            logger.info("🧪 Dry run mode - configuration validation completed")
            return
        
        # Create and setup enhanced trainer
        logger.info("🔧 Creating Enhanced GRPO Trainer...")
        
        trainer = EnhancedGRPOTrainer(config)
        
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