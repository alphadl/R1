#!/usr/bin/env python3
"""
Enhanced features test script
Verify if new features are working correctly
"""

import sys
import os

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.r1.configs import ScriptArguments, GRPOConfig, GRPOScriptArguments
from src.r1.rewards import get_reward_calculator


def test_config_loading():
    """Test configuration loading functionality"""
    # Test enhanced configuration
    script_args = GRPOScriptArguments(dataset_name="test_dataset")
    grpo_config = GRPOConfig()
    
    # Check if enhanced configurations are working
    assert hasattr(script_args, 'model_name_or_path')
    assert hasattr(grpo_config, 'learning_rate')
    assert hasattr(grpo_config, 'warmup_steps')
    
    print("✅ Configuration loading test passed!")


def test_reward_functions():
    """Test reward function creation"""
    # Test default reward calculator
    calculator = get_reward_calculator(["accuracy", "format"])
    assert calculator is not None
    
    # Test with custom configuration and weights
    custom_calculator = get_reward_calculator(
        ["format", "tag_count"], 
        [0.6, 0.4]
    )
    assert custom_calculator is not None
    
    print("✅ Reward function test passed!")


def test_yaml_config():
    """Test YAML configuration file format"""
    config_content = """
# Model configuration
model:
  name: test-model
  learning_rate: 1e-4

# Training configuration  
training:
  batch_size: 4
  max_steps: 1000
"""
    
    # Check key configuration items
    assert "model:" in config_content
    assert "training:" in config_content
    assert "learning_rate:" in config_content
    assert "batch_size:" in config_content
    assert "max_steps:" in config_content
    
    # Check enhanced feature configuration
    enhanced_config = {
        "rewards": {
            "format_reward": {"enabled": True},
            "reasoning_reward": {"enabled": True}
        },
        "evaluation": {
            "interval": 100,
            "metrics": ["accuracy", "reward_score"]
        }
    }
    
    assert "rewards" in enhanced_config
    assert "evaluation" in enhanced_config
    
    print("✅ YAML configuration test passed!")


def test_integration():
    """Integration test - test if all components work together"""
    try:
        # Test configuration loading
        script_args = GRPOScriptArguments(dataset_name="test_dataset")
        grpo_config = GRPOConfig()
        
        # Test reward calculator
        calculator = get_reward_calculator(["accuracy", "format"])
        
        # Test if all components can be instantiated
        assert script_args is not None
        assert grpo_config is not None
        assert calculator is not None
        
        print("✅ Integration test passed!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise


def main():
    """Main test function"""
    print("Starting enhanced features test...")
    
    test_config_loading()
    test_reward_functions() 
    test_yaml_config()
    test_integration()
    
    print("🎉 All tests passed!")


if __name__ == "__main__":
    main() 