#!/usr/bin/env python3
"""
增强功能测试脚本
验证新增功能是否正常工作
"""

import sys
import logging
from pathlib import Path
import yaml

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def test_config_loading():
    """测试配置加载功能"""
    print("🧪 测试配置加载...")
    
    try:
        from src.r1.configs import load_config
        
        # 测试增强版配置
        config_path = "recipes/Qwen2.5-1.5B-Instruct/grpo/config_enhanced_demo.yaml"
        config = load_config(config_path)
        
        print(f"✓ 配置加载成功")
        print(f"  模型: {config.model_name_or_path}")
        print(f"  数据集: {config.dataset_name}")
        
        # 检查新功能配置
        if hasattr(config, 'eval_steps'):
            print(f"  实时评估: 每{config.eval_steps}步")
        if hasattr(config, 'use_reject_sampling'):
            print(f"  拒绝采样: {'启用' if config.use_reject_sampling else '禁用'}")
        if hasattr(config, 'reward_weights'):
            print(f"  奖励权重: {config.reward_weights}")
            
        return True
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def test_reward_functions():
    """测试奖励函数创建"""
    print("\n🧪 测试奖励函数...")
    
    try:
        from src.r1.rewards import create_reward_functions
        
        reward_names = ['accuracy', 'format', 'reasoning_steps', 'tag_count']
        reward_funcs = create_reward_functions(reward_names)
        
        print(f"✓ 创建了{len(reward_funcs)}个奖励函数")
        for name, func in zip(reward_names, reward_funcs):
            print(f"  • {name}: {func.__name__}")
            
        return True
    except Exception as e:
        print(f"❌ 奖励函数创建失败: {e}")
        return False

def test_yaml_config():
    """测试YAML配置文件格式"""
    print("\n🧪 测试YAML配置格式...")
    
    try:
        config_path = "recipes/Qwen2.5-1.5B-Instruct/grpo/config_enhanced_demo.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
            
        print("✓ YAML配置格式正确")
        
        # 检查关键配置项
        required_keys = [
            'model_name_or_path', 'dataset_name', 'reward_funcs',
            'learning_rate', 'num_train_epochs'
        ]
        
        for key in required_keys:
            if key in yaml_data:
                print(f"  • {key}: ✓")
            else:
                print(f"  • {key}: ❌ 缺失")
                
        # 检查新功能配置
        new_features = [
            'eval_steps', 'use_reject_sampling', 'reward_weights',
            'torch_empty_cache_steps', 'log_completion_details'
        ]
        
        print("  新功能配置:")
        for key in new_features:
            if key in yaml_data:
                print(f"    • {key}: {yaml_data[key]}")
            else:
                print(f"    • {key}: 未配置")
                
        return True
    except Exception as e:
        print(f"❌ YAML配置测试失败: {e}")
        return False

def test_imports():
    """Test module imports"""
    print("\n🧪 Testing module imports...")
    
    modules_to_test = [
        'src.r1.configs',
        'src.r1.trainer',
        'src.r1.rewards',
        'src.r1.callbacks',
        'src.r1.utils.logging',
        'src.r1.utils.formatting',
        'src.r1.utils.data'
    ]
    
    success = True
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            success = False
        except Exception as e:
            print(f"  ⚠️ {module}: {e}")
            
    return success

def main():
    """主测试函数"""
    print("🚀 开始增强功能测试\n")
    
    tests = [
        ("模块导入", test_imports),
        ("YAML配置", test_yaml_config),
        ("配置加载", test_config_loading),
        ("奖励函数", test_reward_functions),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"{'='*50}")
        print(f"测试: {test_name}")
        result = test_func()
        results.append((test_name, result))
        
    print(f"\n{'='*50}")
    print("📊 测试结果汇总:")
    
    all_passed = True
    for test_name, result in results:
        status = "✓ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
            
    print(f"\n🎯 总体结果: {'🎉 所有测试通过!' if all_passed else '⚠️ 部分测试失败'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main()) 