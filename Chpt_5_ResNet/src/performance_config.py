"""
性能调优配置解析器
用于解析和应用性能预设配置
"""

import yaml
import copy
from typing import Dict, Any, List
from pathlib import Path


def load_performance_config(config_path: str = "configs/config_performance.yaml") -> Dict[str, Any]:
    """
    加载并解析性能调优配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        解析后的完整配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 获取选择的预设
    preset_name = config.get('performance_preset', 'baseline')
    
    if preset_name not in config['performance_presets']:
        raise ValueError(f"未找到预设配置: {preset_name}")
    
    # 应用预设配置
    preset = config['performance_presets'][preset_name]
    final_config = apply_performance_preset(config, preset)
    
    # 应用手动覆盖
    manual_overrides = config.get('manual_overrides', {})
    if manual_overrides:
        final_config = apply_manual_overrides(final_config, manual_overrides)
    
    # 应用高级参数
    final_config = apply_advanced_settings(final_config, config['advanced_performance'])
    
    print(f"✅ 已加载性能预设: {preset_name}")
    print(f"📝 预设描述: {preset['description']}")
    print(f"🎯 关键参数: epochs={final_config['training']['epochs']}, "
          f"batch_size={final_config['data']['batch_size']}, "
          f"lr={final_config['training']['learning_rate']}")
    
    return final_config


def apply_performance_preset(base_config: Dict[str, Any], preset: Dict[str, Any]) -> Dict[str, Any]:
    """应用性能预设到基础配置"""
    config = copy.deepcopy(base_config)
    
    # 确保必要的配置结构存在
    ensure_nested_dict(config, ['training'])
    ensure_nested_dict(config, ['data'])
    ensure_nested_dict(config, ['device'])
    ensure_nested_dict(config, ['logging'])
    ensure_nested_dict(config, ['checkpoint'])
    ensure_nested_dict(config, ['training_monitoring', 'early_stopping'])
    
    # 映射预设参数到配置结构
    mapping = {
        'epochs': ['training', 'epochs'],
        'batch_size': ['data', 'batch_size'],
        'learning_rate': ['training', 'learning_rate'],
        'weight_decay': ['training', 'weight_decay'],
        'target_accuracy': ['training', 'target_accuracy'],
        'early_stopping_patience': ['training_monitoring', 'early_stopping', 'patience'],
        'scheduler': ['training', 'scheduler'],
        'optimizer': ['training', 'optimizer'],
        'mixed_precision': ['device', 'mixed_precision'],
        'num_workers': ['data', 'num_workers'],
        'log_frequency': ['logging', 'log_frequency'],
        'save_frequency': ['checkpoint', 'save_frequency']
    }
    
    # 应用预设参数
    for preset_key, config_path in mapping.items():
        if preset_key in preset:
            set_nested_value(config, config_path, preset[preset_key])
    
    return config


def apply_manual_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """应用手动覆盖参数"""
    if not overrides:
        return config
    
    mapping = {
        'epochs': ['training', 'epochs'],
        'learning_rate': ['training', 'learning_rate'],
        'batch_size': ['data', 'batch_size'],
        'optimizer': ['training', 'optimizer'],
        'scheduler': ['training', 'scheduler'],
        'weight_decay': ['training', 'weight_decay'],
        'target_accuracy': ['training', 'target_accuracy']
    }
    
    overridden_params = []
    for override_key, value in overrides.items():
        if override_key in mapping:
            set_nested_value(config, mapping[override_key], value)
            overridden_params.append(f"{override_key}={value}")
    
    if overridden_params:
        print(f"🔧 手动覆盖参数: {', '.join(overridden_params)}")
    
    return config


def apply_advanced_settings(config: Dict[str, Any], advanced: Dict[str, Any]) -> Dict[str, Any]:
    """应用高级性能设置"""
    
    # 应用数据增强设置
    aug_strength = advanced.get('augmentation_strength', 'medium')
    aug_config = advanced['augmentation_configs'].get(aug_strength, {})
    
    if 'random_horizontal_flip' in aug_config:
        set_nested_value(config, ['data', 'transforms', 'train', 'random_horizontal_flip'], 
                        aug_config['random_horizontal_flip'])
    if 'random_crop_padding' in aug_config:
        set_nested_value(config, ['data', 'transforms', 'train', 'random_crop', 'padding'], 
                        aug_config['random_crop_padding'])
    
    # 应用优化器配置
    optimizer = config.get('training', {}).get('optimizer', 'adam')
    opt_config = advanced['optimizer_configs'].get(optimizer, {})
    if opt_config:
        ensure_nested_dict(config, ['training', 'optimizer_params'])
        config['training']['optimizer_params'].update(opt_config)
    
    # 应用调度器配置
    scheduler = config.get('training', {}).get('scheduler', 'cosine')
    sched_config = advanced['scheduler_configs'].get(scheduler, {})
    if sched_config:
        ensure_nested_dict(config, ['training', 'scheduler_params'])
        config['training']['scheduler_params'].update(sched_config)
    
    return config


def set_nested_value(config: Dict[str, Any], path: list, value: Any):
    """设置嵌套字典的值"""
    current = config
    for key in path[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[path[-1]] = value


def ensure_nested_dict(config: Dict[str, Any], path: list):
    """确保嵌套字典路径存在"""
    current = config
    for key in path:
        if key not in current:
            current[key] = {}
        current = current[key]


def print_performance_summary(config: Dict[str, Any]):
    """打印性能配置摘要"""
    print("\n" + "="*60)
    print("🚀 性能调优配置摘要")
    print("="*60)
    
    training = config.get('training', {})
    data = config.get('data', {})
    device = config.get('device', {})
    
    print(f"📊 训练参数:")
    print(f"   - 训练轮数: {training.get('epochs', 'N/A')}")
    print(f"   - 批次大小: {data.get('batch_size', 'N/A')}")
    print(f"   - 学习率: {training.get('learning_rate', 'N/A')}")
    print(f"   - 权重衰减: {training.get('weight_decay', 'N/A')}")
    print(f"   - 目标准确率: {training.get('target_accuracy', 'N/A')}")
    
    print(f"\n🔧 优化设置:")
    print(f"   - 优化器: {training.get('optimizer', 'N/A')}")
    print(f"   - 调度器: {training.get('scheduler', 'N/A')}")
    print(f"   - 混合精度: {device.get('mixed_precision', 'N/A')}")
    print(f"   - 数据线程数: {data.get('num_workers', 'N/A')}")
    
    print("="*60)


def get_config_value(config: Dict[str, Any], path: List[str], default=None):
    """安全地获取配置值，支持嵌套路径
    
    Args:
        config: 配置字典
        path: 配置路径列表，如 ['training', 'learning_rate']
        default: 默认值
        
    Returns:
        配置值或默认值
    """
    current = config
    try:
        for key in path:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def get_early_stopping_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """获取早停配置，兼容新旧格式
    
    Args:
        config: 配置字典
        
    Returns:
        早停配置字典
    """
    # 尝试新格式
    early_stopping_config = get_config_value(config, ['training_monitoring', 'early_stopping'], {})
    
    # 如果新格式为空，尝试旧格式
    if not early_stopping_config:
        early_stopping_config = get_config_value(config, ['training', 'early_stopping'], {})
    
    # 设置默认值
    return {
        'patience': early_stopping_config.get('patience', 10),
        'min_delta': early_stopping_config.get('min_delta', 0.001),
        'mode': early_stopping_config.get('mode', 'min')
    }


def get_random_seed(config: Dict[str, Any]) -> int:
    """获取随机种子，兼容新旧格式
    
    Args:
        config: 配置字典
        
    Returns:
        随机种子值
    """
    # 尝试从根级别获取
    seed = get_config_value(config, ['random_seed'])
    if seed is not None:
        return seed
    
    # 尝试从data配置获取
    seed = get_config_value(config, ['data', 'random_seed'])
    if seed is not None:
        return seed
    
    # 默认值
    return 42


if __name__ == "__main__":
    # 测试配置解析器
    try:
        config = load_performance_config()
        print_performance_summary(config)
    except Exception as e:
        print(f"❌ 配置解析失败: {e}")
