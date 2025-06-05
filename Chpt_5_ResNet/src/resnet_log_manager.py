"""
ResNet项目专用日志管理器
专门满足项目的日志要求：
- 训练模式：train_yyyymmdd_hhmmss.log + train_results_yyyymmdd_hhmmss.txt (同步时间戳)
- 评估模式：eval_yyyymmdd_hhmmss.log (仅日志文件)
"""

import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO


class ResNetLogManager:
    """ResNet项目专用日志管理器
    
    简化设计，专注核心需求：
    1. 训练模式：生成同步时间戳的.log和.txt文件
    2. 评估模式：仅生成.log文件
    3. 自动处理文件创建和logger配置
    """
    
    def __init__(self, logs_dir: str):
        """初始化日志管理器
        
        Args:
            logs_dir: 日志目录路径
        """
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 当前会话状态
        self.session_timestamp: Optional[str] = None
        self.mode: Optional[str] = None
        self.log_file_path: Optional[str] = None
        self.result_file_path: Optional[str] = None
        self.result_file_handle: Optional[TextIO] = None
        
        # Logger配置
        self.logger: Optional[logging.Logger] = None
        
    def start_session(self, mode: str) -> str:
        """开始新的日志会话
        
        Args:
            mode: 'train' 或 'eval'
            
        Returns:
            会话时间戳
        """
        if mode not in ['train', 'eval']:
            raise ValueError(f"不支持的模式: {mode}. 只支持 'train' 或 'eval'")
        
        # 生成时间戳
        self.session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.mode = mode
        
        # 设置文件路径
        if mode == 'train':
            self.log_file_path = str(self.logs_dir / f"train_{self.session_timestamp}.log")
            self.result_file_path = str(self.logs_dir / f"train_results_{self.session_timestamp}.txt")
        else:  # eval
            self.log_file_path = str(self.logs_dir / f"eval_{self.session_timestamp}.log")
            self.result_file_path = None
        
        # 配置logger
        self._setup_logger()
        
        # 如果是训练模式，创建结果文件
        if mode == 'train':
            self._create_result_file()
        
        self.logger.info(f"开始{mode}会话: {self.session_timestamp}")
        return self.session_timestamp
    
    def _setup_logger(self):
        """配置logger"""
        # 创建专用logger
        logger_name = f"resnet_{self.mode}_{self.session_timestamp}"
        self.logger = logging.getLogger(logger_name)
        
        # 清除已有handler
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 设置级别
        self.logger.setLevel(logging.INFO)
        
        # 创建文件handler
        file_handler = logging.FileHandler(self.log_file_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加handler
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 禁用传播，避免重复日志
        self.logger.propagate = False
    
    def _create_result_file(self):
        """创建结果文件（仅训练模式）"""
        if self.result_file_path:
            self.result_file_handle = open(self.result_file_path, 'w', encoding='utf-8')
            # 写入文件头
            self.result_file_handle.write(f"ResNet训练结果文件\n")
            self.result_file_handle.write(f"会话时间戳: {self.session_timestamp}\n")
            self.result_file_handle.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.result_file_handle.write("=" * 50 + "\n\n")
            self.result_file_handle.flush()
    
    def log_info(self, message: str):
        """记录信息日志"""
        if self.logger:
            self.logger.info(message)
    
    def log_warning(self, message: str):
        """记录警告日志"""
        if self.logger:
            self.logger.warning(message)
    def log_error(self, message: str):
        """记录错误日志"""
        if self.logger:
            self.logger.error(message)
    
    def write_result(self, content: str):
        """写入结果文件（仅训练模式，纯文本格式，无时间戳）"""
        if self.result_file_handle:
            self.result_file_handle.write(f"{content}\n")
            self.result_file_handle.flush()

    def write_training_summary(self, epoch: int, train_loss: float, train_acc: float, 
                             val_loss: float = None, val_acc: float = None,
                             val_top5_acc: float = None, lr: float = None, 
                             epoch_time: float = None, is_best: bool = False):
        """写入详细的训练摘要"""
        if self.result_file_handle:
            content = f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}"
            if val_loss is not None and val_acc is not None:
                content += f", Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
            if val_top5_acc is not None:
                content += f", Val Top5={val_top5_acc:.4f}"
            if lr is not None:
                content += f", LR={lr:.6f}"
            if epoch_time is not None:
                content += f", Time={epoch_time:.2f}s"
            if is_best:
                content += " ⭐ BEST"
            self.write_result(content)
    
    def write_model_info(self, model_summary: str, total_params: int = None):
        """写入模型信息"""
        if self.result_file_handle:
            self.result_file_handle.write("\n模型信息:\n")
            self.result_file_handle.write("-" * 30 + "\n")
            if total_params:
                self.result_file_handle.write(f"总参数量: {total_params:,}\n")
            self.result_file_handle.write(f"{model_summary}\n")
            self.result_file_handle.write("-" * 30 + "\n\n")
            self.result_file_handle.flush()
    
    def write_config_info(self, config_dict: dict):
        """写入配置信息"""
        if self.result_file_handle:
            self.result_file_handle.write("\n训练配置:\n")
            self.result_file_handle.write("-" * 30 + "\n")
            
            # 主要配置信息
            if 'training' in config_dict:
                train_cfg = config_dict['training']
                self.result_file_handle.write(f"epochs: {train_cfg.get('epochs', 'N/A')}\n")
                self.result_file_handle.write(f"optimizer: {train_cfg.get('optimizer', 'N/A')}\n")
                self.result_file_handle.write(f"learning_rate: {train_cfg.get('learning_rate', 'N/A')}\n")
                self.result_file_handle.write(f"scheduler: {train_cfg.get('scheduler', 'None')}\n")
            
            if 'data' in config_dict:
                data_cfg = config_dict['data']
                self.result_file_handle.write(f"batch_size: {data_cfg.get('batch_size', 'N/A')}\n")
                self.result_file_handle.write(f"num_workers: {data_cfg.get('num_workers', 'N/A')}\n")
            
            if 'model' in config_dict:
                model_cfg = config_dict['model']
                self.result_file_handle.write(f"dropout_rate: {model_cfg.get('dropout_rate', 'N/A')}\n")
            
            self.result_file_handle.write("-" * 30 + "\n\n")
            self.result_file_handle.flush()
    
    def write_early_stopping_info(self, reason: str, stopped_epoch: int, best_epoch: int, 
                                best_val_acc: float):
        """写入早停信息"""
        if self.result_file_handle:
            self.result_file_handle.write("\n早停信息:\n")
            self.result_file_handle.write("-" * 30 + "\n")
            self.result_file_handle.write(f"停止原因: {reason}\n")
            self.result_file_handle.write(f"停止轮次: {stopped_epoch}\n")
            self.result_file_handle.write(f"最佳轮次: {best_epoch}\n")
            self.result_file_handle.write(f"最佳验证准确率: {best_val_acc:.4f}\n")
            self.result_file_handle.write("-" * 30 + "\n\n")
            self.result_file_handle.flush()
    
    def write_final_results(self, best_epoch: int, best_train_acc: float, 
                          best_val_acc: float = None, total_time: float = None):
        """写入最终结果"""
        if self.result_file_handle:
            self.result_file_handle.write("\n" + "=" * 50 + "\n")
            self.result_file_handle.write("最终训练结果\n")
            self.result_file_handle.write("=" * 50 + "\n")
            self.result_file_handle.write(f"最佳轮次: {best_epoch}\n")
            self.result_file_handle.write(f"最佳训练准确率: {best_train_acc:.4f}\n")
            if best_val_acc is not None:
                self.result_file_handle.write(f"最佳验证准确率: {best_val_acc:.4f}\n")
            if total_time is not None:
                self.result_file_handle.write(f"总训练时间: {total_time:.2f}秒\n")
            self.result_file_handle.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.result_file_handle.flush()
    
    def close_session(self):
        """关闭当前会话"""
        if self.logger:
            self.logger.info(f"结束{self.mode}会话: {self.session_timestamp}")
            
            # 关闭所有handler
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
        
        # 关闭结果文件
        if self.result_file_handle:
            self.result_file_handle.close()
            self.result_file_handle = None
        
        # 重置状态
        self.session_timestamp = None
        self.mode = None
        self.log_file_path = None
        self.result_file_path = None
        self.logger = None
    
    def get_log_file_path(self) -> Optional[str]:
        """获取日志文件路径"""
        return self.log_file_path
    
    def get_result_file_path(self) -> Optional[str]:
        """获取结果文件路径"""
        return self.result_file_path
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close_session()


# 使用示例
if __name__ == "__main__":
    # 测试训练模式
    with ResNetLogManager("../outputs/logs") as log_manager:
        log_manager.start_session('train')
        log_manager.log_info("开始训练测试")
        log_manager.write_result("这是一个测试结果")
        log_manager.write_training_summary(1, 0.5, 0.85, 0.3, 0.92)
        log_manager.write_final_results(1, 0.85, 0.92, 120.5)
        log_manager.log_info("训练测试完成")
    
    # 测试评估模式
    with ResNetLogManager("../outputs/logs") as log_manager:
        log_manager.start_session('eval')
        log_manager.log_info("开始评估测试")
        log_manager.log_info("评估测试完成")
