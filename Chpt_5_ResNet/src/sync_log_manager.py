"""
同步日志文件管理器
确保训练日志(.log)和结果文件(.txt)的完全同步管理
支持训练和评估模式的分离日志系统
"""

import os
import re
import glob
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import shutil


class SyncLogManager:
    """同步日志文件管理器
    
    核心功能:
    1. 日志和结果文件的同步生命周期管理
    2. 支持训练(train)和评估(evaluate)模式分离
    3. 自动清理和维护文件一致性
    4. 防止孤立文件产生
    """
    
    def __init__(self, logs_dir: str, max_sessions: int = 10, max_days: int = 30):
        """初始化同步日志管理器
        
        Args:
            logs_dir: 日志目录路径
            max_sessions: 每种模式保留的最大会话数量
            max_days: 文件保留的最大天数
        """
        self.logs_dir = Path(logs_dir)
        self.max_sessions = max_sessions
        self.max_days = max_days
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 当前会话信息
        self.current_session_timestamp: Optional[str] = None
        self.current_mode: Optional[str] = None
        
        # 文件命名模式
        self.file_patterns = {
            'train': {
                'log': 'train_{timestamp}.log',
                'result': 'final_results_{timestamp}.txt'
            },
            'evaluate': {
                'log': 'evaluate_{timestamp}.log', 
                'result': 'evaluation_results_{timestamp}.txt'
            }
        }
        
    def start_new_session(self, mode: str = 'train') -> str:
        """开始新的会话
        
        Args:
            mode: 会话模式，'train' 或 'evaluate'
            
        Returns:
            新会话的时间戳字符串
        """
        if mode not in ['train', 'evaluate']:
            raise ValueError("模式必须是 'train' 或 'evaluate'")
            
        self.current_session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_mode = mode
        
        # 记录会话开始
        logger = logging.getLogger(__name__)
        logger.info(f"开始新的{mode}会话: {self.current_session_timestamp}")
        
        return self.current_session_timestamp
    
    def get_log_file_path(self) -> str:
        """获取当前会话的日志文件路径"""
        if not self.current_session_timestamp or not self.current_mode:
            raise RuntimeError("尚未开始会话，请先调用 start_new_session()")
        
        pattern = self.file_patterns[self.current_mode]['log']
        filename = pattern.format(timestamp=self.current_session_timestamp)
        return str(self.logs_dir / filename)
    
    def get_result_file_path(self) -> str:
        """获取当前会话的结果文件路径"""
        if not self.current_session_timestamp or not self.current_mode:
            raise RuntimeError("尚未开始会话，请先调用 start_new_session()")
        
        pattern = self.file_patterns[self.current_mode]['result']
        filename = pattern.format(timestamp=self.current_session_timestamp)
        return str(self.logs_dir / filename)
    
    def _extract_timestamp_and_mode(self, filename: str) -> Tuple[Optional[str], Optional[str]]:
        """从文件名中提取时间戳和模式
        
        Args:
            filename: 文件名
            
        Returns:
            (timestamp, mode) 元组，如果无法解析则返回 (None, None)
        """
        # 匹配模式: mode_YYYYMMDD_HHMMSS.ext 或 prefix_YYYYMMDD_HHMMSS.ext
        patterns = [
            (r'^train_(\d{8}_\d{6})\.log$', 'train'),
            (r'^evaluate_(\d{8}_\d{6})\.log$', 'evaluate'),
            (r'^final_results_(\d{8}_\d{6})\.txt$', 'train'),
            (r'^evaluation_results_(\d{8}_\d{6})\.txt$', 'evaluate'),
        ]
        
        for pattern, mode in patterns:
            match = re.match(pattern, filename)
            if match:
                return match.group(1), mode
        
        return None, None
    
    def get_session_pairs(self) -> Dict[str, Dict[str, Dict]]:
        """获取所有会话的配对信息
        
        Returns:
            按模式和时间戳组织的会话信息字典
            {
                'train': {
                    'timestamp1': {'log': path, 'result': path, 'complete': bool},
                    ...
                },
                'evaluate': {
                    ...
                }
            }
        """
        sessions = {'train': {}, 'evaluate': {}}
        
        # 扫描所有相关文件
        for file_path in self.logs_dir.glob('*'):
            if file_path.is_file():
                filename = file_path.name
                timestamp, mode = self._extract_timestamp_and_mode(filename)
                
                if timestamp and mode:
                    if timestamp not in sessions[mode]:
                        sessions[mode][timestamp] = {
                            'log': None,
                            'result': None,
                            'complete': False
                        }
                    
                    if filename.endswith('.log'):
                        sessions[mode][timestamp]['log'] = str(file_path)
                    elif filename.endswith('.txt'):
                        sessions[mode][timestamp]['result'] = str(file_path)
                    
                    # 检查是否完整
                    session = sessions[mode][timestamp]
                    session['complete'] = bool(session['log'] and session['result'])
        
        return sessions
    
    def cleanup_old_sessions_by_time(self) -> int:
        """按时间清理旧会话
        
        Returns:
            删除的文件总数
        """
        cutoff_date = datetime.now() - timedelta(days=self.max_days)
        deleted_count = 0
        
        logger = logging.getLogger(__name__)
        
        sessions = self.get_session_pairs()
        
        for mode in ['train', 'evaluate']:
            for timestamp, session_info in sessions[mode].items():
                try:
                    # 解析时间戳
                    session_date = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                    
                    if session_date < cutoff_date:
                        # 删除日志文件
                        if session_info['log'] and os.path.exists(session_info['log']):
                            os.remove(session_info['log'])
                            deleted_count += 1
                            logger.info(f"删除过期日志文件: {os.path.basename(session_info['log'])}")
                        
                        # 删除结果文件
                        if session_info['result'] and os.path.exists(session_info['result']):
                            os.remove(session_info['result'])
                            deleted_count += 1
                            logger.info(f"删除过期结果文件: {os.path.basename(session_info['result'])}")
                            
                except ValueError as e:
                    logger.warning(f"无法解析时间戳 {timestamp}: {e}")
                except Exception as e:
                    logger.warning(f"删除文件失败: {e}")
        
        return deleted_count
    
    def cleanup_old_sessions_by_count(self) -> int:
        """按数量清理旧会话（每种模式分别计算）
        
        Returns:
            删除的文件总数
        """
        deleted_count = 0
        logger = logging.getLogger(__name__)
        
        sessions = self.get_session_pairs()
        
        for mode in ['train', 'evaluate']:
            mode_sessions = sessions[mode]
            
            if len(mode_sessions) <= self.max_sessions:
                continue
            
            # 按时间戳排序，保留最新的
            sorted_timestamps = sorted(mode_sessions.keys(), reverse=True)
            timestamps_to_delete = sorted_timestamps[self.max_sessions:]
            
            for timestamp in timestamps_to_delete:
                session_info = mode_sessions[timestamp]
                
                # 删除日志文件
                if session_info['log'] and os.path.exists(session_info['log']):
                    try:
                        os.remove(session_info['log'])
                        deleted_count += 1
                        logger.info(f"删除多余日志文件: {os.path.basename(session_info['log'])}")
                    except Exception as e:
                        logger.warning(f"删除日志文件失败: {e}")
                
                # 删除结果文件
                if session_info['result'] and os.path.exists(session_info['result']):
                    try:
                        os.remove(session_info['result'])
                        deleted_count += 1
                        logger.info(f"删除多余结果文件: {os.path.basename(session_info['result'])}")
                    except Exception as e:
                        logger.warning(f"删除结果文件失败: {e}")
        
        return deleted_count
    
    def sync_incomplete_sessions(self) -> Tuple[int, int]:
        """同步不完整的会话（创建缺失文件或删除孤立文件）
        
        Returns:
            (删除的孤立文件数, 创建的占位文件数)
        """
        deleted_count = 0
        created_count = 0
        logger = logging.getLogger(__name__)
        
        sessions = self.get_session_pairs()
        
        for mode in ['train', 'evaluate']:
            for timestamp, session_info in sessions[mode].items():
                if session_info['complete']:
                    continue
                
                has_log = session_info['log'] and os.path.exists(session_info['log'])
                has_result = session_info['result'] and os.path.exists(session_info['result'])
                
                if has_log and not has_result:
                    # 只有日志文件，创建结果文件占位符
                    try:
                        result_pattern = self.file_patterns[mode]['result']
                        result_filename = result_pattern.format(timestamp=timestamp)
                        result_path = self.logs_dir / result_filename
                        
                        with open(result_path, 'w', encoding='utf-8') as f:
                            f.write(f"{'训练' if mode == 'train' else '评估'}会话结果\n")
                            f.write("=" * 50 + "\n\n")
                            f.write(f"会话时间戳: {timestamp}\n")
                            f.write(f"会话模式: {mode}\n")
                            f.write("状态: 会话未完成或异常终止\n")
                            f.write("注: 此文件由日志管理器自动生成以保持文件同步\n")
                        
                        created_count += 1
                        logger.info(f"创建结果文件占位符: {result_filename}")
                        
                    except Exception as e:
                        logger.warning(f"创建结果文件占位符失败: {e}")
                
                elif has_result and not has_log:
                    # 只有结果文件，删除孤立文件
                    try:
                        os.remove(session_info['result'])
                        deleted_count += 1
                        logger.info(f"删除孤立结果文件: {os.path.basename(session_info['result'])}")
                    except Exception as e:
                        logger.warning(f"删除孤立结果文件失败: {e}")
        
        return deleted_count, created_count
    
    def get_session_report(self) -> str:
        """生成会话状态报告"""
        sessions = self.get_session_pairs()
        
        report = []
        report.append("=" * 80)
        report.append("同步日志管理器 - 会话状态报告")
        report.append("=" * 80)
        
        for mode in ['train', 'evaluate']:
            mode_sessions = sessions[mode]
            report.append(f"\n{mode.upper()} 模式:")
            report.append("-" * 40)
            
            if not mode_sessions:
                report.append("  暂无会话记录")
                continue
            
            report.append(f"  总会话数: {len(mode_sessions)}")
            report.append(f"  完整会话: {sum(1 for s in mode_sessions.values() if s['complete'])}")
            report.append(f"  不完整会话: {sum(1 for s in mode_sessions.values() if not s['complete'])}")
            report.append("")
            
            # 显示最近的几个会话
            sorted_timestamps = sorted(mode_sessions.keys(), reverse=True)[:5]
            report.append("  最近会话:")
            report.append(f"  {'时间戳':<20} {'状态':<10} {'日志':<6} {'结果'}")
            report.append("  " + "-" * 45)
            
            for timestamp in sorted_timestamps:
                session = mode_sessions[timestamp]
                status = "完整" if session['complete'] else "不完整"
                log_status = "✓" if session['log'] and os.path.exists(session['log']) else "✗"
                result_status = "✓" if session['result'] and os.path.exists(session['result']) else "✗"
                
                report.append(f"  {timestamp:<20} {status:<10} {log_status:<6} {result_status}")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
    
    def perform_full_maintenance(self) -> Dict[str, int]:
        """执行完整的维护操作
        
        Returns:
            维护操作统计信息
        """
        logger = logging.getLogger(__name__)
        logger.info("开始执行同步日志文件维护...")
        
        # 统计信息
        stats = {
            'deleted_by_time': 0,
            'deleted_by_count': 0,
            'deleted_orphaned': 0,
            'created_placeholders': 0
        }
        
        try:
            # 1. 按时间清理
            stats['deleted_by_time'] = self.cleanup_old_sessions_by_time()
            
            # 2. 同步不完整会话
            deleted_orphaned, created_placeholders = self.sync_incomplete_sessions()
            stats['deleted_orphaned'] = deleted_orphaned
            stats['created_placeholders'] = created_placeholders
            
            # 3. 按数量清理（在同步后进行）
            stats['deleted_by_count'] = self.cleanup_old_sessions_by_count()
            
            # 4. 生成报告
            report = self.get_session_report()
            
            logger.info("同步日志文件维护完成")
            logger.info(f"维护统计: 按时间删除={stats['deleted_by_time']}, "
                       f"按数量删除={stats['deleted_by_count']}, "
                       f"删除孤立={stats['deleted_orphaned']}, "
                       f"创建占位={stats['created_placeholders']}")
            
        except Exception as e:
            logger.error(f"日志维护失败: {e}")
        
        return stats


def setup_sync_logging(config: dict, log_manager: SyncLogManager, mode: str = 'train'):
    """使用同步日志管理器设置日志系统
    
    Args:
        config: 配置字典
        log_manager: 同步日志文件管理器实例
        mode: 日志模式 ('train' 或 'evaluate')
    """
    # 开始新的会话
    log_manager.start_new_session(mode)
    
    # 获取日志文件路径
    log_file = log_manager.get_log_file_path()
    
    # 清理现有的处理器以避免重复日志
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 设置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if mode == 'evaluate':
        log_format = '[EVAL] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 配置日志系统
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True  # 强制重新配置
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"{mode.upper()}模式日志已设置，文件保存至: {log_file}")
    logger.info(f"会话时间戳: {log_manager.current_session_timestamp}")
