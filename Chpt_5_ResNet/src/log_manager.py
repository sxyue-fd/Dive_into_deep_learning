"""
日志文件管理器
确保训练日志(.log)和结果文件(.txt)的时间戳同步和数量一致性
"""

import os
import glob
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import shutil


class LogFileManager:
    """日志文件管理器
    
    职责:
    1. 统一管理训练会话的时间戳
    2. 确保.log和.txt文件成对创建和删除
    3. 维护文件数量限制
    4. 提供清理和同步功能
    """
    
    def __init__(self, logs_dir: str, max_sessions: int = 10):
        """初始化日志文件管理器
        
        Args:
            logs_dir: 日志目录路径
            max_sessions: 保留的最大训练会话数量
        """
        self.logs_dir = Path(logs_dir)
        self.max_sessions = max_sessions
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 当前会话的时间戳，在训练开始时设置
        self.current_session_timestamp: Optional[str] = None
        
    def start_new_session(self) -> str:
        """开始新的训练会话
        
        Returns:
            新会话的时间戳字符串
        """
        self.current_session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return self.current_session_timestamp
    
    def get_log_file_path(self) -> str:
        """获取当前会话的日志文件路径
        
        Returns:
            日志文件的完整路径
        """
        if not self.current_session_timestamp:
            raise RuntimeError("尚未开始训练会话，请先调用 start_new_session()")
        
        return str(self.logs_dir / f'training_{self.current_session_timestamp}.log')
    
    def get_result_file_path(self) -> str:
        """获取当前会话的结果文件路径
        
        Returns:
            结果文件的完整路径
        """
        if not self.current_session_timestamp:
            raise RuntimeError("尚未开始训练会话，请先调用 start_new_session()")
        
        return str(self.logs_dir / f'final_results_{self.current_session_timestamp}.txt')
    
    def get_existing_sessions(self) -> List[Tuple[str, str, str]]:
        """获取现有的训练会话信息
        
        Returns:
            (timestamp, log_file, result_file) 的列表，按时间戳排序
        """
        sessions = {}
        
        # 扫描日志文件
        log_pattern = self.logs_dir / 'training_*.log'
        for log_file in glob.glob(str(log_pattern)):
            filename = Path(log_file).name
            if filename.startswith('training_') and filename.endswith('.log'):
                timestamp = filename[9:-4]  # 移除 'training_' 前缀和 '.log' 后缀
                sessions[timestamp] = {'log': log_file, 'result': None}
        
        # 扫描结果文件
        result_pattern = self.logs_dir / 'final_results_*.txt'
        for result_file in glob.glob(str(result_pattern)):
            filename = Path(result_file).name
            if filename.startswith('final_results_') and filename.endswith('.txt'):
                timestamp = filename[14:-4]  # 移除 'final_results_' 前缀和 '.txt' 后缀
                if timestamp in sessions:
                    sessions[timestamp]['result'] = result_file
                else:
                    # 孤立的结果文件
                    sessions[timestamp] = {'log': None, 'result': result_file}
        
        # 转换为列表并排序
        session_list = []
        for timestamp in sorted(sessions.keys(), reverse=True):
            session_data = sessions[timestamp]
            session_list.append((
                timestamp,
                session_data.get('log', ''),
                session_data.get('result', '')
            ))
        
        return session_list
    
    def cleanup_old_sessions(self) -> Tuple[int, int]:
        """清理旧的训练会话，保持数量限制
        
        Returns:
            (删除的文件总数, 保留的会话数)
        """
        sessions = self.get_existing_sessions()
        deleted_count = 0
        
        if len(sessions) <= self.max_sessions:
            logging.info(f"当前有 {len(sessions)} 个会话，未超过限制 {self.max_sessions}")
            return 0, len(sessions)
        
        # 删除超出限制的会话
        sessions_to_delete = sessions[self.max_sessions:]
        
        for timestamp, log_file, result_file in sessions_to_delete:
            # 删除日志文件
            if log_file and os.path.exists(log_file):
                try:
                    os.remove(log_file)
                    deleted_count += 1
                    logging.info(f"已删除日志文件: {os.path.basename(log_file)}")
                except OSError as e:
                    logging.warning(f"删除日志文件失败 {log_file}: {e}")
            
            # 删除结果文件
            if result_file and os.path.exists(result_file):
                try:
                    os.remove(result_file)
                    deleted_count += 1
                    logging.info(f"已删除结果文件: {os.path.basename(result_file)}")
                except OSError as e:
                    logging.warning(f"删除结果文件失败 {result_file}: {e}")
        
        remaining_sessions = min(len(sessions), self.max_sessions)
        logging.info(f"清理完成: 删除了 {deleted_count} 个文件，保留 {remaining_sessions} 个会话")
        
        return deleted_count, remaining_sessions
    
    def sync_orphaned_files(self) -> Tuple[int, int]:
        """同步孤立的文件（只有.log或只有.txt的文件）
        
        Returns:
            (删除的孤立文件数, 创建的占位文件数)
        """
        sessions = self.get_existing_sessions()
        deleted_count = 0
        created_count = 0
        
        for timestamp, log_file, result_file in sessions:
            # 情况1: 只有日志文件，没有结果文件
            if log_file and not result_file:
                # 创建一个简单的结果文件占位符
                result_path = str(self.logs_dir / f'final_results_{timestamp}.txt')
                try:
                    with open(result_path, 'w', encoding='utf-8') as f:
                        f.write("训练会话结果\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(f"会话时间戳: {timestamp}\n")
                        f.write("状态: 训练未完成或异常终止\n")
                        f.write("注: 此文件由日志管理器自动生成以保持文件同步\n")
                    
                    created_count += 1
                    logging.info(f"已创建结果文件占位符: final_results_{timestamp}.txt")
                except Exception as e:
                    logging.warning(f"创建结果文件占位符失败: {e}")
            
            # 情况2: 只有结果文件，没有日志文件
            elif result_file and not log_file:
                # 删除孤立的结果文件
                try:
                    os.remove(result_file)
                    deleted_count += 1
                    logging.info(f"已删除孤立的结果文件: {os.path.basename(result_file)}")
                except OSError as e:
                    logging.warning(f"删除孤立结果文件失败 {result_file}: {e}")
        
        if deleted_count > 0 or created_count > 0:
            logging.info(f"文件同步完成: 删除 {deleted_count} 个孤立文件，创建 {created_count} 个占位文件")
        
        return deleted_count, created_count
    
    def get_session_report(self) -> str:
        """生成会话状态报告
        
        Returns:
            会话状态的文本报告
        """
        sessions = self.get_existing_sessions()
        
        report = []
        report.append("=" * 60)
        report.append("训练会话状态报告")
        report.append("=" * 60)
        report.append(f"总会话数: {len(sessions)}")
        report.append(f"最大保留数: {self.max_sessions}")
        report.append("")
        
        if not sessions:
            report.append("暂无训练会话记录")
            return "\n".join(report)
        
        report.append("会话详情:")
        report.append("-" * 60)
        report.append(f"{'时间戳':<20} {'日志文件':<10} {'结果文件':<10} {'状态'}")
        report.append("-" * 60)
        
        for timestamp, log_file, result_file in sessions:
            log_status = "✓" if log_file and os.path.exists(log_file) else "✗"
            result_status = "✓" if result_file and os.path.exists(result_file) else "✗"
            
            if log_status == "✓" and result_status == "✓":
                status = "完整"
            elif log_status == "✓" and result_status == "✗":
                status = "缺少结果"
            elif log_status == "✗" and result_status == "✓":
                status = "孤立结果"
            else:
                status = "文件缺失"
            
            report.append(f"{timestamp:<20} {log_status:<10} {result_status:<10} {status}")
        
        report.append("-" * 60)
        return "\n".join(report)
    
    def perform_maintenance(self) -> str:
        """执行完整的维护操作
        
        Returns:
            维护操作的总结报告
        """
        logging.info("开始执行日志文件维护...")
        
        # 1. 同步孤立文件
        deleted_orphaned, created_placeholders = self.sync_orphaned_files()
        
        # 2. 清理旧会话
        deleted_files, remaining_sessions = self.cleanup_old_sessions()
        
        # 3. 生成报告
        report = self.get_session_report()
        
        # 4. 生成维护总结
        summary = []
        summary.append("日志文件维护完成")
        summary.append("=" * 40)
        summary.append(f"删除孤立文件: {deleted_orphaned}")
        summary.append(f"创建占位文件: {created_placeholders}")
        summary.append(f"清理旧文件: {deleted_files}")
        summary.append(f"保留会话数: {remaining_sessions}")
        summary.append("")
        summary.append(report)
        
        maintenance_summary = "\n".join(summary)
        logging.info("日志文件维护完成")
        
        return maintenance_summary


def setup_logging_with_manager(config: dict, log_manager: LogFileManager):
    """使用日志管理器设置日志系统
    
    Args:
        config: 配置字典
        log_manager: 日志文件管理器实例
    """
    # 开始新的训练会话
    log_manager.start_new_session()
    
    # 获取日志文件路径
    log_file = log_manager.get_log_file_path()
    
    # 设置日志系统
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志已设置，文件保存至: {log_file}")
    logger.info(f"会话时间戳: {log_manager.current_session_timestamp}")
