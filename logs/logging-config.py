import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime

# 图标常量
INFO_ICON = "ℹ️"
SUCCESS_ICON = "✅"
WARNING_ICON = "⚠️"
ERROR_ICON = "❌"
WAIT_ICON = "⏳"
DEBUG_ICON = "🔍"

# 日志等级颜色（仅终端）
class ColoredFormatter(logging.Formatter):
    """
    彩色日志格式化器
    """
    
    COLORS = {
        'DEBUG': '\033[94m',  # 蓝色
        'INFO': '\033[92m',   # 绿色
        'WARNING': '\033[93m', # 黄色
        'ERROR': '\033[91m',  # 红色
        'CRITICAL': '\033[91m\033[1m', # 粗体红色
        'RESET': '\033[0m'    # 重置
    }
    
    ICONS = {
        'DEBUG': DEBUG_ICON,
        'INFO': INFO_ICON,
        'WARNING': WARNING_ICON,
        'ERROR': ERROR_ICON,
        'CRITICAL': ERROR_ICON
    }
    
    def format(self, record):
        # 获取原始消息格式
        log_message = super().format(record)
        
        # 添加颜色和图标（如果是终端输出）
        if sys.stdout.isatty():  # 检查是否为终端输出
            levelname = record.levelname
            icon = self.ICONS.get(levelname, '')
            color = self.COLORS.get(levelname, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            
            # 应用颜色和图标
            return f"{color}{icon} {log_message}{reset}"
        else:
            # 非终端输出，只添加图标
            levelname = record.levelname
            icon = self.ICONS.get(levelname, '')
            return f"{icon} {log_message}"


def setup_logger(name, level=logging.INFO, log_dir='logs', max_size=10*1024*1024, backup_count=5):
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志等级
        log_dir: 日志文件目录
        max_size: 日志文件最大大小(字节)
        backup_count: 保留的日志文件数量
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 获取日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除已有的处理器
    if logger.handlers:
        logger.handlers = []
    
    # 日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 控制台处理器（彩色）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    colored_formatter = ColoredFormatter(log_format, datefmt=date_format)
    console_handler.setFormatter(colored_formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（普通格式）
    today = datetime.now().strftime('%Y%m%d')
    log_file = os.path.join(log_dir, f'{name}_{today}.log')
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_size, 
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(log_format, datefmt=date_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


# 日志装饰器
def log_function_call(logger=None):
    """
    记录函数调用的装饰器
    
    Args:
        logger: 使用的日志记录器，默认为根记录器
    """
    if logger is None:
        logger = logging.getLogger()
        
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 记录函数调用
            logger.debug(f"{WAIT_ICON} 调用函数: {func.__name__}")
            
            try:
                # 执行函数
                result = func(*args, **kwargs)
                
                # 记录成功完成
                logger.debug(f"{SUCCESS_ICON} 函数 {func.__name__} 执行完成")
                
                return result
            except Exception as e:
                # 记录异常
                logger.error(f"{ERROR_ICON} 函数 {func.__name__} 执行出错: {str(e)}")
                # 重新抛出异常
                raise
                
        return wrapper
    return decorator


# 示例用法
if __name__ == "__main__":
    # 设置日志记录器
    logger = setup_logger("example")
    
    # 测试不同级别的日志
    logger.debug("这是一条调试日志")
    logger.info("这是一条信息日志")
    logger.warning("这是一条警告日志")
    logger.error("这是一条错误日志")
    logger.critical("这是一条严重错误日志")
    
    # 测试日志装饰器
    @log_function_call(logger)
    def example_function(x, y):
        logger.info(f"计算 {x} + {y}")
        return x + y
        
    @log_function_call(logger)
    def error_function():
        logger.info("即将发生错误")
        raise ValueError("测试错误")
        
    # 调用测试函数
    result = example_function(3, 5)
    logger.info(f"结果: {result}")
    
    try:
        error_function()
    except ValueError:
        logger.info("错误已捕获")
