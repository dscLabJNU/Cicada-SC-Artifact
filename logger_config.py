import logging

# 配置根日志记录器，避免重复输出
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[]  # 清空默认处理程序
)


def get_logger(name):
    """
    获取统一配置的 logger
    Args:
        name: 模块名称
    Returns:
        logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)

    # 如果logger已经有处理程序，直接返回
    if logger.handlers:
        return logger

    # 创建控制台处理程序
    handler = logging.StreamHandler()

    # 设置日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 将格式器添加到处理程序
    handler.setFormatter(formatter)

    # 将处理程序添加到logger
    logger.addHandler(handler)

    # 设置日志级别
    logger.setLevel(logging.INFO)

    # 防止日志向上传播到根记录器
    logger.propagate = False

    return logger
