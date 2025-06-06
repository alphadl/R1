"""
Logging utility module
Provides unified logging functionality
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logger(
    name: str = "r1",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger
    
    Args:
        name: Logger name
        level: Log level
        log_file: Optional log file path
        format_string: Optional log format string
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if file path specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "r1") -> logging.Logger:
    """
    Get configured logger
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


class TrainingLogger:
    """Training logger class for structured logging"""
    
    def __init__(self, logger_name: str = "training", log_file: Optional[str] = None):
        self.logger = setup_logger(logger_name, log_file=log_file)
        self.step = 0
        
    def log_step(self, loss: float, metrics: dict = None):
        """Log training step"""
        self.step += 1
        msg = f"Step {self.step}: loss={loss:.4f}"
        
        if metrics:
            metric_str = " ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            msg += f" {metric_str}"
            
        self.logger.info(msg)
        
    def log_epoch(self, epoch: int, avg_loss: float, metrics: dict = None):
        """Log epoch summary"""
        msg = f"Epoch {epoch}: avg_loss={avg_loss:.4f}"
        
        if metrics:
            metric_str = " ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            msg += f" {metric_str}"
            
        self.logger.info(msg)
        
    def log_evaluation(self, eval_metrics: dict):
        """Log evaluation results"""
        metric_str = " ".join([f"{k}={v:.4f}" for k, v in eval_metrics.items()])
        self.logger.info(f"Evaluation: {metric_str}")


def log_system_info(logger: logging.Logger):
    """Log system information"""
    import platform
    import psutil
    
    logger.info("💻 System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  CPU: {psutil.cpu_count()} cores")
    logger.info(f"  Memory: {psutil.virtual_memory().total // (1024**3)} GB")


def log_model_info(logger: logging.Logger, model, tokenizer):
    """Log model information"""
    logger.info("🤖 Model Information:")
    logger.info(f"  Model type: {type(model).__name__}")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(f"  Vocabulary size: {len(tokenizer)}")


def log_dataset_info(logger: logging.Logger, dataset, dataset_name: str = "Dataset"):
    """Log dataset information"""
    logger.info(f"📊 {dataset_name} Information:")
    logger.info(f"  Size: {len(dataset):,} samples")
    if hasattr(dataset, 'features'):
        logger.info(f"  Features: {list(dataset.features.keys())}")


def log_config_summary(logger: logging.Logger, config):
    """Log configuration summary"""
    logger.info("⚙️ Configuration Summary:")
    important_fields = [
        'model_name_or_path', 'dataset_name', 'learning_rate', 
        'num_train_epochs', 'per_device_train_batch_size',
        'reward_funcs', 'eval_steps'
    ]
    
    for field in important_fields:
        if hasattr(config, field):
            value = getattr(config, field)
            logger.info(f"  {field}: {value}") 