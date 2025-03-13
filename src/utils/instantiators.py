import hydra
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger
from omegaconf import DictConfig

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """configからcallbackをインスタンス化

    Args:
        callbacks_cfg (DictConfig): callbackの設定を含むDictConfigオブジェクト
    
    Returns:
        List: インスタンス化されたcallbackのリスト
    """
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks

def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """configからloggerをインスタンス化

    Args:
        logger_cfg (DictConfig): loggerの設定を含むDictConfigオブジェクト

    Returns:
        List: インスタンス化されたloggerのリスト

    """
    logger: list[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        return TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger
