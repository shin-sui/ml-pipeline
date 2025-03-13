import warnings
from importlib.util import find_spec
from typing import Any, Callable

from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

def extras(cfg: DictConfig) -> None:
    """タスクが開始する前にオプションのユーティリティを適用する

    Utilities:
        - Pythonのwarningsを無視する
        - コマンドラインからタグを設定する
        - RichでConfig Tree(設定ツリー)を出力する
    
    Args:
        cfg (DictConfig): Config Treeを含むDictConfigオブジェクト
    """
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.enforce_tags=True>")
        warnings.filterwarnings("ignore")

    if cfg.extras.get("enforcer_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)

def task_wrapper(task_func: Callable) -> Callable:
    """
    Args:
        task_func ():
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")

            raise ex
        
        finally:
            log.info(f"Output dir: {cfg.paths.output_dir}")

        return metric_dict, object_dict

    return wrap

def get_metric_value(metric_dict: dict[str, Any], metric_name: str | None) -> float | None:
    """
    Args:
        metric_dict:
        metric_name:
    
    Returns:

    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}")
