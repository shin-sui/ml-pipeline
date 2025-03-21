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
    # configにextrasがない場合、warning
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # pythonのwarningを非表示にする
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # configにtagがない場合、ユーザーにコマンドラインでの入力を促す
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # Richを使用してconfig treeを表示する
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)

def task_wrapper(task_func: Callable) -> Callable:
    """タスク関数の実行時にエラー処理を制御するためのオプションのデコレーター
    主な用途は以下。
    - タスク関数が例外を出した場合にLoggerが確実にクローズされるようにする（multirun時のエラー防止）
    - 発生した例外の内容を`.log`ファイルとして保存する
    - 実行が失敗したことを示す専用ファイルを`logs/`フォルダ内に作成する

    Args:
        task_func (Callable): ラップしたいタスク関数

    Returns:
        Callable: ラップされたタスク関数
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        # タスク関数を実行
        # 例外が発生した場合、exceptブロックで捕捉
        # exceptionでログを記録し再スロー
        try:
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception:
            log.exception("")
            raise # 単純にraiseすることで元のスタックトレースを維持

        # ログを保存しておくディレクトリのパスをターミナル上で出力
        finally:
            log.info(f"Output dir: {cfg.paths.output_dir}")

        return metric_dict, object_dict

    return wrap

def get_metric_value(metric_dict: dict[str, Any], metric_name: str | None) -> float | None:
    """LightningModuleに記録されたメトリックの値を安全に取得する
    Args:
        metric_dict (dict[str, Any]): メトリックの名前とその値の辞書
        metric_name (str | None): 取得したいメトリックの名前
    
    Returns:
        float | None: メトリックの値
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

    return metric_value
