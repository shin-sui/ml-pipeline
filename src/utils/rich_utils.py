from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Richを使ってDictConfig(設定オブジェクト)の内容をツリー構造として出力する

    :param cfg: Hydraによって構成されるDictConfig
    :param print_order: 出力順の指定
    (default: data -> model -> callbacks -> logger -> trainer -> paths -> extras)
    :param resolve: DictConfigをYAMLで出力する時に補間を行うかどうか(default: False)
    :param save_to_file: Hydraの出力フォルダにエクスポートするかどうか(default: False)
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # 設定項目(field)をprint_orderの順番でqueueに追加する
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field {field}' not found in config. Skipping '{field}' config printing..."
        )

    # queueに設定項目(field)を追加する
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # queueの情報からconfigツリーを作成する
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # configツリーを出力
    rich.print(tree)

    # configツリーを保存
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)

@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """DictConfig(設定オブジェクト)にタグ情報が含まれていない場合、ユーザーにタグの入力を促し設定に追加する。

    :param cfg: Hydraによって構成されるDictConfig
    :param save_to_file: Hydraの出力フォルダにエクスポートするかどうか(default: False)
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)
