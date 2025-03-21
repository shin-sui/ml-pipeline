import logging
from typing import Mapping

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only

class RankedLogger(logging.LoggerAdapter):
    """マルチGPU環境に対応したPythonコマンドラインロガー"""
    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Mapping[str, object] | None = None
    ) -> None:
        """Pythonのコマンドラインロガーの初期化

        Args:
            name (str): ロガーの名前(default=__name__)
            rank_zero_only (bool): ログの出力をランク0のプロセスに限定するかを指定する(default=False)
            extra Mapping[str, object] | None: コンテキスト情報を追加する辞書型オブジェクト。詳細は以下を参照
                https://docs.python.org/3/library/logging.html#logging.LoggerAdapter
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: int | None = None, *args, **kwargs) -> None:
        """
        Args:
            level (int): ログレベル。詳細はhttps://github.com/python/cpython/blob/3.13/Lib/logging/__init__.py
            msg (str): 出力するログメッセージ
            rank (int): ログを出力するプロセスのランク。
            args: 追加の位置引数
            kwargs: 追加のキーワード引数
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)
