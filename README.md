# モダンな深層学習モデルの開発環境
PyTorch LightningとHydraを組み合わせて、機械学習プロジェクトの開発効率化を図る。

## 使用技術
uv, MLFlow, PyTorch Lightning, Hydra, mypy, ruff

- config
    - extras:
        ignore_warnings: Pythonのwarningsを無視するかどうか
        enforce_tags: コマンドラインからタグを設定するかどうか
        print_config:　RichでConfig Tree(設定ツリー)を出力するかどうか


```mermaid
sequenceDiagram
    actor user
    user ->> main: cfg: DictConfig
    main ->>+ extras: cfg: DictConfig
    opt if not.get("extras")
        extras -->> user: warning
    end
    opt cfg.extras.get("ignore_warnings")
        extras --> user: pythonのwarningsを無視
    end
    opt cfg.extras.get("enforce_tags")
        extras ->>- enforce_tags: (cfg: DictConfig, save_to_file: bool)
        opt not cfg.get("tags")
            enforce_tags ->> user: タグ情報の入力を促す
            user -->> enforce_tags: tags
        end
        opt save_to_file
            enforce_tags -->> user: tags.log
        end
    end
    opt cfg.extras.get("print_config")
        extras ->>+ print_config_tree: (cfg: DictConfig, resolve: bool, save_to_file: bool)
        print_config_tree -->> print_config_tree: tree: rich.tree.Tree
        print_config_tree -->> user: treeを出力
        opt save_to_file
            print_config_tree -->>- user: config_tree.log
        end
    end
    main ->>+ train: cfg: DictConfig
    train -->> train: datamodule: LightningDataModule
    train -->> train: model: LightningModule
    train ->>+ instantiate_callbacks: cfg["callbacks"]: DictConfig
    instantiate_callbacks -->>- train: callbacks: list[Callback]
    train ->>+ instantiate_loggers: cfg["logger"]: DictConfig
    instantiate_loggers -->>- train: logger: list[Logger]
    train -->> train: trainer: Trainer
    opt if logger
        train ->> log_hyperparameters: (cfg: DictConfig, datamodule, model, callbacks, logger, trainer)
        log_hyperparameters -->> user: log
    end
    opt if cfg.get("train")
        train ->> train: trainer.fit(model, datamodule)
    end
    opt if cfg.get("test")
        opt if ckpt_path == ""
            train -->> user: warning
        end
        train ->> train: trainer.test(model, datamodule, ckpt_path)
    end
    train -->>- main: metric_dict
    main ->>+ get_metric_value: metric_dict
    get_metric_value -->>- main: metric_value
    main -->> user: metric_value
```
