from typing import Any
import random

import polars as pl
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from transformers import BertJapaneseTokenizer


class PatentDataModule(LightningDataModule):
    def __init__(
        self,
        model_name,
        train_data_path,
        test_data_path,
        train_val_test_split: tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 1

    def setup(self, stage: str | None = None) -> None:
        if not self.data_train and not self.data_val and not self.data_test:
            MODEL_NAME = self.hparams.model_name
            df_train = pl.read_csv(self.hparams.train_data_path)
            df_test = pl.read_csv(self.hparams.test_data_path)

            tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

            G06V30_l_train = []
            G06V30_l_test = []

            for fi_l in df_train["FI"]:
                if "G06V30" in fi_l:
                    encode = 1
                else:
                    encode = 0
                G06V30_l_train.append(encode)

            for fi_l in df_test["FI"]:
                if "G06V30" in fi_l:
                    encode = 1
                else:
                    encode = 0
                G06V30_l_test.append(encode)

            max_length = 256
            train_dataset_for_loader = []
            test_dataset_for_loader = []

            for i in range(len(df_train)):
                text = df_train["summary"][i]
                labels = [G06V30_l_train[i]]
                encoding = tokenizer(
                    text,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True
                )
                encoding['labels'] = labels
                encoding = { k: torch.tensor(v) for k, v in encoding.items() }
                train_dataset_for_loader.append(encoding)

            for i in range(len(df_test)):
                text = df_test["summary"][i]
                labels = [G06V30_l_test[i]]
                encoding = tokenizer(
                    text,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True
                )
                encoding['labels'] = labels
                encoding = { k: torch.tensor(v) for k, v in encoding.items() }
                test_dataset_for_loader.append(encoding)

            # データセットの分割
            random.shuffle(train_dataset_for_loader)
            n = len(train_dataset_for_loader)
            n_train = int(0.7*n)

            self.data_train = train_dataset_for_loader[:n_train]
            self.data_val = train_dataset_for_loader[n_train:]
            self.data_test = test_dataset_for_loader

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            shuffle=False,
        )

    def teardown(self, stage: str | None = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = PatentDataModule()
