import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import Callback

class SlackNotificationCallback(Callback):
    def __init__(self, slack_token: str, channel: str):
        super().__init__()
        self.slack_token = slack_token
        self.channel = channel
        self.client = WebClient(self.slack_token)

    # 学習開始時
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        try:
            model_name = getattr(pl_module, "model_name", pl_module.__class__.__name__)
            response = self.client.chat_postMessage(
                channel=self.channel,
                text=f"🚀{model_name}の学習が開始しました🚀",
            )
            print(f"Slack notification sent: {response['message']['text']}")
        except SlackApiError as e:
            print(f"Error sending Slack notification: {e.response['error']}")

    # 学習終了時
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        try:
            model_name = getattr(pl_module, "model_name", pl_module.__class__.__name__)
            response = self.client.chat_postMessage(
                channel=self.channel,
                text=f"🎉{model_name}の学習が終了しました🎉"
            )
            print(f"Slack notification sent: {response['message']['text']}")
        except SlackApiError as e:
            print(f"Error sending Slack notigication: {e.response['error']}")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        try:
            current_epoch = trainer.current_epoch + 1
            model_name = getattr(pl_module, "model_name", pl_module.__class__.__name__)
            response = self.client.chat_postMessage(
                channel=self.channel,
                text=f"✅{model_name}の学習が{current_epoch}epochまで終了しました"
            )
            print(f"Slack notification sent: {response['message']['text']}")
        except SlackApiError as e:
            print(f"Error sending Slack notification: {e.response['error']}")
