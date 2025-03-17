import torch
from transformers import BertModel
from lightning.pytorch import LightningModule

class BertForSequenceClassificationMultiLabel(torch.nn.Module):

  def __init__(self, model_name, num_labels):
    super() .__init__()
    # BertModelのロード
    self.bert = BertModel.from_pretrained(model_name)
    # 線形変換を初期化しておく
    self.linear = torch.nn.Linear(
        self.bert.config.hidden_size, num_labels
    )

  def forward(
      self,
      input_ids=None,
      attention_mask=None,
      token_type_ids=None,
      labels=None     
  ):
    # データを入力しBERTの最終層の出力を得る
    bert_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids)
    last_hidden_state = bert_output.last_hidden_state

    # [PAD]以外のトークンで隠れ状態の平均をとる
    averaged_hedden_state = \
      (last_hidden_state*attention_mask.unsqueeze(-1)).sum(1) \
      / attention_mask.sum(1, keepdim=True)

    # 線形変換
    scores = self.linear(averaged_hedden_state)

    # 出力の形式を整える
    output = {'logits': scores}

    # labelsが入力に含まれていたら、損失を計算し出力する
    if labels is not None:
      loss = torch.nn.BCEWithLogitsLoss() (scores, labels.float())
      output['loss'] = loss

    # 属性でアクセスできるようにする
    output = type('bert_output', (object,), output)

    return output

class BertForSequenceClassificationMultiLabel_pl(LightningModule):

  def __init__(self, model_name, num_labels, lr):
    super() .__init__()
    self.save_hyperparameters()
    self.bert_scml = BertForSequenceClassificationMultiLabel(
        model_name, num_labels=num_labels
    )

  def training_step(self, batch, batch_idx):
    output = self.bert_scml(**batch)
    loss = output.loss
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    output = self.bert_scml(**batch)
    val_loss = output.loss
    self.log('val_loss', val_loss)

  def test_step(self, batch, batch_idx):
    labels = batch.pop('labels')
    output = self.bert_scml(**batch)
    scores = output.logits
    labels_predicted = ( scores > 0 ).int()
    num_correct = ( labels_predicted == labels ).all(-1).sum().item()
    accuracy = num_correct/scores.size(0)
    self.log('accuracy', accuracy)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

if __name__ == "__main__":
    _ = BertForSequenceClassificationMultiLabel_pl(None, None, None)
