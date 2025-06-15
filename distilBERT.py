from transformers import AutoModel, AdamW
import torch.nn as nn
import torch
import pytorch_lightning as pl
from torchmetrics.functional.classification import multilabel_auroc, multilabel_f1_score, multilabel_recall


class Toxic_Comment_Classifier(pl.LightningModule):

  def __init__(self, config: dict):
    super().__init__()

    self.bert = AutoModel.from_pretrained(config['model_name'], return_dict = True)
    self.dropout = nn.Dropout(0.3)
    self.classifier = nn.Linear(self.bert.config.hidden_size, config['n_labels'])
    self.sigmoid = nn.Sigmoid()
    self.loss_fn = nn.BCEWithLogitsLoss()

  def forward(self, input_ids, attention_mask, labels=None):
      outputs = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
      )
      # Extract the pooled output (use the [CLS] token hidden state)
      pooled_output = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0, :]
      pooled_output = self.dropout(pooled_output)
      logits = self.classifier(pooled_output)

      # Compute loss if labels are provided
      loss = 0
      if labels is not None:
          loss = self.loss_fn(logits, labels)

      probabilities = torch.sigmoid(logits)  # Apply sigmoid for multi-label classification probabilities
      return loss, probabilities


def train_and_validate(optimizer, model, train_data, validation_data, n_epochs, device):
  model = model.to(device)

  for epoch in range(n_epochs):
    train_lossess = []
    model.train()
    for d in train_data:
      optimizer.zero_grad()
    # Move batch data to the GPU
      input_ids = d['input_ids'].to(device)
      attention_mask = d['attention_mask'].to(device)
      labels = d['labels'].to(device)
      loss, output = model(input_ids, attention_mask, labels)
      loss.backward()
      optimizer.step()
      train_lossess.append(loss.item())

    val_lossess = []
    model.eval()
    with torch.no_grad():
      for d in validation_data:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['labels'].to(device)
        loss, output = model(input_ids, attention_mask, labels)
        val_lossess.append(loss.item())

    print(f"Train Loss for epoch {epoch + 1} is {torch.mean(torch.tensor(train_lossess))} & Validation Loss is {torch.mean(torch.tensor(val_lossess))}")
  
def test(model, test_data, device):
  """
  Evaluates and Store the predicted labels in a list for evaluating test data metrics
  """
  model.eval()
  predicted = []
  actual = []

  model.eval()
  with torch.no_grad():
      for d in test_data:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        _, output = model(input_ids, attention_mask)
        output_list = output.tolist()
        for output in output_list:
          predicted.append(output)
        labels = d['labels'].tolist()
        for label in labels:
          actual.append(label)
  return predicted, actual


def evaluate_metrics(predicted, actual, n_labels):
    """
    Evaluate the model on ROC AUC, F1 Score, Recall Score
    """
    print("Metrics for performance Evaluation\n")
    ans = multilabel_auroc(predicted, actual, num_labels=n_labels, average=None, thresholds=None)
    print(f"ROC AUC for each label: {ans}")
    ans = multilabel_auroc(predicted, actual, num_labels=n_labels, average="macro", thresholds=None)
    print(f"ROC AUC macro {ans}")
    ans = multilabel_auroc(predicted, actual, num_labels=n_labels, average="micro", thresholds=None)
    print(f"ROC AUC micro {ans}")
    ans = multilabel_auroc(predicted, actual, num_labels=n_labels, average="weighted", thresholds=None)
    print(f"ROC AUC weighted {ans}")

    ans = multilabel_f1_score(predicted, actual, num_labels=n_labels, average=None)
    print(f"\nF1 score for each label: {ans}")
    ans = multilabel_f1_score(predicted, actual, num_labels=n_labels, average="micro")
    print(f"Micro F1 score {ans}")
    ans = multilabel_f1_score(predicted, actual, num_labels=n_labels, average="macro")
    print(f"Macro F1 score {ans}")
    ans = multilabel_f1_score(predicted, actual, num_labels=n_labels, average="weighted")
    print(f"Weighted F1 score {ans}")

    ans = multilabel_recall(predicted, actual, num_labels=n_labels, average=None)
    print(f"\nRecall for each label: {ans}")
    ans = multilabel_recall(predicted, actual, num_labels=n_labels, average="micro")
    print(f"Micro Recall score {ans}")
    ans = multilabel_recall(predicted, actual, num_labels=n_labels, average="macro")
    print(f"Macro Recall score {ans}")
    ans = multilabel_recall(predicted, actual, num_labels=n_labels, average="weighted")
    print(f"Weighted Recall score {ans}")

def run_distil_bert_toxic_comment_classifier(model, config, train_dataloader, val_dataloader, test_dataloader, device):
    """
    Main function to run distil_bert_toxic_comment_classifier
    """
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    train_and_validate(optimizer=optimizer, model=model, train_data=train_dataloader, validation_data=val_dataloader, n_epochs=config['n_epochs'], device=device)
    print()

    predicted, actual = test(model=model, test_data=test_dataloader, device=device)
    predicted_tensor = torch.tensor(predicted)
    actual_tensor = torch.tensor(actual, dtype=torch.int64)
    evaluate_metrics(predicted=predicted_tensor, actual=actual_tensor, n_labels=config['n_labels'])
  