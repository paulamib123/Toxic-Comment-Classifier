import torch
from dataset import ToxicCommentsDataset, load_huggingface_dataset
from distilBERT import Toxic_Comment_Classifier, run_distil_bert_toxic_comment_classifier
from sklearn.model_selection import train_test_split
import warnings
from torch.utils.data import DataLoader
from baseline import run_baseline_model

warnings.filterwarnings("ignore")

train_df, test_df = load_huggingface_dataset("tcapelle/jigsaw-toxic-comment-classification-challenge")
# split the trainset to create validation set
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

print(f"\nTrain dataset samples: {len(train_df)}")
print(f"Validation dataset samples: {len(val_df)}")
print(f"Test dataset samples: {len(test_df)}\n")

print("--------------------RUNNING BASELINE MODEL---------------------")
# run baseline logistic regression model for toxic comments classification
run_baseline_model(train_df, val_df, test_df)
print()

print("--------------------RUNNING DistilBERT TOXIC COMMENT CLASSIFIER---------------------")
# use gpu if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device is set to: {device}\n")

# hyperparameters for distil_bert_config
distil_bert_config = {
    'model_name': 'distilbert-base-cased',
    'n_labels': 6,
    'batch_size': 32,
    'lr': 2.0e-5,
    'weight_decay': 0.001,
    'n_epochs': 5
}

model = Toxic_Comment_Classifier(distil_bert_config)

train_dataset = ToxicCommentsDataset(train_df)
train_dataloader = DataLoader(train_dataset, batch_size = distil_bert_config['batch_size'], shuffle=True)

val_dataset = ToxicCommentsDataset(val_df)
val_dataloader = DataLoader(val_dataset, batch_size= distil_bert_config['batch_size'])

test_dataset = ToxicCommentsDataset(test_df)
test_dataloader = DataLoader(test_dataset, batch_size= distil_bert_config['batch_size'])

run_distil_bert_toxic_comment_classifier(model=model, config=distil_bert_config, 
                                         train_dataloader=train_dataloader, val_dataloader=val_dataloader, 
                                         test_dataloader=test_dataloader, device=device)
