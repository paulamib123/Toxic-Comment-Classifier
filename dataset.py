import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from datasets import load_dataset

class ToxicCommentsDataset(Dataset):
    def __init__(self, dataframe, tokenizer_name='distilbert-base-cased', max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.data = dataframe
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['comment_text']
        label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        labels = row[label_columns]

        # Convert to numeric, handling potential string or mixed type inputs
        labels = labels.apply(lambda x: float(x) if not pd.isna(x) else 0.0)
        label = torch.tensor(labels.values, dtype=torch.float16)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label
        }

def load_huggingface_dataset(dataset_name):
    dataset_train = load_dataset(dataset_name, split="train")
    balanced_test = load_dataset(dataset_name, split="balanced_test")

    train_df = dataset_train.to_pandas()
    test_df = balanced_test.to_pandas()

    # Filter out toxic comments and non - toxic comments
    column_labels = train_df.columns.tolist()[2:]
    train_toxic = train_df[train_df[column_labels].sum(axis=1) > 0]
    train_clean = train_df[train_df[column_labels].sum(axis=1) == 0]

    print(f"No. of Toxic comments: {len(train_toxic)}")
    print(f"No. of Clean comments: {len(train_clean)}")

    # As we have more non - toxic comments so we randomly sample 15000 non -toxic comments to balance the dataset
    # and have equal no. of toxic and non - toxic comments
    balanced_train_df = pd.concat([train_toxic, train_clean.sample(15000)])

    balanced_train_df = balanced_train_df.drop_duplicates()

    balanced_train_df = balanced_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_train_df, test_df