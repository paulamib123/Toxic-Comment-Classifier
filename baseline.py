import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, hamming_loss, roc_auc_score

label_columns = [
        'toxic', 'severe_toxic', 'obscene', 
        'threat', 'insult', 'identity_hate'
]

def preprocess_text(text):
    """
    Preprocess input text
    """
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    return text

def prepare_multi_label_data(dataframes):
    """
    Prepare multi-label data from train, val, test dataframes
    """

    processed_data = {}
    for split in ['train', 'val', 'test']:
        # Preprocess text
        X = dataframes[split]['comment_text'].apply(preprocess_text)
        
        # Extract multi-label targets
        y = dataframes[split][label_columns].values
        
        processed_data[split] = (X, y)
    
    return processed_data

def create_logistic_regression_pipeline():
    """
    Create a pipeline for multi-label Logistic Regression
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1,2),  # Unigrams and bigrams
            max_features=10000, 
            stop_words='english'  # Remove English stop words
        )),
        ('classifier', MultiOutputClassifier(LogisticRegression(
            multi_class='ovr',  # One-vs-Rest strategy
            max_iter=1000,
            C=1.0,
            random_state=42
        )))
    ])
    
    return pipeline

def train_and_evaluate_model(processed_data):
    """
    Train and evaluate multi-label Logistic Regression model
    """

    X_train, y_train = processed_data['train']
    X_val, y_val = processed_data['val']
    X_test, y_test = processed_data['test']
    
    pipeline = create_logistic_regression_pipeline()
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    # Evaluation metrics
    metrics = {
        'classification_report': classification_report(
            y_test, y_pred, target_names=label_columns
        ),
        'roc_auc': roc_auc_score(y_test, y_pred),
        'micro_f1': f1_score(y_test, y_pred, average='micro'),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'hamming_loss': hamming_loss(y_test, y_pred)
    }
    
    return {
        'model': pipeline,
        'metrics': metrics
    }

def get_actual_and_predicted_labels(model, X, y_true):
    """
    Get actual and predicted labels
    """
    X_processed = X.apply(preprocess_text)
    
    y_pred = model.predict(X_processed)
    
    return y_true, y_pred

def run_baseline_model(train_df, val_df, test_df):
    dataframes = {
    'train': train_df,
    'val': val_df,
    'test': test_df
    }

    processed_data = prepare_multi_label_data(dataframes)

    result = train_and_evaluate_model(processed_data)

    print("Model Performance Metrics:")
    for metric_name, metric_value in result['metrics'].items():
        print(f"{metric_name}:\n{metric_value}")