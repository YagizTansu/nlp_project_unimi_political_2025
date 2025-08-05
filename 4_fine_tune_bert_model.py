import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load emotion definitions
emotions_df = pd.read_csv('/home/yagiz/Desktop/nlp_project/turkish_emotions_datasets/emotions_english_turkish.csv')
valid_emotion_ids = set(emotions_df['emotion_id'].tolist())

# Create emotion mappings for display purposes
emotion_id_to_name = dict(zip(emotions_df['emotion_id'], emotions_df['emotion_name_en']))
emotion_id_to_name_tr = dict(zip(emotions_df['emotion_id'], emotions_df['emotion_name_tr']))

# FIXED: Create proper mapping between emotion IDs and model indices
sorted_emotion_ids = sorted(list(valid_emotion_ids))
emotion_id_to_index = {eid: idx for idx, eid in enumerate(sorted_emotion_ids)}
index_to_emotion_id = {idx: eid for idx, eid in enumerate(sorted_emotion_ids)}

# Load training data
train_df = pd.read_csv('/home/yagiz/Desktop/nlp_project/turkish_emotions_datasets/go_emotions_english_train.csv')
label_column = 'labels' if 'labels' in train_df.columns else train_df.columns[1]
text_column = train_df.columns[0]

def convert_labels(labels_str):
    if pd.isna(labels_str):
        return []
    try:
        if isinstance(labels_str, (int, float)):
            emotion_id = int(labels_str)
            return [emotion_id_to_index[emotion_id]] if emotion_id in emotion_id_to_index else []
        else:
            emotion_ids = [int(x.strip()) for x in str(labels_str).split(',')]
            return [emotion_id_to_index[eid] for eid in emotion_ids if eid in emotion_id_to_index]
    except:
        return []

train_df['label_list'] = train_df[label_column].apply(convert_labels)

# *** BURASI EKLENDİ: Kaldırılan duyguları içeren tüm örnekleri çıkarıyoruz ***

def has_only_valid_emotions(label_list, valid_ids):
    return all([eid in valid_ids for eid in label_list])

train_df = train_df[train_df['label_list'].apply(lambda x: len(x) > 0 and has_only_valid_emotions(x, valid_emotion_ids))].copy()

# Tokenizer and model
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_labels = len(valid_emotion_ids)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="multi_label_classification",
    id2label=index_to_emotion_id,
    label2id=emotion_id_to_index
)
model.gradient_checkpointing_enable()

# Data split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df[text_column].tolist(),
    train_df['label_list'].tolist(),
    test_size=0.2,
    random_state=42
)

def create_multi_hot_labels(label_lists, num_classes):
    labels = np.zeros((len(label_lists), num_classes), dtype=np.float32)
    for i, label_list in enumerate(label_lists):
        for label_id in label_list:
            if label_id < num_classes:
                labels[i][label_id] = 1.0
    return labels

def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

train_labels_multi_hot = create_multi_hot_labels(train_labels, num_labels)
val_labels_multi_hot = create_multi_hot_labels(val_labels, num_labels)

# Dataset class
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = EmotionDataset(train_encodings, train_labels_multi_hot)
val_dataset = EmotionDataset(val_encodings, val_labels_multi_hot)

# Metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    predictions = (predictions > 0.3).astype(int)  # Lower threshold

    # Handle case where no predictions are made
    if predictions.sum() == 0:
        return {
            'micro_f1': 0.0,
            'macro_f1': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }

    micro_f1 = f1_score(labels, predictions, average='micro', zero_division=0)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    precision = precision_score(labels, predictions, average='micro', zero_division=0)
    recall = recall_score(labels, predictions, average='micro', zero_division=0)

    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=16,  # More epochs
    per_device_train_batch_size=32,  # Smaller batch size
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,  # Reduced
    warmup_ratio=0.1,
    weight_decay=0.01,
    learning_rate=7e-5,  # Lower learning rate
    fp16=True,
    dataloader_num_workers=4,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=100,  # More frequent evaluation
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    report_to=None,
    save_total_limit=3,
    eval_accumulation_steps=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train
print("Starting training...")
trainer.train()

# Evaluate
print("Final evaluation...")
eval_results = trainer.evaluate()
print("Evaluation Results:")
for key, value in eval_results.items():
    print(f"  {key}: {value:.4f}")

# Save model
output_dir = "./fine_tuned_turkish_emotions"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to: {output_dir}")