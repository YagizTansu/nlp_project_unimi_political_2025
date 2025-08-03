import json
import pandas as pd
import numpy as np
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
emotions_df = pd.read_csv('/home/yagiz/Desktop/nlp_project/turkish_emotions_datasets/emotions_english_turkish.csv')
valid_emotion_ids = set(emotions_df['emotion_id'].tolist())

train_df = pd.read_csv('/home/yagiz/Desktop/nlp_project/turkish_emotions_datasets/go_emotions_english_train.csv')
label_column = 'labels' if 'labels' in train_df.columns else train_df.columns[1]
text_column = train_df.columns[0]

# Filter and process labels
def has_valid_emotion(labels_str):
    if pd.isna(labels_str):
        return False
    try:
        if isinstance(labels_str, (int, float)):
            emotion_ids = [int(labels_str)]
        else:
            emotion_ids = [int(x.strip()) for x in str(labels_str).split(',')]
        return any(eid in valid_emotion_ids for eid in emotion_ids)
    except:
        return False

train_df_filtered = train_df[train_df[label_column].apply(has_valid_emotion)].copy()

# Create label mapping
old_to_new_label_map = {emotion_id: i for i, emotion_id in enumerate(sorted(valid_emotion_ids))}
new_to_old_label_map = {i: emotion_id for emotion_id, i in old_to_new_label_map.items()}

def convert_labels(labels_str):
    if pd.isna(labels_str):
        return []
    try:
        if isinstance(labels_str, (int, float)):
            emotion_ids = [int(labels_str)]
        else:
            emotion_ids = [int(x.strip()) for x in str(labels_str).split(',')]
        return [old_to_new_label_map[eid] for eid in emotion_ids if eid in valid_emotion_ids]
    except:
        return []

train_df_filtered['new_labels'] = train_df_filtered[label_column].apply(convert_labels)
train_df_filtered = train_df_filtered[train_df_filtered['new_labels'].apply(len) > 0].copy()

# Initialize model
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_labels = len(valid_emotion_ids)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="multi_label_classification"
)
model.gradient_checkpointing_enable()

# Prepare data
def create_multi_hot_labels(label_lists, num_classes):
    labels = np.zeros((len(label_lists), num_classes), dtype=np.float32)
    for i, label_list in enumerate(label_lists):
        for label_id in label_list:
            labels[i][label_id] = 1.0
    return labels

train_texts_split, val_texts_split, train_labels_split, val_labels_split = train_test_split(
    train_df_filtered[text_column].tolist(), 
    train_df_filtered['new_labels'].tolist(), 
    test_size=0.2, 
    random_state=42
)

def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")

train_encodings = tokenize_function(train_texts_split)
val_encodings = tokenize_function(val_texts_split)

train_multi_hot_labels = create_multi_hot_labels(train_labels_split, num_labels)
val_multi_hot_labels = create_multi_hot_labels(val_labels_split, num_labels)

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

train_dataset = EmotionDataset(train_encodings, train_multi_hot_labels)
val_dataset = EmotionDataset(val_encodings, val_multi_hot_labels)

# Verify CSV data is loaded correctly instead of saving mappings
print(f"CSV data loaded successfully. Found {len(emotions_df)} emotion entries.")
print(f"Valid emotion IDs: {sorted(list(valid_emotion_ids))}")
print(f"Sample emotions from CSV: {emotions_df.head(3).to_dict('records')}")

# Define compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Apply sigmoid and threshold at 0.5
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
    predictions = (predictions > 0.5).astype(int)
    
    # Calculate metrics
    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')
    precision = precision_score(labels, predictions, average='micro')
    recall = recall_score(labels, predictions, average='micro')
    
    return {
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    warmup_steps=200,
    weight_decay=0.01,
    learning_rate=3e-5,
    fp16=True,
    dataloader_num_workers=2,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=400,
    save_strategy="steps",
    save_steps=400,
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    report_to=None,
    save_total_limit=2,
    eval_accumulation_steps=4
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train the model
print("Starting training...")
trainer.train()

# Final evaluation
print("Final evaluation...")
eval_results = trainer.evaluate()
print("Evaluation Results:")
for key, value in eval_results.items():
    print(f"  {key}: {value:.4f}")

# Save model and tokenizer
output_dir = "./fine_tuned_turkish_bert_emotions"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to: {output_dir}")

# Test prediction function
def predict_emotions(text, threshold=0.5):
    """Predict emotions for a given text"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    
    # Get predicted emotion IDs
    predicted_emotions = [i for i, score in enumerate(predictions) if score > threshold]
    
    return predicted_emotions, predictions

# Test with a sample
sample_text = "Bu gerçekten harika bir gün!"
predicted_emotions, scores = predict_emotions(sample_text)
print(f"\nSample prediction for: '{sample_text}'")
print(f"Predicted emotions: {predicted_emotions}")
print(f"Top 5 scores: {sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]}")

# Test with multiple samples
sample_texts = [
    "Bu gerçekten harika bir gün!",  # Happiness/Joy
    "Bugün kendimi çok üzgün hissediyorum.",  # Sadness
    "Bu haber beni çok kızdırdı!",  # Anger
    "Yarınki sınavdan çok korkuyorum.",  # Fear
    "Bu yemek gerçekten iğrenç.",  # Disgust
    "Bu film beni çok şaşırttı.",  # Surprise
    "Seninle gurur duyuyorum.",  # Pride
    "Yaptığım hatadan dolayı kendimi suçlu hissediyorum.",  # Guilt
    "Ona bu konuda çok kıskançlık duyuyorum.",  # Jealousy
    "Akşamki toplantı beni endişelendiriyor.",  # Anxiety
    "Bu müzik beni huzurlu hissettiriyor."  # Calm
]

print("\nTesting model with multiple samples:")
for sample_text in sample_texts:
    predicted_emotions, scores = predict_emotions(sample_text)
    print(f"\nSample: '{sample_text}'")
    print(f"Predicted emotions: {predicted_emotions}")
    print(f"Top 3 scores: {sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]}")

print("\nTraining completed successfully!")
