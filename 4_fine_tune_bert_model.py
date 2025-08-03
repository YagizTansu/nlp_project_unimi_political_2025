import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer,  AutoModelForSequenceClassification, TrainingArguments,Trainer,EarlyStoppingCallback)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_df = pd.read_csv('turkish_emotions_datasets/go_emotions_turkish_train.csv')
emotions_df = pd.read_csv('turkish_emotions_datasets/emotions_english_turkish.csv')

print(f"Train data shape: {train_df.shape}")
print(f"Emotions data shape: {emotions_df.shape}")

# Clean emotion_id column - handle multiple labels
def parse_emotion_ids(emotion_str):
    """Parse emotion_id string that may contain multiple comma-separated values"""
    if pd.isna(emotion_str):
        return []
    
    # Convert to string and clean
    emotion_str = str(emotion_str).strip()
    
    # Handle quoted strings with commas
    if emotion_str.startswith('"') and emotion_str.endswith('"'):
        emotion_str = emotion_str[1:-1]
    
    # Split by comma and convert to integers
    try:
        emotion_ids = [int(x.strip()) for x in emotion_str.split(',') if x.strip()]
        return emotion_ids
    except:
        # If parsing fails, try single value
        try:
            return [int(emotion_str)]
        except:
            return []

# Apply parsing to emotion_id column
train_df['emotion_ids'] = train_df['emotion_id'].apply(parse_emotion_ids)

# Remove rows with no valid emotion IDs
train_df = train_df[train_df['emotion_ids'].apply(len) > 0].reset_index(drop=True)

print(f"After cleaning: {len(train_df)} rows")

# Group by text and aggregate emotion labels
print("Creating multi-label dataset...")
grouped_data = []

for text, group in tqdm(train_df.groupby('text')):
    # Collect all emotion IDs for this text
    all_emotion_ids = []
    for emotion_list in group['emotion_ids']:
        all_emotion_ids.extend(emotion_list)
    
    # Remove duplicates
    unique_emotions = list(set(all_emotion_ids))
    
    # Create one-hot vector (28 emotions: 0-27)
    label_vector = [0] * 28
    for emotion_id in unique_emotions:
        if 0 <= emotion_id <= 27:  # Ensure valid range
            label_vector[emotion_id] = 1
    
    grouped_data.append({
        'text': text,
        'labels': label_vector
    })

# Convert to DataFrame
df = pd.DataFrame(grouped_data)
print(f"Final dataset size: {len(df)} unique texts")

# Check label distribution
label_counts = np.array([row for row in df['labels']]).sum(axis=0)
print(f"Label distribution (top 10): {sorted(enumerate(label_counts), key=lambda x: x[1], reverse=True)[:10]}")

# Train-validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['labels'].tolist(),
    test_size=0.2,
    random_state=42
)

print(f"Train size: {len(train_texts)}, Validation size: {len(val_texts)}")

# Initialize tokenizer and model
model_name = "xlm-roberta-base"
print(f"Loading tokenizer and model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=28,
    problem_type="multi_label_classification"
)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Tokenize data
def tokenize_function(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,  # 96'dan 64'e düşür - daha az memory
        return_tensors="pt"
    )

print("Tokenizing data...")
train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

# Create datasets
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

train_dataset = EmotionDataset(train_encodings, train_labels)
val_dataset = EmotionDataset(val_encodings, val_labels)

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
    per_device_train_batch_size=8,   # 32'den 8'e düşür
    per_device_eval_batch_size=8,    # 32'den 8'e düşür
    gradient_accumulation_steps=8,   # 2'den 8'e artır (etkili batch size = 8*8 = 64)
    warmup_steps=200,
    weight_decay=0.01,
    learning_rate=3e-5,
    fp16=True,
    dataloader_num_workers=2,        # 4'den 2'ye düşür
    dataloader_pin_memory=False,     # Memory kullanımını azalt
    remove_unused_columns=False,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=400,                  # 200'den 400'e artır (daha az evaluation)
    save_strategy="steps",
    save_steps=400,                  # 200'den 400'e artır
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    report_to=None,
    save_total_limit=2,              # Sadece en iyi 2 checkpoint'i sakla
    eval_accumulation_steps=4        # Evaluation sırasında memory kullanımını azalt
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
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)  # 128'den 64'e düşür
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

print("\nTraining completed successfully!")
