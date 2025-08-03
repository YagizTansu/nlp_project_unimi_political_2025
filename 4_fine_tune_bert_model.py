import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback
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
    num_train_epochs=5,  # More epochs
    per_device_train_batch_size=8,  # Smaller batch size
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,  # Reduced
    warmup_steps=500,  # More warmup
    weight_decay=0.01,
    learning_rate=2e-5,  # Lower learning rate
    fp16=True,
    dataloader_num_workers=2,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=200,  # More frequent evaluation
    save_strategy="steps",
    save_steps=200,
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

# Prediction function
def predict_emotions(text, threshold=0.3):  # Lower threshold
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    predicted_indices = [i for i, score in enumerate(probs) if score > threshold]
    # Convert model indices back to emotion IDs
    predicted_emotion_ids = [index_to_emotion_id[idx] for idx in predicted_indices]
    return predicted_emotion_ids, probs

# Test predictions
sample_text = "Bu gerçekten harika bir gün!"
predicted_emotions, scores = predict_emotions(sample_text)

print(f"\nSample prediction for: '{sample_text}'")
print(f"Predicted emotion IDs: {predicted_emotions}")
print("Predicted emotions:")
for eid in predicted_emotions:
    print(f"  {eid}: {emotion_id_to_name[eid]} ({emotion_id_to_name_tr[eid]})")

top_5 = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 scores (emotion_id, emotion_name, score):")
for model_idx, score in top_5:
    emotion_id = index_to_emotion_id[model_idx]
    print(f"  {emotion_id}: {emotion_id_to_name[emotion_id]} ({emotion_id_to_name_tr[emotion_id]}) - {score:.4f}")

# Batch test
sample_texts = [
    "Bu gerçekten harika bir gün!",
    "Bugün kendimi çok üzgün hissediyorum.",
    "Bu haber beni çok kızdırdı!",
    "Yarınki sınavdan çok korkuyorum.",
    "Bu yemek gerçekten iğrenç.",
    "Bu film beni çok şaşırttı.",
    "Seninle gurur duyuyorum.",
    "Yaptığım hatadan dolayı kendimi suçlu hissediyorum.",
    "Ona bu konuda çok kıskançlık duyuyorum.",
    "Akşamki toplantı beni endişelendiriyor.",
    "Bu müzik beni huzurlu hissettiriyor."
]

print("\nTesting model with multiple samples:")
for text in sample_texts:
    pred_ids, probs = predict_emotions(text)
    print(f"\nText: {text}")
    print(f"Predicted emotion IDs: {pred_ids}")
    if pred_ids:
        print("Predicted emotions:")
        for eid in pred_ids:
            print(f"  {eid}: {emotion_id_to_name[eid]} ({emotion_id_to_name_tr[eid]})")
    
    top_ids = sorted(list(enumerate(probs)), key=lambda x: x[1], reverse=True)[:3]
    print("Top 3 predictions:")
    for model_idx, score in top_ids:
        emotion_id = index_to_emotion_id[model_idx]
        print(f"  {emotion_id}: {emotion_id_to_name[emotion_id]} ({emotion_id_to_name_tr[emotion_id]}) - {score:.4f}")

print("\nTraining and testing completed successfully!")

# Process full_tweets.csv file
print("\nProcessing full_tweets.csv...")

# Load the full tweets dataset
full_tweets_df = pd.read_csv('/home/yagiz/Desktop/nlp_project/politican_tweets_combined_data/full_tweets.csv')
print(f"Loaded {len(full_tweets_df)} tweets")

# Function to get top 3 emotion IDs for a text
def get_top3_emotions(text):
    try:
        if pd.isna(text):
            return ""
        
        # Convert to string and check if empty
        text_str = str(text).strip()
        if text_str == "" or text_str == "nan":
            return ""
        
        model.eval()
        inputs = tokenizer(text_str, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        
        # Get top 3 emotion indices
        top_3_indices = sorted(list(enumerate(probs)), key=lambda x: x[1], reverse=True)[:3]
        # Convert model indices to emotion IDs
        top_3_emotion_ids = [index_to_emotion_id[idx] for idx, _ in top_3_indices]
        # Return as comma-separated string
        return ",".join(map(str, top_3_emotion_ids))
    
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return ""

# Apply prediction to each row (assuming the text column is the first column or named 'text')
text_col = 'text' if 'text' in full_tweets_df.columns else full_tweets_df.columns[0]
print(f"Using column '{text_col}' for predictions")

# Process in batches for better performance
batch_size = 100
top3_emotions = []

for i in range(0, len(full_tweets_df), batch_size):
    batch_end = min(i + batch_size, len(full_tweets_df))
    batch_texts = full_tweets_df[text_col].iloc[i:batch_end]
    
    batch_predictions = []
    for text in batch_texts:
        prediction = get_top3_emotions(text)
        batch_predictions.append(prediction)
    
    top3_emotions.extend(batch_predictions)
    
    if (i // batch_size + 1) % 10 == 0:
        print(f"Processed {batch_end}/{len(full_tweets_df)} tweets...")

# Add the new column
full_tweets_df['top3_emotion_ids'] = top3_emotions

# Save the updated dataset
output_path = '/home/yagiz/Desktop/nlp_project/turkish_emotions_datasets/full_tweets_with_emotions.csv'
full_tweets_df.to_csv(output_path, index=False)
print(f"Updated dataset saved to: {output_path}")

# Show some sample predictions
print("\nSample predictions from full_tweets.csv:")
for i in range(min(5, len(full_tweets_df))):
    text = str(full_tweets_df[text_col].iloc[i])  # Convert to string
    emotions = full_tweets_df['top3_emotion_ids'].iloc[i]
    print(f"\nText: {text[:100]}...")
    print(f"Top 3 emotion IDs: {emotions}")
    
    if emotions:
        emotion_ids = [int(x) for x in emotions.split(',')]
        print("Emotions:")
        for eid in emotion_ids:
            if eid in emotion_id_to_name:
                print(f"  {eid}: {emotion_id_to_name[eid]} ({emotion_id_to_name_tr[eid]})")

print(f"\nCompleted! Added 'top3_emotion_ids' column to {len(full_tweets_df)} tweets.")
