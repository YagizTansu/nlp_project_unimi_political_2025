import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    micro_f1 = f1_score(labels, predictions, average='micro')
    return {
        'accuracy': accuracy,
        'micro_f1': micro_f1
    }

def main():
    # Model configuration
    model_name = "dbmdz/bert-base-turkish-uncased"
    output_dir = "./fine_tuned_turkish_emotions"
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('/home/yagiz/Desktop/nlp_project/2_turkish_emotions_datasets/tremo_data.csv')
    
    # Data preprocessing
    print("Preprocessing data...")
    # Remove rows with missing text or emotion
    df = df.dropna(subset=['text', 'emotion'])
    
    # Remove empty or whitespace-only texts
    df = df[df['text'].str.strip().str.len() > 0]
    
    # Filter out 'Ambigious' emotions for cleaner training
    df = df[df['emotion'] != 'Ambigious']
    
    print(f"Dataset shape after cleaning: {df.shape}")
    print(f"Emotion distribution:\n{df['emotion'].value_counts()}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['emotion'])
    
    # Get unique labels and create label mappings
    emotion_labels = label_encoder.classes_
    num_labels = len(emotion_labels)
    
    print(f"Number of emotion classes: {num_labels}")
    print(f"Emotion labels: {emotion_labels}")
    
    # Create label mappings
    id2label = {i: label for i, label in enumerate(emotion_labels)}
    label2id = {label: i for i, label in enumerate(emotion_labels)}
    
    # Split data
    print("Splitting data...")
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    
    # Initialize tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
    val_dataset = EmotionDataset(val_texts, val_labels, tokenizer)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./chekpoints',
        num_train_epochs=8,  # More epochs
        per_device_train_batch_size=32,  # Smaller batch size
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=8,  # Reduced
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=3e-5,  # Lower learning rate
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
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save the model
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label mappings
    import json
    mappings = {
        'id2label': id2label,
        'label2id': label2id,
        'emotion_labels': emotion_labels.tolist()
    }
    
    with open(f'{output_dir}/label_mappings.json', 'w', encoding='utf-8') as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)
    
    print("Training completed!")
    print(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    main()