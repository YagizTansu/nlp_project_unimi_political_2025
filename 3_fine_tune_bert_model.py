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
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import EarlyStoppingCallback
import sys

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
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

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

def main(model_name="dbmdz/bert-base-turkish-cased"):
    # Model configuration
    output_dir = "./fine_tuned_turkish_emotions"
    
    print(f"Using model: {model_name}")
    
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
    
    # Calculate class weights
    print("Calculating class weights...")
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(df['label']),
        y=df['label']
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {dict(zip(emotion_labels, class_weights))}")
    
    # Split data
    print("Splitting data...")
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
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
        num_train_epochs=5,  # More epochs
        per_device_train_batch_size=8,  # Smaller batch size
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # Reduced
        warmup_ratio=0.1,
        weight_decay=0.02,
        learning_rate=2e-5,  # Lower learning rate
        fp16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,  # More frequent evaluation
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        report_to=None,
        save_total_limit=3,
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]

    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Get predictions for detailed evaluation
    print("Getting predictions for detailed evaluation...")
    pred_output = trainer.predict(val_dataset)
    preds = np.argmax(pred_output.predictions, axis=1)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(val_labels, preds, target_names=emotion_labels))

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
    # Get model name from command line argument or use default
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        print(f"Using custom model: {model_name}")
    else:
        model_name = "dbmdz/bert-base-turkish-cased"
        print(f"Using default model: {model_name}")
    
    main(model_name)