import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
)
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score
from transformers import EarlyStoppingCallback
from datasets import Dataset, DatasetDict  # Hugging Face datasets importu
import warnings
warnings.filterwarnings('ignore')

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    micro_f1 = f1_score(labels, predictions, average='micro')
    return {
        'accuracy': accuracy,
        'micro_f1': micro_f1
    }

def main(model_name="dbmdz/bert-base-turkish-cased"):
    # Model configuration
    output_dir = "./fine_tuned_model"
    
    print(f"Using model: {model_name}")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('/home/yagiz/Desktop/nlp_project/data_processed//tremo_data.csv')
    
    # Data preprocessing
    print("Preprocessing data...")
    df = df.dropna(subset=['text', 'emotion'])
    df = df[df['text'].str.strip().str.len() > 0]
    print(f"Dataset shape after cleaning: {df.shape}")
    print(f"Emotion distribution:\n{df['emotion'].value_counts()}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['emotion'])
    emotion_labels = label_encoder.classes_
    num_labels = len(emotion_labels)

    print(f"Number of emotion classes: {num_labels}")
    print(f"Emotion labels: {emotion_labels}")
    
    id2label = {i: label for i, label in enumerate(emotion_labels)}
    label2id = {label: i for i, label in enumerate(emotion_labels)}
    
    # Split data (now using Hugging Face datasets)
    print("Splitting data...")
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
    
    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
    dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    print(f"Training samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")

    # Initialize tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"],truncation=True,padding='max_length',max_length=512)
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Set label field for Trainer
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./checkpoints',
        num_train_epochs=5,  
        gradient_accumulation_steps=4,  
        warmup_ratio=0.1,
        weight_decay=0.02,
        learning_rate=2e-5,  
        dataloader_num_workers=4,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        save_total_limit=3,
    )
    
    # Initialize trainer (use default Trainer)
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
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
    pred_output = trainer.predict(tokenized_datasets["validation"])
    preds = np.argmax(pred_output.predictions, axis=1)
    val_labels = val_df['label'].tolist()
    print("\nClassification Report:")
    print(classification_report(val_labels, preds, target_names=emotion_labels))

    # Save the model
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label mappings
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