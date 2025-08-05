import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
import re
import json

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Text cleaning function
def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove RT (retweet indicators) and common retweet patterns
    text = re.sub(r'\bRT\b\s*@[A-Za-z0-9_]+:?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bRT\b\s*@[A-ZazlA-Z0-9_]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bRT\b', '', text, flags=re.IGNORECASE)
    
    # Remove URLs (including t.co links and other patterns)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Specifically target t.co URLs
    text = re.sub(r'https://t\.co/[a-zA-Z0-9]+', '', text)
    text = re.sub(r'http://t\.co/[a-zA-Z0-9]+', '', text)
    text = re.sub(r't\.co/[a-zA-Z0-9]+', '', text)
    
    # Remove mentions (@username)
    text = re.sub(r'@[A-Za-z0-9_]+:?', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    
    # Remove hashtags but keep the text (convert #example to example)
    text = re.sub(r'#([A-Za-z0-9_]+)', r'\1', text)
    
    # Remove extra whitespace and newlines
    text = ' '.join(text.split())
    
    # Remove special characters but keep Turkish characters and basic punctuation
    text = re.sub(r'[^\w\sğüşıöçĞÜŞİÖÇ.,!?-]', '', text)
    
    # Remove standalone words that are too short (1-2 characters) and common artifacts
    words = text.split()
    filtered_words = [word for word in words if len(word) > 2 or word.lower() in ['ve', 'ya', 'da', 'ki', 'mi', 'mu', 'mü']]
    text = ' '.join(filtered_words)
    
    return text.strip()

# Function to check if tweet is single word or too short
def is_single_word_tweet(text):
    if pd.isna(text):
        return True
    
    cleaned_text = clean_text(text)
    
    if not cleaned_text:
        return True
    
    # Remove punctuation and split into words
    words = re.findall(r'\b\w+\b', cleaned_text)
    
    # If less than 2 meaningful words, consider it single word
    if len(words) < 2:
        return True
    
    # Check for location-only tweets (city name + "dayız/dayım" etc.)
    if len(words) == 1 and (
        cleaned_text.endswith('dayız') or 
        cleaned_text.endswith('dayım') or
        cleaned_text.endswith('deyiz') or
        cleaned_text.endswith('deyim') or
        cleaned_text.endswith('tayız') or
        cleaned_text.endswith('tayım')
    ):
        return True
    
    return False

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_turkish_emotions"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# Load label mappings
with open(f'{model_path}/label_mappings.json', 'r', encoding='utf-8') as f:
    mappings = json.load(f)
    id2label = {int(k): v for k, v in mappings['id2label'].items()}
    label2id = mappings['label2id']
    emotion_labels = mappings['emotion_labels']

print(f"Loaded fine-tuned model from: {model_path}")
print(f"Available emotions: {emotion_labels}")

# Prediction function for single-class classification
def predict_emotion(text, return_top_k=3):
    model.eval()
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return None, []
        
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    # Get top predictions
    top_indices = probabilities.argsort()[-return_top_k:][::-1]
    top_predictions = [(id2label[idx], probabilities[idx]) for idx in top_indices]
    
    # Best prediction
    best_emotion = id2label[probabilities.argmax()]
    
    return best_emotion, top_predictions

# Test predictions
sample_text = "Bu gerçekten harika bir gün!"
predicted_emotion, top_predictions = predict_emotion(sample_text)

print(f"\nSample prediction for: '{sample_text}'")
print(f"Predicted emotion: {predicted_emotion}")
print("Top 3 predictions:")
for emotion, score in top_predictions:
    print(f"  {emotion}: {score:.4f}")

# Batch test
sample_political_tweet_texts = [
    "Bugün çok üzgünüm, ülkemiz için endişeliyim.",
    "Bu seçim sonuçları beni çok mutlu etti!",
    "Hükümetin politikalarını eleştiriyorum, daha iyi bir gelecek istiyorum.",
    "Bu konuda kafam çok karışık, ne yapacağımı bilmiyorum.",
    "Partim için çok heyecanlıyım, yeni projelerimiz var!",
    "Bu tweeti yazarken çok kızgınım, adaletsizliklere karşıyım.",
    "Sevgi dolu bir toplum için çalışmalıyız, birlik olmalıyız.",
    "Bu konuda çok şaşkınım, beklemediğim bir durumla karşılaştım.",
    "Ülkemizin geleceği için umutluyum, gençlerimize güveniyorum.",
    "Bu konuda çok endişeliyim, geleceğimiz tehlikede.",
    "Bu tweeti yazarken çok mutluyum, güzel bir haber aldım.",
    "Bu konuda çok kızgınım, adaletsizliklere karşıyım.",
]

print("\nTesting model with multiple samples:")
for text in sample_political_tweet_texts:
    pred_emotion, top_preds = predict_emotion(text)
    print(f"\nText: {text}")
    print(f"Predicted emotion: {pred_emotion}")
    print("Top 3 predictions:")
    for emotion, score in top_preds:
        print(f"  {emotion}: {score:.4f}")

print("\nTesting completed successfully!")

# Process full_tweets.csv file
print("\nProcessing full_tweets.csv...")

# Load the full tweets dataset
full_tweets_df = pd.read_csv('/home/yagiz/Desktop/nlp_project/1_politican_tweets_combined_data/full_tweets.csv')
print(f"Loaded {len(full_tweets_df)} tweets")

# Function to get emotion prediction for a text
def get_emotion_prediction(text):
    try:
        cleaned_text = clean_text(text)
        if not cleaned_text:
            return "", ""
        
        model.eval()
        inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        
        # Get best prediction
        best_idx = probabilities.argmax()
        best_emotion = id2label[best_idx]
        best_confidence = probabilities[best_idx]
        
        # Get top 3 predictions as comma-separated string
        top_3_indices = probabilities.argsort()[-3:][::-1]
        top_3_emotions = [id2label[idx] for idx in top_3_indices]
        top_3_str = ",".join(top_3_emotions)
        
        return best_emotion, top_3_str
    
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return "", ""

# Apply prediction to each row
text_col = 'Text' if 'Text' in full_tweets_df.columns else full_tweets_df.columns[0]
print(f"Using column '{text_col}' for predictions")

# Clean texts and create new dataframe with only necessary columns
print("Cleaning texts and preparing data...")
cleaned_tweets_df = full_tweets_df[['ID', 'Author', 'party', 'political_side', 'Date', text_col]].copy()

# Add cleaned text column
cleaned_tweets_df['cleaned_text'] = full_tweets_df[text_col].apply(clean_text)

# Filter out single word tweets
print("Filtering out single word tweets...")
initial_count = len(cleaned_tweets_df)
cleaned_tweets_df = cleaned_tweets_df[~cleaned_tweets_df[text_col].apply(is_single_word_tweet)]
filtered_count = len(cleaned_tweets_df)
print(f"Filtered out {initial_count - filtered_count} single word tweets")
print(f"Remaining tweets: {filtered_count}")

# Reset index after filtering
cleaned_tweets_df = cleaned_tweets_df.reset_index(drop=True)

# Process in batches for better performance
batch_size = 100
predicted_emotions = []
top3_emotions = []

print("Processing emotion predictions...")
for i in range(0, len(cleaned_tweets_df), batch_size):
    batch_end = min(i + batch_size, len(cleaned_tweets_df))
    batch_texts = cleaned_tweets_df['cleaned_text'].iloc[i:batch_end]
    
    batch_predictions = []
    batch_top3 = []
    for text in batch_texts:
        emotion, top3 = get_emotion_prediction(text)
        batch_predictions.append(emotion)
        batch_top3.append(top3)
    
    predicted_emotions.extend(batch_predictions)
    top3_emotions.extend(batch_top3)
    
    if (i // batch_size + 1) % 10 == 0:
        print(f"Processed {batch_end}/{len(cleaned_tweets_df)} tweets...")

# Add the emotion prediction columns
cleaned_tweets_df['predicted_emotion'] = predicted_emotions
cleaned_tweets_df['top3_emotions'] = top3_emotions

# Save the cleaned dataset
output_path = '/home/yagiz/Desktop/nlp_project/3_tweets_with_emotions/cleaned_tweets_with_emotions.csv'
cleaned_tweets_df.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to: {output_path}")

# Print emotion distribution
print(f"\nEmotion distribution in the dataset:")
emotion_counts = pd.Series(predicted_emotions).value_counts()
print(emotion_counts)