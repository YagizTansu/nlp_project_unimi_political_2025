import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
import re

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Text cleaning function
def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove URLs (including t.co links and other patterns)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Specifically target t.co URLs
    text = re.sub(r'https://t\.co/[a-zA-Z0-9]+', '', text)
    text = re.sub(r'http://t\.co/[a-zA-Z0-9]+', '', text)
    text = re.sub(r't\.co/[a-zA-Z0-9]+', '', text)
    
    # Remove mentions (@username)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    
    # Remove hashtags but keep the text (convert #example to example)
    text = re.sub(r'#([A-Za-z0-9_]+)', r'\1', text)
    
    # Remove extra whitespace and newlines
    text = ' '.join(text.split())
    
    # Remove special characters but keep Turkish characters and basic punctuation
    text = re.sub(r'[^\w\sğüşıöçĞÜŞİÖÇ.,!?-]', '', text)
    
    return text.strip()

# Load emotion definitions
emotions_df = pd.read_csv('/home/yagiz/Desktop/nlp_project/2_turkish_emotions_datasets/emotions_english_turkish.csv')
valid_emotion_ids = set(emotions_df['emotion_id'].tolist())

# Create emotion mappings for display purposes
emotion_id_to_name = dict(zip(emotions_df['emotion_id'], emotions_df['emotion_name_en']))
emotion_id_to_name_tr = dict(zip(emotions_df['emotion_id'], emotions_df['emotion_name_tr']))

# Create proper mapping between emotion IDs and model indices
sorted_emotion_ids = sorted(list(valid_emotion_ids))
emotion_id_to_index = {eid: idx for idx, eid in enumerate(sorted_emotion_ids)}
index_to_emotion_id = {idx: eid for idx, eid in enumerate(sorted_emotion_ids)}

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_turkish_emotions"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

print(f"Loaded fine-tuned model from: {model_path}")

# Prediction function
def predict_emotions(text, threshold=0.1):
    model.eval()
    cleaned_text = clean_text(text)
    if not cleaned_text:
        return [], np.zeros(len(sorted_emotion_ids))
        
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
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

print("\nTesting completed successfully!")

# Process full_tweets.csv file
print("\nProcessing full_tweets.csv...")

# Load the full tweets dataset
full_tweets_df = pd.read_csv('/home/yagiz/Desktop/nlp_project/1_politican_tweets_combined_data/full_tweets.csv')
print(f"Loaded {len(full_tweets_df)} tweets")

# Function to get top 3 emotion IDs for a text
def get_top3_emotions(text):
    try:
        cleaned_text = clean_text(text)
        if not cleaned_text:
            return ""
        
        model.eval()
        inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
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
text_col = 'Text' if 'Text' in full_tweets_df.columns else full_tweets_df.columns[0]
print(f"Using column '{text_col}' for predictions")

# Clean texts and create new dataframe with only necessary columns
print("Cleaning texts and preparing data...")
cleaned_tweets_df = full_tweets_df[['ID', 'Author', 'party', 'political_side', 'Date']].copy()

# Add cleaned text column (replace original Text column)
cleaned_tweets_df['cleaned_text'] = full_tweets_df[text_col].apply(clean_text)

# Process in batches for better performance
batch_size = 100
top3_emotions = []

print("Processing emotion predictions...")
for i in range(0, len(cleaned_tweets_df), batch_size):
    batch_end = min(i + batch_size, len(cleaned_tweets_df))
    batch_texts = cleaned_tweets_df['cleaned_text'].iloc[i:batch_end]
    
    batch_predictions = []
    for text in batch_texts:
        prediction = get_top3_emotions(text)
        batch_predictions.append(prediction)
    
    top3_emotions.extend(batch_predictions)
    
    if (i // batch_size + 1) % 10 == 0:
        print(f"Processed {batch_end}/{len(cleaned_tweets_df)} tweets...")

# Add the emotion prediction column
cleaned_tweets_df['top3_emotion_ids'] = top3_emotions

# Save the cleaned dataset
output_path = '/home/yagiz/Desktop/nlp_project/3_tweets_with_emotions/cleaned_tweets_with_emotions.csv'
cleaned_tweets_df.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to: {output_path}")